const btnStart = document.getElementById("btnStart");
const btnEndTurn = document.getElementById("btnEndTurn");
const btnStop = document.getElementById("btnStop");
const stateValue = document.getElementById("stateValue");
const hint = document.getElementById("hint");
const transcript = document.getElementById("transcript");
const emotionBox = document.getElementById("emotionBox");
const preview = document.getElementById("preview");
const userIdInput = document.getElementById("userId");

let ws = null;
let mediaStream = null;
let audioContext = null;
let playbackContext = null;
let playbackScheduleTime = 0;
let processor = null;
let micSource = null;
let muteNode = null;
let videoTimer = null;
let canvas = null;
let canvasCtx = null;
let turnActive = false;
let turnStartMs = 0;
let lastVoiceMs = 0;
let autoEndSent = false;
let vadSilenceMs = 1300;
let vadMinTurnMs = 900;
let vadThreshold = 0.015;

function toWsUrl(relativePath) {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}${relativePath}`;
}

function setState(state, detail = "") {
  stateValue.textContent = state;
  if (detail) hint.textContent = detail;
}

function appendTranscript(line) {
  transcript.textContent += `${line}\n`;
  transcript.scrollTop = transcript.scrollHeight;
}

function b64FromInt16(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const slice = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

function int16FromB64(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

function floatToInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const v = Math.max(-1, Math.min(1, float32[i]));
    out[i] = v < 0 ? v * 32768 : v * 32767;
  }
  return out;
}

function computeRms(float32) {
  if (!float32.length) return 0;
  let sum = 0;
  for (let i = 0; i < float32.length; i += 1) {
    sum += float32[i] * float32[i];
  }
  return Math.sqrt(sum / float32.length);
}

function int16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) out[i] = int16[i] / 32768;
  return out;
}

async function startMedia() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: { width: 640, height: 360, facingMode: "user" },
  });
  preview.srcObject = mediaStream;

  audioContext = new AudioContext();
  playbackContext = audioContext;
  playbackScheduleTime = playbackContext.currentTime;

  micSource = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  muteNode = audioContext.createGain();
  muteNode.gain.value = 0;

  processor.onaudioprocess = (event) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const samples = event.inputBuffer.getChannelData(0);
    const int16 = floatToInt16(samples);
    const now = Date.now();
    const rms = computeRms(samples);

    if (rms >= vadThreshold) {
      if (!turnActive) {
        turnActive = true;
        turnStartMs = now;
      }
      lastVoiceMs = now;
      autoEndSent = false;
    }

    if (
      turnActive &&
      !autoEndSent &&
      now - turnStartMs >= vadMinTurnMs &&
      now - lastVoiceMs >= vadSilenceMs
    ) {
      ws.send(JSON.stringify({ type: "control", action: "end_turn" }));
      autoEndSent = true;
      turnActive = false;
      setState("thinking", "silence detected, ending turn");
    }

    ws.send(
      JSON.stringify({
        type: "audio_chunk",
        ts: Date.now(),
        sample_rate: audioContext.sampleRate,
        pcm16_base64: b64FromInt16(int16),
      })
    );
  };

  micSource.connect(processor);
  processor.connect(muteNode);
  muteNode.connect(audioContext.destination);

  canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 240;
  canvasCtx = canvas.getContext("2d");

  videoTimer = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN || !canvasCtx) return;
    canvasCtx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    const b64 = dataUrl.split(",")[1];
    ws.send(
      JSON.stringify({
        type: "video_frame",
        ts: Date.now(),
        width: canvas.width,
        height: canvas.height,
        jpeg_base64: b64,
      })
    );
  }, 500);
}

function stopMedia() {
  if (videoTimer) {
    clearInterval(videoTimer);
    videoTimer = null;
  }
  if (processor) processor.disconnect();
  if (micSource) micSource.disconnect();
  if (muteNode) muteNode.disconnect();

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  preview.srcObject = null;
  turnActive = false;
  autoEndSent = false;
}

function playAssistantChunk(pcm16Base64, sampleRate) {
  if (!playbackContext) return;
  const int16 = int16FromB64(pcm16Base64);
  const float32 = int16ToFloat32(int16);
  const buffer = playbackContext.createBuffer(1, float32.length, sampleRate);
  buffer.copyToChannel(float32, 0);

  const src = playbackContext.createBufferSource();
  src.buffer = buffer;
  src.connect(playbackContext.destination);

  const now = playbackContext.currentTime;
  if (playbackScheduleTime < now) playbackScheduleTime = now + 0.01;
  src.start(playbackScheduleTime);
  playbackScheduleTime += buffer.duration;
}

async function startSession() {
  const userId = userIdInput.value.trim() || "demo-user";
  const res = await fetch("/api/v1/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!res.ok) {
    throw new Error(`start session failed: ${res.status}`);
  }
  const data = await res.json();
  const wsUrl = toWsUrl(data.ws_url);

  ws = new WebSocket(wsUrl);
  ws.onopen = async () => {
    await startMedia();
    setState("connected", "session connected");
    btnStart.disabled = true;
    btnEndTurn.disabled = false;
    btnStop.disabled = false;
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "status") {
      setState(msg.state, msg.detail || "");
      if (msg.state === "listening" || msg.state === "idle") {
        autoEndSent = false;
      }
      return;
    }
    if (msg.type === "assistant_text_delta") {
      appendTranscript(msg.text);
      return;
    }
    if (msg.type === "assistant_audio_chunk") {
      playAssistantChunk(msg.pcm16_base64, msg.sample_rate);
      return;
    }
    if (msg.type === "emotion_update") {
      const faces = msg.faces || [];
      if (faces.length === 0) {
        emotionBox.textContent = "No face";
      } else {
        emotionBox.textContent = JSON.stringify(faces, null, 2);
      }
      return;
    }
    if (msg.type === "error") {
      appendTranscript(`[ERROR] ${msg.code}: ${msg.message}`);
    }
  };

  ws.onerror = () => {
    setState("error", "websocket error");
  };

  ws.onclose = () => {
    stopMedia();
    setState("disconnected", "session disconnected");
    btnStart.disabled = false;
    btnEndTurn.disabled = true;
    btnStop.disabled = true;
  };
}

function stopSession() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "control", action: "stop" }));
    ws.close();
  }
  ws = null;
  stopMedia();
}

btnStart.addEventListener("click", async () => {
  try {
    transcript.textContent = "";
    emotionBox.textContent = "-";
    await startSession();
  } catch (err) {
    setState("error", String(err));
  }
});

btnEndTurn.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  autoEndSent = true;
  turnActive = false;
  ws.send(JSON.stringify({ type: "control", action: "end_turn" }));
});

btnStop.addEventListener("click", () => {
  stopSession();
});
