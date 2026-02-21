# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Aura.py'],
    pathex=[],
    binaries=[],
    datas=[('yolo', 'yolo'), ('whitelist', 'whitelist'), ('daily_log.json', '.'), ('user_profile.json', '.'), ('event_stream.json', '.'), ('memcells.json', '.'), ('conversations.json', '.')],
    hiddenimports=['edge_tts', 'google.genai', 'comtypes'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tensorboard', 'tensorflow', 'keras', 'tf_keras', 'deepface', 'retinaface', 'torch.utils.tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Aura',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Aura',
)
