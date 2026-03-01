# app_v2.spec
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

project_dir = os.getcwd()

# --- Collect dynamic packages (data + binaries + hidden imports) ---
ey_datas, ey_bins, ey_hidden = collect_all("eyetrax")
mp_datas, mp_bins, mp_hidden = collect_all("mediapipe")
sk_datas, sk_bins, sk_hidden = collect_all("sklearn")
sp_datas, sp_bins, sp_hidden = collect_all("scipy")
cv_datas, cv_bins, cv_hidden = collect_all("cv2")
pg_datas, pg_bins, pg_hidden = collect_all("pygame")

datas = []
binaries = []
hiddenimports = []

datas += ey_datas + mp_datas + sk_datas + sp_datas + cv_datas + pg_datas
binaries += ey_bins + mp_bins + sk_bins + sp_bins + cv_bins + pg_bins
hiddenimports += ey_hidden + mp_hidden + sk_hidden + sp_hidden + cv_hidden + pg_hidden

# --- Catch lazy/dynamic imports ---
hiddenimports += collect_submodules("eyetrax")
hiddenimports += collect_submodules("mediapipe")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("scipy")

# --- Known dynamic module referenced by your errors ---
hiddenimports += [
    "eyetrax.gaze",
    "mediapipe.tasks.c",
]

# --- Your bundled audio files ---
datas += [
    (os.path.join(project_dir, "sound.mp3"), "."),
    (os.path.join(project_dir, "cheer.mp3"), "."),
]

a = Analysis(
    ["app_v2.py"],
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- Build an EXE (no console window) ---
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="app_v2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # <- no cmd window
)

# --- ONEDIR output (recommended for mediapipe/sklearn/scipy) ---
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="app_v2",
)