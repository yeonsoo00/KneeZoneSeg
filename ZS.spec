# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


PROJECT_DIR = Path(SPECPATH)
ICON_PNG = PROJECT_DIR / "UI" / "ZS_icon.png"
ICON_ICNS = PROJECT_DIR / "build_macos" / "ZS.icns"
CHECKPOINT_CANDIDATES = (
    PROJECT_DIR / "ckpt" / "best_model_ch48_super.pth",
    PROJECT_DIR.parent
    / "SemiSeparatedUnetFusion"
    / "ckpt"
    / "groupnorm_deepsuper"
    / "best_model_ch48_super.pth",
)
CHECKPOINT = next((path for path in CHECKPOINT_CANDIDATES if path.is_file()), None)

if CHECKPOINT is None:
    raise SystemExit(
        "Missing ckpt/best_model_ch48_super.pth. Copy the checkpoint into "
        "KneeZoneSeg/ckpt before building ZS.app."
    )
if not ICON_PNG.is_file():
    raise SystemExit(f"Missing application icon: {ICON_PNG}")
if not ICON_ICNS.is_file():
    raise SystemExit(
        f"Missing generated macOS icon: {ICON_ICNS}. Run build_macos_app.sh."
    )

hidden_imports = collect_submodules("psd_tools")
hidden_imports.extend(["cv2", "scipy.ndimage"])

analysis = Analysis(
    [str(PROJECT_DIR / "zs_app.py")],
    pathex=[str(PROJECT_DIR), str(PROJECT_DIR / "UI")],
    binaries=[],
    datas=[
        (str(CHECKPOINT), "ckpt"),
        (str(ICON_PNG), "UI"),
        *collect_data_files("psd_tools"),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "jupyterlab", "matplotlib", "notebook", "pandas"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="ZS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=os.environ.get("ZS_CODESIGN_IDENTITY") or None,
    entitlements_file=None,
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ZS",
)
application = BUNDLE(
    collection,
    name="ZS.app",
    icon=str(ICON_ICNS),
    bundle_identifier="org.kneezoneseg.zs",
    version="1.0.0",
    info_plist={
        "CFBundleDisplayName": "ZS",
        "CFBundleName": "ZS",
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.medical",
    },
)
