#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ICON_SOURCE="$PROJECT_DIR/UI/ZS_icon.png"
ICON_OUTPUT_DIR="$PROJECT_DIR/build_macos"
ICONSET_DIR="$ICON_OUTPUT_DIR/ZS.iconset"
ICON_OUTPUT="$ICON_OUTPUT_DIR/ZS.icns"
LOCAL_CHECKPOINT="$PROJECT_DIR/ckpt/best_model_ch48_super.pth"
WORKSPACE_CHECKPOINT="$PROJECT_DIR/../SemiSeparatedUnetFusion/ckpt/groupnorm_deepsuper/best_model_ch48_super.pth"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: ZS.app must be built on macOS; PyInstaller cannot cross-build it." >&2
    exit 1
fi

PYTHON_MACHINE="$(python3 -c 'import platform; print(platform.machine())')"
if [[ "$PYTHON_MACHINE" != "arm64" ]]; then
    echo "Error: use a native Apple-Silicon arm64 Python, not a Rosetta x86_64 Python." >&2
    exit 1
fi

if [[ ! -f "$ICON_SOURCE" ]]; then
    echo "Error: missing icon $ICON_SOURCE" >&2
    exit 1
fi
if [[ ! -f "$LOCAL_CHECKPOINT" && ! -f "$WORKSPACE_CHECKPOINT" ]]; then
    echo "Error: copy best_model_ch48_super.pth to $LOCAL_CHECKPOINT" >&2
    exit 1
fi

python3 -c 'import PyInstaller, PyQt6, torch' >/dev/null || {
    echo "Error: install requirements-macos-app.txt in the active environment first." >&2
    exit 1
}

rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

make_icon() {
    local pixels="$1"
    local filename="$2"
    sips -z "$pixels" "$pixels" "$ICON_SOURCE" --out "$ICONSET_DIR/$filename" >/dev/null
}

make_icon 16 icon_16x16.png
make_icon 32 icon_16x16@2x.png
make_icon 32 icon_32x32.png
make_icon 64 icon_32x32@2x.png
make_icon 128 icon_128x128.png
make_icon 256 icon_128x128@2x.png
make_icon 256 icon_256x256.png
make_icon 512 icon_256x256@2x.png
make_icon 512 icon_512x512.png
make_icon 1024 icon_512x512@2x.png
iconutil -c icns "$ICONSET_DIR" -o "$ICON_OUTPUT"

cd "$PROJECT_DIR"
python3 -m PyInstaller --noconfirm --clean ZS.spec

echo
echo "Built $PROJECT_DIR/dist/ZS.app"
echo "Test it with: open '$PROJECT_DIR/dist/ZS.app'"
