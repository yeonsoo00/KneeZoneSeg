# ZS macOS application

`ZS.app` is an ARM64 Apple-Silicon application built with PyInstaller. The app
contains Python, PyTorch, PyQt6, PSD support, the UI icon, and the model
checkpoint. End users do not install Python or any packages.

PyInstaller builds are operating-system specific. Build this application on a
Mac with an Apple M-series processor, ideally using the oldest macOS version
used by the target machines.

## Build ZS.app on an Apple-Silicon Mac

Install the Xcode command-line tools once:

```bash
xcode-select --install
```

Use a native ARM64 Python 3.11 environment. The following commands use `venv`:

```bash
cd KneeZoneSeg
python3.11 -m venv .venv-app
source .venv-app/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-app.txt
```

Place the model at the following path. The model is bundled into the app and is
not required separately on end-user machines:

```text
KneeZoneSeg/ckpt/best_model_ch48_super.pth
```

Then build:

```bash
chmod +x build_macos_app.sh
./build_macos_app.sh
```

The build script runs an import and checkpoint self-test against the completed
bundle. It stops instead of installing a broken application if that test fails.

The result is:

```text
KneeZoneSeg/dist/ZS.app
```

Test the bundle on the build Mac:

```bash
open dist/ZS.app
```

With blank UI fields, the bundled checkpoint is used automatically and output
is written to `~/KneeZoneSeg/Prediction`. ZS automatically uses the Apple MPS
device when available.

## Install on another Apple-Silicon Mac

For a quick internal installation:

1. Compress the app on the build Mac with `ditto`, which preserves bundle
   metadata:

   ```bash
   ditto -c -k --keepParent dist/ZS.app ZS-macos-arm64.zip
   ```

2. Copy `ZS-macos-arm64.zip` to the target Mac and extract it.
3. Drag `ZS.app` into `/Applications`.
4. Open `ZS` from Applications. For an ad-hoc-signed internal build, macOS may
   require Control-clicking the app and selecting **Open** on first launch.

No Python environment, checkpoint file, or command-line setup is needed on the
target Mac.

## Replace an older ZS build

Do not merge a new bundle into an existing `/Applications/ZS.app`, because
obsolete packaged libraries can remain inside it. Quit ZS, move the old app to
Trash, and then copy the newly built `dist/ZS.app` into `/Applications`.

You can verify the new bundle before copying it:

```bash
dist/ZS.app/Contents/MacOS/ZS --self-test
```

The command must report `ZS self-test passed`. If Finder blocks an internally
built, ad-hoc-signed copy, Control-click **ZS**, choose **Open**, and confirm the
prompt. Developer-ID-signed and notarized releases do not require this step.

## Signed distribution

PyInstaller ad-hoc signs the application when no certificate is supplied. For
normal distribution across multiple Macs, use an Apple Developer ID
Application certificate and notarize the result.

List available signing identities:

```bash
security find-identity -v -p codesigning
```

Build with a Developer ID identity:

```bash
export ZS_CODESIGN_IDENTITY="Developer ID Application: Organization Name (TEAMID)"
./build_macos_app.sh
```

Submit the signed archive using your organization's Apple notarization
workflow, staple the successful notarization ticket to `dist/ZS.app`, and then
create the final ZIP with `ditto`. A signed and notarized build opens normally
on target Macs without the first-launch exception.

## Files used by the build

- `zs_app.py`: dual GUI/inference application entry point.
- `ZS.spec`: ARM64 PyInstaller bundle definition.
- `build_macos_app.sh`: icon conversion and repeatable build command.
- `requirements-macos-app.txt`: isolated app build dependencies.
- `UI/ZS_icon.png`: source application icon.
