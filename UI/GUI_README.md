# Zone analysis interface (macOS)

Install the GUI dependencies in the same Python environment used for inference:

```bash
python3 -m pip install -r requirements-macos-gui.txt
```

Launch the application:

```bash
python3 analyze_qt6.py
```

1. Choose any one of the numbered PSD files, or choose their directory.
2. Confirm the prediction output directory and model checkpoint.
3. Click **Predict**. Converted PNG inputs are retained in
   `<prediction output>/_formatted_input`.
4. Choose a sample and two predicted stain zones, select background signals,
   and click **Analyze gap**.
5. Click **Save image** to export the displayed overlay.

## Measurement definition

All predicted subzones with the selected stain's leading digit are unioned. For
example, selecting Calcein unions `3a.png` through `3d.png`. The zone with the
smaller median vertical coordinate is treated as the upper zone. In every image
column occupied by both selected zones, thickness is the number of pixels between
the bottommost upper-zone pixel and topmost lower-zone pixel. This shared-column
intersection constrains the measurement to the shorter horizontal support.

The reported mean removes the lowest 10% and highest 10% of column thicknesses.
Standard deviation and maximum are calculated from all shared-column thicknesses.
Measurements are reported in pixels because the source PSDs do not supply a
physical pixel-size calibration to the application.
