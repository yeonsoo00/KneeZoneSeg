# Five-zone analysis interface

Python 3.11 is recommended. Install the complete inference, GUI, and notebook
environment from the `KneeZoneSeg` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r environment.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
For an NVIDIA-specific PyTorch build, install PyTorch using the command from
<https://pytorch.org/get-started/locally/> before installing `environment.txt`.

Place the final checkpoint at `KneeZoneSeg/ckpt/best_model_ch48_deepsuper.pth`,
or select the existing checkpoint in the GUI. When this repository is kept
inside the ZoneSeg workspace, the GUI automatically finds
`SemiSeparatedUnetFusion/ckpt/best_model_ch48_deepsuper.pth`.

Launch the application from the `KneeZoneSeg/UI` directory:

```bash
python3 analyze_qt6.py
```

1. Choose one layered PSD/PSB/PXD file, or a directory. Directory input is
   recursive and may mix layered files with sample folders of JPG/PNG signals.
   Each layered document is one sample. Its layers must start with `1_` through `9_`:
   1=Mineral, 2=AC, 3=Calcein, 4=TRAP, 5=DAPI, 6=AP, 7=EdU, 8=CFO,
   and 9=SFO. Text after the underscore is ignored.
   Each JPG/PNG sample folder must contain those stains using either their
   stain names (`mineral.png`, `ac.jpg`, and so on) or numbered names. EdU is
   optional and becomes a black placeholder when absent.
2. Confirm the prediction output directory and model checkpoint. The default
   output is `KneeZoneSeg/Prediction`.
3. Click **Predict**. Converted PNG inputs are retained in
   `<prediction output>/_formatted_input`. Nested input paths are preserved for
   both formatted signals and prediction folders, preventing duplicate sample
   names in different directories from overwriting one another.
4. Choose a sample and analysis background (SFO by default), then click
   **Analyze five zones**.
5. Use **Export measurements to CSV** and **Save zone visualizations** to save
   the table, binary zone masks, individual measurement images, and the total
   colored contour visualization.

The active GUI is implemented in `zone_gui.py`; `analyze_qt6.py` remains the
compatible launcher name.

## Zone definitions

| Zone | Predicted boundaries |
|---|---|
| Germinal | `9a`–`6b` |
| Columnar | `6a`–`6d` |
| Pre-hypertrophic | `6c`–`1d` |
| Hypertrophic | `1c`–`9d` |
| GP | `9a`–`9d` |

Each zone is filled column-by-column between the interpolated centerlines of
its two boundary masks. The analysis uses their common horizontal support and
automatically removes 5% from the left edge and 5% from the right edge. There
is no manual range/reset control and no pairwise gap analysis.

## Measurements and exports

The table and CSV contain zone area, analyzed width, mean/median/standard
deviation/minimum/maximum thickness, and mean RGB/grayscale background signal.
Pixel values and physical units are both exported. The default calibration is
3.45 µm/pixel and can be changed in the GUI before analysis.

**Save zone visualizations** writes:

- the five binary zone masks;
- one measurement visualization per zone, with a thin red contour and no
  measurement or edge guide lines;
- `All_zones_contour_overlay.png`, containing all five colored zones and their
  contour outlines;
- `zone_measurements.csv`.

The same presentation workflow is available in `test.ipynb`. Its final cell is
named `# With contour outline — save the total five-zone visualization` and
saves the combined contour image.
