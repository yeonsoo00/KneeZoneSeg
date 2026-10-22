# Five-zone analysis interface

Install the GUI dependencies in the same Python environment used for inference:

```bash
python3 -m pip install -r requirements-macos-gui.txt
```

Place the final checkpoint at `KneeZoneSeg/ckpt/best_model_ch48_deepsuper.pth`,
or select the existing checkpoint in the GUI. When this repository is kept
inside the ZoneSeg workspace, the GUI automatically finds
`SemiSeparatedUnetFusion/ckpt/best_model_ch48_deepsuper.pth`.

Launch the application from this directory:

```bash
python3 analyze_qt6.py
```

1. Choose one layered PSD/PSB file, or a directory containing layered PSDs.
   Each PSD is one sample. Its layers must start with `1_` through `9_`:
   1=Mineral, 2=AC, 3=Calcein, 4=TRAP, 5=DAPI, 6=AP, 7=EdU, 8=CFO,
   and 9=SFO. Text after the underscore is ignored.
2. Confirm the prediction output directory and model checkpoint.
3. Click **Predict**. Converted PNG inputs are retained in
   `<prediction output>/_formatted_input`.
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
- one measurement visualization per zone, with red contour, blue analyzed
  edges, and sampled green thickness lines;
- `All_zones_contour_overlay.png`, containing all five colored zones and their
  contour outlines;
- `zone_measurements.csv`.

The same presentation workflow is available in `test.ipynb`. Its final cell is
named `# With contour outline — save the total five-zone visualization` and
saves the combined contour image.
