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

Place the default checkpoint at `KneeZoneSeg/ckpt/best_model_ch48_super.pth`.
The checkpoint field may be left blank; inside the full ZoneSeg workspace, the
GUI also automatically finds
`SemiSeparatedUnetFusion/ckpt/groupnorm_deepsuper/best_model_ch48_super.pth`.

Launch the application from the `KneeZoneSeg/UI` directory:

```bash
python3 analyze_qt6.py
```

1. Use the single input **Browse** button to choose one PSD/PSB/PXD file, one
   PNG/JPG signal from a raster sample folder, or a directory. Choosing a
   raster file uses its containing signal folder as the sample. Directory input
   is recursive and may mix layered files with sample folders of JPG/PNG signals.
   Each layered document is one sample. Its layers must start with `1_` through `9_`:
   1=Mineral, 2=AC, 3=Calcein, 4=TRAP, 5=DAPI, 6=AP, 7=EdU, 8=CFO,
   and 9=SFO. Text after the underscore is ignored.
   Each JPG/PNG sample folder must contain those stains using either their
   stain names (`mineral.png`, `ac.jpg`, and so on) or numbered names. EdU is
   optional and becomes a black placeholder when absent.
2. The prediction output and checkpoint fields are optional. Leave them blank
   to use `KneeZoneSeg/Prediction` and `ckpt/best_model_ch48_super.pth`.
3. Click **Predict**. Converted PNG inputs are retained in
   `<prediction output>/_formatted_input`. Nested input paths are preserved for
   both formatted signals and prediction folders, preventing duplicate sample
   names in different directories from overwriting one another.
4. Choose a sample and click **Analyze five zones**. SFO's native image size is
   used as the common measurement grid; visualization backgrounds use the fixed
   recipes below.
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

Each boundary centerline is the median y-coordinate of its mask in each image
column; missing columns are linearly interpolated. Each zone is filled
column-by-column between its two centerlines. The analysis uses the boundaries'
common horizontal support and automatically removes 5% from the left edge and
5% from the right edge. There is no manual range/reset control and no pairwise
gap analysis.

Thickness is measured vertically in every retained x-column as
`bottom y - top y + 1`. Mean, median, population standard deviation, minimum,
and maximum are calculated across those column thicknesses. Values in µm are
the pixel measurements multiplied by the selected pixel size (3.45 µm/pixel by
default). Thickness is therefore an image-column measurement, not a distance
measured perpendicular to the local curved boundary.

## Measurements and exports

Before zones are constructed, every model prediction mask is resized to the
SFO image's native `(height, width)` using nearest-neighbor sampling. All area,
width, and thickness measurements are calculated from these uniformly resized
binary masks. The table and CSV contain image size, zone area, analyzed width,
mean/median/standard deviation/minimum/maximum thickness, and mean thickness as
a percentage of GP mean thickness. GP is therefore 100%. RGB and grayscale
signal measurements are not included. Pixel values and physical units are both
exported. The default calibration is 3.45 µm/pixel and can be changed in the GUI
before analysis.

**Save zone visualizations** writes:

- the five binary zone masks;
- GP on an SFO+CFO background;
- Germinal on both SFO+CFO and AP+DAPI backgrounds;
- Columnar on an AP+DAPI background;
- Pre-hypertrophic on both AP+DAPI and Mineral backgrounds;
- Hypertrophic on a Mineral+SFO background;
- `All_zones_SFO_contour_overlay.png`, containing all five colored zones and
  their contour outlines on SFO;
- `zone_measurements.csv`.

For a two-stain background, each stain image is contrast-normalized separately
and the two RGB arrays are averaged equally. Individual zone visualizations
retain the thin red contour and do not draw measurement or edge guide lines.
Visualizations use the formatted background's native resolution, capped at
4096 pixels on the longest edge for memory safety. Analysis masks already use
the native SFO measurement grid; if a visualization background has different
dimensions, a display copy of the mask is aligned with nearest-neighbor
sampling. Exported binary zone masks use the high-resolution SFO dimensions.
The in-app preview is rendered separately with a 440-pixel background limit so
the complete image fits in the visualization panel without scrolling. Saved
files retain the high-resolution dimensions. Individual and all-zones contour
outlines use the same relative line thickness.

The same presentation workflow is available in `test.ipynb`. Its final cell is
named `# With contour outline — save the total five-zone visualization` and
saves the combined contour image.
