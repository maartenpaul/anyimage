# HCS Plates

anyimage supports OME-Zarr HCS (High-Content Screening) plates. The widget adds **Well** and **FOV** dropdowns for navigation.

## Loading a plate

```python
from anyimage import BioImageViewer

viewer = BioImageViewer()
viewer.set_plate("plate.zarr")
viewer  # shows Well / FOV dropdowns in Jupyter
```

In marimo:

```python
mo.ui.anywidget(viewer)
```

## Navigation

- **Well dropdown** — lists all wells in the plate (e.g. A1, B2, …)
- **FOV dropdown** — lists fields of view for the selected well

Selecting a well reloads the FOV list. Selecting a FOV loads the corresponding image with full T/Z/channel support.

## Plate traitlets

Access or set the current position programmatically:

```python
viewer.plate_wells   # list of well names
viewer.plate_fovs    # list of FOV names for the current well
viewer.current_well  # e.g. "A/1"
viewer.current_fov   # e.g. "0"
```

## Example plate structure

The widget expects an OME-Zarr v0.4 plate (`zarr.json` with `plate` key at root). A typical structure:

```
plate.zarr/
  zarr.json          # contains plate metadata with well list
  A/1/0/             # well A1, FOV 0
  A/2/0/             # well A2, FOV 0
  B/1/0/
  ...
```

## Custom well viewers (non-OME-Zarr plates)

For custom folder structures (e.g. per-well TIFF directories), you can build the navigation yourself and call `set_image` on each selection:

```python
import marimo as mo
from anyimage import BioImageViewer
from bioio import BioImage

wells = ["A01", "A02", "B01"]
well_select = mo.ui.dropdown(options=wells, label="Well")

viewer = BioImageViewer()

@app.cell
def _(well_select):
    path = f"data/{well_select.value}/image.tif"
    viewer.set_image(BioImage(path))
    return mo.ui.anywidget(viewer)
```
