# Getting Started

## Installation

```bash
uv pip install anyimage
```

### Optional extras

| Extra | Adds | Notes |
|-------|------|-------|
| `bioio` | `bioio`, `bioio-tifffile` | TIFF and OME-Zarr file reading |
| `contours` | `scipy` | Contour-only mask rendering |
| `sam` | `ultralytics` | SAM segmentation (PyTorch required) |
| `all` | bioio + contours | Recommended for most users |
| `complete` | all + sam | Full install |

```bash
uv pip install "anyimage[all]"       # recommended
uv pip install "anyimage[complete]"  # includes SAM (Python 3.10–3.12)
```

!!! note "SAM and Python 3.13+"
    SAM support requires PyTorch via `ultralytics`. Use Python 3.10–3.12 for SAM features.

## Usage in Jupyter

Just evaluate the viewer in a cell — it renders inline like any other widget:

```python
from anyimage import BioImageViewer
from bioio import BioImage

viewer = BioImageViewer()
viewer.set_image(BioImage("image.tif"))
viewer
```

## Usage in marimo

Wrap with `mo.ui.anywidget()` to make the widget reactive:

```python
import marimo as mo
from anyimage import BioImageViewer
from bioio import BioImage

viewer = BioImageViewer()
viewer.set_image(BioImage("image.tif"))
mo.ui.anywidget(viewer)
```

!!! tip "Reactive annotations"
    In marimo, annotation data is reactive. Any cell referencing `widget.value` will
    re-run automatically when you draw or delete annotations.

## Loading images

`set_image` accepts a numpy array or a `BioImage` object:

```python
# NumPy array (2D, CYX, CZYX, or TCZYX)
viewer.set_image(np.array(...))

# BioImage — lazy loading, full 5D support
from bioio import BioImage
import bioio_tifffile, bioio_ome_zarr

viewer.set_image(BioImage("image.tif",  reader=bioio_tifffile.Reader))
viewer.set_image(BioImage("image.zarr", reader=bioio_ome_zarr.Reader))
```

When a multi-dimensional image is loaded, sliders for T and Z appear automatically, and the Layers panel shows per-channel controls.
