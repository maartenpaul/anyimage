# anyimage

Interactive bioimage viewer widget for Jupyter and marimo notebooks. Built on [anywidget](https://anywidget.dev), it supports multi-dimensional images, multi-channel composites, mask overlays, annotation tools, and HCS plate navigation.

## Installation

```bash
uv pip install anyimage

# With all recommended dependencies (excludes SAM/PyTorch)
uv pip install "anyimage[all]"

# With SAM support (Python 3.10–3.12, requires PyTorch)
uv pip install "anyimage[complete]"
```

## Quick Start

### Jupyter

```python
from anyimage import BioImageViewer
from bioio import BioImage
import bioio_tifffile

viewer = BioImageViewer()
viewer.set_image(BioImage("image.tif", reader=bioio_tifffile.Reader))
viewer  # renders inline
```

### marimo

```python
import marimo as mo
from anyimage import BioImageViewer
from bioio import BioImage
import bioio_tifffile

viewer = BioImageViewer()
viewer.set_image(BioImage("image.tif", reader=bioio_tifffile.Reader))
mo.ui.anywidget(viewer)
```

## Features

### Multi-dimensional images

Supports 5D arrays (TCZYX: Time, Channel, Z-stack, Y, X) with sliders for T, Z, and per-channel controls. Pass a `BioImage` object for lazy loading — efficient for large TIFF and OME-Zarr files.

```python
from bioio import BioImage
import bioio_tifffile
import bioio_ome_zarr

img = BioImage("image.tif",  reader=bioio_tifffile.Reader)
img = BioImage("image.zarr", reader=bioio_ome_zarr.Reader)
viewer.set_image(img)  # activates T/Z sliders, per-channel LUT controls
```

### Multi-channel composites

Each channel has independent color, brightness/contrast (LUT), and visibility controls via the **Layers** panel in the toolbar. Channel settings can also be set programmatically:

```python
# Access and modify channel settings
settings = list(viewer._channel_settings)
settings[0] = {**settings[0], "name": "DAPI", "color": "#0000ff"}
viewer._channel_settings = settings
```

### Mask overlays

Add segmentation masks as overlay layers with configurable color, opacity, and contour rendering:

```python
viewer.add_mask(labels, name="Nuclei", color="#ff0000", opacity=0.5)
viewer.add_mask(cells, name="Cells", color="#00ff00", contours_only=True)

# Manage masks
viewer.update_mask_settings(mask_id, opacity=0.3)
viewer.remove_mask(mask_id)
viewer.clear_masks()
```

### HCS plate support

Load OME-Zarr HCS plates with well and FOV navigation dropdowns built into the widget:

```python
viewer = BioImageViewer()
viewer.set_plate("plate.zarr")
viewer  # shows Well / FOV dropdowns
```

### Annotation tools

| Tool | Shortcut | Description |
|------|----------|-------------|
| Pan | `P` | Navigate and zoom |
| Select | `V` | Select annotations; `Delete` to remove |
| Rectangle | `R` | Draw bounding boxes |
| Polygon | `G` | Click vertices, double-click to close |
| Point | `O` | Place point markers |

Export annotations as DataFrames:

```python
viewer.rois_df      # rectangles: id, x, y, width, height
viewer.polygons_df  # polygons: id, points, num_vertices
viewer.points_df    # points: id, x, y
```

### SAM integration

Automatic segmentation with [Segment Anything Model](https://segment-anything.com) when drawing rectangles or placing points:

```python
viewer.enable_sam(model_type="mobile_sam")  # ~40 MB, fastest
viewer.enable_sam(model_type="sam_b")       # SAM base, ~375 MB
```

Requires `uv pip install "anyimage[sam]"` (Python 3.10–3.12).

## Optional dependencies

| Extra | Installs | Use case |
|-------|----------|----------|
| `bioio` | `bioio`, `bioio-tifffile` | TIFF / OME-Zarr loading |
| `contours` | `scipy` | Contour-only mask rendering |
| `sam` | `ultralytics` (PyTorch) | SAM segmentation |
| `all` | bioio + contours | Recommended (no PyTorch) |
| `complete` | all + sam | Everything |

## License

MIT
