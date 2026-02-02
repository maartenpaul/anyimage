# anyimage

Interactive anywidget for visualizing biological images in Jupyter and marimo notebooks.

## Features

- **Multi-dimensional support** - 5D images (TCZYX: Time, Channel, Z-stack, Y, X)
- **Multi-layer mask overlays** - Customizable colors, opacity, and contour rendering
- **Annotation tools** - Rectangles, polygons, and points with visibility controls
- **SAM integration** - Segment Anything Model for interactive segmentation
- **Tile-based rendering** - Efficient handling of large images with lazy loading
- **BioImage format support** - TIFF, OME-Zarr, and other formats via bioio

## Project Structure

```
anyimage/
├── __init__.py           # Public API exports
├── utils.py              # Image processing helpers and color constants
├── viewer.py             # Main BioImageViewer widget (JS/CSS frontend + traitlets)
└── mixins/
    ├── __init__.py       # Mixin exports
    ├── image_loading.py  # Image loading, caching, and prefetching
    ├── mask_management.py # Mask layer operations
    ├── annotations.py    # Annotation data management (ROIs, polygons, points)
    └── sam_integration.py # SAM model integration
```

## Development

This project uses `uv` for package management:

```bash
uv pip install -e ".[all]"   # Full install with all dependencies
uv pip install -e ".[dev]"   # Development dependencies only
uv pip install -e ".[sam]"   # SAM model support
```

## Usage

```python
from anyimage import BioImageViewer
from bioio import BioImage

# Load and display an image
viewer = BioImageViewer()
img = BioImage("image.tif")
viewer.set_image(img)
viewer
```

## Key Classes

### BioImageViewer

The main widget class with these capabilities:

- `set_image(data)` - Load numpy array or BioImage object
- `add_mask(labels, name, color, opacity)` - Add mask overlay layer
- `enable_sam(model_type)` - Enable SAM segmentation
- `rois_df`, `polygons_df`, `points_df` - Access annotation data as DataFrames

### Annotation Tools

- **Pan** - Navigate the image
- **Select** - Select existing annotations
- **Rectangle** - Draw bounding boxes (triggers SAM when enabled)
- **Polygon** - Draw polygon regions
- **Point** - Place point markers (triggers SAM when enabled)

## Marimo Integration

When editing marimo notebooks, only edit contents inside `@app.cell` decorators:

```python
@app.cell
def _():
    viewer = BioImageViewer()
    viewer.set_image(image_data)
    mo.ui.anywidget(viewer)
    return
```

Run `marimo check --fix` after editing to catch formatting issues.
