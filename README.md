# BioImageViewer

An interactive image viewer widget for Jupyter/marimo notebooks with support for multiple mask layers and annotation tools.

## Installation

```bash
# Install from source (minimal dependencies)
uv pip install -e .

# With all recommended dependencies (excludes SAM/PyTorch)
uv pip install -e ".[all]"

# With SAM support (requires PyTorch, may not work on Python 3.13+)
uv pip install -e ".[sam]"

# With everything including SAM
uv pip install -e ".[complete]"
```

### Optional Dependencies
- `sam` - SAM/MobileSAM segmentation (`ultralytics`, requires PyTorch)
- `contours` - Contour rendering (`scipy`)
- `bioio` - BioImage file reading (`bioio`, `bioio-tifffile`)
- `dev` - Development tools (`marimo`, `pytest`, `ruff`)
- `all` - All dependencies except SAM (recommended for Python 3.13+)
- `complete` - All dependencies including SAM

## Quick Start

```python
import marimo as mo
from anyimage import BioImageViewer
import numpy as np

# Create viewer
viewer = BioImageViewer()
viewer.set_image(image_array)  # 2D numpy array

# Wrap for marimo
widget = mo.ui.anywidget(viewer)
widget
```

## Features

### Image Display
- Pan and zoom with mouse (scroll to zoom, drag to pan)
- Keyboard shortcuts for tool selection

### Multiple Mask Layers
```python
# Add masks with custom settings
viewer.add_mask(labels, name="Nuclei", color="#ff0000", opacity=0.5)
viewer.add_mask(cells, name="Cells", color="#00ff00", contours_only=True)

# Manage masks
viewer.remove_mask(mask_id)
viewer.clear_masks()
viewer.update_mask_settings(mask_id, opacity=0.3, visible=False)
```

### Annotation Tools
| Tool | Shortcut | Description |
|------|----------|-------------|
| Pan | P | Navigate and zoom |
| Select | V | Click to select, Delete to remove |
| Rectangle | R | Click and drag |
| Polygon | G | Click to add vertices, double-click to close |
| Point | O | Click to place |

### Access Annotation Data
```python
# As DataFrames
viewer.rois_df       # Rectangles: id, x, y, width, height
viewer.polygons_df   # Polygons: id, points, num_vertices
viewer.points_df     # Points: id, x, y
viewer.masks_df      # Masks: id, name, visible, opacity, color

# Clear annotations
viewer.clear_rois()
viewer.clear_polygons()
viewer.clear_points()
viewer.clear_all_annotations()
```

## API Reference

### Image Methods
- `set_image(data)` - Set the base image from a 2D numpy array

### Mask Methods
- `add_mask(labels, name, color, opacity, visible, contours_only, contour_width)` - Add a mask layer, returns mask ID
- `set_mask(labels, ...)` - Set a single mask (clears existing)
- `remove_mask(mask_id)` - Remove a mask by ID
- `clear_masks()` - Remove all masks
- `update_mask_settings(mask_id, **kwargs)` - Update mask properties
- `get_mask_ids()` - List all mask IDs

### Properties
| Property | Type | Description |
|----------|------|-------------|
| `image_visible` | bool | Toggle image visibility |
| `rois_visible` | bool | Toggle rectangle visibility |
| `polygons_visible` | bool | Toggle polygon visibility |
| `points_visible` | bool | Toggle point visibility |
| `roi_color` | str | Rectangle color (hex) |
| `polygon_color` | str | Polygon color (hex) |
| `point_color` | str | Point color (hex) |
| `point_radius` | int | Point radius in pixels |
| `tool_mode` | str | Current tool: pan, select, draw, polygon, point |

## UI Controls

The toolbar provides:
- Tool buttons (Pan, Select, Rectangle, Polygon, Point)
- Reset view and Clear all buttons
- Layers dropdown for visibility, opacity, and color controls

The status bar shows current tool, cursor position, and zoom level.

## SAM Integration

Automatic segmentation using Segment Anything Model (SAM) when drawing rectangles.

### Installation
```bash
# Requires PyTorch (may not work on Python 3.13+)
uv pip install -e ".[sam]"
```

**Note:** SAM support requires PyTorch via ultralytics. If you encounter installation issues on Python 3.13+, use Python 3.10-3.12 or install without the `sam` extra.

### Usage
```python
viewer = BioImageViewer()
viewer.set_image(image_array)

# Enable SAM - rectangle ROIs trigger segmentation
viewer.enable_sam(model_type="mobile_sam")

widget = mo.ui.anywidget(viewer)
```

### Available Models
| Model | Size | Speed | Command |
|-------|------|-------|---------|
| MobileSAM | ~40MB | ~10ms | `enable_sam("mobile_sam")` |
| FastSAM | ~140MB | ~40ms | `enable_sam("fast_sam")` |
| SAM Base | ~375MB | ~50ms | `enable_sam("sam_b")` |
| SAM Large | ~1.2GB | ~100ms | `enable_sam("sam_l")` |

### How it Works
1. Draw a rectangle around an object
2. SAM automatically generates a segmentation mask
3. Mask is added as a new layer

```python
# Disable SAM when done
viewer.disable_sam()
```

## License

MIT
