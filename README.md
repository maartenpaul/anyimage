# BioImageViewer

An interactive image viewer widget for Jupyter/marimo notebooks with support for multiple mask layers and annotation tools.

## Installation

```bash
uv pip install anywidget numpy pandas pillow
# Optional: scipy (for contour rendering)
```

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

## License

MIT
