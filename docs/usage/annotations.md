# Annotations

Draw and manage rectangles, polygons, and points directly on the image.

## Tools

| Tool | Shortcut | How to use |
|------|----------|------------|
| **Pan** | `P` | Scroll to zoom, drag to pan |
| **Select** | `V` | Click to select; `Delete` to remove |
| **Rectangle** | `R` | Click and drag |
| **Polygon** | `G` | Click vertices; double-click or click near start to close |
| **Point** | `O` | Click to place |

Select a tool from the toolbar or press the keyboard shortcut.

## Exporting annotations

```python
viewer.rois_df      # pd.DataFrame: id, x, y, width, height
viewer.polygons_df  # pd.DataFrame: id, points, num_vertices
viewer.points_df    # pd.DataFrame: id, x, y
```

## Clearing annotations

```python
viewer.clear_rois()
viewer.clear_polygons()
viewer.clear_points()
viewer.clear_all_annotations()
```

## Styling

```python
viewer.roi_color     = "#ff0000"  # rectangle color (hex)
viewer.polygon_color = "#00ff00"  # polygon color (hex)
viewer.point_color   = "#0000ff"  # point color (hex)
viewer.point_radius  = 5          # point radius in pixels

viewer.rois_visible     = False   # toggle visibility
viewer.polygons_visible = True
viewer.points_visible   = True
```

## Reactive annotations in marimo

In marimo, wrap the viewer and reference `widget.value` to reactively access annotations in downstream cells:

```python
@app.cell
def _(viewer, mo):
    widget = mo.ui.anywidget(viewer)
    return widget

@app.cell
def _(widget):
    import pandas as pd
    rois = widget.value.get("_rois_data", [])
    rois_df = pd.DataFrame(rois)
    return rois_df
```

The downstream cell re-runs automatically whenever you draw or delete an annotation.
