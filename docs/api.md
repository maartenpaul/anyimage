# API Reference

## BioImageViewer

```python
from anyimage import BioImageViewer
viewer = BioImageViewer()
```

### Image methods

| Method | Description |
|--------|-------------|
| `set_image(data)` | Load a numpy array or `BioImage` object |
| `set_plate(path)` | Load an OME-Zarr HCS plate |

### Mask methods

| Method | Description |
|--------|-------------|
| `add_mask(labels, name, color, opacity, visible, contours_only, contour_width)` | Add a mask layer; returns `mask_id` |
| `set_mask(labels, ...)` | Replace all masks with a single mask |
| `remove_mask(mask_id)` | Remove a mask by ID |
| `clear_masks()` | Remove all masks |
| `update_mask_settings(mask_id, **kwargs)` | Update mask properties |
| `get_mask_ids()` | List all active mask IDs |

### Annotation methods

| Method | Description |
|--------|-------------|
| `clear_rois()` | Remove all rectangles |
| `clear_polygons()` | Remove all polygons |
| `clear_points()` | Remove all points |
| `clear_all_annotations()` | Remove all annotations |

### SAM methods

| Method | Description |
|--------|-------------|
| `enable_sam(model_type)` | Enable SAM segmentation |
| `disable_sam()` | Disable SAM |

### Properties — image display

| Property | Type | Description |
|----------|------|-------------|
| `image_visible` | `bool` | Toggle image layer |
| `image_brightness` | `float` | Brightness adjustment (−1 to 1) |
| `image_contrast` | `float` | Contrast adjustment (−1 to 1) |
| `dim_t` | `int` | Number of time points (read-only) |
| `dim_c` | `int` | Number of channels (read-only) |
| `dim_z` | `int` | Number of Z slices (read-only) |
| `current_t` | `int` | Current time index |
| `current_z` | `int` | Current Z index |
| `_channel_settings` | `list[dict]` | Per-channel name, color, min, max, visible |

### Properties — annotations

| Property | Type | Description |
|----------|------|-------------|
| `rois_df` | `DataFrame` | Rectangles: id, x, y, width, height |
| `polygons_df` | `DataFrame` | Polygons: id, points, num_vertices |
| `points_df` | `DataFrame` | Points: id, x, y |
| `masks_df` | `DataFrame` | Masks: id, name, visible, opacity, color |
| `tool_mode` | `str` | Active tool: `pan`, `select`, `draw`, `polygon`, `point` |
| `roi_color` | `str` | Rectangle color (hex) |
| `polygon_color` | `str` | Polygon color (hex) |
| `point_color` | `str` | Point color (hex) |
| `point_radius` | `int` | Point radius in pixels |
| `rois_visible` | `bool` | Toggle rectangle visibility |
| `polygons_visible` | `bool` | Toggle polygon visibility |
| `points_visible` | `bool` | Toggle point visibility |

### Properties — HCS plate

| Property | Type | Description |
|----------|------|-------------|
| `plate_wells` | `list[str]` | Available well names |
| `plate_fovs` | `list[str]` | FOVs for current well |
| `current_well` | `str` | Active well |
| `current_fov` | `str` | Active FOV |
