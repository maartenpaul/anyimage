# Masks & Overlays

Overlay segmentation or labeling results as colored mask layers on top of the base image.

## Adding masks

```python
mask_id = viewer.add_mask(
    labels,           # 2D integer numpy array (label image)
    name="Nuclei",
    color="#ff0000",  # hex color
    opacity=0.5,      # 0.0–1.0
    contours_only=False,  # True = draw borders only
)
```

Multiple masks can be added, each tracked by a unique `mask_id`:

```python
id1 = viewer.add_mask(nuclei, name="Nuclei", color="#0000ff", opacity=0.5)
id2 = viewer.add_mask(cells,  name="Cells",  color="#00ff00", contours_only=True)
```

## Managing masks

```python
# Update settings
viewer.update_mask_settings(mask_id, opacity=0.3, color="#ff8800")
viewer.update_mask_settings(mask_id, visible=False)

# Remove
viewer.remove_mask(mask_id)
viewer.clear_masks()

# List
viewer.get_mask_ids()       # → ['mask_0_...', 'mask_1_...']
viewer.masks_df             # DataFrame: id, name, visible, opacity, color
```

## Layers panel

The **Layers** dropdown in the toolbar lets you toggle visibility, adjust opacity sliders, and change colors for each mask — without writing any code.

## Contour rendering

With `contours_only=True`, only the outlines of labeled regions are drawn. Requires `scipy`:

```bash
pip install "anyimage[contours]"
```

```python
viewer.add_mask(labels, name="Outlines", color="#ffff00", contours_only=True, contour_width=2)
```
