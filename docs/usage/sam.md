# SAM Integration

anyimage integrates [Segment Anything Model (SAM)](https://segment-anything.com) for interactive segmentation. Draw a rectangle or place a point, and SAM generates a segmentation mask automatically.

## Installation

```bash
uv pip install "anyimage[sam]"
```

!!! warning "Python version"
    SAM requires PyTorch via `ultralytics`. Use **Python 3.10–3.12**. Python 3.13+ is not supported.

## Usage

```python
from anyimage import BioImageViewer
from bioio import BioImage

viewer = BioImageViewer()
viewer.set_image(BioImage("image.tif"))
viewer.enable_sam(model_type="mobile_sam")

viewer  # Jupyter / mo.ui.anywidget(viewer) for marimo
```

1. Select the **Rectangle** tool (`R`) or **Point** tool (`O`)
2. Draw a box around an object or click on it
3. SAM generates a mask and adds it as a new layer

## Available models

| Model | Size | Notes |
|-------|------|-------|
| `mobile_sam` | ~40 MB | Fastest, default |
| `fast_sam` | ~140 MB | CNN-based, no encoder |
| `sam_b` | ~375 MB | SAM base ViT |
| `sam_l` | ~1.2 GB | SAM large ViT |

```python
viewer.enable_sam("mobile_sam")  # recommended for interactive use
viewer.enable_sam("sam_b")       # higher quality
```

## Managing SAM masks

SAM-generated masks are added via `add_mask` and are accessible like any other mask:

```python
print(viewer.get_mask_ids())   # includes SAM-generated masks
viewer.clear_masks()           # removes all masks including SAM results
```

## Disabling SAM

```python
viewer.disable_sam()
```

After disabling, rectangle and point tools revert to annotation-only mode.
