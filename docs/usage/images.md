# Images

## Supported formats

Via [bioio](https://github.com/bioio-devs/bioio):

| Format | Reader |
|--------|--------|
| TIFF / OME-TIFF | `bioio-tifffile` |
| OME-Zarr | `bioio-ome-zarr` |
| Any bioio-supported format | appropriate `bioio-*` plugin |

```python
from bioio import BioImage
import bioio_tifffile
import bioio_ome_zarr

img = BioImage("image.tif",  reader=bioio_tifffile.Reader)
img = BioImage("image.zarr", reader=bioio_ome_zarr.Reader)
```

Or pass a NumPy array directly — the viewer infers the dimension order:

```python
viewer.set_image(array)  # 2D, CYX, CZYX, or TCZYX
```

## Multi-dimensional navigation

When the image has T > 1 or Z > 1, the widget shows time and Z sliders. Channel count is reflected in the Layers panel.

```python
img = BioImage("image.zarr", reader=bioio_ome_zarr.Reader)
print(img.shape)  # e.g. (10, 3, 5, 2048, 2048) → T=10, C=3, Z=5

viewer.set_image(img)
# → T slider (0–9), Z slider (0–4), 3 channels in Layers panel
```

## Multi-channel display

Channels are rendered as a composite. Each channel has:

- **Color** (hex) — mapped from intensity to that hue
- **Min / Max** — contrast limits (0–1 normalized)
- **Visible** toggle

Controls are in the **Layers** dropdown in the toolbar. They can also be set programmatically:

```python
settings = list(viewer._channel_settings)
settings[0] = {**settings[0], "name": "DAPI",  "color": "#0000ff"}
settings[1] = {**settings[1], "name": "GFP",   "color": "#00ff00"}
settings[2] = {**settings[2], "name": "mCherry","color": "#ff0000"}
viewer._channel_settings = settings
```

## Performance

For large images, `BioImage` with an OME-Zarr reader enables lazy loading. The viewer uses a tile-based rendering pipeline that precomputes visible tiles across all T/Z positions in the background — T/Z scrubbing becomes instant after the initial precompute pass.

Pan and zoom send the current viewport to Python so the precompute prioritizes visible tiles.
