# anyimage

Interactive bioimage viewer widget for Jupyter and marimo notebooks. Built on [anywidget](https://anywidget.dev).

## Key features

- **5D image support** — Time, Channel, Z-stack, Y, X with interactive sliders
- **Multi-channel composites** — Per-channel color, LUT, and visibility controls
- **OME-Zarr** — Lazy loading for large images; multi-resolution tile rendering
- **HCS plates** — Navigate OME-Zarr HCS plates by well and FOV
- **Mask overlays** — Multiple labeled layers with color, opacity, and contour options
- **Annotation tools** — Draw rectangles, polygons, and points; export as DataFrames
- **SAM integration** — Segment Anything Model triggered by rectangle or point annotations

## Installation

```bash
uv pip install anyimage

# With BioImage file support and contour rendering (recommended)
uv pip install "anyimage[all]"

# Including SAM (requires PyTorch, Python 3.10–3.12)
uv pip install "anyimage[complete]"
```

## Minimal example

=== "Jupyter"

    ```python
    from anyimage import BioImageViewer
    from bioio import BioImage

    viewer = BioImageViewer()
    viewer.set_image(BioImage("image.zarr"))
    viewer
    ```

=== "marimo"

    ```python
    import marimo as mo
    from anyimage import BioImageViewer
    from bioio import BioImage

    viewer = BioImageViewer()
    viewer.set_image(BioImage("image.zarr"))
    mo.ui.anywidget(viewer)
    ```

=== "NumPy array"

    ```python
    import numpy as np
    from anyimage import BioImageViewer

    data = np.random.uint8(np.random.rand(3, 512, 512) * 255)  # CYX
    viewer = BioImageViewer()
    viewer.set_image(data)
    viewer
    ```
