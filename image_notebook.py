import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from bioio import BioImage
    import bioio_tifffile
    from anyimage import BioImageViewer
    return BioImage, BioImageViewer, bioio_tifffile, mo, pd


@app.cell
def _(BioImage, bioio_tifffile):
    img = BioImage("image.tif", reader=bioio_tifffile.Reader)
    mask = BioImage("mask.tif", reader=bioio_tifffile.Reader)
    return img, mask


@app.cell
def _(BioImageViewer, img, mask, mo):
    viewer = BioImageViewer()
    viewer.set_image(img.data)

    # Add multiple mask layers with different colors and settings
    # Each mask can have its own name, color, opacity, and visibility
    viewer.add_mask(
        mask.data,
        name="Segmentation",
        color="#ff0000",
        opacity=0.5,
        contours_only=False
    )
    flipped = mask.data.copy()
    flipped = flipped[:, ::-1]
    viewer.add_mask(
        flipped,
        name="Segmentation",
        color="#ff0000",
        opacity=0.5,
        contours_only=False
    )

    # You can add additional masks with different settings:
    # viewer.add_mask(another_mask, name="Nuclei", color="#00ff00", opacity=0.3)
    # viewer.add_mask(cell_mask, name="Cells", color="#0000ff", contours_only=True)

    widget = mo.ui.anywidget(viewer)
    widget
    return (widget,)


@app.cell
def _(mo):
    mo.md("""
    ## Annotation Tools

    Use the toolbar to select different annotation modes:
    - **Pan (P)**: Navigate and zoom the image
    - **Select (V)**: Click annotations to select, Delete to remove
    - **Rectangle (R)**: Click and drag to draw rectangles
    - **Polygon (G)**: Click to add vertices, double-click or click near start to close
    - **Point (O)**: Click to place points

    ## Layers

    Use the **Layers** dropdown to:
    - Toggle visibility of image, masks, and annotations
    - Adjust opacity for each mask layer
    - Change mask colors

    Multiple mask layers can be added programmatically:
    ```python
    viewer.add_mask(labels, name="Nuclei", color="#ff0000", opacity=0.5)
    viewer.add_mask(cells, name="Cells", color="#00ff00", contours_only=True)
    ```
    """)
    return


@app.cell
def _(mo, pd, widget):
    rois_data = widget.value.get("_rois_data", [])
    polygons_data = widget.value.get("_polygons_data", [])
    points_data = widget.value.get("_points_data", [])
    masks_data = widget.value.get("_masks_data", [])

    rois_df = pd.DataFrame(rois_data) if rois_data else pd.DataFrame(columns=['id', 'x', 'y', 'width', 'height'])
    polygons_df = pd.DataFrame([{"id": p["id"], "num_vertices": len(p["points"])} for p in polygons_data]) if polygons_data else pd.DataFrame(columns=['id', 'num_vertices'])
    points_df = pd.DataFrame(points_data) if points_data else pd.DataFrame(columns=['id', 'x', 'y'])
    masks_df = pd.DataFrame([{"id": m["id"], "name": m["name"], "visible": m["visible"], "opacity": m["opacity"], "color": m["color"]} for m in masks_data]) if masks_data else pd.DataFrame(columns=['id', 'name', 'visible', 'opacity', 'color'])

    mo.vstack([
        mo.md("### Mask Layers"),
        masks_df,
        mo.md("### Rectangles"),
        rois_df,
        mo.md("### Polygons"),
        polygons_df,
        mo.md("### Points"),
        points_df
    ])
    return


if __name__ == "__main__":
    app.run()
