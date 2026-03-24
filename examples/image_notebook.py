import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from bioio import BioImage
    import bioio_tifffile
    import bioio_ome_zarr
    from anybioimage import BioImageViewer

    return BioImage, BioImageViewer, bioio_ome_zarr, mo, pd


@app.cell
def _(BioImage, bioio_ome_zarr):
    #img = BioImage("examples/fluocell.tif", reader=bioio_tifffile.Reader)

    #img = BioImage("examples/image.tif", reader=bioio_tifffile.Reader)
    img = BioImage("examples/image.zarr", reader=bioio_ome_zarr.Reader)

    #mask = BioImage("examples/mask.tif", reader=bioio_tifffile.Reader)

    img.shape
    return


@app.cell
def _(BioImage):
    path = "https://allencell.s3.amazonaws.com/aics/nuc-morph-dataset/hipsc_fov_nuclei_timelapse_dataset/hipsc_fov_nuclei_timelapse_data_used_for_analysis/baseline_colonies_fov_timelapse_dataset/20200323_09_small/raw.ome.zarr"
    image = BioImage(path)
    print(image.get_image_dask_data())
    return (image,)


@app.cell
def _(BioImageViewer, image, mo):
    viewer = BioImageViewer()
    # Pass BioImage directly for lazy loading and 5D support
    # This enables T, Z, C sliders when the image has multiple dimensions
    viewer.set_image(image)

    # Add multiple mask layers with different colors and settings
    # Each mask can have its own name, color, opacity, and visibility
    #viewer.add_mask(
    #    mask.data,
     #   name="Segmentation",
     ##   color="#ff0000",
     #   opacity=0.5,
     #   contours_only=False
    #)

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
