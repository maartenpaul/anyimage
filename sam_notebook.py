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
    img2 = BioImage("/var/home/maartenpaul/Nextcloud2/LACDR_NL-Bioimaging (Projectfolder)/Maarten Paul/Projects/vanDijkBrechtje_PBMC_AI_classification/Experimental data/Brechje/20251124_BvD003_D3/p1/tiffs/20251124_172329_075/wellxy001c5t1.tif", reader=bioio_tifffile.Reader)
    return


@app.cell
def _(mo):
    # Create a file browser  
    file_browser = mo.ui.file_browser(multiple=False,filetypes=[".tif", ".tiff"],label="Select a TIFF image",initial_path="/var/home/maartenpaul/Nextcloud2/LACDR_NL-Bioimaging (Projectfolder)/Maarten Paul/Projects/vanDijkBrechtje_PBMC_AI_classification/Experimental data/Brechje/")  
  
    # Display it  
    file_browser  

    return (file_browser,)


@app.cell
def _(BioImage, bioio_tifffile, file_browser):
    img = BioImage(file_browser.path(), reader=bioio_tifffile.Reader)
    return (img,)


@app.cell
def _(BioImageViewer, img, mo):
    viewer = BioImageViewer()
    viewer.set_image(img.data)

    # Enable SAM - draws rectangle to trigger segmentation
    # Options: "mobile_sam" (default), "sam_b", "sam_l", "fast_sam"
    viewer.enable_sam(model_type="mobile_sam")

    widget = mo.ui.anywidget(viewer)
    widget
    return (widget,)


@app.cell
def _(mo):
    mo.md("""
    ## SAM Segmentation Demo

    **How to use:**
    1. Select the **Rectangle** tool (R)
    2. Draw a bounding box around an object you want to segment
    3. SAM will automatically generate a segmentation mask

    The mask will appear as a new layer in the **Layers** dropdown.

    **Available SAM models:**
    - `mobile_sam` - Fastest, ~40MB (default)
    - `fast_sam` - Fast, CNN-based
    - `sam_b` - SAM base model
    - `sam_l` - SAM large model

    ```python
    viewer.enable_sam(model_type="mobile_sam")
    ```
    """)
    return


@app.cell
def _(mo, pd, widget):
    masks_data = widget.value.get("_masks_data", [])
    rois_data = widget.value.get("_rois_data", [])

    masks_df = pd.DataFrame([{"id": m["id"], "name": m["name"], "visible": m["visible"], "opacity": m["opacity"]} for m in masks_data]) if masks_data else pd.DataFrame(columns=['id', 'name', 'visible', 'opacity'])
    rois_df = pd.DataFrame(rois_data) if rois_data else pd.DataFrame(columns=['id', 'x', 'y', 'width', 'height'])

    mo.vstack([
        mo.md("### SAM Masks"),
        masks_df,
        mo.md("### Bounding Boxes"),
        rois_df
    ])
    return


if __name__ == "__main__":
    app.run()
