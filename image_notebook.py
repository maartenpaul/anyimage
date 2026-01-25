import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from bioio import BioImage
    import bioio_tifffile
    from anyimage import BioImageViewer
    return BioImage, BioImageViewer, bioio_tifffile, mo


@app.cell
def _(BioImage, bioio_tifffile):
    img = BioImage("image.tif", reader=bioio_tifffile.Reader)
    mask = BioImage("mask.tif", reader=bioio_tifffile.Reader)
    return img, mask


@app.cell
def _(BioImageViewer, img, mask, mo):
    viewer = BioImageViewer()
    viewer.set_image(img.data)
    viewer.set_mask(mask.data)
    mo.ui.anywidget(viewer)
    return


if __name__ == "__main__":
    app.run()
