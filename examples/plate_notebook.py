import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from anyimage import BioImageViewer

    return BioImageViewer, mo


@app.cell
def _(BioImageViewer, mo):
    viewer = BioImageViewer()
    viewer.set_plate("examples/test_plate.zarr")

    widget = mo.ui.anywidget(viewer)
    widget
    return (widget,)


@app.cell
def _(mo):
    mo.md("""
    ## HCS Plate Viewer

    Use the **Well** and **FOV** dropdowns to navigate between wells and fields of view.

    - **Well**: Select a well (e.g., A1, A2, B1, B2)
    - **FOV**: Select a field of view within the well (e.g., 0, 1)
    - **T/Z sliders**: Navigate time and z-stack dimensions
    - **Channel chips**: Toggle channels and adjust contrast
    """)
    return


if __name__ == "__main__":
    app.run()
