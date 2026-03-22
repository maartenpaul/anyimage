# anybioimage

Interactive anywidget for visualizing biological images in Jupyter and marimo notebooks.

## Features

- **Multi-dimensional support** - 5D images (TCZYX: Time, Channel, Z-stack, Y, X)
- **Multi-layer mask overlays** - Customizable colors, opacity, and contour rendering
- **Annotation tools** - Rectangles, polygons, and points with visibility controls
- **SAM integration** - Segment Anything Model for interactive segmentation
- **Tile-based rendering** - Viewport-aware caching: precomputes visible tiles first across all T/Z, then fills the rest off-screen
- **BioImage format support** - TIFF, OME-Zarr, and other formats via bioio
- **HCS plate support** - OME-Zarr plate visualization with well and FOV selection

## Project Structure

```
anybioimage/
├── __init__.py           # Public API exports
├── utils.py              # Image processing helpers and color constants
├── viewer.py             # Main BioImageViewer widget (JS/CSS frontend + traitlets)
└── mixins/
    ├── __init__.py       # Mixin exports
    ├── image_loading.py  # Image loading, caching, and prefetching
    ├── mask_management.py # Mask layer operations
    ├── annotations.py    # Annotation data management (ROIs, polygons, points)
    ├── plate_loading.py  # HCS OME-Zarr plate loading (well/FOV selection)
    └── sam_integration.py # SAM model integration
```

## Development

This project uses `uv` for package management:

```bash
uv pip install -e ".[all]"      # Recommended: all dependencies except SAM
uv pip install -e ".[dev]"      # Development dependencies only
uv pip install -e ".[sam]"      # SAM model support (requires PyTorch)
uv pip install -e ".[complete]" # Everything including SAM
```

**Note:** The `sam` extra requires PyTorch and may not work on Python 3.13+. Use Python 3.10-3.12 for SAM features.

## Usage

```python
from anybioimage import BioImageViewer
from bioio import BioImage

# Load and display an image
viewer = BioImageViewer()
img = BioImage("image.tif")
viewer.set_image(img)
viewer
```

### HCS Plate Usage

```python
from anybioimage import BioImageViewer

viewer = BioImageViewer()
viewer.set_plate("plate.zarr")  # OME-Zarr HCS plate
viewer
```

The widget adds **Well** and **FOV** dropdown selectors. Changing the well loads its FOV list; changing the FOV loads the corresponding image with full T/Z/channel support.

**Plate traitlets:** `plate_wells`, `plate_fovs`, `current_well`, `current_fov`

## Key Classes

### BioImageViewer

The main widget class with these capabilities:

- `set_image(data)` - Load numpy array or BioImage object
- `set_plate(path)` - Load HCS OME-Zarr plate with well/FOV selection dropdowns
- `add_mask(labels, name, color, opacity)` - Add mask overlay layer
- `enable_sam(model_type)` - Enable SAM segmentation
- `rois_df`, `polygons_df`, `points_df` - Access annotation data as DataFrames

### Annotation Tools

- **Pan** - Navigate the image
- **Select** - Select existing annotations
- **Rectangle** - Draw bounding boxes (triggers SAM when enabled)
- **Polygon** - Draw polygon regions
- **Point** - Place point markers (triggers SAM when enabled)

## Marimo Integration

When editing marimo notebooks, only edit contents inside `@app.cell` decorators:

```python
@app.cell
def _():
    viewer = BioImageViewer()
    viewer.set_image(image_data)
    mo.ui.anywidget(viewer)
    return
```

Run `marimo check --fix` after editing to catch formatting issues.

## Testing with Playwright

Use the `playwright-cli` skill to test the widget in a real browser against a running marimo server.

### Screenshots

Store all Playwright screenshots in a temporary folder at `/tmp/anybioimage-screenshots/`. Create it at the start of a testing session and delete it when done:

```bash
mkdir -p /tmp/anybioimage-screenshots
# ... run tests, save screenshots to /tmp/anybioimage-screenshots/
rm -rf /tmp/anybioimage-screenshots
```

In playwright-cli, pass the temp path when taking screenshots:
```javascript
await page.screenshot({ path: '/tmp/anybioimage-screenshots/step-name.png' });
```

### Setup

```bash
# Start marimo server (note the access token in the URL printed to terminal)
marimo edit examples/image_notebook.py

# Open browser (use chromium, not chrome — chrome requires root to install)
playwright-cli open "http://localhost:2718?access_token=<token>" --browser=chromium
```

### Key patterns for testing anybioimage widgets

**The widget renders inside a shadow DOM** (`MARIMO-ANYWIDGET` element). Regular DOM queries won't find the canvas — use `element.shadowRoot`:

```javascript
// Find the canvas
for (const el of document.querySelectorAll('*')) {
  if (el.tagName === 'MARIMO-ANYWIDGET' && el.shadowRoot) {
    const canvas = el.shadowRoot.querySelector('canvas');
    // canvas is here
  }
}
```

**The widget output is in a scrollable container**, not `window`. Scroll it directly:

```javascript
// Scroll to widget
let parent = widgetElement.parentElement;
while (parent) {
  if (parent.scrollHeight > 2000 && parent.clientHeight < 1000) {
    parent.scrollTo(0, targetY);
    break;
  }
  parent = parent.parentElement;
}
```

**Sliders in the widget** (in order):
- Index 0: Brightness (min=-1, max=1)
- Index 1: Contrast (min=-1, max=1)
- Index 2: T/time slider (min=0, max=dim_t-1)
- Index 3: Z slider (min=0, max=dim_z-1)

**Set a slider value** (React-style, needs native setter + events):

```javascript
const sliders = shadowRoot.querySelectorAll('input[type="range"]');
const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
setter.call(sliders[2], '1');  // Set T=1
sliders[2].dispatchEvent(new Event('input', { bubbles: true }));
sliders[2].dispatchEvent(new Event('change', { bubbles: true }));
```

**Measure render time** by polling canvas pixel changes:

```javascript
await page.waitForFunction(([br, bg, bb, x, y]) => {
  for (const el of document.querySelectorAll('*')) {
    if (el.tagName === 'MARIMO-ANYWIDGET' && el.shadowRoot) {
      const p = el.shadowRoot.querySelector('canvas').getContext('2d').getImageData(x, y, 1, 1).data;
      return p[0] !== br || p[1] !== bg || p[2] !== bb;
    }
  }
}, [before[0], before[1], before[2], 300, 300], { timeout: 10000, polling: 50 });
```

### Kernel restart after code changes

After modifying Python code, the marimo kernel must be restarted to pick up changes (code is loaded at kernel start):

1. Click **Restart** in the "Reconnected" banner (appears after reconnecting to an existing session)
2. Confirm in the **"Restart Kernel"** dialog that appears
3. Wait ~20s for kernel to restart and run all cells

Or use playwright:

```javascript
// Click banner Restart, then confirm dialog
await page.getByTestId('restart-session-button').click();
// Dialog appears — find Restart after Cancel
const buttons = await page.locator('button').all();
let cancelFound = false;
for (const btn of buttons) {
  const text = await btn.textContent();
  if (text?.trim() === 'Cancel') { cancelFound = true; continue; }
  if (cancelFound && text?.trim() === 'Restart') { await btn.click(); break; }
}
await page.waitForTimeout(20000);  // wait for cells to run
```

### Caching architecture

**Python side (image_loading.py):**
- `_full_array`: full TCZYX numpy array in RAM if ≤2GB (avoids per-slice zarr I/O); sized using actual `img.dtype.itemsize` (not assumed uint16)
- `_composite_cache`: `(t, z, res)` → uint8 RGB array; size is dynamic — set in `_set_bioimage` to fit as many T×Z composites as 512 MB allows (up to 1024), recalculated on resolution/scene change
- `_composite_cache_lock`: `threading.Lock()` — guards cache writes during parallel precompute
- `_tile_cache`: `(t, z, tx, ty, res)` → `{w, h, data: base64[, fmt: "jpeg"]}`, max 2048 entries FIFO
- `_tile_cache_lock`: `threading.Lock()` — guards tile cache writes
- `_pyramid`: list of TCZYX arrays (or `None` placeholders) for synthetic pyramid levels; built in `_set_bioimage` for flat files (TIFF, CZI, ND2) that have no native resolution levels; levels materialised on first access via `_get_pyramid_level(level)`
- `_pyramid_has_native`: `True` when the BioImage has native resolution levels (OME-Zarr pyramids); `False` for flat files using the synthetic pyramid
- `_precompute_all_composites(cancel_event)`: background task, two-pass, **parallel** (4 workers via `ThreadPoolExecutor`):
  - **Pass 1** — viewport tiles only, all T/Z slices sorted by Manhattan distance from current position. Makes Z/T scrubbing instant at the current zoom/pan.
  - **Pass 2** — off-screen tiles for all slices. Fills the full cache in the background.
- Restarts on: `set_image()`, channel settings change, viewport change (pan/zoom settles after 300ms)
- `_viewport_tiles` traitlet: JS sends `{tx0, ty0, tx1, ty1}` tile range; Python uses it to scope Pass 1
- `_viewport_tiles_all_cached(t, z)`: returns True when every viewport tile for (t,z) is in `_tile_cache` — used by `_update_slice()` to skip sending a thumbnail when tiles are already ready
- `use_jpeg_tiles` traitlet (default `False`): opt-in JPEG encoding for tiles (~15–25 KB vs ~349 KB RGBA); useful for remote JupyterHub. Toggle with `viewer.use_jpeg_tiles = True`.

**Synthetic pyramid (`_pyramid`) — flat file zoom-out performance:**
- Built automatically in `_set_bioimage` when the image has no native resolution levels
- Level 0 = `_full_array` (full resolution); levels 1 and 2 are `None` until first access (step ×2, ×4)
- `_get_pyramid_level(level)` materialises on demand: `full_array[:, :, :, ::step, ::step]`
- `_on_resolution_change` detects synthetic vs native and dispatches accordingly
- `_clear_caches` resets materialised levels 1+ but preserves `_full_array`
- JS auto-pyramid selection (already in `_esm`) picks the right level from zoom scale — works for both native and synthetic pyramids

**`_update_slice()` fast path:**
When `_viewport_tiles_all_cached(t, z)` is True, `_update_slice()` returns immediately without sending `image_data`. This means:
- No thumbnail PNG is encoded or transferred
- No `loadBaseImage` fires in JS
- `change:current_t/z` triggers `renderCanvas()` directly
- `requestTiles()` finds all tiles in `tileCache` and sends nothing
- Result: **instant render from JS cache, zero round-trips**

**JS side (viewer.py `_esm`):**
- `tileCache`: `Map<key, ImageBitmap>`, max 4096 entries, true LRU (delete+re-insert on `getTile()`)
- `pendingTiles`: `Set<key>` prevents duplicate in-flight requests
- `getVisibleTiles()`: computes only tiles intersecting current viewport (scale/translateX/translateY)
- `requestTiles()`: debounced 30ms — coalesces rapid slider calls into one websocket message
- `reportViewport()`: debounced 300ms after pan/zoom ends, sends tile range to Python to restart precompute
- `decodeRawTile`: uses `Uint8ClampedArray.from(binary, c => c.charCodeAt(0))` (native V8, ~4× faster than explicit loop); handles `fmt: "jpeg"` via `createImageBitmap(new Blob([bytes], {type: "image/jpeg"}))`

**Thumbnail (baseImage):**
- Sent as PNG with `compress_level=1` (~16ms vs 81ms for default level 6) on cold T/Z navigation
- Drawn as canvas background while high-res tiles load — prevents blank canvas
- Tiles overdraw it at full resolution as they arrive
- Skipped entirely when viewport tiles are already cached (fast path above)

**Remote zarr precompute (`_precompute_composites_remote`):**
- Axis-aware: scrubs T or Z only (whichever dimension the user last moved)
- 4 parallel workers; each pair uses parallel channel fetching → 4×C concurrent S3 GETs
- Sorted nearest-first so the most useful frames arrive first

### Expected performance (image.zarr, 10T×3Z×2048×2048)

**Fit-to-screen zoom** (64 tiles/slice visible, all slices = 1920 tiles):
- `set_image`: ~2s (loads 252MB into RAM, starts precompute)
- First T/Z navigation: thumbnail in ~20ms, full tiles in ~500ms
- After precompute Pass 1 (~8s with 4 workers): all T/Z instant from cache (~50ms)

**4× zoom** (4 tiles/slice visible):
- Viewport reported after 300ms of holding still
- Pass 1 caches 4 × 30 = 120 tiles in ~2s
- All T/Z navigation and play after that: **instant, zero round-trips**

**Play mode:**
- While precompute Pass 1 is running: thumbnail flash on each frame
- After Pass 1 complete: pure JS render, no Python involvement, smooth playback

**Flat TIFF/CZI at fit-to-screen (synthetic pyramid level 2, 512×512):**
- Composite is 16× smaller → ~0.4ms vs ~6ms per slice
- Pass 1 precompute for 30 T/Z pairs: ~12ms vs ~180ms at full resolution
