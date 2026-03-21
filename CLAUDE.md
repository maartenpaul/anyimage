# anyimage

Interactive anywidget for visualizing biological images in Jupyter and marimo notebooks.

## Features

- **Multi-dimensional support** - 5D images (TCZYX: Time, Channel, Z-stack, Y, X)
- **Multi-layer mask overlays** - Customizable colors, opacity, and contour rendering
- **Annotation tools** - Rectangles, polygons, and points with visibility controls
- **SAM integration** - Segment Anything Model for interactive segmentation
- **Tile-based rendering** - Viewport-aware caching: precomputes visible tiles first across all T/Z, then fills the rest off-screen
- **BioImage format support** - TIFF, OME-Zarr, and other formats via bioio

## Project Structure

```
anyimage/
├── __init__.py           # Public API exports
├── utils.py              # Image processing helpers and color constants
├── viewer.py             # Main BioImageViewer widget (JS/CSS frontend + traitlets)
└── mixins/
    ├── __init__.py       # Mixin exports
    ├── image_loading.py  # Image loading, caching, and prefetching
    ├── mask_management.py # Mask layer operations
    ├── annotations.py    # Annotation data management (ROIs, polygons, points)
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
from anyimage import BioImageViewer
from bioio import BioImage

# Load and display an image
viewer = BioImageViewer()
img = BioImage("image.tif")
viewer.set_image(img)
viewer
```

## Key Classes

### BioImageViewer

The main widget class with these capabilities:

- `set_image(data)` - Load numpy array or BioImage object
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

Store all Playwright screenshots in a temporary folder at `/tmp/anyimage-screenshots/`. Create it at the start of a testing session and delete it when done:

```bash
mkdir -p /tmp/anyimage-screenshots
# ... run tests, save screenshots to /tmp/anyimage-screenshots/
rm -rf /tmp/anyimage-screenshots
```

In playwright-cli, pass the temp path when taking screenshots:
```javascript
await page.screenshot({ path: '/tmp/anyimage-screenshots/step-name.png' });
```

### Setup

```bash
# Start marimo server (note the access token in the URL printed to terminal)
marimo edit examples/image_notebook.py

# Open browser (use chromium, not chrome — chrome requires root to install)
playwright-cli open "http://localhost:2718?access_token=<token>" --browser=chromium
```

### Key patterns for testing anyimage widgets

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
- `_full_array`: full TCZYX numpy array in RAM if ≤2GB (avoids per-slice zarr I/O)
- `_composite_cache`: `(t, z, res)` → uint8 RGB array (2048×2048), max 64 entries FIFO
- `_tile_cache`: `(t, z, tx, ty, res)` → `{w, h, data: base64}`, max 2048 entries FIFO
- `_precompute_all_composites`: background task, two-pass:
  - Pass 1: viewport tiles only, all T/Z slices sorted by Manhattan distance from current → fast Z/T scrubbing at current zoom/pan
  - Pass 2: off-screen tiles for all slices → full cache fill in background
- Restarts on: `set_image()`, channel settings change, viewport change (pan/zoom settles after 300ms)
- `_viewport_tiles` traitlet: JS reports `{tx0, ty0, tx1, ty1}` tile range; Python uses it to focus Pass 1

**JS side (viewer.py `_esm`):**
- `tileCache`: `Map<key, ImageBitmap>`, max 4096 entries, true LRU (delete+re-insert on access)
- `pendingTiles`: `Set<key>` prevents duplicate requests
- `getVisibleTiles()`: computes only tiles in current viewport from scale/translateX/translateY
- `requestTiles()`: debounced 30ms to coalesce rapid slider movements
- `reportViewport()`: debounced 300ms after pan/zoom, sends tile range to Python

**Thumbnail (baseImage):**
- Sent as fast PNG (`compress_level=1`, ~16ms) on every T/Z change
- Shown as background while high-res tiles load — prevents blank canvas
- Tiles overdraw it at full resolution as they arrive

### Expected performance (image.zarr, 10T×3Z×2048×2048)

At fit-to-screen zoom (64 tiles/slice visible):
- `set_image`: ~2s (loads 252MB into RAM)
- Thumbnail appears: ~20ms after navigation
- Full tile render (cold, precompute running): ~500ms
- Full tile render (cached, precompute done): ~50ms

At 4× zoom (4 tiles/slice visible):
- After viewport settles (300ms), Pass 1 caches 4 tiles × 30 slices = 120 tiles in ~2s
- Z/T scrubbing after that: ~50ms (all viewport tiles already cached)
