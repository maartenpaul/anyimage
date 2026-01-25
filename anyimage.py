import anywidget
import traitlets
import numpy as np
import base64
from io import BytesIO


def _normalize_image(data: np.ndarray) -> np.ndarray:
    """Normalize image data to uint8 range."""
    if data.dtype == np.uint8:
        return data
    data = data.astype(np.float64)
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        data = np.zeros_like(data)
    return data.astype(np.uint8)


def _array_to_base64(data: np.ndarray) -> str:
    """Convert numpy array to base64 encoded PNG."""
    from PIL import Image

    if data.ndim == 2:
        img = Image.fromarray(data, mode='L')
    elif data.ndim == 3 and data.shape[2] == 3:
        img = Image.fromarray(data, mode='RGB')
    elif data.ndim == 3 and data.shape[2] == 4:
        img = Image.fromarray(data, mode='RGBA')
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _labels_to_rgba(labels: np.ndarray) -> np.ndarray:
    """Convert label array to RGBA with unique colors per label."""
    height, width = labels.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue  # Keep background transparent
        # Deterministic color using same hash as JS
        seed = int(label) * 2654435761
        r = (seed >> 16) & 0xFF
        g = (seed >> 8) & 0xFF
        b = seed & 0xFF
        mask = labels == label
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        rgba[mask, 3] = 255

    return rgba


class BioImageViewer(anywidget.AnyWidget):
    """Anywidget for viewing bioimages with label mask overlays."""

    _esm = """
    async function loadImage(base64Data) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = 'data:image/png;base64,' + base64Data;
        });
    }

    async function render({ model, el }) {
        const container = document.createElement('div');
        container.className = 'bioimage-viewer';

        // Controls
        const controls = document.createElement('div');
        controls.className = 'controls';

        // Image toggle
        const imageToggle = document.createElement('label');
        imageToggle.className = 'toggle';
        const imageCheck = document.createElement('input');
        imageCheck.type = 'checkbox';
        imageCheck.checked = model.get('image_visible');
        imageCheck.addEventListener('change', () => {
            model.set('image_visible', imageCheck.checked);
            model.save_changes();
        });
        imageToggle.appendChild(imageCheck);
        imageToggle.appendChild(document.createTextNode(' Image'));

        // Mask toggle
        const maskToggle = document.createElement('label');
        maskToggle.className = 'toggle';
        const maskCheck = document.createElement('input');
        maskCheck.type = 'checkbox';
        maskCheck.checked = model.get('mask_visible');
        maskCheck.addEventListener('change', () => {
            model.set('mask_visible', maskCheck.checked);
            model.save_changes();
        });
        maskToggle.appendChild(maskCheck);
        maskToggle.appendChild(document.createTextNode(' Mask'));

        // Opacity slider
        const opacityLabel = document.createElement('label');
        opacityLabel.className = 'opacity-control';
        opacityLabel.appendChild(document.createTextNode('Opacity: '));
        const opacitySlider = document.createElement('input');
        opacitySlider.type = 'range';
        opacitySlider.min = '0';
        opacitySlider.max = '1';
        opacitySlider.step = '0.05';
        opacitySlider.value = model.get('mask_opacity');
        opacitySlider.addEventListener('input', () => {
            model.set('mask_opacity', parseFloat(opacitySlider.value));
            model.save_changes();
        });
        opacityLabel.appendChild(opacitySlider);

        // Reset view button
        const resetBtn = document.createElement('button');
        resetBtn.className = 'reset-btn';
        resetBtn.textContent = 'Reset View';

        controls.appendChild(imageToggle);
        controls.appendChild(maskToggle);
        controls.appendChild(opacityLabel);
        controls.appendChild(resetBtn);

        // Canvas wrapper for overflow handling
        const canvasWrapper = document.createElement('div');
        canvasWrapper.className = 'canvas-wrapper';

        const canvas = document.createElement('canvas');
        canvas.className = 'viewer-canvas';
        const ctx = canvas.getContext('2d');

        canvasWrapper.appendChild(canvas);
        container.appendChild(controls);
        container.appendChild(canvasWrapper);
        el.appendChild(container);

        // Transform state
        let scale = 1;
        let translateX = 0;
        let translateY = 0;
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;

        let baseImage = null;
        let maskCanvasEl = null;

        function renderCanvas() {
            const imgWidth = model.get('width');
            const imgHeight = model.get('height');

            if (imgWidth === 0 || imgHeight === 0) return;

            canvas.width = canvasWrapper.clientWidth || imgWidth;
            canvas.height = canvasWrapper.clientHeight || imgHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.translate(translateX, translateY);
            ctx.scale(scale, scale);

            if (model.get('image_visible') && baseImage) {
                ctx.drawImage(baseImage, 0, 0);
            }

            if (model.get('mask_visible') && maskCanvasEl) {
                ctx.globalAlpha = model.get('mask_opacity');
                ctx.drawImage(maskCanvasEl, 0, 0);
                ctx.globalAlpha = 1.0;
            }

            ctx.restore();
        }

        function resetView() {
            const imgWidth = model.get('width');
            const imgHeight = model.get('height');
            const wrapperWidth = canvasWrapper.clientWidth || imgWidth;
            const wrapperHeight = canvasWrapper.clientHeight || imgHeight;

            scale = Math.min(wrapperWidth / imgWidth, wrapperHeight / imgHeight, 1);
            translateX = (wrapperWidth - imgWidth * scale) / 2;
            translateY = (wrapperHeight - imgHeight * scale) / 2;
            renderCanvas();
        }

        resetBtn.addEventListener('click', resetView);

        // Zoom with mouse wheel
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const zoom = e.deltaY < 0 ? 1.1 : 0.9;
            const newScale = Math.min(Math.max(scale * zoom, 0.1), 20);

            // Zoom centered on mouse position
            translateX = mouseX - (mouseX - translateX) * (newScale / scale);
            translateY = mouseY - (mouseY - translateY) * (newScale / scale);
            scale = newScale;

            renderCanvas();
        });

        // Pan with mouse drag
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.style.cursor = 'grabbing';
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            translateX += e.clientX - lastX;
            translateY += e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            renderCanvas();
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
            canvas.style.cursor = 'grab';
        });

        canvas.style.cursor = 'grab';

        async function loadBaseImage() {
            const imageData = model.get('image_data');
            if (imageData) {
                baseImage = await loadImage(imageData);
            }
            resetView();
        }

        async function loadMaskImage() {
            const maskData = model.get('mask_data');
            const imgWidth = model.get('width');
            const imgHeight = model.get('height');

            if (!maskData || imgWidth === 0 || imgHeight === 0) return;

            const maskImg = await loadImage(maskData);

            maskCanvasEl = document.createElement('canvas');
            maskCanvasEl.width = imgWidth;
            maskCanvasEl.height = imgHeight;
            const maskCtx = maskCanvasEl.getContext('2d');
            maskCtx.drawImage(maskImg, 0, 0);
            renderCanvas();
        }

        await loadBaseImage();
        await loadMaskImage();

        model.on('change:image_data', loadBaseImage);
        model.on('change:mask_data', loadMaskImage);
        model.on('change:image_visible', renderCanvas);
        model.on('change:mask_visible', renderCanvas);
        model.on('change:mask_opacity', renderCanvas);
        model.on('change:width', resetView);
        model.on('change:height', resetView);

        // Handle resize
        new ResizeObserver(() => renderCanvas()).observe(canvasWrapper);
    }

    export default { render };
    """

    _css = """
    .bioimage-viewer {
        font-family: system-ui, -apple-system, sans-serif;
        padding: 8px;
    }
    .controls {
        display: flex;
        gap: 16px;
        margin-bottom: 8px;
        align-items: center;
        flex-wrap: wrap;
    }
    .toggle {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
    }
    .opacity-control {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .opacity-control input[type="range"] {
        width: 100px;
    }
    .reset-btn {
        padding: 4px 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background: #f5f5f5;
        cursor: pointer;
    }
    .reset-btn:hover {
        background: #e8e8e8;
    }
    .canvas-wrapper {
        border: 1px solid #ccc;
        overflow: hidden;
        width: 100%;
        height: 500px;
        position: relative;
    }
    .viewer-canvas {
        display: block;
    }
    @media (prefers-color-scheme: dark) {
        .bioimage-viewer {
            color: #e0e0e0;
        }
        .canvas-wrapper {
            border-color: #444;
        }
        .reset-btn {
            background: #333;
            border-color: #555;
            color: #e0e0e0;
        }
        .reset-btn:hover {
            background: #444;
        }
    }
    """

    # Image data as base64 encoded PNG
    image_data = traitlets.Unicode("").tag(sync=True)
    mask_data = traitlets.Unicode("").tag(sync=True)

    # Layer controls
    image_visible = traitlets.Bool(True).tag(sync=True)
    mask_visible = traitlets.Bool(True).tag(sync=True)
    mask_opacity = traitlets.Float(0.5).tag(sync=True)

    # Image dimensions
    width = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)

    def set_image(self, data: np.ndarray):
        """Set the base image from a numpy array."""
        if data.ndim > 2:
            # Take first 2D slice if multi-dimensional
            data = data.squeeze()
            if data.ndim > 2:
                data = data[0] if data.ndim == 3 else data[0, 0]

        self.height, self.width = data.shape[:2]
        normalized = _normalize_image(data)
        self.image_data = _array_to_base64(normalized)

    def set_mask(self, labels: np.ndarray):
        """Set the label mask from a numpy array."""
        if labels.ndim > 2:
            labels = labels.squeeze()
            if labels.ndim > 2:
                labels = labels[0] if labels.ndim == 3 else labels[0, 0]

        # Convert labels to pre-colored RGBA (handles arbitrary label counts)
        rgba = _labels_to_rgba(labels)
        self.mask_data = _array_to_base64(rgba)
