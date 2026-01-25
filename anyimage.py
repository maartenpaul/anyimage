import anywidget
import traitlets
import numpy as np
import pandas as pd
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
        img = Image.fromarray(data, mode="L")
    elif data.ndim == 3 and data.shape[2] == 3:
        img = Image.fromarray(data, mode="RGB")
    elif data.ndim == 3 and data.shape[2] == 4:
        img = Image.fromarray(data, mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape: {data.shape}")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _labels_to_rgba(
    labels: np.ndarray, contours_only: bool = False, contour_width: int = 1
) -> np.ndarray:
    """Convert label array to RGBA with unique colors per label.

    Args:
        labels: 2D array of label values
        contours_only: If True, only draw contours instead of filled regions
        contour_width: Width of contour lines in pixels (only used if contours_only=True)
    """
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

        if contours_only:
            # Detect boundaries using morphological gradient
            from scipy import ndimage

            # Create binary mask for this label
            binary_mask = mask.astype(np.uint8)
            # Dilate and subtract to get boundary
            dilated = ndimage.binary_dilation(binary_mask, iterations=contour_width)
            eroded = ndimage.binary_erosion(binary_mask, iterations=1)
            boundary = dilated & ~eroded

            rgba[boundary, 0] = r
            rgba[boundary, 1] = g
            rgba[boundary, 2] = b
            rgba[boundary, 3] = 255
        else:
            # Fill entire region
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = 255

    return rgba


class BioImageViewer(anywidget.AnyWidget):
    """Anywidget for viewing bioimages with label mask overlays."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._labels_array = None  # Store original labels for regeneration
        self._mask_cache = {}  # Cache for filled and contour versions
        # Set up observers for contour settings
        self.observe(
            self._on_contour_settings_changed, names=["show_contours", "contour_width"]
        )

    def _on_contour_settings_changed(self, change):
        """Called when contour settings change - regenerate mask."""
        self._regenerate_mask()

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

        // Controls row 1: layers
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

        // Contours toggle
        const contoursToggle = document.createElement('label');
        contoursToggle.className = 'toggle';
        const contoursCheck = document.createElement('input');
        contoursCheck.type = 'checkbox';
        contoursCheck.checked = model.get('show_contours');
        contoursCheck.addEventListener('change', () => {
            model.set('show_contours', contoursCheck.checked);
            model.save_changes();
        });
        contoursToggle.appendChild(contoursCheck);
        contoursToggle.appendChild(document.createTextNode(' Contours'));

        // ROIs toggle
        const roisToggle = document.createElement('label');
        roisToggle.className = 'toggle';
        const roisCheck = document.createElement('input');
        roisCheck.type = 'checkbox';
        roisCheck.checked = model.get('rois_visible');
        roisCheck.addEventListener('change', () => {
            model.set('rois_visible', roisCheck.checked);
            model.save_changes();
        });
        roisToggle.appendChild(roisCheck);
        roisToggle.appendChild(document.createTextNode(' ROIs'));

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

        controls.appendChild(imageToggle);
        controls.appendChild(maskToggle);
        controls.appendChild(contoursToggle);
        controls.appendChild(roisToggle);
        controls.appendChild(opacityLabel);

        // Controls row 2: tools
        const toolbar = document.createElement('div');
        toolbar.className = 'toolbar';

        // Pan tool button
        const panBtn = document.createElement('button');
        panBtn.className = 'tool-btn active';
        panBtn.textContent = 'Pan';
        panBtn.dataset.mode = 'pan';

        // Draw tool button
        const drawBtn = document.createElement('button');
        drawBtn.className = 'tool-btn';
        drawBtn.textContent = 'Draw ROI';
        drawBtn.dataset.mode = 'draw';

        // Reset view button
        const resetBtn = document.createElement('button');
        resetBtn.className = 'tool-btn';
        resetBtn.textContent = 'Reset View';

        // Clear ROIs button
        const clearRoisBtn = document.createElement('button');
        clearRoisBtn.className = 'tool-btn';
        clearRoisBtn.textContent = 'Clear ROIs';

        toolbar.appendChild(panBtn);
        toolbar.appendChild(drawBtn);
        toolbar.appendChild(resetBtn);
        toolbar.appendChild(clearRoisBtn);

        // Canvas wrapper
        const canvasWrapper = document.createElement('div');
        canvasWrapper.className = 'canvas-wrapper';

        const canvas = document.createElement('canvas');
        canvas.className = 'viewer-canvas';
        const ctx = canvas.getContext('2d');

        canvasWrapper.appendChild(canvas);
        container.appendChild(controls);
        container.appendChild(toolbar);
        container.appendChild(canvasWrapper);
        el.appendChild(container);

        // Transform state
        let scale = 1;
        let translateX = 0;
        let translateY = 0;

        // Interaction state
        let isDragging = false;
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        let drawStartX = 0;
        let drawStartY = 0;
        let currentDrawRect = null;

        let baseImage = null;
        let maskCanvasEl = null;

        // Coordinate conversion
        function screenToImage(screenX, screenY) {
            return {
                x: (screenX - translateX) / scale,
                y: (screenY - translateY) / scale
            };
        }

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

            // Draw base image
            if (model.get('image_visible') && baseImage) {
                ctx.drawImage(baseImage, 0, 0);
            }

            // Draw mask overlay
            if (model.get('mask_visible') && maskCanvasEl) {
                ctx.globalAlpha = model.get('mask_opacity');
                ctx.drawImage(maskCanvasEl, 0, 0);
                ctx.globalAlpha = 1.0;
            }

            // Draw ROIs
            if (model.get('rois_visible')) {
                const rois = model.get('_rois_data') || [];
                const roiColor = model.get('roi_color');
                ctx.strokeStyle = roiColor;
                ctx.lineWidth = 2 / scale;
                for (const roi of rois) {
                    ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
                }
            }

            // Draw current drawing preview
            if (currentDrawRect && model.get('rois_visible')) {
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2 / scale;
                ctx.setLineDash([5 / scale, 5 / scale]);
                ctx.strokeRect(
                    currentDrawRect.x, currentDrawRect.y,
                    currentDrawRect.width, currentDrawRect.height
                );
                ctx.setLineDash([]);
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

        function updateToolMode(mode) {
            model.set('tool_mode', mode);
            model.save_changes();
            panBtn.classList.toggle('active', mode === 'pan');
            drawBtn.classList.toggle('active', mode === 'draw');
            canvas.style.cursor = mode === 'pan' ? 'grab' : 'crosshair';
        }

        // Tool button handlers
        panBtn.addEventListener('click', () => updateToolMode('pan'));
        drawBtn.addEventListener('click', () => updateToolMode('draw'));
        resetBtn.addEventListener('click', resetView);
        clearRoisBtn.addEventListener('click', () => {
            model.set('_rois_data', []);
            model.save_changes();
            renderCanvas();
        });

        // Zoom with mouse wheel (always active)
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const zoom = e.deltaY < 0 ? 1.1 : 0.9;
            const newScale = Math.min(Math.max(scale * zoom, 0.1), 20);

            translateX = mouseX - (mouseX - translateX) * (newScale / scale);
            translateY = mouseY - (mouseY - translateY) * (newScale / scale);
            scale = newScale;

            renderCanvas();
        });

        // Mouse down handler
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;

            if (model.get('tool_mode') === 'draw') {
                // Start drawing ROI
                isDrawing = true;
                const imgCoords = screenToImage(screenX, screenY);
                drawStartX = imgCoords.x;
                drawStartY = imgCoords.y;
                currentDrawRect = { x: drawStartX, y: drawStartY, width: 0, height: 0 };
            } else {
                // Start panning
                isDragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
                canvas.style.cursor = 'grabbing';
            }
        });

        // Mouse move handler
        window.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;

            if (isDrawing) {
                const imgCoords = screenToImage(screenX, screenY);
                currentDrawRect = {
                    x: Math.min(drawStartX, imgCoords.x),
                    y: Math.min(drawStartY, imgCoords.y),
                    width: Math.abs(imgCoords.x - drawStartX),
                    height: Math.abs(imgCoords.y - drawStartY)
                };
                renderCanvas();
            } else if (isDragging) {
                translateX += e.clientX - lastX;
                translateY += e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                renderCanvas();
            }
        });

        // Mouse up handler
        window.addEventListener('mouseup', () => {
            if (isDrawing && currentDrawRect) {
                // Finalize ROI if it has reasonable size
                if (currentDrawRect.width > 5 && currentDrawRect.height > 5) {
                    const rois = model.get('_rois_data') || [];
                    const newRoi = {
                        id: 'roi_' + Date.now(),
                        x: Math.round(currentDrawRect.x),
                        y: Math.round(currentDrawRect.y),
                        width: Math.round(currentDrawRect.width),
                        height: Math.round(currentDrawRect.height)
                    };
                    rois.push(newRoi);
                    model.set('_rois_data', [...rois]);
                    model.save_changes();
                }
                currentDrawRect = null;
                isDrawing = false;
                renderCanvas();
            }

            if (isDragging) {
                isDragging = false;
                if (model.get('tool_mode') === 'pan') {
                    canvas.style.cursor = 'grab';
                }
            }
        });

        // Set initial cursor
        canvas.style.cursor = model.get('tool_mode') === 'pan' ? 'grab' : 'crosshair';

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
        model.on('change:rois_visible', renderCanvas);
        model.on('change:mask_opacity', renderCanvas);
        model.on('change:show_contours', () => {
            contoursCheck.checked = model.get('show_contours');
        });
        model.on('change:_rois_data', renderCanvas);
        model.on('change:roi_color', renderCanvas);
        model.on('change:width', resetView);
        model.on('change:height', resetView);
        model.on('change:tool_mode', () => {
            const mode = model.get('tool_mode');
            panBtn.classList.toggle('active', mode === 'pan');
            drawBtn.classList.toggle('active', mode === 'draw');
            canvas.style.cursor = mode === 'pan' ? 'grab' : 'crosshair';
        });

        new ResizeObserver(() => renderCanvas()).observe(canvasWrapper);
    }

    export default { render };
    """

    _css = """
    .bioimage-viewer {
        font-family: system-ui, -apple-system, sans-serif;
        padding: 8px;
    }
    .controls, .toolbar {
        display: flex;
        gap: 12px;
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
    .tool-btn {
        padding: 4px 10px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background: #f5f5f5;
        cursor: pointer;
        font-size: 13px;
    }
    .tool-btn:hover {
        background: #e8e8e8;
    }
    .tool-btn.active {
        background: #0066cc;
        color: white;
        border-color: #0066cc;
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
        .tool-btn {
            background: #333;
            border-color: #555;
            color: #e0e0e0;
        }
        .tool-btn:hover {
            background: #444;
        }
        .tool-btn.active {
            background: #0066cc;
            border-color: #0066cc;
            color: white;
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
    show_contours = traitlets.Bool(False).tag(sync=True)
    contour_width = traitlets.Int(1).tag(sync=True)

    # Image dimensions
    width = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)

    # ROI annotation
    tool_mode = traitlets.Unicode("pan").tag(sync=True)  # 'pan' or 'draw'
    _rois_data = traitlets.List(traitlets.Dict()).tag(sync=True)
    rois_visible = traitlets.Bool(True).tag(sync=True)
    roi_color = traitlets.Unicode("#ff0000").tag(sync=True)

    @property
    def rois_df(self) -> pd.DataFrame:
        """Get ROIs as a pandas DataFrame."""
        if not self._rois_data:
            return pd.DataFrame(columns=["id", "x", "y", "width", "height"])
        return pd.DataFrame(self._rois_data)

    @rois_df.setter
    def rois_df(self, df: pd.DataFrame):
        """Set ROIs from a pandas DataFrame."""
        self._rois_data = df.to_dict("records")

    def clear_rois(self):
        """Clear all ROIs."""
        self._rois_data = []

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

    def set_mask(
        self,
        labels: np.ndarray,
        contours_only: bool | None = None,
        contour_width: int | None = None,
    ):
        """Set the label mask from a numpy array.

        Args:
            labels: 2D numpy array of label values
            contours_only: If True, show only contours. If None, uses self.show_contours
            contour_width: Width of contours in pixels. If None, uses self.contour_width
        """
        if labels.ndim > 2:
            labels = labels.squeeze()
            if labels.ndim > 2:
                labels = labels[0] if labels.ndim == 3 else labels[0, 0]

        # Store original labels for later regeneration
        self._labels_array = labels

        # Clear cache when new labels are set
        self._mask_cache = {}

        # Use instance attributes if not specified
        use_contours = (
            contours_only if contours_only is not None else self.show_contours
        )
        use_width = contour_width if contour_width is not None else self.contour_width

        # Generate the requested version immediately for instant display
        current_key = (use_contours, use_width)
        rgba_current = _labels_to_rgba(
            labels, contours_only=use_contours, contour_width=use_width
        )
        self._mask_cache[current_key] = _array_to_base64(rgba_current)
        self.mask_data = self._mask_cache[current_key]

        # Pre-generate the other version asynchronously in background
        import threading

        other_key = (not use_contours, use_width)

        def _pregenerate_other():
            rgba_other = _labels_to_rgba(
                labels, contours_only=not use_contours, contour_width=use_width
            )
            self._mask_cache[other_key] = _array_to_base64(rgba_other)

        # Start background thread to pre-generate the alternate version
        thread = threading.Thread(target=_pregenerate_other, daemon=True)
        thread.start()

    def _regenerate_mask(self):
        """Regenerate mask from cached or stored labels array."""
        if self._labels_array is None:
            return

        cache_key = (self.show_contours, self.contour_width)

        # Check if we have it cached
        if cache_key in self._mask_cache:
            self.mask_data = self._mask_cache[cache_key]
        else:
            # Generate and cache if not available
            rgba = _labels_to_rgba(
                self._labels_array,
                contours_only=self.show_contours,
                contour_width=self.contour_width,
            )
            self._mask_cache[cache_key] = _array_to_base64(rgba)
            self.mask_data = self._mask_cache[cache_key]
