import base64
from io import BytesIO

import anywidget
import numpy as np
import pandas as pd
import traitlets


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
    """Convert label array to RGBA with unique colors per label."""
    height, width = labels.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue
        seed = int(label) * 2654435761
        r = (seed >> 16) & 0xFF
        g = (seed >> 8) & 0xFF
        b = seed & 0xFF
        mask = labels == label

        if contours_only:
            from scipy import ndimage
            binary_mask = mask.astype(np.uint8)
            dilated = ndimage.binary_dilation(binary_mask, iterations=contour_width)
            eroded = ndimage.binary_erosion(binary_mask, iterations=1)
            boundary = dilated & ~eroded
            rgba[boundary, 0] = r
            rgba[boundary, 1] = g
            rgba[boundary, 2] = b
            rgba[boundary, 3] = 255
        else:
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = 255

    return rgba


# Default colors for mask layers
MASK_COLORS = [
    "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
    "#dfe6e9", "#fd79a8", "#a29bfe", "#6c5ce7", "#00b894"
]


class BioImageViewer(anywidget.AnyWidget):
    """Anywidget for viewing bioimages with multiple mask layers and annotation tools."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mask_arrays = {}  # Store raw label arrays by mask id
        self._mask_caches = {}  # Cache rendered versions by mask id

        # Observer for SAM label deletion
        self.observe(self._on_delete_sam_at, names=["_delete_sam_at"])

    _esm = """
    async function loadImage(base64Data) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = 'data:image/png;base64,' + base64Data;
        });
    }

    const ICONS = {
        pan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>',
        select: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m3 3 7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/><path d="m13 13 6 6"/></svg>',
        rect: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>',
        polygon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l8 5v10l-8 5-8-5V7l8-5z"/></svg>',
        point: '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
        reset: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>',
        trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
        eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
        eyeOff: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>',
        layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        mask: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>',
        download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
    };

    async function render({ model, el }) {
        const container = document.createElement('div');
        container.className = 'bioimage-viewer';
        container.tabIndex = 0;

        const toolbar = document.createElement('div');
        toolbar.className = 'toolbar';

        const toolGroup = document.createElement('div');
        toolGroup.className = 'tool-group';

        function createToolBtn(icon, mode, title) {
            const btn = document.createElement('button');
            btn.className = 'tool-btn' + (model.get('tool_mode') === mode ? ' active' : '');
            btn.innerHTML = icon;
            btn.title = title;
            btn.dataset.mode = mode;
            return btn;
        }

        const panBtn = createToolBtn(ICONS.pan, 'pan', 'Pan (P)');
        const selectBtn = createToolBtn(ICONS.select, 'select', 'Select (V)');
        const rectBtn = createToolBtn(ICONS.rect, 'draw', 'Rectangle (R)');
        const polygonBtn = createToolBtn(ICONS.polygon, 'polygon', 'Polygon (G)');
        const pointBtn = createToolBtn(ICONS.point, 'point', 'Point (O)');

        toolGroup.appendChild(panBtn);
        toolGroup.appendChild(selectBtn);
        toolGroup.appendChild(rectBtn);
        toolGroup.appendChild(polygonBtn);
        toolGroup.appendChild(pointBtn);

        const sep1 = document.createElement('div');
        sep1.className = 'toolbar-separator';

        const actionGroup = document.createElement('div');
        actionGroup.className = 'tool-group';

        const resetBtn = document.createElement('button');
        resetBtn.className = 'tool-btn';
        resetBtn.innerHTML = ICONS.reset;
        resetBtn.title = 'Reset View';

        const clearBtn = document.createElement('button');
        clearBtn.className = 'tool-btn danger';
        clearBtn.innerHTML = ICONS.trash;
        clearBtn.title = 'Clear All Annotations';

        actionGroup.appendChild(resetBtn);
        actionGroup.appendChild(clearBtn);

        const sep2 = document.createElement('div');
        sep2.className = 'toolbar-separator';

        const layersGroup = document.createElement('div');
        layersGroup.className = 'layers-group';

        const layersBtn = document.createElement('button');
        layersBtn.className = 'layers-btn';
        layersBtn.innerHTML = ICONS.layers + '<span>Layers</span>';

        const layersDropdown = document.createElement('div');
        layersDropdown.className = 'layers-dropdown';

        function downloadMask(mask) {
            if (!mask.data) return;
            
            // Create a temporary canvas to render the mask
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = model.get('width');
            tempCanvas.height = model.get('height');
            const tempCtx = tempCanvas.getContext('2d');
            
            // Load and draw the mask image
            const maskImg = new Image();
            maskImg.onload = () => {
                tempCtx.drawImage(maskImg, 0, 0);
                
                // Convert canvas to blob and download
                tempCanvas.toBlob((blob) => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${mask.name.replace(/\s+/g, '_')}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }, 'image/png');
            };
            maskImg.src = 'data:image/png;base64,' + mask.data;
        }

        function rebuildLayersDropdown() {
            layersDropdown.innerHTML = '';

            // Image layer
            const imageItem = document.createElement('div');
            imageItem.className = 'layer-item';
            const imageToggle = document.createElement('button');
            imageToggle.className = 'layer-toggle' + (model.get('image_visible') ? ' visible' : '');
            imageToggle.innerHTML = model.get('image_visible') ? ICONS.eye : ICONS.eyeOff;
            imageToggle.addEventListener('click', () => {
                model.set('image_visible', !model.get('image_visible'));
                model.save_changes();
                imageToggle.classList.toggle('visible', model.get('image_visible'));
                imageToggle.innerHTML = model.get('image_visible') ? ICONS.eye : ICONS.eyeOff;
            });
            const imageLabel = document.createElement('span');
            imageLabel.textContent = 'Image';
            imageItem.appendChild(imageToggle);
            imageItem.appendChild(imageLabel);
            layersDropdown.appendChild(imageItem);

            // Brightness slider
            const brightnessItem = document.createElement('div');
            brightnessItem.className = 'layer-item sub-item slider-item';
            const brightnessLabel = document.createElement('span');
            brightnessLabel.className = 'slider-label';
            brightnessLabel.textContent = 'Brightness';
            const brightnessSlider = document.createElement('input');
            brightnessSlider.type = 'range';
            brightnessSlider.min = '-1';
            brightnessSlider.max = '1';
            brightnessSlider.step = '0.05';
            brightnessSlider.value = model.get('image_brightness') || 0;
            brightnessSlider.className = 'adjustment-slider';
            brightnessSlider.addEventListener('input', () => {
                model.set('image_brightness', parseFloat(brightnessSlider.value));
                model.save_changes();
            });
            brightnessItem.appendChild(brightnessLabel);
            brightnessItem.appendChild(brightnessSlider);
            layersDropdown.appendChild(brightnessItem);

            // Contrast slider
            const contrastItem = document.createElement('div');
            contrastItem.className = 'layer-item sub-item slider-item';
            const contrastLabel = document.createElement('span');
            contrastLabel.className = 'slider-label';
            contrastLabel.textContent = 'Contrast';
            const contrastSlider = document.createElement('input');
            contrastSlider.type = 'range';
            contrastSlider.min = '-1';
            contrastSlider.max = '1';
            contrastSlider.step = '0.05';
            contrastSlider.value = model.get('image_contrast') || 0;
            contrastSlider.className = 'adjustment-slider';
            contrastSlider.addEventListener('input', () => {
                model.set('image_contrast', parseFloat(contrastSlider.value));
                model.save_changes();
            });
            contrastItem.appendChild(contrastLabel);
            contrastItem.appendChild(contrastSlider);
            layersDropdown.appendChild(contrastItem);

            // Mask layers section
            const masks = model.get('_masks_data') || [];
            if (masks.length > 0) {
                const maskDivider = document.createElement('div');
                maskDivider.className = 'layer-divider';
                layersDropdown.appendChild(maskDivider);

                const maskHeader = document.createElement('div');
                maskHeader.className = 'layer-header';
                maskHeader.innerHTML = ICONS.mask + '<span>Masks</span>';
                layersDropdown.appendChild(maskHeader);

                for (const mask of masks) {
                    const maskItem = document.createElement('div');
                    maskItem.className = 'layer-item mask-layer';

                    const maskToggle = document.createElement('button');
                    maskToggle.className = 'layer-toggle' + (mask.visible ? ' visible' : '');
                    maskToggle.innerHTML = mask.visible ? ICONS.eye : ICONS.eyeOff;
                    maskToggle.addEventListener('click', () => {
                        const updatedMasks = model.get('_masks_data').map(m =>
                            m.id === mask.id ? { ...m, visible: !m.visible } : m
                        );
                        model.set('_masks_data', updatedMasks);
                        model.save_changes();
                    });

                    const maskLabel = document.createElement('span');
                    maskLabel.textContent = mask.name;
                    maskLabel.className = 'mask-name';

                    const maskActions = document.createElement('div');
                    maskActions.className = 'mask-actions';

                    const downloadBtn = document.createElement('button');
                    downloadBtn.className = 'layer-action-btn';
                    downloadBtn.innerHTML = ICONS.download;
                    downloadBtn.title = 'Download mask';
                    downloadBtn.addEventListener('click', () => {
                        downloadMask(mask);
                    });

                    const maskColor = document.createElement('input');
                    maskColor.type = 'color';
                    maskColor.value = mask.color || '#ff0000';
                    maskColor.className = 'color-swatch';
                    maskColor.addEventListener('input', () => {
                        const updatedMasks = model.get('_masks_data').map(m =>
                            m.id === mask.id ? { ...m, color: maskColor.value } : m
                        );
                        model.set('_masks_data', updatedMasks);
                        model.save_changes();
                    });

                    maskActions.appendChild(downloadBtn);
                    maskActions.appendChild(maskColor);

                    maskItem.appendChild(maskToggle);
                    maskItem.appendChild(maskLabel);
                    maskItem.appendChild(maskActions);
                    layersDropdown.appendChild(maskItem);

                    // Opacity slider for this mask
                    const opacityItem = document.createElement('div');
                    opacityItem.className = 'layer-item opacity-item sub-item';
                    const opacitySlider = document.createElement('input');
                    opacitySlider.type = 'range';
                    opacitySlider.min = '0';
                    opacitySlider.max = '1';
                    opacitySlider.step = '0.05';
                    opacitySlider.value = mask.opacity || 0.5;
                    opacitySlider.addEventListener('input', () => {
                        const updatedMasks = model.get('_masks_data').map(m =>
                            m.id === mask.id ? { ...m, opacity: parseFloat(opacitySlider.value) } : m
                        );
                        model.set('_masks_data', updatedMasks);
                        model.save_changes();
                    });
                    opacityItem.appendChild(opacitySlider);
                    layersDropdown.appendChild(opacityItem);

                    // Contours toggle
                    const contoursItem = document.createElement('div');
                    contoursItem.className = 'layer-item sub-item';
                    const contoursCheck = document.createElement('input');
                    contoursCheck.type = 'checkbox';
                    contoursCheck.checked = mask.contours || false;
                    contoursCheck.addEventListener('change', () => {
                        const updatedMasks = model.get('_masks_data').map(m =>
                            m.id === mask.id ? { ...m, contours: contoursCheck.checked } : m
                        );
                        model.set('_masks_data', updatedMasks);
                        model.save_changes();
                    });
                    const contoursLabel = document.createElement('span');
                    contoursLabel.textContent = 'Contours only';
                    contoursItem.appendChild(contoursCheck);
                    contoursItem.appendChild(contoursLabel);
                    layersDropdown.appendChild(contoursItem);
                }
            }

            // Annotations section
            const annotDivider = document.createElement('div');
            annotDivider.className = 'layer-divider';
            layersDropdown.appendChild(annotDivider);

            const annotHeader = document.createElement('div');
            annotHeader.className = 'layer-header';
            annotHeader.textContent = 'Annotations';
            layersDropdown.appendChild(annotHeader);

            // Rectangles
            const rectItem = createAnnotationLayerItem('Rectangles', 'rois_visible', 'roi_color');
            layersDropdown.appendChild(rectItem);

            // Polygons
            const polyItem = createAnnotationLayerItem('Polygons', 'polygons_visible', 'polygon_color');
            layersDropdown.appendChild(polyItem);

            // Points
            const pointItem = createAnnotationLayerItem('Points', 'points_visible', 'point_color');
            layersDropdown.appendChild(pointItem);
        }

        function createAnnotationLayerItem(label, visibleProp, colorProp) {
            const item = document.createElement('div');
            item.className = 'layer-item';

            const toggle = document.createElement('button');
            toggle.className = 'layer-toggle' + (model.get(visibleProp) ? ' visible' : '');
            toggle.innerHTML = model.get(visibleProp) ? ICONS.eye : ICONS.eyeOff;
            toggle.addEventListener('click', () => {
                model.set(visibleProp, !model.get(visibleProp));
                model.save_changes();
                toggle.classList.toggle('visible', model.get(visibleProp));
                toggle.innerHTML = model.get(visibleProp) ? ICONS.eye : ICONS.eyeOff;
            });

            const labelEl = document.createElement('span');
            labelEl.textContent = label;

            const colorSwatch = document.createElement('input');
            colorSwatch.type = 'color';
            colorSwatch.value = model.get(colorProp);
            colorSwatch.className = 'color-swatch';
            colorSwatch.addEventListener('input', () => {
                model.set(colorProp, colorSwatch.value);
                model.save_changes();
            });

            item.appendChild(toggle);
            item.appendChild(labelEl);
            item.appendChild(colorSwatch);
            return item;
        }

        rebuildLayersDropdown();

        layersGroup.appendChild(layersBtn);
        layersGroup.appendChild(layersDropdown);

        let dropdownOpen = false;
        layersBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdownOpen = !dropdownOpen;
            if (dropdownOpen) rebuildLayersDropdown();
            layersDropdown.classList.toggle('open', dropdownOpen);
        });
        document.addEventListener('click', () => {
            dropdownOpen = false;
            layersDropdown.classList.remove('open');
        });
        layersDropdown.addEventListener('click', (e) => e.stopPropagation());

        toolbar.appendChild(toolGroup);
        toolbar.appendChild(sep1);
        toolbar.appendChild(actionGroup);
        toolbar.appendChild(sep2);
        toolbar.appendChild(layersGroup);

        const canvasWrapper = document.createElement('div');
        canvasWrapper.className = 'canvas-wrapper';

        const canvas = document.createElement('canvas');
        canvas.className = 'viewer-canvas';
        const ctx = canvas.getContext('2d');

        canvasWrapper.appendChild(canvas);

        const statusBar = document.createElement('div');
        statusBar.className = 'status-bar';

        const toolStatus = document.createElement('span');
        toolStatus.className = 'status-item';
        toolStatus.textContent = 'Tool: Pan';

        const posStatus = document.createElement('span');
        posStatus.className = 'status-item';
        posStatus.textContent = 'Position: --';

        const zoomStatus = document.createElement('span');
        zoomStatus.className = 'status-item';
        zoomStatus.textContent = 'Zoom: 100%';

        statusBar.appendChild(toolStatus);
        statusBar.appendChild(posStatus);
        statusBar.appendChild(zoomStatus);

        container.appendChild(toolbar);
        container.appendChild(canvasWrapper);
        container.appendChild(statusBar);
        el.appendChild(container);

        let scale = 1;
        let translateX = 0;
        let translateY = 0;
        let isDragging = false;
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;
        let drawStartX = 0;
        let drawStartY = 0;
        let currentDrawRect = null;
        let currentPolygonPoints = [];
        let currentMousePos = null;
        let baseImage = null;
        let maskCanvases = {};  // Cache for mask canvas elements
        let lastClickedSamCoords = null;  // For SAM mask deletion

        const TOOL_NAMES = {
            'pan': 'Pan',
            'select': 'Select',
            'draw': 'Rectangle',
            'polygon': 'Polygon',
            'point': 'Point'
        };

        function screenToImage(screenX, screenY) {
            return {
                x: (screenX - translateX) / scale,
                y: (screenY - translateY) / scale
            };
        }

        function distance(p1, p2) {
            return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
        }

        function pointInRect(px, py, rect) {
            return px >= rect.x && px <= rect.x + rect.width &&
                   py >= rect.y && py <= rect.y + rect.height;
        }

        function pointInPolygon(px, py, points) {
            if (points.length < 3) return false;
            let inside = false;
            for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
                const xi = points[i].x, yi = points[i].y;
                const xj = points[j].x, yj = points[j].y;
                if (((yi > py) !== (yj > py)) &&
                    (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
                    inside = !inside;
                }
            }
            return inside;
        }

        function pointNearPoint(px, py, point, threshold = 10) {
            const d = Math.sqrt((px - point.x) ** 2 + (py - point.y) ** 2);
            return d <= threshold / scale;
        }

        function findAnnotationAt(imgX, imgY) {
            if (model.get('points_visible')) {
                const points = model.get('_points_data') || [];
                for (const pt of points) {
                    if (pointNearPoint(imgX, imgY, pt, model.get('point_radius') + 5)) {
                        return { type: 'point', id: pt.id };
                    }
                }
            }
            if (model.get('polygons_visible')) {
                const polygons = model.get('_polygons_data') || [];
                for (const poly of polygons) {
                    if (pointInPolygon(imgX, imgY, poly.points)) {
                        return { type: 'polygon', id: poly.id };
                    }
                }
            }
            if (model.get('rois_visible')) {
                const rois = model.get('_rois_data') || [];
                for (const roi of rois) {
                    if (pointInRect(imgX, imgY, roi)) {
                        return { type: 'roi', id: roi.id };
                    }
                }
            }
            return null;
        }

        function deleteSelectedAnnotation() {
            const selectedId = model.get('selected_annotation_id');
            const selectedType = model.get('selected_annotation_type');
            if (!selectedId || !selectedType) return;

            if (selectedType === 'roi') {
                const rois = model.get('_rois_data') || [];
                model.set('_rois_data', rois.filter(r => r.id !== selectedId));
            } else if (selectedType === 'polygon') {
                const polygons = model.get('_polygons_data') || [];
                model.set('_polygons_data', polygons.filter(p => p.id !== selectedId));
            } else if (selectedType === 'point') {
                const points = model.get('_points_data') || [];
                model.set('_points_data', points.filter(p => p.id !== selectedId));
            }

            model.set('selected_annotation_id', '');
            model.set('selected_annotation_type', '');
            model.save_changes();
            renderCanvas();
        }

        async function loadMaskCanvas(mask) {
            if (!mask.data) return null;
            const maskImg = await loadImage(mask.data);
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = model.get('width');
            maskCanvas.height = model.get('height');
            const maskCtx = maskCanvas.getContext('2d');
            maskCtx.drawImage(maskImg, 0, 0);
            return maskCanvas;
        }

        async function updateMaskCanvases() {
            const masks = model.get('_masks_data') || [];
            const newCanvases = {};
            for (const mask of masks) {
                if (maskCanvases[mask.id] && maskCanvases[mask.id].dataHash === mask.data) {
                    newCanvases[mask.id] = maskCanvases[mask.id];
                } else if (mask.data) {
                    const canvas = await loadMaskCanvas(mask);
                    if (canvas) {
                        newCanvases[mask.id] = { canvas, dataHash: mask.data };
                    }
                }
            }
            maskCanvases = newCanvases;
            renderCanvas();
        }

        function renderCanvas() {
            const imgWidth = model.get('width');
            const imgHeight = model.get('height');

            if (imgWidth === 0 || imgHeight === 0) return;

            canvas.width = canvasWrapper.clientWidth || imgWidth;
            canvas.height = canvasWrapper.clientHeight || imgHeight;

            const checkSize = 10;
            ctx.fillStyle = '#2a2a2a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#3a3a3a';
            for (let y = 0; y < canvas.height; y += checkSize) {
                for (let x = 0; x < canvas.width; x += checkSize) {
                    if ((Math.floor(x / checkSize) + Math.floor(y / checkSize)) % 2 === 0) {
                        ctx.fillRect(x, y, checkSize, checkSize);
                    }
                }
            }

            ctx.save();
            ctx.translate(translateX, translateY);
            ctx.scale(scale, scale);

            if (model.get('image_visible') && baseImage) {
                const brightness = model.get('image_brightness') || 0;
                const contrast = model.get('image_contrast') || 0;
                
                // Apply brightness and contrast filters
                const brightnessPercent = (brightness * 100);
                const contrastPercent = ((contrast + 1) * 100);
                ctx.filter = `brightness(${100 + brightnessPercent}%) contrast(${contrastPercent}%)`;
                
                ctx.drawImage(baseImage, 0, 0);
                
                // Reset filter
                ctx.filter = 'none';
            }

            // Draw mask overlays
            const masks = model.get('_masks_data') || [];
            for (const mask of masks) {
                if (mask.visible && maskCanvases[mask.id]) {
                    ctx.globalAlpha = mask.opacity || 0.5;
                    ctx.drawImage(maskCanvases[mask.id].canvas, 0, 0);
                    ctx.globalAlpha = 1.0;
                }
            }

            const selectedId = model.get('selected_annotation_id');
            const selectedType = model.get('selected_annotation_type');

            if (model.get('rois_visible')) {
                const rois = model.get('_rois_data') || [];
                const roiColor = model.get('roi_color');
                for (const roi of rois) {
                    const isSelected = selectedType === 'roi' && selectedId === roi.id;
                    ctx.strokeStyle = isSelected ? '#ffffff' : roiColor;
                    ctx.lineWidth = (isSelected ? 3 : 2) / scale;
                    ctx.fillStyle = roiColor + '33';
                    ctx.fillRect(roi.x, roi.y, roi.width, roi.height);
                    ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
                    if (isSelected) {
                        ctx.setLineDash([5 / scale, 5 / scale]);
                        ctx.strokeStyle = roiColor;
                        ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
                        ctx.setLineDash([]);
                    }
                }
            }

            if (model.get('polygons_visible')) {
                const polygons = model.get('_polygons_data') || [];
                const polyColor = model.get('polygon_color');
                for (const poly of polygons) {
                    if (poly.points.length < 2) continue;
                    const isSelected = selectedType === 'polygon' && selectedId === poly.id;
                    ctx.beginPath();
                    ctx.moveTo(poly.points[0].x, poly.points[0].y);
                    for (let i = 1; i < poly.points.length; i++) {
                        ctx.lineTo(poly.points[i].x, poly.points[i].y);
                    }
                    ctx.closePath();
                    ctx.fillStyle = polyColor + '33';
                    ctx.fill();
                    ctx.strokeStyle = isSelected ? '#ffffff' : polyColor;
                    ctx.lineWidth = (isSelected ? 3 : 2) / scale;
                    ctx.stroke();
                    if (isSelected) {
                        ctx.setLineDash([5 / scale, 5 / scale]);
                        ctx.strokeStyle = polyColor;
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                }
            }

            if (model.get('points_visible')) {
                const points = model.get('_points_data') || [];
                const ptColor = model.get('point_color');
                const ptRadius = model.get('point_radius');
                for (const pt of points) {
                    const isSelected = selectedType === 'point' && selectedId === pt.id;
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, ptRadius / scale, 0, Math.PI * 2);
                    ctx.fillStyle = ptColor;
                    ctx.fill();
                    if (isSelected) {
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2 / scale;
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(pt.x, pt.y, (ptRadius + 4) / scale, 0, Math.PI * 2);
                        ctx.strokeStyle = ptColor;
                        ctx.setLineDash([3 / scale, 3 / scale]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                }
            }

            if (currentDrawRect) {
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2 / scale;
                ctx.setLineDash([5 / scale, 5 / scale]);
                ctx.strokeRect(currentDrawRect.x, currentDrawRect.y, currentDrawRect.width, currentDrawRect.height);
                ctx.setLineDash([]);
            }

            if (currentPolygonPoints.length > 0) {
                ctx.beginPath();
                ctx.moveTo(currentPolygonPoints[0].x, currentPolygonPoints[0].y);
                for (let i = 1; i < currentPolygonPoints.length; i++) {
                    ctx.lineTo(currentPolygonPoints[i].x, currentPolygonPoints[i].y);
                }
                if (currentMousePos) {
                    ctx.lineTo(currentMousePos.x, currentMousePos.y);
                }
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2 / scale;
                ctx.setLineDash([5 / scale, 5 / scale]);
                ctx.stroke();
                ctx.setLineDash([]);

                for (const pt of currentPolygonPoints) {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 4 / scale, 0, Math.PI * 2);
                    ctx.fillStyle = '#00ff00';
                    ctx.fill();
                }

                if (currentPolygonPoints.length >= 3 && currentMousePos) {
                    const d = distance(currentMousePos, currentPolygonPoints[0]) * scale;
                    if (d < 15) {
                        ctx.beginPath();
                        ctx.arc(currentPolygonPoints[0].x, currentPolygonPoints[0].y, 8 / scale, 0, Math.PI * 2);
                        ctx.strokeStyle = '#00ff00';
                        ctx.lineWidth = 2 / scale;
                        ctx.stroke();
                    }
                }
            }

            ctx.restore();
            zoomStatus.textContent = 'Zoom: ' + Math.round(scale * 100) + '%';
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
            currentPolygonPoints = [];
            currentDrawRect = null;
            currentMousePos = null;

            model.set('tool_mode', mode);
            model.save_changes();

            [panBtn, selectBtn, rectBtn, polygonBtn, pointBtn].forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });

            const cursors = { 'pan': 'grab', 'select': 'default', 'draw': 'crosshair', 'polygon': 'crosshair', 'point': 'crosshair' };
            canvas.style.cursor = cursors[mode] || 'default';
            toolStatus.textContent = 'Tool: ' + TOOL_NAMES[mode];
            renderCanvas();
        }

        panBtn.addEventListener('click', () => updateToolMode('pan'));
        selectBtn.addEventListener('click', () => updateToolMode('select'));
        rectBtn.addEventListener('click', () => updateToolMode('draw'));
        polygonBtn.addEventListener('click', () => updateToolMode('polygon'));
        pointBtn.addEventListener('click', () => updateToolMode('point'));
        resetBtn.addEventListener('click', resetView);
        clearBtn.addEventListener('click', () => {
            model.set('_rois_data', []);
            model.set('_polygons_data', []);
            model.set('_points_data', []);
            model.set('selected_annotation_id', '');
            model.set('selected_annotation_type', '');
            model.save_changes();
            renderCanvas();
        });

        container.addEventListener('keydown', (e) => {
            if (e.key === 'p' || e.key === 'P') updateToolMode('pan');
            else if (e.key === 'v' || e.key === 'V') updateToolMode('select');
            else if (e.key === 'r' || e.key === 'R') updateToolMode('draw');
            else if (e.key === 'g' || e.key === 'G') updateToolMode('polygon');
            else if (e.key === 'o' || e.key === 'O') updateToolMode('point');
            else if (e.key === 'Escape') {
                currentPolygonPoints = [];
                currentDrawRect = null;
                lastClickedSamCoords = null;
                model.set('selected_annotation_id', '');
                model.set('selected_annotation_type', '');
                model.save_changes();
                renderCanvas();
            }
            else if (e.key === 'Delete' || e.key === 'Backspace') {
                const selectedId = model.get('selected_annotation_id');
                if (selectedId) {
                    deleteSelectedAnnotation();
                } else if (lastClickedSamCoords) {
                    // Delete SAM mask at clicked location
                    model.set('_delete_sam_at', lastClickedSamCoords);
                    model.save_changes();
                    lastClickedSamCoords = null;
                }
            }
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const zoom = e.deltaY < 0 ? 1.1 : 0.9;
            const newScale = Math.min(Math.max(scale * zoom, 0.1), 50);

            translateX = mouseX - (mouseX - translateX) * (newScale / scale);
            translateY = mouseY - (mouseY - translateY) * (newScale / scale);
            scale = newScale;

            renderCanvas();
        });

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;
            const imgCoords = screenToImage(screenX, screenY);
            const mode = model.get('tool_mode');

            if (mode === 'select') {
                const found = findAnnotationAt(imgCoords.x, imgCoords.y);
                if (found) {
                    model.set('selected_annotation_id', found.id);
                    model.set('selected_annotation_type', found.type);
                    // Clear SAM selection
                    lastClickedSamCoords = null;
                } else {
                    model.set('selected_annotation_id', '');
                    model.set('selected_annotation_type', '');
                    // Store coordinates for potential SAM mask deletion
                    lastClickedSamCoords = { x: Math.round(imgCoords.x), y: Math.round(imgCoords.y) };
                }
                model.save_changes();
                renderCanvas();
            } else if (mode === 'draw') {
                isDrawing = true;
                drawStartX = imgCoords.x;
                drawStartY = imgCoords.y;
                currentDrawRect = { x: drawStartX, y: drawStartY, width: 0, height: 0 };
            } else if (mode === 'point') {
                const points = model.get('_points_data') || [];
                const newPoint = {
                    id: 'pt_' + Date.now(),
                    x: Math.round(imgCoords.x),
                    y: Math.round(imgCoords.y)
                };
                points.push(newPoint);
                model.set('_points_data', [...points]);
                model.save_changes();
                renderCanvas();
            } else if (mode === 'pan') {
                isDragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
                canvas.style.cursor = 'grabbing';
            }
        });

        canvas.addEventListener('click', (e) => {
            if (model.get('tool_mode') !== 'polygon') return;

            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;
            const imgCoords = screenToImage(screenX, screenY);

            if (currentPolygonPoints.length >= 3) {
                const d = distance(imgCoords, currentPolygonPoints[0]) * scale;
                if (d < 15) {
                    const polygons = model.get('_polygons_data') || [];
                    const newPoly = {
                        id: 'poly_' + Date.now(),
                        points: currentPolygonPoints.map(p => ({ x: Math.round(p.x), y: Math.round(p.y) }))
                    };
                    polygons.push(newPoly);
                    model.set('_polygons_data', [...polygons]);
                    model.save_changes();
                    currentPolygonPoints = [];
                    renderCanvas();
                    return;
                }
            }

            currentPolygonPoints.push({ x: imgCoords.x, y: imgCoords.y });
            renderCanvas();
        });

        canvas.addEventListener('dblclick', (e) => {
            if (model.get('tool_mode') !== 'polygon') return;
            if (currentPolygonPoints.length < 3) return;

            const polygons = model.get('_polygons_data') || [];
            const newPoly = {
                id: 'poly_' + Date.now(),
                points: currentPolygonPoints.map(p => ({ x: Math.round(p.x), y: Math.round(p.y) }))
            };
            polygons.push(newPoly);
            model.set('_polygons_data', [...polygons]);
            model.save_changes();
            currentPolygonPoints = [];
            renderCanvas();
        });

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const screenX = e.clientX - rect.left;
            const screenY = e.clientY - rect.top;
            const imgCoords = screenToImage(screenX, screenY);

            const imgWidth = model.get('width');
            const imgHeight = model.get('height');
            if (imgCoords.x >= 0 && imgCoords.x < imgWidth && imgCoords.y >= 0 && imgCoords.y < imgHeight) {
                posStatus.textContent = 'Position: (' + Math.round(imgCoords.x) + ', ' + Math.round(imgCoords.y) + ')';
            } else {
                posStatus.textContent = 'Position: --';
            }

            if (model.get('tool_mode') === 'polygon' && currentPolygonPoints.length > 0) {
                currentMousePos = imgCoords;
                renderCanvas();
            }
        });

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

        window.addEventListener('mouseup', () => {
            if (isDrawing && currentDrawRect) {
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

        const initMode = model.get('tool_mode');
        const cursors = { 'pan': 'grab', 'select': 'default', 'draw': 'crosshair', 'polygon': 'crosshair', 'point': 'crosshair' };
        canvas.style.cursor = cursors[initMode] || 'default';
        toolStatus.textContent = 'Tool: ' + TOOL_NAMES[initMode];

        async function loadBaseImage() {
            const imageData = model.get('image_data');
            if (imageData) {
                baseImage = await loadImage(imageData);
            }
            resetView();
        }

        await loadBaseImage();
        await updateMaskCanvases();

        model.on('change:image_data', loadBaseImage);
        model.on('change:_masks_data', updateMaskCanvases);
        model.on('change:image_visible', renderCanvas);
        model.on('change:image_brightness', renderCanvas);
        model.on('change:image_contrast', renderCanvas);
        model.on('change:rois_visible', renderCanvas);
        model.on('change:polygons_visible', renderCanvas);
        model.on('change:points_visible', renderCanvas);
        model.on('change:_rois_data', renderCanvas);
        model.on('change:_polygons_data', renderCanvas);
        model.on('change:_points_data', renderCanvas);
        model.on('change:roi_color', renderCanvas);
        model.on('change:polygon_color', renderCanvas);
        model.on('change:point_color', renderCanvas);
        model.on('change:width', resetView);
        model.on('change:height', resetView);
        model.on('change:tool_mode', () => {
            const mode = model.get('tool_mode');
            [panBtn, selectBtn, rectBtn, polygonBtn, pointBtn].forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            const cursors = { 'pan': 'grab', 'select': 'default', 'draw': 'crosshair', 'polygon': 'crosshair', 'point': 'crosshair' };
            canvas.style.cursor = cursors[mode] || 'default';
            toolStatus.textContent = 'Tool: ' + TOOL_NAMES[mode];
        });

        new ResizeObserver(() => renderCanvas()).observe(canvasWrapper);
    }

    export default { render };
    """

    _css = """
    .bioimage-viewer {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f8f9fa;
        border-radius: 8px;
        overflow: hidden;
        outline: none;
    }
    .bioimage-viewer:focus {
        box-shadow: 0 0 0 2px #0d6efd33;
    }
    .toolbar {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #ffffff;
        border-bottom: 1px solid #e0e0e0;
    }
    .tool-group {
        display: flex;
        gap: 2px;
    }
    .toolbar-separator {
        width: 1px;
        height: 24px;
        background: #e0e0e0;
        margin: 0 4px;
    }
    .tool-btn {
        width: 32px;
        height: 32px;
        padding: 6px;
        border: none;
        border-radius: 6px;
        background: transparent;
        color: #555;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
    }
    .tool-btn:hover {
        background: #f0f0f0;
        color: #333;
    }
    .tool-btn.active {
        background: #0d6efd;
        color: white;
    }
    .tool-btn.danger:hover {
        background: #dc3545;
        color: white;
    }
    .tool-btn svg {
        width: 18px;
        height: 18px;
    }
    .layers-group {
        position: relative;
        margin-left: auto;
    }
    .layers-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fff;
        color: #555;
        cursor: pointer;
        font-size: 13px;
    }
    .layers-btn:hover {
        background: #f8f8f8;
    }
    .layers-btn svg {
        width: 16px;
        height: 16px;
    }
    .layers-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 4px;
        min-width: 220px;
        max-height: 400px;
        overflow-y: auto;
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 8px 0;
        display: none;
        z-index: 100;
    }
    .layers-dropdown.open {
        display: block;
    }
    .layer-header {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        font-size: 11px;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .layer-header svg {
        width: 14px;
        height: 14px;
    }
    .layer-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        font-size: 13px;
        color: #333;
    }
    .layer-item:hover {
        background: #f8f9fa;
    }
    .layer-item.sub-item {
        padding-left: 44px;
        padding-top: 4px;
        padding-bottom: 4px;
    }
    .layer-item.mask-layer {
        background: #f8f9fa;
    }
    .mask-name {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .mask-actions {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .layer-action-btn {
        width: 24px;
        height: 24px;
        padding: 4px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: #666;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
    }
    .layer-action-btn:hover {
        background: #e0e0e0;
        color: #333;
    }
    .layer-action-btn svg {
        width: 14px;
        height: 14px;
    }
    .layer-toggle {
        width: 24px;
        height: 24px;
        padding: 4px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: #999;
        cursor: pointer;
    }
    .layer-toggle.visible {
        color: #0d6efd;
    }
    .layer-toggle svg {
        width: 16px;
        height: 16px;
    }
    .layer-divider {
        height: 1px;
        background: #e0e0e0;
        margin: 8px 0;
    }
    .color-swatch {
        width: 24px;
        height: 24px;
        padding: 0;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
        flex-shrink: 0;
    }
    .opacity-item {
        flex-direction: column;
        align-items: stretch;
        gap: 4px;
    }
    .opacity-item input[type="range"] {
        width: 100%;
        height: 4px;
        border-radius: 2px;
        -webkit-appearance: none;
        background: #e0e0e0;
    }
    .opacity-item input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
    }
    .slider-item {
        flex-direction: column;
        align-items: stretch;
        gap: 6px;
    }
    .slider-label {
        font-size: 11px;
        color: #666;
        font-weight: 500;
    }
    .adjustment-slider {
        width: 100%;
        height: 4px;
        border-radius: 2px;
        -webkit-appearance: none;
        background: linear-gradient(to right, #666 0%, #e0e0e0 50%, #fff 100%);
    }
    .adjustment-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
    }
    .adjustment-slider::-moz-range-thumb {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
        border: none;
    }
    .canvas-wrapper {
        position: relative;
        width: 100%;
        height: 500px;
        overflow: hidden;
    }
    .viewer-canvas {
        display: block;
    }
    .status-bar {
        display: flex;
        gap: 24px;
        padding: 6px 12px;
        background: #f0f0f0;
        border-top: 1px solid #e0e0e0;
        font-size: 12px;
        color: #666;
    }
    .status-item {
        white-space: nowrap;
    }

    @media (prefers-color-scheme: dark) {
        .bioimage-viewer {
            background: #1e1e1e;
        }
        .toolbar {
            background: #2d2d2d;
            border-color: #404040;
        }
        .toolbar-separator {
            background: #404040;
        }
        .tool-btn {
            color: #aaa;
        }
        .tool-btn:hover {
            background: #3d3d3d;
            color: #fff;
        }
        .tool-btn.active {
            background: #0d6efd;
            color: white;
        }
        .layers-btn {
            background: #2d2d2d;
            border-color: #404040;
            color: #aaa;
        }
        .layers-btn:hover {
            background: #3d3d3d;
        }
        .layers-dropdown {
            background: #2d2d2d;
            border-color: #404040;
        }
        .layer-header {
            color: #888;
        }
        .layer-item {
            color: #e0e0e0;
        }
        .layer-item:hover {
            background: #3d3d3d;
        }
        .layer-item.mask-layer {
            background: #353535;
        }
        .layer-toggle {
            color: #666;
        }
        .layer-toggle.visible {
            color: #0d6efd;
        }
        .layer-action-btn {
            color: #888;
        }
        .layer-action-btn:hover {
            background: #404040;
            color: #fff;
        }
        .layer-divider {
            background: #404040;
        }
        .status-bar {
            background: #252525;
            border-color: #404040;
            color: #888;
        }
        .opacity-item input[type="range"] {
            background: #404040;
        }
        .slider-label {
            color: #888;
        }
        .adjustment-slider {
            background: linear-gradient(to right, #333 0%, #666 50%, #999 100%);
        }
    }
    """

    # Image data
    image_data = traitlets.Unicode("").tag(sync=True)

    # Multiple mask layers: [{id, name, data, visible, opacity, color, contours}]
    _masks_data = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Layer controls
    image_visible = traitlets.Bool(True).tag(sync=True)
    image_brightness = traitlets.Float(0.0).tag(sync=True)  # -1.0 to 1.0
    image_contrast = traitlets.Float(0.0).tag(sync=True)    # -1.0 to 1.0

    # Image dimensions
    width = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)

    # Tool mode
    tool_mode = traitlets.Unicode("pan").tag(sync=True)

    # Annotations
    _rois_data = traitlets.List(traitlets.Dict()).tag(sync=True)
    rois_visible = traitlets.Bool(True).tag(sync=True)
    roi_color = traitlets.Unicode("#ff0000").tag(sync=True)

    _polygons_data = traitlets.List(traitlets.Dict()).tag(sync=True)
    polygons_visible = traitlets.Bool(True).tag(sync=True)
    polygon_color = traitlets.Unicode("#00ff00").tag(sync=True)

    _points_data = traitlets.List(traitlets.Dict()).tag(sync=True)
    points_visible = traitlets.Bool(True).tag(sync=True)
    point_color = traitlets.Unicode("#0066ff").tag(sync=True)
    point_radius = traitlets.Int(5).tag(sync=True)

    # Selection
    selected_annotation_id = traitlets.Unicode("").tag(sync=True)
    selected_annotation_type = traitlets.Unicode("").tag(sync=True)

    # SAM label deletion - set coordinates to delete SAM label at that position
    _delete_sam_at = traitlets.Dict(allow_none=True).tag(sync=True)

    def set_image(self, data: np.ndarray):
        """Set the base image from a numpy array."""
        if data.ndim > 2:
            data = data.squeeze()
            if data.ndim > 2:
                data = data[0] if data.ndim == 3 else data[0, 0]

        self.height, self.width = data.shape[:2]
        normalized = _normalize_image(data)

        # Store for SAM integration
        self._image_array = normalized

        self.image_data = _array_to_base64(normalized)

    def add_mask(
        self,
        labels: np.ndarray,
        name: str | None = None,
        color: str | None = None,
        opacity: float = 0.5,
        visible: bool = True,
        contours_only: bool = False,
        contour_width: int = 1,
    ) -> str:
        """Add a mask layer.

        Args:
            labels: 2D numpy array of label values
            name: Display name for the mask layer
            color: Hex color for the mask (auto-assigned if None)
            opacity: Opacity value 0-1
            visible: Whether the mask is visible
            contours_only: If True, show only contours
            contour_width: Width of contours in pixels

        Returns:
            The mask ID
        """
        if labels.ndim > 2:
            labels = labels.squeeze()
            if labels.ndim > 2:
                labels = labels[0] if labels.ndim == 3 else labels[0, 0]

        mask_id = f"mask_{len(self._masks_data)}_{id(labels)}"

        # Store raw labels
        self._mask_arrays[mask_id] = labels

        # Auto-assign color
        if color is None:
            color = MASK_COLORS[len(self._masks_data) % len(MASK_COLORS)]

        # Auto-assign name
        if name is None:
            name = f"Mask {len(self._masks_data) + 1}"

        # Generate RGBA
        rgba = _labels_to_rgba(labels, contours_only=contours_only, contour_width=contour_width)
        data_b64 = _array_to_base64(rgba)

        # Cache
        self._mask_caches[mask_id] = {
            (contours_only, contour_width): data_b64
        }

        mask_entry = {
            "id": mask_id,
            "name": name,
            "data": data_b64,
            "visible": visible,
            "opacity": opacity,
            "color": color,
            "contours": contours_only,
            "contour_width": contour_width,
        }

        self._masks_data = [*self._masks_data, mask_entry]
        return mask_id

    def set_mask(
        self,
        labels: np.ndarray,
        name: str | None = None,
        contours_only: bool = False,
        contour_width: int = 1,
    ):
        """Set a single mask (convenience method, clears existing masks).

        For backward compatibility with single-mask API.
        """
        self._masks_data = []
        self._mask_arrays = {}
        self._mask_caches = {}
        self.add_mask(labels, name=name or "Mask", contours_only=contours_only, contour_width=contour_width)

    def remove_mask(self, mask_id: str):
        """Remove a mask layer by ID."""
        self._masks_data = [m for m in self._masks_data if m["id"] != mask_id]
        if mask_id in self._mask_arrays:
            del self._mask_arrays[mask_id]
        if mask_id in self._mask_caches:
            del self._mask_caches[mask_id]

    def clear_masks(self):
        """Remove all mask layers."""
        self._masks_data = []
        self._mask_arrays = {}
        self._mask_caches = {}

    def update_mask_settings(self, mask_id: str, **kwargs):
        """Update settings for a mask layer.

        Args:
            mask_id: The mask ID to update
            **kwargs: Settings to update (name, color, opacity, visible, contours, contour_width)
        """
        updated = []
        for mask in self._masks_data:
            if mask["id"] == mask_id:
                new_mask = {**mask, **kwargs}

                # Regenerate if contour settings changed
                if "contours" in kwargs or "contour_width" in kwargs:
                    if mask_id in self._mask_arrays:
                        contours = new_mask.get("contours", False)
                        width = new_mask.get("contour_width", 1)
                        cache_key = (contours, width)

                        if mask_id in self._mask_caches and cache_key in self._mask_caches[mask_id]:
                            new_mask["data"] = self._mask_caches[mask_id][cache_key]
                        else:
                            rgba = _labels_to_rgba(
                                self._mask_arrays[mask_id],
                                contours_only=contours,
                                contour_width=width
                            )
                            data_b64 = _array_to_base64(rgba)
                            if mask_id not in self._mask_caches:
                                self._mask_caches[mask_id] = {}
                            self._mask_caches[mask_id][cache_key] = data_b64
                            new_mask["data"] = data_b64

                updated.append(new_mask)
            else:
                updated.append(mask)
        self._masks_data = updated

    def get_mask_ids(self) -> list[str]:
        """Get list of all mask IDs."""
        return [m["id"] for m in self._masks_data]

    @property
    def masks_df(self) -> pd.DataFrame:
        """Get mask layers as a pandas DataFrame."""
        if not self._masks_data:
            return pd.DataFrame(columns=["id", "name", "visible", "opacity", "color", "contours"])
        return pd.DataFrame([
            {k: v for k, v in m.items() if k != "data"}
            for m in self._masks_data
        ])

    @property
    def rois_df(self) -> pd.DataFrame:
        """Get ROIs as a pandas DataFrame."""
        if not self._rois_data:
            return pd.DataFrame(columns=["id", "x", "y", "width", "height"])
        return pd.DataFrame(self._rois_data)

    @rois_df.setter
    def rois_df(self, df: pd.DataFrame):
        self._rois_data = df.to_dict("records")

    @property
    def polygons_df(self) -> pd.DataFrame:
        """Get polygons as a pandas DataFrame."""
        if not self._polygons_data:
            return pd.DataFrame(columns=["id", "points", "num_vertices"])
        data = []
        for poly in self._polygons_data:
            data.append({
                "id": poly["id"],
                "points": poly["points"],
                "num_vertices": len(poly["points"])
            })
        return pd.DataFrame(data)

    @polygons_df.setter
    def polygons_df(self, df: pd.DataFrame):
        records = df.to_dict("records")
        self._polygons_data = [{"id": r["id"], "points": r["points"]} for r in records]

    @property
    def points_df(self) -> pd.DataFrame:
        """Get points as a pandas DataFrame."""
        if not self._points_data:
            return pd.DataFrame(columns=["id", "x", "y"])
        return pd.DataFrame(self._points_data)

    @points_df.setter
    def points_df(self, df: pd.DataFrame):
        self._points_data = df.to_dict("records")

    def clear_rois(self):
        """Clear all rectangle ROIs."""
        self._rois_data = []

    def clear_polygons(self):
        """Clear all polygons."""
        self._polygons_data = []

    def clear_points(self):
        """Clear all points."""
        self._points_data = []

    def clear_all_annotations(self):
        """Clear all annotations."""
        self._rois_data = []
        self._polygons_data = []
        self._points_data = []
        self.selected_annotation_id = ""
        self.selected_annotation_type = ""

    # ==================== SAM Integration ====================

    def enable_sam(self, model_type: str = "mobile_sam"):
        """Enable SAM segmentation triggered by rectangle ROIs.

        Args:
            model_type: SAM model to use. Options:
                - "mobile_sam" (default, ~40MB, fastest)
                - "sam_b" (SAM base)
                - "sam_l" (SAM large)
                - "fast_sam" (FastSAM)

        Requires: pip install ultralytics
        """
        try:
            from ultralytics import SAM
        except ImportError:
            raise ImportError(
                "SAM requires ultralytics. Install with: pip install ultralytics"
            )

        # Model file mapping
        model_files = {
            "mobile_sam": "mobile_sam.pt",
            "sam_b": "sam_b.pt",
            "sam_l": "sam_l.pt",
            "fast_sam": "FastSAM-s.pt",
        }

        if model_type not in model_files:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(model_files.keys())}")

        self._sam_model = SAM(model_files[model_type])
        self._sam_enabled = True
        self._sam_model_type = model_type
        self._processed_roi_ids = set()
        self._processed_point_ids = set()
        self._sam_label_counter = 0
        self._sam_mask_id = None
        self._sam_labels_array = None

        # Set up observers for ROI and point changes
        self.observe(self._on_rois_changed, names=["_rois_data"])
        self.observe(self._on_points_changed, names=["_points_data"])

    def disable_sam(self):
        """Disable SAM segmentation."""
        self._sam_enabled = False
        self._sam_model = None
        self._sam_mask_id = None
        self._sam_labels_array = None
        self._sam_label_counter = 0
        try:
            self.unobserve(self._on_rois_changed, names=["_rois_data"])
        except ValueError:
            pass  # Observer might not be registered
        try:
            self.unobserve(self._on_points_changed, names=["_points_data"])
        except ValueError:
            pass

    def clear_sam_masks(self):
        """Clear all SAM-generated masks and reset the label counter."""
        if self._sam_mask_id and self._sam_mask_id in [m["id"] for m in self._masks_data]:
            self.remove_mask(self._sam_mask_id)
        self._sam_mask_id = None
        self._sam_labels_array = None
        self._sam_label_counter = 0
        self._processed_roi_ids = set()
        self._processed_point_ids = set()

    def _on_delete_sam_at(self, change):
        """Observer callback to delete SAM label at given coordinates."""
        coords = change.get("new")
        if coords and isinstance(coords, dict) and "x" in coords and "y" in coords:
            self.delete_sam_label_at(int(coords["x"]), int(coords["y"]))

    def delete_sam_label_at(self, x: int, y: int):
        """Delete the SAM label at the given coordinates."""
        if self._sam_labels_array is None:
            return

        # Check bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        # Get the label at this position
        label = self._sam_labels_array[y, x]
        if label == 0:
            return  # No label at this position

        # Remove this label from the array
        self._sam_labels_array[self._sam_labels_array == label] = 0

        # Check if any labels remain
        if np.max(self._sam_labels_array) == 0:
            # No labels left, remove the mask layer
            if self._sam_mask_id:
                self.remove_mask(self._sam_mask_id)
            self._sam_mask_id = None
            self._sam_labels_array = None
        else:
            # Update the mask layer
            self._update_sam_mask_layer()

    def _on_rois_changed(self, change):
        """Observer callback when ROIs change."""
        if not getattr(self, "_sam_enabled", False):
            return

        new_rois = change.get("new", [])
        if not new_rois:
            return

        # Find new ROIs that haven't been processed
        for roi in new_rois:
            roi_id = roi.get("id", "")
            if roi_id and roi_id not in self._processed_roi_ids:
                self._processed_roi_ids.add(roi_id)
                self._run_sam_on_roi(roi)

    def _on_points_changed(self, change):
        """Observer callback when points change."""
        if not getattr(self, "_sam_enabled", False):
            return

        new_points = change.get("new", [])
        if not new_points:
            return

        # Find new points that haven't been processed
        for point in new_points:
            point_id = point.get("id", "")
            if point_id and point_id not in self._processed_point_ids:
                self._processed_point_ids.add(point_id)
                self._run_sam_on_point(point)

    def _run_sam_on_point(self, point: dict):
        """Run SAM segmentation on a point prompt."""
        if not hasattr(self, "_sam_model") or self._sam_model is None:
            return

        if not hasattr(self, "_image_array") or self._image_array is None:
            return

        x = int(point["x"])
        y = int(point["y"])

        try:
            # Ensure array is contiguous
            image = np.ascontiguousarray(self._image_array).copy()

            # Convert grayscale to RGB if needed
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            # Run SAM prediction with point prompt (label=1 means foreground)
            results = self._sam_model.predict(
                image,
                points=[[x, y]],
                labels=[1],
                verbose=False
            )

            if results and len(results) > 0 and results[0].masks is not None:
                # Get the binary mask
                mask_data = results[0].masks.data[0].cpu().numpy().astype(bool)

                # Increment label counter for unique color
                self._sam_label_counter += 1

                # Initialize or update the combined labels array
                if self._sam_labels_array is None:
                    self._sam_labels_array = np.zeros(
                        (self.height, self.width), dtype=np.uint16
                    )

                # Add new mask with unique label
                new_mask_region = mask_data & (self._sam_labels_array == 0)
                self._sam_labels_array[new_mask_region] = self._sam_label_counter

                # Update or create the SAM mask layer
                if self._sam_mask_id is None:
                    self._sam_mask_id = self.add_mask(
                        self._sam_labels_array,
                        name="SAM Masks",
                        opacity=0.5,
                        contours_only=False
                    )
                else:
                    self._update_sam_mask_layer()

                # Remove the point after processing
                self._points_data = [p for p in self._points_data if p["id"] != point["id"]]

        except Exception as e:
            print(f"SAM point prediction failed: {e}")

    def _run_sam_on_roi(self, roi: dict):
        """Run SAM segmentation on a bounding box ROI."""
        if not hasattr(self, "_sam_model") or self._sam_model is None:
            return

        if not hasattr(self, "_image_array") or self._image_array is None:
            return

        # Convert ROI to bounding box format [x1, y1, x2, y2]
        x1 = int(roi["x"])
        y1 = int(roi["y"])
        x2 = int(roi["x"] + roi["width"])
        y2 = int(roi["y"] + roi["height"])

        try:
            # Ensure array is contiguous (fixes negative stride error)
            image = np.ascontiguousarray(self._image_array).copy()

            # Convert grayscale to RGB if needed (SAM expects RGB)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            # Run SAM prediction
            results = self._sam_model.predict(
                image,
                bboxes=[[x1, y1, x2, y2]],
                verbose=False
            )

            if results and len(results) > 0 and results[0].masks is not None:
                # Get the binary mask
                mask_data = results[0].masks.data[0].cpu().numpy().astype(bool)

                # Increment label counter for unique color
                self._sam_label_counter += 1

                # Initialize or update the combined labels array
                if self._sam_labels_array is None:
                    self._sam_labels_array = np.zeros(
                        (self.height, self.width), dtype=np.uint16
                    )

                # Add new mask with unique label (don't overwrite existing labels)
                new_mask_region = mask_data & (self._sam_labels_array == 0)
                self._sam_labels_array[new_mask_region] = self._sam_label_counter

                # Update or create the SAM mask layer
                if self._sam_mask_id is None:
                    self._sam_mask_id = self.add_mask(
                        self._sam_labels_array,
                        name="SAM Masks",
                        opacity=0.5,
                        contours_only=False
                    )
                else:
                    # Update existing mask layer
                    self._update_sam_mask_layer()

                # Remove the ROI after processing
                self._rois_data = [r for r in self._rois_data if r["id"] != roi["id"]]

        except Exception as e:
            print(f"SAM prediction failed: {e}")

    def _update_sam_mask_layer(self):
        """Update the SAM mask layer with new labels."""
        if self._sam_mask_id is None or self._sam_labels_array is None:
            return

        # Store raw labels
        self._mask_arrays[self._sam_mask_id] = self._sam_labels_array

        # Generate new RGBA
        rgba = _labels_to_rgba(self._sam_labels_array, contours_only=False, contour_width=1)
        data_b64 = _array_to_base64(rgba)

        # Update cache
        self._mask_caches[self._sam_mask_id] = {(False, 1): data_b64}

        # Update the mask data
        updated_masks = []
        for mask in self._masks_data:
            if mask["id"] == self._sam_mask_id:
                updated_masks.append({**mask, "data": data_b64})
            else:
                updated_masks.append(mask)
        self._masks_data = updated_masks

