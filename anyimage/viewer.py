"""BioImageViewer - Main anywidget for viewing bioimages with multi-dimensional support."""

from concurrent.futures import ThreadPoolExecutor

import anywidget
import traitlets

from .mixins import (
    AnnotationsMixin,
    ImageLoadingMixin,
    MaskManagementMixin,
    SAMIntegrationMixin,
)
from .mixins.image_loading import LRUCache
from .profiling import Profiler


class BioImageViewer(
    ImageLoadingMixin,
    MaskManagementMixin,
    AnnotationsMixin,
    SAMIntegrationMixin,
    anywidget.AnyWidget,
):
    """Anywidget for viewing bioimages with multiple mask layers and annotation tools."""

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

    # 5D dimension sizes (TCZYX)
    dim_t = traitlets.Int(1).tag(sync=True)
    dim_c = traitlets.Int(1).tag(sync=True)
    dim_z = traitlets.Int(1).tag(sync=True)

    # Current position in each dimension
    current_t = traitlets.Int(0).tag(sync=True)
    current_c = traitlets.Int(0).tag(sync=True)
    current_z = traitlets.Int(0).tag(sync=True)

    # Multi-resolution support
    resolution_levels = traitlets.List(traitlets.Int()).tag(sync=True)
    current_resolution = traitlets.Int(0).tag(sync=True)
    _preview_mode = traitlets.Bool(False).tag(sync=True)  # True when actively scrubbing

    # Scene support
    scenes = traitlets.List(traitlets.Unicode()).tag(sync=True)
    current_scene = traitlets.Unicode("").tag(sync=True)

    # Channel settings for composite view: [{name, color, visible, min, max}]
    # Colors are hex strings, min/max are 0-1 normalized contrast limits
    _channel_settings = traitlets.List(traitlets.Dict()).tag(sync=True)

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

    # Viewer layout
    canvas_height = traitlets.Int(800).tag(sync=True)

    # Tile-based loading
    _tile_size = traitlets.Int(256).tag(sync=True)
    _tile_request = traitlets.Dict(allow_none=True).tag(sync=True)
    _tiles_data = traitlets.Dict({}).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mask_arrays = {}  # Store raw label arrays by mask id
        self._mask_caches = {}  # Cache rendered versions by mask id
        self._bioimage = None  # Store BioImage reference for lazy loading
        self._slice_cache = LRUCache(max_size=128)  # LRU cache: (T, C, Z) -> np.ndarray
        self._tile_cache = LRUCache(max_size=2048)  # LRU cache: (t, z, tx, ty, res) -> tile data
        self._prefetch_executor = ThreadPoolExecutor(max_workers=4)  # Background prefetching
        self._profiler = Profiler.get_instance()  # Performance profiler

        # Observer for SAM label deletion
        self.observe(self._on_delete_sam_at, names=["_delete_sam_at"])

        # Observers for dimension changes
        self.observe(self._on_dimension_change, names=["current_t", "current_z"])
        self.observe(self._on_resolution_change, names=["current_resolution"])
        self.observe(self._on_scene_change, names=["current_scene"])
        self.observe(self._on_channel_settings_change, names=["_channel_settings"])

        # Observer for tile-based loading
        self.observe(self._on_tile_request, names=["_tile_request"])

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
        const TILE_SIZE = model.get('_tile_size') || 256;
        const tileCache = new Map();
        const pendingTiles = new Set();
        const MAX_CACHE_SIZE = 2048;
        let useTileMode = false;

        function getVisibleTiles(scale, translateX, translateY, canvasWidth, canvasHeight, imageWidth, imageHeight, t, z) {
            const minX = Math.max(0, -translateX / scale);
            const minY = Math.max(0, -translateY / scale);
            const maxX = Math.min(imageWidth, (-translateX + canvasWidth) / scale);
            const maxY = Math.min(imageHeight, (-translateY + canvasHeight) / scale);

            const tiles = [];
            const startTileX = Math.max(0, Math.floor(minX / TILE_SIZE));
            const startTileY = Math.max(0, Math.floor(minY / TILE_SIZE));
            const endTileX = Math.ceil(maxX / TILE_SIZE);
            const endTileY = Math.ceil(maxY / TILE_SIZE);

            for (let ty = startTileY; ty < endTileY; ty++) {
                for (let tx = startTileX; tx < endTileX; tx++) {
                    tiles.push({ tx, ty, key: `${t}_${z}_${tx}_${ty}` });
                }
            }
            return tiles;
        }

        function cacheTile(key, img) {
            if (tileCache.size >= MAX_CACHE_SIZE) {
                let oldestKey = null;
                let oldestTime = Infinity;
                for (const [k, v] of tileCache) {
                    if (v.lastAccess < oldestTime) {
                        oldestTime = v.lastAccess;
                        oldestKey = k;
                    }
                }
                if (oldestKey) tileCache.delete(oldestKey);
            }
            tileCache.set(key, { img, lastAccess: Date.now() });
        }

        function requestTiles(tiles, t, z) {
            const cached = tiles.filter(tile => tileCache.has(tile.key)).length;
            const missing = tiles.filter(tile => !tileCache.has(tile.key) && !pendingTiles.has(tile.key));
            if (missing.length === 0) {
                if (cached > 0) console.log(`[JS] All ${cached} tiles from cache`);
                return;
            }

            console.log(`[JS] Requesting ${missing.length} tiles (${cached} cached) for T=${t} Z=${z}`);
            missing.forEach(tile => pendingTiles.add(tile.key));
            model.set('_tile_request', {
                tiles: missing.map(t => ({ tx: t.tx, ty: t.ty })),
                t, z, timestamp: Date.now()
            });
            model.save_changes();
        }

        // Prefetch tiles for adjacent T/Z slices
        function prefetchAdjacentTiles(visibleTiles, t, z) {
            const dimT = model.get('dim_t') || 1;
            const dimZ = model.get('dim_z') || 1;
            const adjacentSlices = [];

            // Adjacent T (t-1, t+1)
            if (t > 0) adjacentSlices.push({ t: t - 1, z });
            if (t < dimT - 1) adjacentSlices.push({ t: t + 1, z });
            // Adjacent Z (z-1, z+1)
            if (z > 0) adjacentSlices.push({ t, z: z - 1 });
            if (z < dimZ - 1) adjacentSlices.push({ t, z: z + 1 });

            for (const slice of adjacentSlices) {
                const prefetchTiles = visibleTiles.map(tile => ({
                    tx: tile.tx, ty: tile.ty,
                    key: `${slice.t}_${slice.z}_${tile.tx}_${tile.ty}`
                }));
                const missing = prefetchTiles.filter(tile => !tileCache.has(tile.key) && !pendingTiles.has(tile.key));
                if (missing.length > 0) {
                    missing.forEach(tile => pendingTiles.add(tile.key));
                    model.set('_tile_request', {
                        tiles: missing.map(t => ({ tx: t.tx, ty: t.ty })),
                        t: slice.t, z: slice.z, timestamp: Date.now(), prefetch: true
                    });
                    model.save_changes();
                    break; // Only prefetch one slice at a time to avoid overload
                }
            }
        }

        function clearTileCache() {
            tileCache.clear();
            pendingTiles.clear();
        }

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
                    a.download = `${mask.name.replace(/\\s+/g, '_')}.png`;
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
        canvasWrapper.style.height = (model.get('canvas_height') || 800) + 'px';

        const canvas = document.createElement('canvas');
        canvas.className = 'viewer-canvas';
        const ctx = canvas.getContext('2d');

        canvasWrapper.appendChild(canvas);

        // Dimension controls panel
        const dimControls = document.createElement('div');
        dimControls.className = 'dimension-controls';

        // Play state for time series
        let playInterval = null;
        let playSpeed = 200; // ms per frame

        function createDimSlider(label, dimKey, currentKey, maxVal, showPlayBtn = false) {
            const wrapper = document.createElement('div');
            wrapper.className = 'dim-slider-wrapper';

            const labelEl = document.createElement('span');
            labelEl.className = 'dim-label';
            labelEl.textContent = label;

            // Play button for time series
            let playBtn = null;
            if (showPlayBtn && maxVal > 1) {
                playBtn = document.createElement('button');
                playBtn.className = 'play-btn';
                playBtn.innerHTML = '\u25B6';
                playBtn.title = 'Play/Pause';
                playBtn.addEventListener('click', () => {
                    if (playInterval) {
                        clearInterval(playInterval);
                        playInterval = null;
                        playBtn.innerHTML = '\u25B6';
                    } else {
                        playBtn.innerHTML = '\u23F8';
                        playInterval = setInterval(() => {
                            let val = model.get(currentKey) + 1;
                            if (val >= maxVal) val = 0;
                            model.set(currentKey, val);
                            model.save_changes();
                        }, playSpeed);
                    }
                });
            }

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = '0';
            slider.max = String(maxVal - 1);
            slider.value = String(model.get(currentKey) || 0);
            slider.className = 'dim-slider';

            const valueEl = document.createElement('span');
            valueEl.className = 'dim-value';
            valueEl.textContent = `${model.get(currentKey) || 0}/${maxVal}`;

            let debounceTimer = null;
            let previewTimer = null;

            slider.addEventListener('input', () => {
                const val = parseInt(slider.value);
                valueEl.textContent = `${val}/${maxVal}`;

                if (!model.get('_preview_mode')) {
                    model.set('_preview_mode', true);
                    model.save_changes();
                }

                if (previewTimer) clearTimeout(previewTimer);
                previewTimer = setTimeout(() => {
                    model.set(currentKey, val);
                    model.save_changes();
                }, 30);

                if (debounceTimer) clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    model.set('_preview_mode', false);
                    model.set(currentKey, val);
                    model.save_changes();
                }, 200);
            });

            model.on(`change:${currentKey}`, () => {
                slider.value = String(model.get(currentKey));
                valueEl.textContent = `${model.get(currentKey)}/${maxVal}`;
            });

            wrapper.appendChild(labelEl);
            if (playBtn) wrapper.appendChild(playBtn);
            wrapper.appendChild(slider);
            wrapper.appendChild(valueEl);
            return wrapper;
        }

        function createSceneSelector() {
            const wrapper = document.createElement('div');
            wrapper.className = 'scene-selector-wrapper';

            const labelEl = document.createElement('span');
            labelEl.className = 'dim-label';
            labelEl.textContent = 'Scene';

            const select = document.createElement('select');
            select.className = 'scene-select';

            const scenes = model.get('scenes') || [];
            scenes.forEach(scene => {
                const opt = document.createElement('option');
                opt.value = scene;
                opt.textContent = scene;
                if (scene === model.get('current_scene')) opt.selected = true;
                select.appendChild(opt);
            });

            select.addEventListener('change', () => {
                model.set('current_scene', select.value);
                model.save_changes();
            });

            wrapper.appendChild(labelEl);
            wrapper.appendChild(select);
            return wrapper;
        }

        function createChannelControls() {
            const wrapper = document.createElement('div');
            wrapper.className = 'channel-controls';

            const channelSettings = model.get('_channel_settings') || [];
            if (channelSettings.length <= 1) return wrapper;

            const label = document.createElement('span');
            label.className = 'dim-label';
            label.textContent = 'Channels';
            wrapper.appendChild(label);

            channelSettings.forEach((ch, idx) => {
                const chItem = document.createElement('div');
                chItem.className = 'channel-chip';

                const toggle = document.createElement('input');
                toggle.type = 'checkbox';
                toggle.checked = ch.visible !== false;
                toggle.className = 'channel-toggle';
                toggle.addEventListener('change', (e) => {
                    e.stopPropagation();
                    const settings = [...model.get('_channel_settings')];
                    settings[idx] = { ...settings[idx], visible: toggle.checked };
                    model.set('_channel_settings', settings);
                    model.save_changes();
                });

                const colorDot = document.createElement('span');
                colorDot.className = 'channel-dot';
                colorDot.style.backgroundColor = ch.color || '#ffffff';

                const name = document.createElement('span');
                name.className = 'channel-name';
                name.textContent = ch.name || `Ch ${idx}`;

                chItem.appendChild(toggle);
                chItem.appendChild(colorDot);
                chItem.appendChild(name);

                // Popup for contrast controls
                const popup = document.createElement('div');
                popup.className = 'channel-popup';

                // Prevent popup from closing when interacting inside it
                popup.addEventListener('click', (e) => e.stopPropagation());
                popup.addEventListener('mousedown', (e) => e.stopPropagation());
                popup.addEventListener('mouseup', (e) => e.stopPropagation());

                const popupHeader = document.createElement('div');
                popupHeader.className = 'popup-header';
                popupHeader.textContent = ch.name || `Channel ${idx}`;

                const colorRow = document.createElement('div');
                colorRow.className = 'popup-row';
                const colorLabel = document.createElement('span');
                colorLabel.textContent = 'Color';
                const colorPicker = document.createElement('input');
                colorPicker.type = 'color';
                colorPicker.value = ch.color || '#ffffff';
                colorPicker.className = 'popup-color';
                colorPicker.addEventListener('click', (e) => e.stopPropagation());
                colorPicker.addEventListener('input', (e) => {
                    const settings = [...model.get('_channel_settings')];
                    settings[idx] = { ...settings[idx], color: colorPicker.value };
                    model.set('_channel_settings', settings);
                    model.save_changes();
                    colorDot.style.backgroundColor = colorPicker.value;
                });
                colorRow.appendChild(colorLabel);
                colorRow.appendChild(colorPicker);

                const minRow = document.createElement('div');
                minRow.className = 'popup-row';
                const minLabel = document.createElement('span');
                minLabel.textContent = 'Min';
                const minSlider = document.createElement('input');
                minSlider.type = 'range';
                minSlider.min = '0';
                minSlider.max = '100';
                minSlider.value = String((ch.min || 0) * 100);
                minSlider.className = 'popup-slider';
                const minValue = document.createElement('span');
                minValue.className = 'popup-value';
                minValue.textContent = Math.round((ch.min || 0) * 100) + '%';
                minSlider.addEventListener('input', (e) => {
                    const settings = [...model.get('_channel_settings')];
                    settings[idx] = { ...settings[idx], min: parseInt(minSlider.value) / 100 };
                    model.set('_channel_settings', settings);
                    model.save_changes();
                    minValue.textContent = minSlider.value + '%';
                });
                minRow.appendChild(minLabel);
                minRow.appendChild(minSlider);
                minRow.appendChild(minValue);

                const maxRow = document.createElement('div');
                maxRow.className = 'popup-row';
                const maxLabel = document.createElement('span');
                maxLabel.textContent = 'Max';
                const maxSlider = document.createElement('input');
                maxSlider.type = 'range';
                maxSlider.min = '0';
                maxSlider.max = '100';
                maxSlider.value = String((ch.max || 1) * 100);
                maxSlider.className = 'popup-slider';
                const maxValue = document.createElement('span');
                maxValue.className = 'popup-value';
                maxValue.textContent = Math.round((ch.max || 1) * 100) + '%';
                maxSlider.addEventListener('input', (e) => {
                    const settings = [...model.get('_channel_settings')];
                    settings[idx] = { ...settings[idx], max: parseInt(maxSlider.value) / 100 };
                    model.set('_channel_settings', settings);
                    model.save_changes();
                    maxValue.textContent = maxSlider.value + '%';
                });
                maxRow.appendChild(maxLabel);
                maxRow.appendChild(maxSlider);
                maxRow.appendChild(maxValue);

                popup.appendChild(popupHeader);
                popup.appendChild(colorRow);
                popup.appendChild(minRow);
                popup.appendChild(maxRow);

                chItem.appendChild(popup);

                // Toggle popup on click (only on the chip itself, not the popup)
                chItem.addEventListener('click', (e) => {
                    if (e.target === toggle || popup.contains(e.target)) return;
                    e.stopPropagation();
                    // Close other popups
                    wrapper.querySelectorAll('.channel-popup.open').forEach(p => {
                        if (p !== popup) p.classList.remove('open');
                    });
                    popup.classList.toggle('open');
                });

                wrapper.appendChild(chItem);
            });

            return wrapper;
        }

        function rebuildDimControls() {
            dimControls.innerHTML = '';

            const dimT = model.get('dim_t') || 1;
            const dimC = model.get('dim_c') || 1;
            const dimZ = model.get('dim_z') || 1;
            const scenes = model.get('scenes') || [];

            // Only show controls if we have multi-dimensional data
            const hasMultiDim = dimT > 1 || dimC > 1 || dimZ > 1 || scenes.length > 1;
            dimControls.style.display = hasMultiDim ? 'flex' : 'none';

            if (scenes.length > 1) {
                dimControls.appendChild(createSceneSelector());
            }
            if (dimT > 1) {
                dimControls.appendChild(createDimSlider('T', 'dim_t', 'current_t', dimT, true));
            }
            if (dimZ > 1) {
                dimControls.appendChild(createDimSlider('Z', 'dim_z', 'current_z', dimZ));
            }
            // Channel controls (replaces C slider for composite view)
            if (dimC > 1) {
                dimControls.appendChild(createChannelControls());
            }
        }

        rebuildDimControls();

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

        const dimStatus = document.createElement('span');
        dimStatus.className = 'status-item dim-status';

        function updateDimStatus() {
            const dimT = model.get('dim_t') || 1;
            const dimC = model.get('dim_c') || 1;
            const dimZ = model.get('dim_z') || 1;
            const hasMultiDim = dimT > 1 || dimZ > 1;

            if (hasMultiDim) {
                const parts = [];
                if (dimT > 1) parts.push(`T:${model.get('current_t')}/${dimT}`);
                if (dimZ > 1) parts.push(`Z:${model.get('current_z')}/${dimZ}`);
                dimStatus.textContent = parts.join(' | ');
            } else {
                dimStatus.textContent = '';
            }

            // Show active channels count
            if (dimC > 1) {
                const settings = model.get('_channel_settings') || [];
                const visible = settings.filter(ch => ch.visible !== false).length;
                dimStatus.textContent += (dimStatus.textContent ? ' | ' : '') + `Ch:${visible}/${dimC}`;
            }
        }
        updateDimStatus();

        statusBar.appendChild(toolStatus);
        statusBar.appendChild(posStatus);
        statusBar.appendChild(zoomStatus);
        statusBar.appendChild(dimStatus);

        container.appendChild(toolbar);
        container.appendChild(canvasWrapper);
        container.appendChild(dimControls);
        container.appendChild(statusBar);
        el.appendChild(container);

        // Close channel popups when clicking on canvas or toolbar
        canvasWrapper.addEventListener('click', () => {
            dimControls.querySelectorAll('.channel-popup.open').forEach(p => p.classList.remove('open'));
        });
        toolbar.addEventListener('click', () => {
            dimControls.querySelectorAll('.channel-popup.open').forEach(p => p.classList.remove('open'));
        });

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

            ctx.imageSmoothingEnabled = scale <= 1;

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

            // Set up brightness/contrast filters once
            const brightness = model.get('image_brightness') || 0;
            const contrast = model.get('image_contrast') || 0;
            const brightnessPercent = (brightness * 100);
            const contrastPercent = ((contrast + 1) * 100);

            if (model.get('image_visible')) {
                ctx.filter = `brightness(${100 + brightnessPercent}%) contrast(${contrastPercent}%)`;

                if (useTileMode) {
                    const t = model.get('current_t'), z = model.get('current_z');
                    const visibleTiles = getVisibleTiles(scale, translateX, translateY, canvas.width, canvas.height, imgWidth, imgHeight, t, z);
                    requestTiles(visibleTiles, t, z);

                    let allCached = true;
                    for (const tile of visibleTiles) {
                        const entry = tileCache.get(tile.key);
                        if (entry) {
                            entry.lastAccess = Date.now();
                            const screenX = Math.round(tile.tx * TILE_SIZE * scale + translateX);
                            const screenY = Math.round(tile.ty * TILE_SIZE * scale + translateY);
                            const drawW = Math.ceil(entry.img.width * scale);
                            const drawH = Math.ceil(entry.img.height * scale);
                            ctx.drawImage(entry.img, screenX, screenY, drawW, drawH);
                        } else {
                            allCached = false;
                        }
                    }

                    // Prefetch adjacent T/Z when current view is fully loaded
                    if (allCached && visibleTiles.length > 0) {
                        prefetchAdjacentTiles(visibleTiles, t, z);
                    }
                } else if (baseImage) {
                    ctx.translate(translateX, translateY);
                    ctx.scale(scale, scale);
                    ctx.drawImage(baseImage, 0, 0);
                    ctx.restore();
                    ctx.save();
                }

                ctx.filter = 'none';
            }

            // Apply transform for mask overlays and annotations
            ctx.translate(translateX, translateY);
            ctx.scale(scale, scale);

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

            // Auto-select resolution level based on zoom
            const resLevels = model.get('resolution_levels') || [];
            if (resLevels.length > 1) {
                const optimalLevel = Math.max(0, Math.min(
                    Math.floor(-Math.log2(scale)),
                    resLevels.length - 1
                ));
                if (optimalLevel !== model.get('current_resolution')) {
                    console.log(`[JS] Auto LOD: scale=${scale.toFixed(2)} -> level ${optimalLevel}`);
                    model.set('current_resolution', optimalLevel);
                    model.save_changes();
                }
            }

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

        let isInitialLoad = true;

        function checkTileMode() {
            const imageWidth = model.get('width');
            const imageHeight = model.get('height');
            useTileMode = (imageWidth * imageHeight) >= (1024 * 1024);  // 1MP+ uses tiles
            console.log(`[JS] Tile mode: ${useTileMode} (${imageWidth}x${imageHeight})`);
            if (useTileMode) clearTileCache();
        }

        async function loadBaseImage() {
            const imageData = model.get('image_data');
            if (imageData) {
                baseImage = await loadImage(imageData);
            }
            checkTileMode();
            if (isInitialLoad) {
                resetView();
                isInitialLoad = false;
            } else {
                renderCanvas();
            }
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
        model.on('change:width', () => { checkTileMode(); resetView(); });
        model.on('change:height', () => { checkTileMode(); resetView(); });
        model.on('change:canvas_height', () => {
            canvasWrapper.style.height = (model.get('canvas_height') || 800) + 'px';
            renderCanvas();
        });
        model.on('change:tool_mode', () => {
            const mode = model.get('tool_mode');
            [panBtn, selectBtn, rectBtn, polygonBtn, pointBtn].forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            const cursors = { 'pan': 'grab', 'select': 'default', 'draw': 'crosshair', 'polygon': 'crosshair', 'point': 'crosshair' };
            canvas.style.cursor = cursors[mode] || 'default';
            toolStatus.textContent = 'Tool: ' + TOOL_NAMES[mode];
        });

        // Dimension observers
        model.on('change:dim_t', rebuildDimControls);
        model.on('change:dim_c', rebuildDimControls);
        model.on('change:_channel_settings', () => {
            clearTileCache();  // Contrast/color changed
            updateDimStatus();
            renderCanvas();
        });
        model.on('change:dim_z', rebuildDimControls);
        model.on('change:scenes', rebuildDimControls);
        model.on('change:current_t', updateDimStatus);
        model.on('change:current_c', updateDimStatus);
        model.on('change:current_z', updateDimStatus);

        // Decode raw RGBA bytes to ImageBitmap (much faster than PNG)
        async function decodeRawTile(tileData) {
            const { w, h, data } = tileData;
            const binary = atob(data);
            const bytes = new Uint8ClampedArray(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            const imageData = new ImageData(bytes, w, h);
            return await createImageBitmap(imageData);
        }

        model.on('change:_tiles_data', async () => {
            const start = performance.now();
            const tilesData = model.get('_tiles_data') || {};
            const keys = Object.keys(tilesData);
            let decoded = 0;
            for (const [key, tileData] of Object.entries(tilesData)) {
                if (tileData && !tileCache.has(key)) {
                    try {
                        const img = await decodeRawTile(tileData);
                        cacheTile(key, img);
                        pendingTiles.delete(key);
                        decoded++;
                    } catch (e) {
                        console.error('Tile decode error:', e);
                        pendingTiles.delete(key);
                    }
                }
            }
            console.log(`[JS] Decoded ${decoded} tiles in ${(performance.now() - start).toFixed(0)}ms`);
            renderCanvas();
        });

        model.on('change:current_t', renderCanvas);
        model.on('change:current_z', renderCanvas);
        model.on('change:current_resolution', clearTileCache);

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
        height: 800px;
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
    .dim-status {
        margin-left: auto;
        font-weight: 500;
    }
    .dimension-controls {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 8px 12px;
        background: #f4f4f4;
        border-top: 1px solid #e0e0e0;
    }
    .dim-slider-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dim-label {
        font-size: 12px;
        font-weight: 600;
        color: #555;
        min-width: 16px;
    }
    .play-btn {
        width: 24px;
        height: 24px;
        border: none;
        border-radius: 4px;
        background: #0d6efd;
        color: white;
        cursor: pointer;
        font-size: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .play-btn:hover {
        background: #0b5ed7;
    }
    .dim-slider {
        width: 100px;
        height: 4px;
        border-radius: 2px;
        -webkit-appearance: none;
        background: #ddd;
    }
    .dim-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
    }
    .dim-slider::-moz-range-thumb {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
        border: none;
    }
    .dim-value {
        font-size: 11px;
        color: #666;
        min-width: 40px;
    }
    .scene-selector-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .scene-select {
        padding: 4px 8px;
        font-size: 12px;
        border: 1px solid #ccc;
        border-radius: 4px;
        background: white;
        cursor: pointer;
    }
    .channel-controls {
        display: flex;
        align-items: center;
        gap: 8px;
        padding-left: 8px;
        border-left: 1px solid #ddd;
    }
    .channel-chip {
        position: relative;
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
        background: #f0f0f0;
        border-radius: 4px;
        cursor: pointer;
        user-select: none;
    }
    .channel-chip:hover {
        background: #e8e8e8;
    }
    .channel-toggle {
        width: 14px;
        height: 14px;
        cursor: pointer;
    }
    .channel-dot {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        border: 1px solid rgba(0,0,0,0.2);
    }
    .channel-name {
        font-size: 11px;
        color: #555;
    }
    .channel-popup {
        position: absolute;
        bottom: 100%;
        left: 0;
        margin-bottom: 4px;
        min-width: 180px;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 12px;
        display: none;
        z-index: 200;
    }
    .channel-popup.open {
        display: block;
    }
    .popup-header {
        font-size: 12px;
        font-weight: 600;
        color: #333;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #eee;
    }
    .popup-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .popup-row span:first-child {
        font-size: 11px;
        color: #666;
        min-width: 35px;
    }
    .popup-color {
        width: 28px;
        height: 28px;
        padding: 0;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
    }
    .popup-slider {
        flex: 1;
        height: 4px;
        -webkit-appearance: none;
        background: #e0e0e0;
        border-radius: 2px;
    }
    .popup-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
    }
    .popup-slider::-moz-range-thumb {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #0d6efd;
        cursor: pointer;
        border: none;
    }
    .popup-value {
        font-size: 11px;
        color: #666;
        min-width: 32px;
        text-align: right;
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
        .dimension-controls {
            background: #252525;
            border-color: #404040;
        }
        .dim-label {
            color: #aaa;
        }
        .dim-slider {
            background: #404040;
        }
        .dim-value {
            color: #888;
        }
        .scene-select {
            background: #333;
            border-color: #555;
            color: #eee;
        }
        .channel-controls {
            border-color: #404040;
        }
        .channel-chip {
            background: #3a3a3a;
        }
        .channel-chip:hover {
            background: #444;
        }
        .channel-dot {
            border-color: rgba(255,255,255,0.2);
        }
        .channel-name {
            color: #bbb;
        }
        .channel-popup {
            background: #2d2d2d;
            border-color: #404040;
        }
        .popup-header {
            color: #eee;
            border-color: #404040;
        }
        .popup-row span:first-child {
            color: #aaa;
        }
        .popup-color {
            border-color: #555;
        }
        .popup-slider {
            background: #404040;
        }
        .popup-value {
            color: #aaa;
        }
    }
    """
