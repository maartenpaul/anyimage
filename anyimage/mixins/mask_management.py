"""Mask management mixin for BioImageViewer."""

import numpy as np
import pandas as pd

from ..utils import (
    MASK_COLORS,
    array_to_base64,
    labels_to_rgba,
)


class MaskManagementMixin:
    """Mixin class providing mask layer management for BioImageViewer.

    This mixin handles adding, removing, and updating mask layers,
    including support for contour rendering and opacity control.

    Attributes expected to be defined by the main class:
        - _mask_arrays: Dict storing raw label arrays by mask id
        - _mask_caches: Dict storing rendered versions by mask id
        - _masks_data: List of mask layer dicts (traitlet)
    """

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
            labels: 2D numpy array of label values (0 is background)
            name: Display name for the mask layer (auto-assigned if None)
            color: Hex color for the mask (auto-assigned if None)
            opacity: Opacity value 0-1
            visible: Whether the mask is visible
            contours_only: If True, show only contours instead of filled regions
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
        rgba = labels_to_rgba(labels, contours_only=contours_only, contour_width=contour_width)
        data_b64 = array_to_base64(rgba)

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

        Args:
            labels: 2D numpy array of label values
            name: Display name for the mask layer
            contours_only: If True, show only contours
            contour_width: Width of contours in pixels
        """
        self._masks_data = []
        self._mask_arrays = {}
        self._mask_caches = {}
        self.add_mask(labels, name=name or "Mask", contours_only=contours_only, contour_width=contour_width)

    def remove_mask(self, mask_id: str):
        """Remove a mask layer by ID.

        Args:
            mask_id: The ID of the mask to remove
        """
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
                            rgba = labels_to_rgba(
                                self._mask_arrays[mask_id],
                                contours_only=contours,
                                contour_width=width
                            )
                            data_b64 = array_to_base64(rgba)
                            if mask_id not in self._mask_caches:
                                self._mask_caches[mask_id] = {}
                            self._mask_caches[mask_id][cache_key] = data_b64
                            new_mask["data"] = data_b64

                updated.append(new_mask)
            else:
                updated.append(mask)
        self._masks_data = updated

    def get_mask_ids(self) -> list[str]:
        """Get list of all mask IDs.

        Returns:
            List of mask ID strings
        """
        return [m["id"] for m in self._masks_data]

    @property
    def masks_df(self) -> pd.DataFrame:
        """Get mask layers as a pandas DataFrame.

        Returns:
            DataFrame with columns: id, name, visible, opacity, color, contours
        """
        if not self._masks_data:
            return pd.DataFrame(columns=["id", "name", "visible", "opacity", "color", "contours"])
        return pd.DataFrame([
            {k: v for k, v in m.items() if k != "data"}
            for m in self._masks_data
        ])
