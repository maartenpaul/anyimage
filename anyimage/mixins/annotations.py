"""Annotations mixin for BioImageViewer."""

from __future__ import annotations

from typing import Any

import pandas as pd


class AnnotationsMixin:
    """Mixin class providing annotation management for BioImageViewer.

    This mixin handles ROIs (rectangles), polygons, and point annotations,
    providing DataFrame interfaces for easy data access and manipulation.
    """

    # Type annotations for attributes provided by the composite BioImageViewer class
    _rois_data: list[dict[str, Any]]
    _polygons_data: list[dict[str, Any]]
    _points_data: list[dict[str, Any]]
    selected_annotation_id: str
    selected_annotation_type: str

    @property
    def rois_df(self) -> pd.DataFrame:
        """Get ROIs as a pandas DataFrame.

        Returns:
            DataFrame with columns: id, x, y, width, height
        """
        if not self._rois_data:
            return pd.DataFrame(columns=["id", "x", "y", "width", "height"])
        return pd.DataFrame(self._rois_data)

    @rois_df.setter
    def rois_df(self, df: pd.DataFrame):
        """Set ROIs from a pandas DataFrame.

        Args:
            df: DataFrame with columns: id, x, y, width, height
        """
        self._rois_data = df.to_dict("records")

    @property
    def polygons_df(self) -> pd.DataFrame:
        """Get polygons as a pandas DataFrame.

        Returns:
            DataFrame with columns: id, points, num_vertices
            where points is a list of {x, y} dicts
        """
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
        """Set polygons from a pandas DataFrame.

        Args:
            df: DataFrame with columns: id, points
                where points is a list of {x, y} dicts
        """
        records = df.to_dict("records")
        self._polygons_data = [{"id": r["id"], "points": r["points"]} for r in records]

    @property
    def points_df(self) -> pd.DataFrame:
        """Get points as a pandas DataFrame.

        Returns:
            DataFrame with columns: id, x, y
        """
        if not self._points_data:
            return pd.DataFrame(columns=["id", "x", "y"])
        return pd.DataFrame(self._points_data)

    @points_df.setter
    def points_df(self, df: pd.DataFrame):
        """Set points from a pandas DataFrame.

        Args:
            df: DataFrame with columns: id, x, y
        """
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
        """Clear all annotations (ROIs, polygons, and points)."""
        self.clear_rois()
        self.clear_polygons()
        self.clear_points()
        self.selected_annotation_id = ""
        self.selected_annotation_type = ""
