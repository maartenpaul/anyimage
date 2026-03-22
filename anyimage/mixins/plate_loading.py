"""HCS OME-Zarr plate loading mixin for BioImageViewer."""

import zarr


class PlateLoadingMixin:
    """Mixin class providing HCS OME-Zarr plate loading for BioImageViewer.

    Adds support for loading High Content Screening plates stored in OME-Zarr
    format. Provides well and field-of-view (FOV) selection via dropdown
    selectors in the widget UI.

    HCS plate structure:
        plate.zarr/
        ├── .zattrs          # plate metadata (rows, columns, wells)
        ├── A/1/             # well A1
        │   ├── .zattrs      # well metadata (images/FOVs)
        │   ├── 0/           # FOV 0 (OME-Zarr multiscale image)
        │   └── 1/           # FOV 1
        └── B/3/             # well B3

    Attributes expected to be defined by the main class:
        - plate_wells: List of well names (traitlet)
        - plate_fovs: List of FOV names (traitlet)
        - current_well: Current well name (traitlet)
        - current_fov: Current FOV name (traitlet)
    """

    def set_plate(self, path):
        """Load an HCS OME-Zarr plate and display the first well/FOV.

        Args:
            path: Path to the OME-Zarr plate directory (local or remote).
        """
        store = zarr.open_group(path, mode="r")
        attrs = dict(store.attrs)

        # Support both v0.4 (plate at root) and v0.5 (plate under "ome")
        if "plate" in attrs:
            plate_meta = attrs["plate"]
        elif "ome" in attrs and "plate" in attrs["ome"]:
            plate_meta = attrs["ome"]["plate"]
        else:
            raise ValueError(f"No HCS plate metadata found in {path}")

        self._plate_path = str(path)
        self._plate_store = store
        self._plate_metadata = plate_meta

        # Build well list from plate metadata
        # Wells are stored as {"path": "A/1"} → display as "A1"
        well_entries = plate_meta.get("wells", [])
        wells = []
        for entry in well_entries:
            well_path = entry["path"]  # e.g. "A/1"
            wells.append(well_path)

        # Sort wells naturally (A/1, A/2, ..., B/1, ...)
        wells.sort()
        self._plate_well_paths = wells  # store raw paths for zarr access

        # Display names: "A/1" → "A1"
        display_wells = [w.replace("/", "") for w in wells]
        self.plate_wells = display_wells

        if wells:
            self.current_well = display_wells[0]

    def _on_well_change(self, change):
        """Observer callback when current_well changes."""
        new_well = change.get("new", "")
        old_well = change.get("old", "")
        if not new_well or not hasattr(self, "_plate_path") or self._plate_path is None:
            return
        if new_well == old_well:
            return

        self._load_well_fovs(new_well)

    def _load_well_fovs(self, well_display_name):
        """Load FOV list for the given well and auto-select the first FOV.

        Args:
            well_display_name: Display name of the well (e.g., "A1").
        """
        # Convert display name back to path: "A1" → find matching path
        well_path = None
        for wp in self._plate_well_paths:
            if wp.replace("/", "") == well_display_name:
                well_path = wp
                break

        if well_path is None:
            return

        self._current_well_path = well_path

        # Read well metadata to get FOV list
        well_group = self._plate_store[well_path]
        well_attrs = dict(well_group.attrs)

        if "well" in well_attrs:
            well_meta = well_attrs["well"]
        elif "ome" in well_attrs and "well" in well_attrs["ome"]:
            well_meta = well_attrs["ome"]["well"]
        else:
            # Fallback: list subgroups as FOVs
            well_meta = {"images": []}

        images = well_meta.get("images", [])
        fov_paths = [img["path"] for img in images]

        if not fov_paths:
            # Fallback: enumerate subgroups that look like image groups
            fov_paths = sorted(
                k for k in well_group.keys()
                if not k.startswith(".")
            )

        self._current_well_fov_paths = fov_paths
        self.plate_fovs = fov_paths

        if fov_paths:
            self.current_fov = fov_paths[0]

    def _on_fov_change(self, change):
        """Observer callback when current_fov changes."""
        new_fov = change.get("new", "")
        old_fov = change.get("old", "")
        if not new_fov or not hasattr(self, "_plate_path") or self._plate_path is None:
            return
        if new_fov == old_fov:
            return

        self._load_plate_image(new_fov)

    def _load_plate_image(self, fov):
        """Load the image for the current well and given FOV.

        Args:
            fov: FOV path within the well (e.g., "0").
        """
        if not hasattr(self, "_current_well_path"):
            return

        image_path = f"{self._plate_path}/{self._current_well_path}/{fov}"

        try:
            import bioio_ome_zarr
            from bioio import BioImage

            img = BioImage(image_path, reader=bioio_ome_zarr.Reader)
            self._set_bioimage(img)
        except ImportError:
            raise ImportError(
                "bioio and bioio-ome-zarr are required for plate loading. "
                "Install with: pip install bioio bioio-ome-zarr"
            )
