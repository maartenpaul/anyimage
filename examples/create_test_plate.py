"""Create a test HCS OME-Zarr plate for development and testing.

Usage:
    python examples/create_test_plate.py [output_path]

Creates a small HCS plate with 4 wells (A1, A2, B1, B2), 2 FOVs each,
and 5D image data (T=2, C=3, Z=2, Y=256, X=256) with distinct patterns
per well/FOV so visual differences are easy to verify.
"""

import sys

import numpy as np
import zarr
from PIL import Image, ImageDraw


def create_test_plate(output_path="examples/test_plate.zarr"):
    """Create a minimal HCS OME-Zarr plate."""
    store = zarr.open_group(output_path, mode="w")

    rows = ["A", "B"]
    cols = ["1", "2"]

    # Plate-level metadata
    store.attrs["plate"] = {
        "columns": [{"name": c} for c in cols],
        "rows": [{"name": r} for r in rows],
        "wells": [{"path": f"{r}/{c}"} for r in rows for c in cols],
        "field_count": 2,
        "name": "test_plate",
    }

    for row in rows:
        for col in cols:
            well_path = f"{row}/{col}"
            well_group = store.create_group(well_path)
            well_group.attrs["well"] = {
                "images": [{"path": "0"}, {"path": "1"}],
            }

            for fov in ["0", "1"]:
                # T=2, C=3, Z=2, Y=256, X=256
                data = np.zeros((2, 3, 2, 256, 256), dtype=np.uint16)

                for t in range(2):
                    for z in range(2):
                        label = f"{row}{col} FOV{fov}\nT={t} Z={z}"
                        img = Image.new("L", (256, 256), color=128)
                        draw = ImageDraw.Draw(img)
                        draw.text((10, 10), label, fill=65535 // 256)
                        arr = (np.array(img, dtype=np.uint16) * 256)
                        data[t, 0, z] = arr
                        data[t, 1, z] = arr
                        data[t, 2, z] = arr

                fov_group = well_group.create_group(fov)
                fov_group.attrs["multiscales"] = [
                    {
                        "version": "0.4",
                        "axes": [
                            {"name": "t", "type": "time"},
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}
                                ],
                            }
                        ],
                        "name": f"{row}{col}_FOV{fov}",
                    }
                ]
                fov_group.create_array("0", data=data, chunks=(1, 1, 1, 256, 256))

    print(f"Created test plate at {output_path}")
    print(f"  Wells: {[f'{r}{c}' for r in rows for c in cols]}")
    print(f"  FOVs per well: 2")
    print(f"  Image shape: (2, 3, 2, 256, 256) = T×C×Z×Y×X")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/test_plate.zarr"
    create_test_plate(path)
