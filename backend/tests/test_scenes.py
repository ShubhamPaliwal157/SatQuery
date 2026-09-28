"""Run from backend/:  python -m unittest tests.test_scenes -v"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.transform import from_origin

import scenes


def _tif(path: Path, size=16, pixel=0.0001, lon=77.0, lat=13.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint16",
                       crs="EPSG:4326", transform=from_origin(lon, lat, pixel, pixel)) as dst:
        dst.write(np.ones((size, size), dtype="uint16"), 1)


class SceneTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        a = self.root / "aois" / "flood_test"
        for d in ("2019-03-05", "2019-03-20"):
            _tif(a / "acquisitions" / d / "sar" / "VV.tif")
        _tif(a / "acquisitions" / "2019-03-20" / "optical" / "B04.tif")
        _tif(a / "acquisitions" / "2019-03-20" / "optical" / "B08.tif")
        _tif(a / "acquisitions" / "2019-03-20" / "optical" / "notes.tif")   # unknown name
        (a / "acquisitions" / "2019-03-20" / "optical" / "readme.txt").write_text("x")
        _tif(a / "dem" / "dem.tif")
        (self.root / "aois" / "_template").mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_scene_and_skips_template(self):
        ids = [s.id for s in scenes.list_scenes(self.root)]
        self.assertEqual(ids, ["flood_test"])

    def test_capabilities_reflect_real_files(self):
        s = scenes.get_scene("flood_test", self.root)
        cap = s.capabilities
        self.assertTrue(cap["flood_extent"]["available"])      # VV on two dates
        self.assertTrue(cap["ndvi"]["available"])              # B04 + B08
        self.assertTrue(cap["terrain_3d"]["available"])        # dem.tif
        self.assertTrue(cap["sensor_swipe"]["available"])
        self.assertFalse(cap["true_colour"]["available"])      # no B02/B03
        self.assertFalse(cap["time_series"]["available"])      # only one NDVI date
        self.assertFalse(cap["dnbr"]["available"])
        self.assertIn("B02", cap["true_colour"]["requires"])

    def test_warnings_for_bad_files_and_missing_meta(self):
        s = scenes.get_scene("flood_test", self.root)
        joined = " ".join(s.warnings)
        self.assertIn("notes.tif", joined)
        self.assertIn("readme.txt", joined)
        self.assertIn("No aoi.json", joined)

    def test_grid_and_bounds_read_from_header(self):
        s = scenes.get_scene("flood_test", self.root)
        acq = s.acquisitions[0]
        self.assertEqual((acq.grid.width, acq.grid.height), (16, 16))
        self.assertEqual(acq.grid.geo_bounds[0], 77.0)

    def test_path_traversal_and_private_names_rejected(self):
        self.assertIsNone(scenes.get_scene("../etc", self.root))
        self.assertIsNone(scenes.get_scene("_template", self.root))
        self.assertIsNone(scenes.get_scene("nope", self.root))

    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(scenes.list_scenes(Path(d)), [])


if __name__ == "__main__":
    unittest.main()
