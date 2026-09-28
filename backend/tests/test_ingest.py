"""Run from backend/:  python -m unittest tests.test_ingest -v

Two layers:
  * Logic tests use FakeBackend, so they need neither rasterio nor GDAL. They
    cover recognition, dates, duplicates, grid maths, overlap checks, the
    manifest, re-runs and the report.
  * RasterioIntegration writes real GeoTIFFs and runs the real warp path
    (10 m + 20 m bands, SCL codes, a SAR raster on an offset grid, a DEM in
    EPSG:4326). It is skipped when rasterio isn't installed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest
import scenes
from ingest import Grid, RasterMeta

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

UTM = "EPSG:32736"


def meta(crs=UTM, left=700000.0, top=7800000.0, px=10.0, w=100, h=100, count=1, dtype="uint16", pixel_m=None):
    return RasterMeta(crs=crs, width=w, height=h, count=count, dtype=dtype, nodata=None,
                      bounds=(left, top - h * px, left + w * px, top), pixel=(px, px),
                      pixel_m=px if pixel_m is None else pixel_m)


def touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


class FakeBackend:
    """Headers come from a dict keyed by inbox-relative path; warp() writes a stub."""

    def __init__(self, inbox: Path, metas: dict, cross=None, bbox_bounds=None):
        self.inbox, self.metas = inbox, metas
        self.cross = cross or {}          # source CRS -> its footprint in the grid CRS
        self.bbox_bounds = bbox_bounds
        self.calls = []                   # (dst name, resampling, grid)

    def inspect(self, path):
        rel = path.relative_to(self.inbox).as_posix()
        if rel not in self.metas:
            raise ingest.RasterProblem("couldn't be opened: corrupt")
        m = self.metas[rel]
        if isinstance(m, Exception):
            raise m
        return m

    def footprint_in(self, m, crs):
        return m.bounds if m.crs == crs else self.cross[m.crs]

    def lonlat_bounds_in(self, bbox, crs):
        return self.bbox_bounds

    def warp(self, src, dst, grid, resampling):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"stub")
        self.calls.append((dst.name, resampling, grid))
        return {"dtype": "uint16", "nodata": 0, "n_valid": 100, "nodata_fraction": 0.0,
                "min": 1.0, "max": 9.0, "mean": 5.0}


SAFE = "S2A_MSIL2A_20190305T072051_N0500_R006_T36KWA_20190307T101010.SAFE/GRANULE/L2A_T36KWA/IMG_DATA"


class Recognition(unittest.TestCase):
    def check(self, rel, kind, band, date):
        c = ingest.classify(Path("/in") / rel, Path("/in"))
        self.assertIsInstance(c, ingest.Candidate, rel)
        self.assertEqual((c.kind, c.band, c.date), (kind, band, date), rel)
        return c

    def test_sentinel2_safe_uses_sensing_date_from_filename(self):
        # folder carries a *processing* date 2 days later; the file name must win
        self.check(f"{SAFE}/R10m/T36KWA_20190305T072051_B04_10m.jp2", "optical", "B04", "2019-03-05")
        self.check(f"{SAFE}/R20m/T36KWA_20190305T072051_SCL_20m.jp2", "optical", "SCL", "2019-03-05")
        self.check(f"{SAFE}/R20m/T36KWA_20190305T072051_B8A_20m.jp2", "optical", "B8A", "2019-03-05")

    def test_copernicus_browser_names(self):
        self.check("2019-03-05-00_00_2019-03-05-23_59_Sentinel-2_L2A_B11_(Raw).tiff", "optical", "B11", "2019-03-05")
        # dual-pol export: the trailing token is the band
        self.check("2019-03-06-00_00_2019-03-06-23_59_Sentinel-1_IW_VV+VH_VV_(Raw).tiff", "sar", "VV", "2019-03-06")

    def test_date_from_folder_and_case_insensitive(self):
        self.check("2019-03-05/b04.TIF", "optical", "B04", "2019-03-05")
        self.check("20190320/VH.tif", "sar", "VH", "2019-03-20")

    def test_sentinel1_safe_measurement_file(self):
        self.check("S1A_IW_GRDH_1SDV_20190306T042751.SAFE/measurement/"
                   "s1a-iw-grd-vv-20190306t042751-20190306t042816-026123-02eb9d-001.tiff", "sar", "VV", "2019-03-06")

    def test_dem_by_name_or_folder_and_needs_no_date(self):
        self.check("dem.tif", "dem", "DEM", None)
        self.check("Copernicus_DSM_COG_10_S20_00_E034_00_DEM.tif", "dem", "DEM", None)
        self.check("dem/glo30_tile.tif", "dem", "DEM", None)

    def test_unrecognised_names_are_skipped_with_a_reason(self):
        for rel in (f"{SAFE}/R10m/T36KWA_20190305T072051_TCI_10m.jp2", "holiday_photo.tif"):
            r = ingest.classify(Path("/in") / rel, Path("/in"))
            self.assertIsInstance(r, ingest.Skip)
            self.assertIn("not a recognised band", r.reason)

    def test_date_range_gets_a_note(self):
        c = ingest.classify(Path("/in/2019-03-05-00_00_2019-03-09-23_59_Sentinel-2_L2A_B04_(Raw).tiff"), Path("/in"))
        self.assertEqual(c.date, "2019-03-05")
        self.assertIn("2019-03-09", c.date_note)

    def test_impossible_dates_are_ignored(self):
        self.assertEqual(ingest.find_dates("x_20191341_y"), ([], []))
        self.assertEqual(ingest.date_from_path(Path("B04.tif")), (None, None))

    def test_l1c_is_flagged(self):
        c = ingest.classify(Path("/in/S2A_MSIL1C_20190305T072051_x.SAFE/T36KWA_20190305T072051_B04.jp2"), Path("/in"))
        self.assertEqual(c.level, "L1C")

    def test_discover_counts_archives_and_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            inbox = Path(d)
            touch(inbox, "2019-03-05/B04.tif")
            touch(inbox, "2019-03-05/readme.txt")
            touch(inbox, "2019-03-05/MTD_MSIL2A.xml")
            touch(inbox, "S2_download.zip")
            touch(inbox, ".DS_Store")
            cands, skips, ignored = ingest.discover(inbox)
            self.assertEqual([c.band for c in cands], ["B04"])
            self.assertEqual(ignored, 2)
            self.assertEqual(len(skips), 1)
            self.assertIn("unzip", skips[0].reason)


class GridMaths(unittest.TestCase):
    def test_snap_is_inward_and_on_the_reference_lattice(self):
        ref = meta()
        g = ingest.snap_grid(ref, (700005.0, 7799000.0, 701000.0, 7799995.0))
        self.assertEqual((g.left, g.top, g.width, g.height), (700010.0, 7799990.0, 99, 99))
        b = g.bounds
        self.assertGreaterEqual(b[0], 700005.0)
        self.assertLessEqual(b[3], 7799995.0)

    def test_snap_exact_bounds_keep_every_pixel(self):
        ref = meta()
        g = ingest.snap_grid(ref, ref.bounds)
        self.assertEqual((g.left, g.top, g.width, g.height), (700000.0, 7800000.0, 100, 100))

    def test_sub_pixel_area_is_an_error(self):
        with self.assertRaises(ingest.IngestError):
            ingest.snap_grid(meta(), (700001.0, 7799001.0, 700004.0, 7799004.0))

    def test_resampling_choice(self):
        self.assertEqual(ingest.choose_resampling("SCL", 20, 10), "nearest")
        self.assertEqual(ingest.choose_resampling("B11", 20, 10), "bilinear")
        self.assertEqual(ingest.choose_resampling("B04", 10, 10), "bilinear")
        self.assertEqual(ingest.choose_resampling("VV", 5, 10), "average")

    def test_grid_round_trips_through_json(self):
        g = Grid(UTM, 700010.0, 7799990.0, 10.0, 10.0, 99, 99, 10.0)
        self.assertEqual(Grid.from_dict(json.loads(json.dumps(g.to_dict()))), g)


class Pipeline(unittest.TestCase):
    """End to end against FakeBackend: real folders, real capability rules."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.inbox = self.root / "inbox"
        self.files = {}
        # Sentinel-2 SAFE, 2019-03-05: 10 m and 20 m bands, plus a coarser duplicate and junk
        for band in ("B02", "B03", "B04", "B08"):
            self._add(f"{SAFE}/R10m/T36KWA_20190305T072051_{band}_10m.jp2", meta())
        for band in ("B11", "SCL"):
            self._add(f"{SAFE}/R20m/T36KWA_20190305T072051_{band}_20m.jp2", meta(px=20.0, w=50, h=50))
        self._add(f"{SAFE}/R20m/T36KWA_20190305T072051_B02_20m.jp2", meta(px=20.0, w=50, h=50))
        self._add(f"{SAFE}/R10m/T36KWA_20190305T072051_TCI_10m.jp2", meta())
        # second optical date, same grid
        for band in ("B04", "B08"):
            self._add(f"2019-03-20/{band}.tif", meta())
        # Sentinel-1 on a grid offset by 5 m in x and y (same CRS)
        for d in ("2019-03-05", "2019-03-20"):
            self._add(f"{d}/VV.tif", meta(left=700005.0, top=7799995.0))
        # DEM in lon/lat, comfortably covering the scene once expressed in UTM
        self._add("dem.tif", meta(crs="EPSG:4326", left=34.85, top=-19.80, px=0.002, pixel_m=222.0))
        self.cross = {"EPSG:4326": (699000.0, 7797000.0, 703000.0, 7801000.0)}

    def tearDown(self):
        self._tmp.cleanup()

    def _add(self, rel, m):
        touch(self.inbox, rel)
        self.files[rel] = m

    def backend(self, **kw):
        return FakeBackend(self.inbox, self.files, cross=kw.pop("cross", self.cross), **kw)

    def run_ingest(self, aoi="test_aoi", backend=None, **kw):
        return ingest.run(aoi, data_root=self.root, inbox=self.inbox, backend=backend or self.backend(), **kw)

    # -- the happy path -----------------------------------------------------
    def test_full_run_places_files_on_one_grid(self):
        be = self.backend()
        r = self.run_ingest(backend=be)
        self.assertTrue(r.ok, r.errors)
        # grid = optical B02 lattice, clipped to the S1 offset -> 99 x 99
        self.assertEqual((r.grid.crs, r.grid.left, r.grid.top, r.grid.width, r.grid.height),
                         (UTM, 700010.0, 7799990.0, 99, 99))
        self.assertEqual({g.width for _, _, g in be.calls}, {99})   # every file targets the same grid
        aoi = self.root / "aois" / "test_aoi"
        for rel in ("acquisitions/2019-03-05/optical/B02.tif", "acquisitions/2019-03-05/optical/B11.tif",
                    "acquisitions/2019-03-05/optical/SCL.tif", "acquisitions/2019-03-05/sar/VV.tif",
                    "acquisitions/2019-03-20/optical/B08.tif", "dem/dem.tif"):
            self.assertTrue((aoi / rel).is_file(), rel)

    def test_resampling_and_notes(self):
        be = self.backend()
        r = self.run_ingest(backend=be)
        used = {name: how for name, how, _ in be.calls}
        self.assertEqual(used["SCL.tif"], "nearest")
        self.assertEqual(used["B11.tif"], "bilinear")
        notes = {p.cand.band: p.note for p in r.planned}
        self.assertEqual(notes["B11"], "20 m -> 10 m")
        self.assertEqual(notes["VV"], "shifted onto the grid")
        self.assertEqual(notes["DEM"], "reprojected from EPSG:4326, 222 m -> 10 m")
        self.assertEqual(notes["B02"], "")

    def test_finest_copy_wins_and_junk_is_reported_not_placed(self):
        r = self.run_ingest()
        b02 = [p for p in r.planned if p.cand.band == "B02"]
        self.assertEqual(len(b02), 1)
        self.assertTrue(b02[0].cand.rel.name.endswith("_10m.jp2"))
        reasons = {s.reason.split(" - ")[0] for s in r.skipped}
        self.assertTrue(any("coarser copy" in x for x in reasons))
        self.assertTrue(any("not a recognised band" in x for x in reasons))

    def test_projected_capabilities_match_a_real_rescan(self):
        r = self.run_ingest()
        scanned = scenes.get_scene("test_aoi", self.root)
        rescanned = sorted(k for k, v in scanned.capabilities.items() if v["available"])
        self.assertEqual(r.available_after, rescanned)
        for name in ("true_colour", "ndvi", "cloud_mask", "flood_extent", "terrain_3d", "sensor_swipe"):
            self.assertIn(name, r.available_after)
        # date 2 only has B04+B08, so true-colour exists on ONE date: no optical change, no time series
        for name in ("optical_change", "time_series", "dnbr"):
            self.assertNotIn(name, r.available_after)
        self.assertIn("SAR flood extent", r.unlocked)

    def test_manifest_holds_grid_and_provenance(self):
        self.run_ingest()
        m = json.loads((self.root / "aois" / "test_aoi" / ingest.MANIFEST_NAME).read_text())
        self.assertEqual(m["grid"]["width"], 99)
        rec = m["files"]["acquisitions/2019-03-05/optical/B11.tif"]
        self.assertEqual((rec["resampling"], rec["source_pixel_m"]), ("bilinear", 20.0))
        self.assertTrue(rec["source"].endswith("B11_20m.jp2"))
        self.assertTrue((self.root / "aois" / "test_aoi" / "aoi.json").is_file())

    def test_inbox_is_never_modified(self):
        before = sorted(p.relative_to(self.inbox).as_posix() for p in self.inbox.rglob("*"))
        self.run_ingest()
        after = sorted(p.relative_to(self.inbox).as_posix() for p in self.inbox.rglob("*"))
        self.assertEqual(before, after)

    # -- dry run, re-run, overwrite -----------------------------------------
    def test_dry_run_writes_nothing_but_reports_the_plan(self):
        be = self.backend()
        r = self.run_ingest(backend=be, dry_run=True)
        self.assertTrue(r.ok)
        self.assertFalse((self.root / "aois").exists())
        self.assertEqual(be.calls, [])
        self.assertTrue(r.planned and r.unlocked)
        self.assertIn("DRY RUN", ingest.format_report(r))

    def test_rerun_keeps_existing_files_and_overwrite_redoes_them(self):
        self.run_ingest()
        be = self.backend()
        r2 = self.run_ingest(backend=be)
        self.assertTrue(r2.ok)
        self.assertEqual(be.calls, [])
        self.assertTrue(all(p.action == "exists" for p in r2.planned))
        self.assertIn("already there", ingest.format_report(r2))
        be3 = self.backend()
        self.run_ingest(backend=be3, overwrite=True)
        self.assertEqual(len(be3.calls), len(r2.planned))

    def test_later_run_aligns_new_data_to_the_stored_grid(self):
        first = self.run_ingest()
        # a new date arrives whose own footprint would give a different grid
        # (offset by a fractional number of pixels, and it only covers ~92% of the stored grid)
        self._add("2019-04-10/B04.tif", meta(left=700053.0, top=7799947.0))
        be = self.backend()
        r = self.run_ingest(backend=be)
        self.assertTrue(r.ok, r.errors)
        self.assertTrue(any("B04.tif" in w and "covers only" in w and "of the grid" in w for w in r.warnings))
        self.assertEqual(r.grid, first.grid)
        self.assertEqual([c[0] for c in be.calls], ["B04.tif"])
        self.assertEqual(be.calls[0][2], first.grid)
        self.assertIn("existing AOI grid", r.grid_note)
        self.assertEqual(r.planned[[p.dst_rel for p in r.planned].index("acquisitions/2019-04-10/optical/B04.tif")].note,
                         "shifted onto the grid")

    # -- refusals and drops --------------------------------------------------
    def test_acquisition_over_a_different_place_is_dropped_not_fatal(self):
        self._add("2019-05-01/B04.tif", meta(left=800000.0, top=7900000.0))
        r = self.run_ingest()
        self.assertTrue(r.ok, r.errors)
        self.assertNotIn("acquisitions/2019-05-01/optical/B04.tif", [p.dst_rel for p in r.planned])
        self.assertTrue(any("does not overlap" in s.reason for s in r.skipped))
        self.assertEqual(r.grid.width, 99)   # the good data's grid was not shrunk to nothing

    def test_partial_overlap_warns(self):
        self._add("2019-05-01/B04.tif", meta(left=700500.0, top=7800000.0))   # covers ~50% of the reference
        r = self.run_ingest()
        self.assertTrue(any("covers only" in w and "reference scene" in w for w in r.warnings))

    def test_dem_covering_part_of_the_grid_warns(self):
        cross = {"EPSG:4326": (700000.0, 7799000.0, 700500.0, 7800000.0)}     # left half only
        r = self.run_ingest(backend=self.backend(cross=cross))
        self.assertTrue(any("dem.tif" in w and "nodata" in w for w in r.warnings))

    def test_bbox_clips_the_grid(self):
        be = self.backend(bbox_bounds=(700200.0, 7799200.0, 700600.0, 7799600.0))
        r = self.run_ingest(backend=be, bbox=(34.9, -19.9, 34.95, -19.85))
        self.assertTrue(r.ok, r.errors)
        self.assertEqual((r.grid.left, r.grid.top, r.grid.width, r.grid.height), (700200.0, 7799600.0, 40, 40))
        self.assertIn("--bbox", r.grid_note)

    def test_bbox_outside_the_data_is_an_error(self):
        be = self.backend(bbox_bounds=(900000.0, 7000000.0, 900500.0, 7000500.0))
        r = self.run_ingest(backend=be, bbox=(10.0, 10.0, 10.1, 10.1))
        self.assertFalse(r.ok)
        self.assertIn("--bbox", r.errors[0])

    def test_oversized_grid_is_refused_with_advice(self):
        r = self.run_ingest(max_pixels=5000)
        self.assertFalse(r.ok)
        self.assertIn("--bbox", r.errors[0])

    def test_hand_placed_data_without_manifest_is_refused(self):
        p = self.root / "aois" / "test_aoi" / "acquisitions" / "2019-03-05" / "optical"
        p.mkdir(parents=True)
        (p / "B04.tif").write_bytes(b"x")
        r = self.run_ingest()
        self.assertFalse(r.ok)
        self.assertIn("ingest_manifest.json", r.errors[0])
        self.assertEqual([x.name for x in p.iterdir()], ["B04.tif"])   # untouched

    def test_empty_template_folder_is_fine(self):
        (self.root / "aois" / "test_aoi").mkdir(parents=True)
        (self.root / "aois" / "test_aoi" / "aoi.json").write_text('{"title": "Mine"}')
        r = self.run_ingest()
        self.assertTrue(r.ok, r.errors)
        self.assertFalse(r.created_aoi_json)
        self.assertEqual(json.loads((self.root / "aois" / "test_aoi" / "aoi.json").read_text())["title"], "Mine")

    def test_bad_inputs(self):
        for kw, word in (({"aoi": "Bad Name"}, "lowercase"), ({"aoi": "_x"}, "lowercase"),
                         ({"default_date": "2019-13-45"}, "--date"), ({"bbox": (5, 5, 1, 9)}, "--bbox")):
            aoi = kw.pop("aoi", "test_aoi")
            r = self.run_ingest(aoi=aoi, **kw)
            self.assertFalse(r.ok)
            self.assertIn(word, r.errors[0])
        r = ingest.run("test_aoi", data_root=self.root, inbox=self.root / "nope", backend=self.backend())
        self.assertIn("Inbox folder not found", r.errors[0])

    def test_unusable_files_are_skipped_with_their_own_reason(self):
        self._add("2019-03-05/B12.tif", ingest.RasterProblem("isn't georeferenced (no CRS / map transform)."))
        self._add("2019-03-05/B06.tif", meta(count=3))
        touch(self.inbox, "2019-03-05/B07.tif")           # not in the fake's header table -> unreadable
        r = self.run_ingest()
        reasons = {s.name: s.reason for s in r.skipped}
        self.assertIn("georeferenced", reasons["2019-03-05/B12.tif"])
        self.assertIn("3 bands", reasons["2019-03-05/B06.tif"])
        self.assertIn("couldn't be opened", reasons["2019-03-05/B07.tif"])
        placed = {p.cand.band for p in r.planned}
        self.assertFalse({"B12", "B06", "B07"} & placed)

    def test_undated_files_need_date_flag(self):
        touch(self.inbox, "B04.tif")
        self.files["B04.tif"] = meta()
        r = self.run_ingest()
        self.assertTrue(any("no acquisition date" in s.reason for s in r.skipped))
        r2 = self.run_ingest(aoi="other_aoi", default_date="2019-06-01")
        self.assertIn("acquisitions/2019-06-01/optical/B04.tif", [p.dst_rel for p in r2.planned])

    def test_same_resolution_duplicates_warn(self):
        self._add(f"{SAFE}/R10m/dupe/T36KWA_20190305T072051_B04_10m.jp2", meta())
        r = self.run_ingest()
        self.assertTrue(any("same resolution" in w for w in r.warnings))

    def test_l1c_warning(self):
        self._add("S2A_MSIL1C_20190401T072051.SAFE/T36KWA_20190401T072051_B04.jp2", meta())
        r = self.run_ingest()
        self.assertTrue(any("Level-1C" in w for w in r.warnings))

    def test_per_file_failure_is_reported_and_others_still_written(self):
        class Flaky(FakeBackend):
            def warp(self, src, dst, grid, resampling):
                if dst.name == "B03.tif":
                    raise RuntimeError("disk full")
                return super().warp(src, dst, grid, resampling)
        r = self.run_ingest(backend=Flaky(self.inbox, self.files, cross=self.cross))
        self.assertFalse(r.ok)
        self.assertTrue(any("B03.tif" in e and "disk full" in e for e in r.errors))
        self.assertTrue((self.root / "aois" / "test_aoi" / "acquisitions/2019-03-05/optical/B02.tif").is_file())

    def test_report_shows_readable_bounds_and_which_file_was_dropped(self):
        self._add("2019-05-01/B04.tif", meta(left=800000.0, top=7900000.0))
        text = ingest.format_report(self.run_ingest())
        self.assertNotIn("e+", text)                                   # no 7.799e+06 style coordinates
        self.assertIn("700010, 7799000, 701000, 7799990", text)
        self.assertIn("e.g. 2019-05-01/B04.tif", text)                 # folder disambiguates the many B04.tif
        self.assertIn("AOI-wide", text)                                # the DEM group

    def test_report_is_plain_ascii_and_lists_what_matters(self):
        text = ingest.format_report(self.run_ingest())
        text.encode("ascii")                      # must survive a legacy Windows console
        for needle in ("Grid", "Placed", "2019-03-05", "resampled (20 m -> 10 m): B11 SCL", "Skipped",
                       "Analyses newly available", "check_scenes.py"):
            self.assertIn(needle, text)


@unittest.skipUnless(HAS_RASTERIO, "rasterio not installed - the real warp path is untested here")
class RasterioIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.inbox = self.root / "inbox"
        rng = np.random.default_rng(0)
        self.b02 = (rng.integers(1, 5000, (100, 100))).astype("uint16")           # never 0 (= nodata)
        self.b11 = (rng.integers(1, 3000, (50, 50))).astype("uint16")
        self.scl = np.full((50, 50), 4, "uint8")
        self.scl[:25, :25] = 9
        s2 = from_origin(700000, 7800000, 10, 10)
        s2_20 = from_origin(700000, 7800000, 20, 20)
        for b in ("B03", "B04", "B08"):
            self._write(f"2019-03-05/{b}.tif", self.b02, UTM, s2)
        self._write("2019-03-05/B02.tif", self.b02, UTM, s2)
        self._write("2019-03-05/B11.tif", self.b11, UTM, s2_20)
        self._write("2019-03-05/SCL.tif", self.scl, UTM, s2_20)
        vv = rng.random((100, 100)).astype("float32") + 0.01
        self._write("2019-03-06/VV.tif", vv, UTM, from_origin(700005, 7799995, 10, 10))
        self.dem = (rng.random((100, 100)) * 50 + 10).astype("float32")
        self._write("dem.tif", self.dem, "EPSG:4326", from_origin(34.85, -19.80, 0.002, 0.002))

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, arr, crs, transform):
        p = self.inbox / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(p, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                           dtype=str(arr.dtype), crs=crs, transform=transform) as dst:
            dst.write(arr, 1)

    def _read(self, rel):
        with rasterio.open(self.root / "aois" / "int_aoi" / rel) as src:
            return src.read(1), src.crs, src.transform

    def test_end_to_end(self):
        r = ingest.run("int_aoi", data_root=self.root, inbox=self.inbox)
        self.assertTrue(r.ok, r.errors)
        self.assertEqual((r.grid.width, r.grid.height), (99, 99))

        names = ["acquisitions/2019-03-05/optical/B02.tif", "acquisitions/2019-03-05/optical/B11.tif",
                 "acquisitions/2019-03-05/optical/SCL.tif", "acquisitions/2019-03-06/sar/VV.tif", "dem/dem.tif"]
        ref_crs = ref_tr = None
        for n in names:
            arr, crs, tr = self._read(n)
            self.assertEqual(arr.shape, (99, 99), n)
            ref_crs, ref_tr = ref_crs or crs, ref_tr or tr
            self.assertEqual((crs, tr), (ref_crs, ref_tr), n)

        # reference band copied cell-for-cell, shifted one pixel by the clip
        b02, _, _ = self._read(names[0])
        self.assertLessEqual(np.abs(b02.astype(int) - self.b02[1:, 1:].astype(int)).max(), 1)
        # 20 m band upsampled: bilinear cannot overshoot the source range
        b11, _, _ = self._read(names[1])
        self.assertGreaterEqual(int(b11.min()), int(self.b11.min()))
        self.assertLessEqual(int(b11.max()), int(self.b11.max()))
        # class codes stay class codes
        scl, _, _ = self._read(names[2])
        self.assertTrue(set(np.unique(scl).tolist()) <= {4, 9})
        # DEM reprojected from lon/lat, fully covered, values inside the source range
        dem, _, _ = self._read(names[4])
        self.assertFalse(np.isnan(dem).any())
        self.assertGreaterEqual(float(dem.min()), float(self.dem.min()) - 1e-3)
        self.assertLessEqual(float(dem.max()), float(self.dem.max()) + 1e-3)

        # the real registry agrees: one grid, and it can see the analyses
        s = scenes.get_scene("int_aoi", self.root)
        self.assertFalse(any(a.mixed_grids for a in s.acquisitions))
        self.assertEqual((s.acquisitions[0].grid.width, s.acquisitions[0].grid.height), (99, 99))
        self.assertEqual(sorted(k for k, v in s.capabilities.items() if v["available"]), r.available_after)

        # idempotent
        r2 = ingest.run("int_aoi", data_root=self.root, inbox=self.inbox)
        self.assertTrue(r2.ok)
        self.assertTrue(all(p.action == "exists" for p in r2.planned))

    def test_dry_run_writes_nothing(self):
        r = ingest.run("int_aoi", data_root=self.root, inbox=self.inbox, dry_run=True)
        self.assertTrue(r.ok, r.errors)
        self.assertFalse((self.root / "aois").exists())

    def test_non_georeferenced_and_multiband_files_are_skipped(self):
        p = self.inbox / "2019-03-05" / "B12.tif"
        with rasterio.open(p, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint16") as dst:
            dst.write(np.ones((8, 8), "uint16"), 1)                       # no CRS
        q = self.inbox / "2019-03-05" / "B06.tif"
        with rasterio.open(q, "w", driver="GTiff", height=8, width=8, count=2, dtype="uint16",
                           crs=UTM, transform=from_origin(700000, 7800000, 10, 10)) as dst:
            dst.write(np.ones((2, 8, 8), "uint16"))
        r = ingest.run("int_aoi", data_root=self.root, inbox=self.inbox, dry_run=True)
        reasons = {s.name: s.reason for s in r.skipped}
        self.assertIn("georeferenced", reasons["2019-03-05/B12.tif"])
        self.assertIn("2 bands", reasons["2019-03-05/B06.tif"])


if __name__ == "__main__":
    unittest.main()
