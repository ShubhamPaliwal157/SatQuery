"""
Ingest tool (analyst-grade build plan, step 2)
==============================================
Turns whatever is sitting in `data/inbox/` (loose band files, an unzipped
Sentinel-2 L2A .SAFE folder, Copernicus Browser exports, a DEM) into the
folder convention `scenes.py` expects:

    data/aois/<aoi>/acquisitions/<YYYY-MM-DD>/optical/B04.tif ...
                                              /sar/VV.tif ...
                    dem/dem.tif
                    ingest_manifest.json     <- grid + provenance + per-file stats

What it does, in order (each step can only *drop* a file with a stated
reason; nothing is ever guessed silently):

  1. discover   recognise band / polarisation / DEM from names, date from the
                file name or folders (or --date as a last resort)
  2. inspect    read raster headers; skip anything that isn't a single-band,
                georeferenced, unrotated raster
  3. select     when a band exists at 10/20/60 m, keep the finest copy
  4. grid       ONE grid for the whole AOI (see below)
  5. align      warp every raster onto that grid (nearest for SCL, bilinear
                when upsampling, average when downsampling)
  6. report     what was placed, skipped and why, and which analyses this
                unlocked (computed with the real capability rules in scenes.py)

Design decision — one AOI-wide grid, not one grid per date. Change detection,
time series, index maths across dates and the optical/SAR swipe canvas all
become plain pixel-for-pixel operations once every raster shares a grid, so
no later step needs its own co-registration code. The price: SAR/DEM are
resampled onto the optical grid (documented per file in the manifest).
The grid is chosen once — the finest optical raster, clipped to the area every
acquisition covers (and to --bbox) — then stored in the manifest, so data
ingested in a later run is aligned to the *same* grid.

Not handled (reported honestly rather than approximated): multi-band files,
raw Sentinel-1 GRD in radar geometry (needs terrain correction first),
mosaicking several tiles of one date, zip archives (unzip first), rotated
grids. Raster values are copied as-is: no DN->reflectance scaling, no
linear->dB conversion (those belong to the layer steps, which record what
they assume).

All raster I/O sits behind `Backend` so the decision logic above is testable
without GDAL; `RasterioBackend` is the real implementation.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date as _date, datetime, timezone
from pathlib import Path
from typing import Protocol

import scenes

logger = logging.getLogger("satquery.ingest")

RASTER_EXTS = (".tif", ".tiff", ".jp2")
ARCHIVE_EXTS = (".zip", ".tar", ".gz", ".tgz")
MANIFEST_NAME = "ingest_manifest.json"
DEFAULT_MAX_PIXELS = 25_000_000   # ~5000 x 5000; bigger needs --bbox or --max-pixels
SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_]*")

MIN_OVERLAP = 0.05    # below this a raster is "a different place" and is left out
WARN_OVERLAP = 0.80   # below this (vs the reference scene) the shared area is much smaller
FULL_COVER = 0.98     # below this (vs the final grid) part of the grid is nodata

Bounds = tuple[float, float, float, float]  # left, bottom, right, top


class IngestError(Exception):
    """Fatal for this run; the message is shown to the user as-is."""


class RasterProblem(Exception):
    """A single file can't be used; the message becomes its skip reason."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RasterMeta:
    crs: str | None
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | None
    bounds: Bounds                 # in the raster's own CRS
    pixel: tuple[float, float]     # x, y size in CRS units (positive)
    pixel_m: float                 # approximate metres, for ranking and messages only


@dataclass
class Candidate:
    path: Path
    rel: Path                      # relative to the inbox
    kind: str                      # optical | sar | dem
    band: str                      # B04 | VV | DEM ...
    date: str | None               # ISO, from the name or folders
    date_note: str | None = None
    level: str | None = None       # "L1C" when the name says top-of-atmosphere

    @property
    def label(self) -> str:
        """Short but unambiguous: '2019-03-05/B04.tif', 'R10m/T36KWA_..._B04_10m.jp2'."""
        return "/".join(self.rel.parts[-2:])


@dataclass
class Skip:
    name: str
    reason: str


@dataclass
class Located:
    cand: Candidate
    meta: RasterMeta
    date: str | None               # cand.date, or the --date fallback


@dataclass(frozen=True)
class Grid:
    crs: str
    left: float
    top: float
    px: float
    py: float
    width: int
    height: int
    pixel_m: float

    @property
    def bounds(self) -> Bounds:
        return (self.left, self.top - self.height * self.py, self.left + self.width * self.px, self.top)

    def coeffs(self) -> tuple[float, float, float, float, float, float]:
        """Affine coefficients (a, b, c, d, e, f), north-up."""
        return (self.px, 0.0, self.left, 0.0, -self.py, self.top)

    def to_dict(self) -> dict:
        return {"crs": self.crs, "left": self.left, "top": self.top, "pixel_size": [self.px, self.py],
                "width": self.width, "height": self.height, "pixel_m": self.pixel_m,
                "bounds": list(self.bounds)}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(d["crs"], d["left"], d["top"], d["pixel_size"][0], d["pixel_size"][1],
                   d["width"], d["height"], d.get("pixel_m", d["pixel_size"][0]))


@dataclass
class Planned:
    cand: Candidate
    meta: RasterMeta
    date: str | None
    dst_rel: str                   # relative to the AOI folder
    resampling: str
    note: str                      # what alignment did to it ("" = copied cell-for-cell)
    action: str                    # write | exists


@dataclass
class IngestResult:
    aoi: str
    aoi_dir: Path
    inbox: Path
    dry_run: bool
    grid: Grid | None = None
    grid_note: str = ""
    n_seen: int = 0
    n_ignored: int = 0
    planned: list[Planned] = field(default_factory=list)
    written: list[dict] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unlocked: list[str] = field(default_factory=list)          # labels of newly available analyses
    available_after: list[str] = field(default_factory=list)   # capability names once done
    created_aoi_json: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.planned)


# ---------------------------------------------------------------------------
# 1. Discovery: names -> (kind, band, date)
# ---------------------------------------------------------------------------
_BAND_NAMES = set(scenes.OPTICAL_BANDS) | set(scenes.SAR_BANDS) | {"DEM"}
_TOKEN = re.compile(r"[^A-Za-z0-9]+")
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_COMPACT = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")
_L1C = re.compile(r"MSIL1C|(?<![A-Za-z0-9])L1C(?![A-Za-z0-9])", re.I)


def _valid_date(y: str, m: str, d: str) -> str | None:
    try:
        if not 1990 <= int(y) <= 2100:
            return None
        return _date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def find_dates(text: str) -> tuple[list[str], list[str]]:
    """(ISO-format dates, compact YYYYMMDD dates) in order of appearance."""
    iso = [d for d in (_valid_date(*m.groups()) for m in _ISO.finditer(text)) if d]
    compact = [d for d in (_valid_date(*m.groups()) for m in _COMPACT.finditer(text)) if d]
    return iso, compact


def date_from_path(rel: Path) -> tuple[str | None, str | None]:
    """(date, note). The file name is tried first, then folders nearest-first,
    so a Sentinel-2 file name beats its (processing-date) parent folder."""
    for part in [rel.name, *[p.name for p in rel.parents if p.name]]:
        iso, compact = find_dates(part)
        dates = iso or compact
        if dates:
            note = None
            if len(set(iso)) > 1:  # Copernicus Browser names carry a from-to range
                note = f"{'/'.join(rel.parts[-2:])} spans several dates ({', '.join(dict.fromkeys(iso))}); using {dates[0]}."
            return dates[0], note
    return None, None


def classify(path: Path, inbox: Path) -> Candidate | Skip:
    rel = path.relative_to(inbox)
    # The LAST band token wins: '..._Sentinel-1_IW_VV+VH_VV_(Raw)' is the VV file.
    hits = [t for t in (t.upper() for t in _TOKEN.split(rel.stem) if t) if t in _BAND_NAMES]
    if not hits and any(p.lower() == "dem" for p in rel.parts[:-1]):
        hits = ["DEM"]
    if not hits:
        return Skip("/".join(rel.parts[-2:]), "not a recognised band name (B01-B12, B8A, SCL, VV, VH, DEM) - rename it if it should be used")
    band = hits[-1]
    kind = "dem" if band == "DEM" else "sar" if band in scenes.SAR_BANDS else "optical"
    date, note = (None, None) if kind == "dem" else date_from_path(rel)
    level = "L1C" if kind == "optical" and _L1C.search(str(rel)) else None
    return Candidate(path=path, rel=rel, kind=kind, band=band, date=date, date_note=note, level=level)


def discover(inbox: Path) -> tuple[list[Candidate], list[Skip], int]:
    """Walk the inbox -> (candidates, skipped, count of non-raster files ignored)."""
    cands: list[Candidate] = []
    skips: list[Skip] = []
    ignored = 0
    for dirpath, dirnames, filenames in os.walk(inbox):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in ARCHIVE_EXTS:
                skips.append(Skip(fn, "archive - unzip it first (ingest reads folders and .tif/.jp2 files)"))
            elif ext in RASTER_EXTS:
                r = classify(p, inbox)
                (skips if isinstance(r, Skip) else cands).append(r)
            else:
                ignored += 1
    return cands, skips, ignored


# ---------------------------------------------------------------------------
# 3. Selection: finest copy of each band
# ---------------------------------------------------------------------------
def _same_res(a: float, b: float) -> bool:
    return abs(a - b) <= 0.01 * max(a, b)


def select_best(located: list[Located]) -> tuple[list[Located], list[Skip], list[str]]:
    groups: dict[tuple, list[Located]] = defaultdict(list)
    for item in located:
        groups[(item.cand.kind, item.date or "", item.cand.band)].append(item)
    chosen: list[Located] = []
    skipped: list[Skip] = []
    warns: list[str] = []
    for (kind, date, band), items in sorted(groups.items()):
        items.sort(key=lambda i: (round(i.meta.pixel_m, 2), str(i.cand.rel)))
        best = items[0]
        chosen.append(best)
        for other in items[1:]:
            if _same_res(other.meta.pixel_m, best.meta.pixel_m):
                warns.append(f"{band} on {date or 'n/a'}: '{best.cand.label}' and '{other.cand.label}' have the "
                             f"same resolution - used the first. If they are different tiles, ingest them separately "
                             f"(mosaicking isn't supported).")
                skipped.append(Skip(other.cand.label, "duplicate of a band already taken at the same resolution"))
            else:
                skipped.append(Skip(other.cand.label, "coarser copy of a band already taken at finer resolution"))
    return chosen, skipped, warns


# ---------------------------------------------------------------------------
# 4. Grid maths (pure)
# ---------------------------------------------------------------------------
def _area(b: Bounds | None) -> float:
    return 0.0 if b is None else max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersect(a: Bounds | None, b: Bounds | None) -> Bounds | None:
    if a is None or b is None:
        return None
    l, bt, r, t = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (l, bt, r, t) if r > l and t > bt else None


def coverage(footprint: Bounds, grid_bounds: Bounds) -> float:
    """Share of the grid's area that `footprint` covers."""
    return _area(_intersect(footprint, grid_bounds)) / _area(grid_bounds)


def snap_grid(ref: RasterMeta, bounds: Bounds) -> Grid:
    """Grid inside `bounds`, on the pixel lattice of the reference raster (so
    reference bands copy across cell-for-cell). Snaps inward, never outward."""
    px, py = ref.pixel
    rl, rt = ref.bounds[0], ref.bounds[3]
    eps = 1e-9
    left = rl + math.ceil((bounds[0] - rl) / px - eps) * px
    right = rl + math.floor((bounds[2] - rl) / px + eps) * px
    top = rt - math.ceil((rt - bounds[3]) / py - eps) * py
    bottom = rt - math.floor((rt - bounds[1]) / py + eps) * py
    width = round((right - left) / px)
    height = round((top - bottom) / py)
    if width < 1 or height < 1:
        raise IngestError("The area shared by all the data is smaller than one pixel - check the dates/tiles and --bbox.")
    return Grid(ref.crs, left, top, px, py, width, height, ref.pixel_m)


def _new_grid(selected: list[Located], backend: "Backend", bbox, max_pixels: int):
    """-> (grid, note, dropped[(Located, reason)], warnings)."""
    images = [i for i in selected if i.cand.kind != "dem"]
    pool = images or [i for i in selected if i.cand.kind == "dem"]
    # Optical beats SAR beats DEM as the reference; then finest pixels, earliest date.
    ref = min(pool, key=lambda i: (i.cand.kind != "optical", round(i.meta.pixel_m, 2), i.date or "", i.cand.band))
    crs = ref.meta.crs
    ref_fp = ref.meta.bounds
    common: Bounds | None = ref_fp
    dropped: list[tuple[Located, str]] = []
    warns: list[str] = []
    for item in pool:
        if item is ref:
            continue
        fp = backend.footprint_in(item.meta, crs)
        frac = _area(_intersect(ref_fp, fp)) / _area(ref_fp)
        if frac < MIN_OVERLAP:
            dropped.append((item, f"does not overlap the reference scene ({ref.cand.label}) - different area?"))
            continue
        if frac < WARN_OVERLAP:
            warns.append(f"'{item.cand.label}' covers only {frac:.0%} of the reference scene; the shared area is clipped to it.")
        common = _intersect(common, fp)
        if common is None:
            raise IngestError("The acquisitions have no common area - they cover different places.")
    share = _area(common) / _area(ref_fp)
    if share < WARN_OVERLAP:
        warns.append(f"The area shared by all acquisitions is only {share:.0%} of the reference scene.")
    if bbox is not None:
        common = _intersect(common, backend.lonlat_bounds_in(tuple(bbox), crs))
        if common is None:
            raise IngestError("--bbox doesn't overlap the data. Give it as WEST SOUTH EAST NORTH in degrees (lon/lat).")
    grid = snap_grid(ref.meta, common)
    if grid.width * grid.height > max_pixels:
        raise IngestError(f"The grid would be {grid.width} x {grid.height} px ({grid.width * grid.height / 1e6:.0f} M) - "
                          f"too big. Clip it with --bbox WEST SOUTH EAST NORTH, or raise --max-pixels.")
    note = (f"new grid: reference is {ref.cand.band}"
            + (f" on {ref.date}" if ref.date else "")
            + f" ({ref.cand.label}), clipped to the area every acquisition covers"
            + (" and to --bbox" if bbox is not None else ""))
    return grid, note, dropped, warns


def choose_resampling(band: str, src_pixel_m: float, grid_pixel_m: float) -> str:
    if band == "SCL":                       # class codes: never blend
        return "nearest"
    if src_pixel_m < 0.9 * grid_pixel_m:    # source finer than the grid
        return "average"
    return "bilinear"


def _short_crs(crs: str | None) -> str:
    """'EPSG:32736' as-is; a CRS with no EPSG code is a long WKT string - don't print that."""
    return crs if crs and len(crs) <= 30 else "a different CRS"


def alignment_note(meta: RasterMeta, grid: Grid) -> str:
    notes = []
    if meta.crs != grid.crs:
        notes.append(f"reprojected from {_short_crs(meta.crs)}")
    if abs(meta.pixel_m - grid.pixel_m) > 0.02 * grid.pixel_m:
        notes.append(f"{meta.pixel_m:.0f} m -> {grid.pixel_m:.0f} m")
    if not notes and meta.crs == grid.crs:
        dx = (meta.bounds[0] - grid.left) / grid.px
        dy = (grid.top - meta.bounds[3]) / grid.py
        if abs(dx - round(dx)) > 0.01 or abs(dy - round(dy)) > 0.01:
            notes.append("shifted onto the grid")
    return ", ".join(notes)


# ---------------------------------------------------------------------------
# Raster I/O boundary
# ---------------------------------------------------------------------------
class Backend(Protocol):
    def inspect(self, path: Path) -> RasterMeta: ...
    def footprint_in(self, meta: RasterMeta, crs: str) -> Bounds: ...
    def lonlat_bounds_in(self, bbox: tuple[float, float, float, float], crs: str) -> Bounds: ...
    def warp(self, src: Path, dst: Path, grid: Grid, resampling: str) -> dict: ...


class RasterioBackend:
    """The real thing. Kept thin on purpose: every decision is made above."""

    def __init__(self):
        try:
            import rasterio  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("rasterio is required for ingest - pip install -r backend/requirements.txt") from exc

    def inspect(self, path: Path) -> RasterMeta:
        import rasterio
        try:
            with rasterio.open(path) as src:
                tr = src.transform
                if not src.crs or tr.is_identity:
                    raise RasterProblem(
                        "isn't georeferenced (no CRS / map transform). Raw Sentinel-1 GRD 'measurement' files are in "
                        "radar geometry - terrain-correct them first, or download an already-orthorectified product")
                if abs(tr.b) > 1e-9 or abs(tr.d) > 1e-9:
                    raise RasterProblem("has a rotated/sheared grid, which ingest doesn't support")
                px = (abs(tr.a), abs(tr.e))
                b = tuple(float(v) for v in src.bounds)
                if src.crs.is_geographic:
                    lat = math.radians((b[1] + b[3]) / 2)
                    pixel_m = 0.5 * (px[0] * math.cos(lat) + px[1]) * 111_320
                else:
                    pixel_m = 0.5 * (px[0] + px[1])
                return RasterMeta(crs=src.crs.to_string(), width=src.width, height=src.height, count=src.count,
                                  dtype=src.dtypes[0], nodata=src.nodata, bounds=b, pixel=px, pixel_m=pixel_m)
        except RasterProblem:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad file mustn't stop the batch
            hint = " (JP2 needs a GDAL build with OpenJPEG)" if path.suffix.lower() == ".jp2" else ""
            raise RasterProblem(f"couldn't be opened: {exc}{hint}") from exc

    def footprint_in(self, meta: RasterMeta, crs: str) -> Bounds:
        if meta.crs == crs:
            return meta.bounds
        from rasterio.warp import transform_bounds
        return tuple(transform_bounds(meta.crs, crs, *meta.bounds, densify_pts=21))

    def lonlat_bounds_in(self, bbox, crs: str) -> Bounds:
        from rasterio.warp import transform_bounds
        return tuple(transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21))

    def warp(self, src_path: Path, dst_path: Path, grid: Grid, resampling: str) -> dict:
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import reproject

        tmp = dst_path.with_name(dst_path.name + ".part")   # a crash never leaves a truncated B04.tif
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(src_path) as src:
                dtype = src.dtypes[0]
                is_float = np.issubdtype(np.dtype(dtype), np.floating)
                # Integer bands without a declared nodata use 0 (the Sentinel-2 convention).
                nodata = src.nodata if src.nodata is not None else (float("nan") if is_float else 0)
                dest = np.full((grid.height, grid.width), nodata, dtype=dtype)
                transform = Affine(*grid.coeffs())
                reproject(source=rasterio.band(src, 1), destination=dest,
                          src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                          dst_transform=transform, dst_crs=grid.crs, dst_nodata=nodata,
                          resampling=Resampling[resampling])
            if is_float:
                valid = np.isfinite(dest)
                if not math.isnan(nodata):
                    valid &= dest != nodata
            else:
                valid = dest != nodata
            n_valid = int(valid.sum())
            profile = dict(driver="GTiff", height=grid.height, width=grid.width, count=1, dtype=dtype,
                           crs=grid.crs, transform=transform, nodata=nodata, compress="deflate",
                           predictor=3 if is_float else 2, tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(tmp, "w", **profile) as out:
                out.write(dest, 1)
                out.update_tags(SATQUERY_SOURCE=src_path.name, SATQUERY_RESAMPLING=resampling)
            os.replace(tmp, dst_path)
        finally:
            if tmp.exists():
                tmp.unlink()
        vals = dest[valid]
        return {
            "dtype": dtype,
            "nodata": "nan" if (isinstance(nodata, float) and math.isnan(nodata)) else nodata,
            "n_valid": n_valid,
            "nodata_fraction": round(1 - n_valid / dest.size, 4),
            "min": float(vals.min()) if n_valid else None,
            "max": float(vals.max()) if n_valid else None,
            "mean": round(float(vals.mean()), 4) if n_valid else None,
        }


# ---------------------------------------------------------------------------
# Manifest + bookkeeping
# ---------------------------------------------------------------------------
def _clean(obj):
    """Make JSON-safe: non-finite floats -> None."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _load_manifest(aoi_dir: Path) -> dict | None:
    p = aoi_dir / MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise IngestError(f"{p} exists but can't be read. Fix or delete it (deleting it means the grid is chosen again).")


def _save_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(_clean(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _projected_capabilities(existing: scenes.Scene | None, planned: list[Planned], aoi: str) -> dict:
    """What scenes.py would report once `planned` is on disk — same rules, no rescan."""
    acqs: dict[str, tuple[set, set]] = {}
    has_dem = False
    labels: list[str] = []
    if existing:
        acqs = {a.date: (set(a.optical), set(a.sar)) for a in existing.acquisitions}
        has_dem, labels = existing.has_dem, list(existing.label_files)
    for p in planned:
        if p.cand.kind == "dem":
            has_dem = True
        else:
            opt, sar = acqs.setdefault(p.date, (set(), set()))
            (opt if p.cand.kind == "optical" else sar).add(p.cand.band)
    scene = scenes.Scene(id=aoi, title=aoi, has_dem=has_dem, label_files=labels,
                         acquisitions=[scenes.Acquisition(date=d, optical=sorted(o), sar=sorted(s))
                                       for d, (o, s) in sorted(acqs.items())])
    return scenes.compute_capabilities(scene)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def run(aoi: str, *, data_root: Path | None = None, inbox: Path | None = None,
        default_date: str | None = None, bbox=None, dry_run: bool = False, overwrite: bool = False,
        max_pixels: int = DEFAULT_MAX_PIXELS, backend: Backend | None = None) -> IngestResult:
    root = Path(data_root) if data_root else scenes.DATA_ROOT
    inbox = Path(inbox) if inbox else root / "inbox"
    res = IngestResult(aoi=aoi, aoi_dir=root / "aois" / aoi, inbox=inbox, dry_run=dry_run)
    try:
        _run(res, default_date, bbox, overwrite, max_pixels, backend)
    except IngestError as exc:
        res.errors.append(str(exc))
    return res


def _run(res: IngestResult, default_date, bbox, overwrite: bool, max_pixels: int, backend) -> None:
    if not SLUG_RE.fullmatch(res.aoi):
        raise IngestError("AOI name must be lowercase letters, digits and underscores, e.g. beira_flood_2019.")
    if default_date and not (re.fullmatch(r"\d{4}-\d{2}-\d{2}", default_date) and _valid_date(*default_date.split("-"))):
        raise IngestError("--date must be a real date written YYYY-MM-DD.")
    if bbox is not None:
        w, s, e, n = bbox
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise IngestError("--bbox must be WEST SOUTH EAST NORTH in degrees, with west < east and south < north.")
    if not res.inbox.is_dir():
        raise IngestError(f"Inbox folder not found: {res.inbox}")

    # 1. discover
    cands, skips, res.n_ignored = discover(res.inbox)
    res.skipped += skips
    res.n_seen = len(cands) + len(skips)
    if not cands:
        raise IngestError("Nothing in the inbox looks like a band file. Expected names containing B02, B08, SCL, VV, "
                          "VH or DEM (see data/README.md), inside any folder layout.")
    backend = backend or RasterioBackend()

    # 2. inspect
    located: list[Located] = []
    for c in cands:
        try:
            meta = backend.inspect(c.path)
        except Exception as exc:  # noqa: BLE001
            res.skipped.append(Skip(c.label, str(exc) if isinstance(exc, RasterProblem) else f"couldn't be opened: {exc}"))
            continue
        if meta.count != 1:
            res.skipped.append(Skip(c.label, f"has {meta.count} bands; ingest reads single-band files (export one band per file)"))
            continue
        date = c.date or default_date
        if date is None and c.kind != "dem":
            res.skipped.append(Skip(c.label, "no acquisition date in its name or folders - put it in a YYYY-MM-DD folder or pass --date"))
            continue
        if c.date_note:
            res.warnings.append(c.date_note)
        located.append(Located(c, meta, date))

    # 3. select
    selected, dup_skips, dup_warns = select_best(located)
    res.skipped += dup_skips
    res.warnings += dup_warns
    if not selected:
        raise IngestError("No usable rasters left - see the skipped list below.")

    # 4. grid
    manifest = _load_manifest(res.aoi_dir)
    has_data = any(p.suffix.lower() in scenes.RASTER_EXTS
                   for sub in ("acquisitions", "dem") for p in (res.aoi_dir / sub).rglob("*"))
    dropped: list[tuple[Located, str]] = []
    if manifest and manifest.get("grid"):
        res.grid = Grid.from_dict(manifest["grid"])
        res.grid_note = "existing AOI grid from an earlier ingest - new data is aligned to it"
        if bbox is not None:
            res.warnings.append("--bbox ignored: this AOI already has a grid. Delete ingest_manifest.json and the "
                                "acquisitions/dem folders to start over.")
    elif has_data:
        raise IngestError("This AOI already holds data that wasn't made by this tool (no ingest_manifest.json), so I can't "
                          "tell which grid to align to. Use a new AOI name, or move its acquisitions/ and dem/ folders aside.")
    else:
        res.grid, res.grid_note, dropped, warns = _new_grid(selected, backend, bbox, max_pixels)
        res.warnings += warns
    grid = res.grid

    # 4b. does everything actually cover the grid?
    dropped_ids = {id(i) for i, _ in dropped}
    for item in selected:
        if id(item) in dropped_ids:
            continue
        frac = coverage(backend.footprint_in(item.meta, grid.crs), grid.bounds)
        if frac < MIN_OVERLAP:
            dropped.append((item, "does not overlap the AOI grid - different area?"))
        elif frac < FULL_COVER:
            res.warnings.append(f"'{item.cand.label}' covers only {frac:.0%} of the grid; the rest will be nodata.")
    dropped_ids = {id(i) for i, _ in dropped}
    res.skipped += [Skip(i.cand.label, why) for i, why in dropped]
    final = [i for i in selected if id(i) not in dropped_ids]
    if not final:
        raise IngestError("Nothing left to ingest after the overlap check.")

    # 5. plan
    for item in final:
        c = item.cand
        folder = "dem" if c.kind == "dem" else f"acquisitions/{item.date}/{c.kind}"
        name = "dem.tif" if c.kind == "dem" else f"{c.band}.tif"
        dst_rel = f"{folder}/{name}"
        exists = (res.aoi_dir / dst_rel).exists()
        res.planned.append(Planned(
            cand=c, meta=item.meta, date=item.date, dst_rel=dst_rel,
            resampling=choose_resampling(c.band, item.meta.pixel_m, grid.pixel_m),
            note=alignment_note(item.meta, grid),
            action="exists" if exists and not overwrite else "write"))
    if any(p.cand.level == "L1C" for p in res.planned):
        res.warnings.append("Some optical files are Sentinel-2 Level-1C (top-of-atmosphere, no SCL cloud mask). "
                            "Indices won't be comparable across dates - Level-2A is recommended.")

    # what does this unlock?
    existing = scenes.get_scene(res.aoi, res.aoi_dir.parent.parent) if res.aoi_dir.is_dir() else None
    before = _projected_capabilities(existing, [], res.aoi)
    after = _projected_capabilities(existing, res.planned, res.aoi)
    res.available_after = sorted(k for k, v in after.items() if v["available"])
    res.unlocked = [after[k]["label"] for k in after if after[k]["available"] and not before[k]["available"]]

    if res.dry_run:
        return

    # 6. align + write
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for p in res.planned:
        if p.action != "write":
            continue
        try:
            stats = backend.warp(p.cand.path, res.aoi_dir / p.dst_rel, grid, p.resampling)
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"{p.dst_rel}: {exc}")
            continue
        res.written.append({"dst": p.dst_rel, "source": str(p.cand.rel), "source_pixel_m": round(p.meta.pixel_m, 3),
                            "source_crs": p.meta.crs, "resampling": p.resampling, "note": p.note,
                            "ingested_at": stamp, **stats})
        if stats.get("n_valid") == 0:
            res.warnings.append(f"{p.dst_rel} has no valid pixels after alignment - check the data really covers this area.")
        elif stats.get("nodata_fraction", 0) > 0.5:
            res.warnings.append(f"{p.dst_rel} is {stats['nodata_fraction']:.0%} nodata after alignment.")

    if res.written:
        aoi_json = res.aoi_dir / "aoi.json"
        if not aoi_json.exists():
            _save_json(aoi_json, {"title": res.aoi.replace("_", " ").title(), "place": "", "event": "", "notes": ""})
            res.created_aoi_json = True
        m = manifest or {"version": 1, "files": {}}
        m["grid"] = grid.to_dict()
        for rec in res.written:
            m["files"][rec["dst"]] = {k: v for k, v in rec.items() if k != "dst"}
        _save_json(res.aoi_dir / MANIFEST_NAME, m)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _num(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".")


def format_report(r: IngestResult) -> str:
    """Plain ASCII on purpose: Windows consoles choke on arrows and em-dashes."""
    L: list[str] = []
    L.append(f"Ingest report - AOI '{r.aoi}'" + ("   [DRY RUN - nothing was written]" if r.dry_run else ""))
    if r.n_seen or not r.errors:
        L.append(f"Inbox: {r.inbox}  ({r.n_seen} files considered, {r.n_ignored} non-raster files ignored)")
    for e in r.errors:
        L.append(f"ERROR: {e}")
    if r.grid:
        g = r.grid
        L += ["", "Grid",
              f"  {_short_crs(g.crs)}, about {g.pixel_m:g} m pixels, {g.width} x {g.height} px ({g.width * g.height / 1e6:.2f} M px)",
              f"  bounds (left, bottom, right, top): {', '.join(_num(v) for v in g.bounds)}",
              f"  {r.grid_note}"]
    if r.planned:
        verb = "Would place" if r.dry_run else "Placed"
        n_new = sum(p.action == "write" for p in r.planned)
        L += ["", f"{verb} {n_new} file(s)" + (f", kept {len(r.planned) - n_new} already there (use --overwrite to redo)" if n_new != len(r.planned) else "")]
        groups: dict[tuple, list[Planned]] = defaultdict(list)
        for p in r.planned:
            groups[(p.date or "AOI-wide", p.cand.kind)].append(p)
        for (date, kind), items in sorted(groups.items()):
            bands = " ".join(p.cand.band if p.cand.kind != "dem" else "dem" for p in items)
            L.append(f"  {date:<10}  {kind:<7} {bands}")
            for note in sorted({p.note for p in items if p.note}):
                names = " ".join(p.cand.band for p in items if p.note == note)
                L.append(f"              resampled ({note}): {names}")
    if r.skipped:
        by_reason: dict[str, list[str]] = defaultdict(list)
        for s in r.skipped:
            by_reason[s.reason].append(s.name)
        L += ["", f"Skipped {len(r.skipped)} file(s)"]
        for reason, names in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            eg = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
            L.append(f"  {len(names)} x {reason}")
            L.append(f"      e.g. {eg}")
    if r.warnings:
        L += ["", "Warnings"] + [f"  ! {w}" for w in r.warnings]
    if r.planned:
        L += ["", "Analyses newly available: " + (", ".join(r.unlocked) if r.unlocked else "none (nothing new was unlocked)")]
        if r.created_aoi_json:
            L.append(f"Created {r.aoi}/aoi.json - fill in place/event and paste the data source's attribution into notes.")
        L += ["The inbox was not modified; delete its files yourself once you've checked the result.",
              "See everything still missing:  python tools/check_scenes.py"]
    return "\n".join(L)
