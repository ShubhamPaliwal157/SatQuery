"""
Scene registry (Roadmap step 1)
===============================
Scans `data/aois/<aoi_slug>/` for real satellite data the user dropped in
and works out — from what files actually exist — which analyses are
possible. Nothing here reads pixel data; only file names and raster
headers, so listing scenes stays fast even with big rasters.

Folder convention (full guide in data/README.md):

    data/aois/<aoi_slug>/
        aoi.json                      optional: title, place, event, notes
        acquisitions/<YYYY-MM-DD>/
            optical/  B02.tif B03.tif B04.tif B08.tif B11.tif B12.tif SCL.tif ...
            sar/      VV.tif VH.tif
        dem/dem.tif
        labels/*.geojson              optional

Capability gating: each analysis declares the files it needs. If they're
missing, the API says exactly what to add — the chatbot will use this to
refuse honestly ("NDVI needs a NIR band, B08") instead of guessing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger("satquery.scenes")

DATA_ROOT = Path(os.environ.get("SATQUERY_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

OPTICAL_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL")
SAR_BANDS = ("VV", "VH")
RASTER_EXTS = (".tif", ".tiff")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GridInfo:
    crs: str | None
    width: int
    height: int
    pixel_size: tuple[float, float]  # in CRS units
    geo_bounds: tuple[float, float, float, float] | None  # west,south,east,north EPSG:4326


@dataclass
class Acquisition:
    date: str
    optical: list[str] = field(default_factory=list)   # band names present
    sar: list[str] = field(default_factory=list)       # polarisations present
    grid: GridInfo | None = None                       # from the first raster found
    mixed_grids: bool = False                          # bands disagree on size/CRS


@dataclass
class Scene:
    id: str
    title: str
    place: str | None = None
    event: str | None = None
    notes: str | None = None
    acquisitions: list[Acquisition] = field(default_factory=list)
    has_dem: bool = False
    label_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Raster header reading (no pixel reads)
# ---------------------------------------------------------------------------
def read_grid(path: Path) -> GridInfo | None:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:  # pragma: no cover
        return None
    try:
        with rasterio.open(path) as src:
            bounds = None
            if src.crs and not src.transform.is_identity:
                try:
                    w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    bounds = (round(w, 6), round(s, 6), round(e, 6), round(n, 6))
                except Exception:  # noqa: BLE001
                    pass
            return GridInfo(
                crs=str(src.crs) if src.crs else None,
                width=src.width,
                height=src.height,
                pixel_size=(abs(src.transform.a), abs(src.transform.e)),
                geo_bounds=bounds,
            )
    except Exception as exc:  # noqa: BLE001 — a bad file must not break listing
        logger.warning("Could not read raster header %s: %s", path, exc)
        return None


def _band_files(folder: Path, allowed: tuple[str, ...], warnings: list[str], label: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not folder.is_dir():
        return found
    for f in sorted(folder.iterdir()):
        if f.name.startswith(".") or not f.is_file():
            continue
        stem, ext = f.stem, f.suffix.lower()
        if ext not in RASTER_EXTS:
            warnings.append(f"{label}: '{f.name}' is not a .tif — ignored.")
        elif stem.upper() in allowed:
            found[stem.upper()] = f
        else:
            warnings.append(f"{label}: '{f.name}' isn't a recognised band name ({', '.join(allowed)}) — ignored.")
    return found


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def _scan_acquisition(acq_dir: Path, warnings: list[str]) -> Acquisition:
    date = acq_dir.name
    label = f"{acq_dir.parent.parent.name}/{date}"
    if not DATE_RE.match(date):
        warnings.append(f"{label}: folder name isn't an ISO date (YYYY-MM-DD).")
    optical = _band_files(acq_dir / "optical", OPTICAL_BANDS, warnings, label + "/optical")
    sar = _band_files(acq_dir / "sar", SAR_BANDS, warnings, label + "/sar")

    acq = Acquisition(date=date, optical=sorted(optical), sar=sorted(sar))
    grids = [g for g in (read_grid(p) for p in [*optical.values(), *sar.values()]) if g]
    if grids:
        acq.grid = grids[0]
        # Optical bands at 10 m vs 20 m legitimately differ in size — flag it
        # so the ingest step (roadmap step 2) knows to resample.
        acq.mixed_grids = any((g.width, g.height, g.crs) != (grids[0].width, grids[0].height, grids[0].crs) for g in grids)
        if acq.mixed_grids:
            warnings.append(f"{label}: bands differ in size or CRS — the ingest tool will resample to one grid.")
    return acq


def scan_scene(aoi_dir: Path) -> Scene:
    warnings: list[str] = []
    meta: dict = {}
    meta_path = aoi_dir / "aoi.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"aoi.json couldn't be read ({exc}) — using defaults.")
    else:
        warnings.append("No aoi.json — title derived from the folder name.")

    scene = Scene(
        id=aoi_dir.name,
        title=meta.get("title") or aoi_dir.name.replace("_", " ").title(),
        place=meta.get("place"),
        event=meta.get("event"),
        notes=meta.get("notes"),
    )

    acq_root = aoi_dir / "acquisitions"
    if acq_root.is_dir():
        for d in sorted(p for p in acq_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            scene.acquisitions.append(_scan_acquisition(d, warnings))

    scene.has_dem = any((aoi_dir / "dem").glob("dem.tif")) or any((aoi_dir / "dem").glob("dem.tiff"))
    labels_dir = aoi_dir / "labels"
    if labels_dir.is_dir():
        scene.label_files = sorted(f.name for f in labels_dir.iterdir() if f.suffix.lower() in (".geojson", ".json"))

    scene.warnings = warnings
    scene.capabilities = compute_capabilities(scene)
    return scene


def list_scenes(root: Path | None = None) -> list[Scene]:
    aois = (root or DATA_ROOT) / "aois"
    if not aois.is_dir():
        return []
    return [scan_scene(d) for d in sorted(aois.iterdir()) if d.is_dir() and not d.name.startswith(("_", "."))]


def get_scene(scene_id: str, root: Path | None = None) -> Scene | None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", scene_id) or scene_id.startswith((".", "_")):
        return None  # blocks path traversal like "../../etc"
    d = (root or DATA_ROOT) / "aois" / scene_id
    return scan_scene(d) if d.is_dir() else None


def list_quicklooks(root: Path | None = None) -> list[str]:
    d = (root or DATA_ROOT) / "quicklooks"
    if not d.is_dir():
        return []
    ok = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in ok and not f.name.startswith("."))


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------
def _any_acq_has(scene: Scene, bands: tuple[str, ...], kind: str = "optical") -> bool:
    return any(all(b in getattr(a, kind) for b in bands) for a in scene.acquisitions)


def _count_acq_has(scene: Scene, bands: tuple[str, ...], kind: str = "optical") -> int:
    return sum(all(b in getattr(a, kind) for b in bands) for a in scene.acquisitions)


# name -> (what it does, what's required in plain words, checker)
CAPABILITY_RULES: dict[str, tuple[str, str, callable]] = {
    "true_colour":  ("True-colour image", "B02, B03, B04 in one acquisition",
                     lambda s: _any_acq_has(s, ("B02", "B03", "B04"))),
    "ndvi":         ("Vegetation index (NDVI)", "B04 + B08",
                     lambda s: _any_acq_has(s, ("B04", "B08"))),
    "ndwi":         ("Water index (NDWI)", "B03 + B08",
                     lambda s: _any_acq_has(s, ("B03", "B08"))),
    "mndwi":        ("Open-water index (MNDWI)", "B03 + B11",
                     lambda s: _any_acq_has(s, ("B03", "B11"))),
    "ndbi":         ("Built-up index (NDBI)", "B08 + B11",
                     lambda s: _any_acq_has(s, ("B08", "B11"))),
    "nbr":          ("Burn ratio (NBR)", "B08 + B12",
                     lambda s: _any_acq_has(s, ("B08", "B12"))),
    "dnbr":         ("Burn severity (dNBR)", "B08 + B12 on two dates",
                     lambda s: _count_acq_has(s, ("B08", "B12")) >= 2),
    "cloud_mask":   ("Cloud / shadow masking", "SCL band",
                     lambda s: _any_acq_has(s, ("SCL",))),
    "optical_change": ("Optical before/after change detection", "true-colour on two dates",
                     lambda s: _count_acq_has(s, ("B02", "B03", "B04")) >= 2),
    "time_series":  ("Vegetation time series", "B04 + B08 on three or more dates",
                     lambda s: _count_acq_has(s, ("B04", "B08")) >= 3),
    "sar_backscatter": ("SAR backscatter view", "VV or VH in one acquisition",
                     lambda s: any(a.sar for a in s.acquisitions)),
    "flood_extent": ("SAR flood extent", "VV on two dates (before + after the event)",
                     lambda s: _count_acq_has(s, ("VV",), "sar") >= 2),
    "sensor_swipe": ("Optical vs SAR comparison", "any optical band and any SAR band",
                     lambda s: any(a.optical for a in s.acquisitions) and any(a.sar for a in s.acquisitions)),
    "terrain_3d":   ("3D terrain / hillshade / slope", "dem/dem.tif",
                     lambda s: s.has_dem),
    "supervised_landcover": ("Supervised land-cover classification", "labels/*.geojson training samples + true-colour",
                     lambda s: bool(s.label_files) and _any_acq_has(s, ("B02", "B03", "B04"))),
}


def compute_capabilities(scene: Scene) -> dict:
    """{name: {available, label, requires}} — `requires` is shown to the
    user verbatim when something's unavailable."""
    out = {}
    for name, (label, requires, check) in CAPABILITY_RULES.items():
        out[name] = {"available": bool(check(scene)), "label": label, "requires": requires}
    return out


def scene_to_dict(scene: Scene) -> dict:
    return asdict(scene)


def scene_summary(scene: Scene) -> dict:
    """Small payload for list views."""
    dates = [a.date for a in scene.acquisitions]
    return {
        "id": scene.id,
        "title": scene.title,
        "place": scene.place,
        "event": scene.event,
        "dates": dates,
        "has_sar": any(a.sar for a in scene.acquisitions),
        "has_optical": any(a.optical for a in scene.acquisitions),
        "has_dem": scene.has_dem,
        "available": sorted(k for k, v in scene.capabilities.items() if v["available"]),
        "warning_count": len(scene.warnings),
    }
