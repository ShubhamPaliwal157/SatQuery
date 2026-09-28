"""
Image loading utilities — Step 2
-----------------------------------
Real remote-sensing imagery is frequently NOT a plain 8-bit 3-channel
JPEG/PNG — it's often a multi-band, 16-bit (or higher) GeoTIFF. Plain
`PIL.Image.open()` silently mishandles or outright fails on a lot of
these, which was a latent bug in the Step 1 MVP: it only ever worked
correctly on already web-friendly images, not the actual data this
project needs to analyze.

This module tries rasterio first (the geospatial-aware reader the rest of
the planned stack already depends on), and falls back to plain PIL for
files rasterio doesn't need to be involved with. Whatever it reads is
normalized to a standard 8-bit RGB PIL.Image so every downstream vision
model gets a consistent, sane input regardless of source format.

Known limitation (flagged rather than silently glossed over): the
percentile stretch below doesn't mask out nodata pixels, so a GeoTIFF
with large nodata fill regions will get a skewed stretch. Fine for an
MVP; worth revisiting once real mission data is being tested.
"""

from __future__ import annotations
from dataclasses import dataclass
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.image_utils")

try:
    import rasterio
    from rasterio.warp import transform_bounds
    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - rasterio is in requirements.txt, but degrade gracefully
    _HAS_RASTERIO = False


@dataclass
class LoadedImage:
    image: Image.Image          # normalized 8-bit RGB, ready for any vision model
    source_bands: int
    source_dtype: str
    crs: str | None             # geospatial reference, if the file carried one
    stretched: bool             # True if pixel values were rescaled for display
    geo_bounds: tuple[float, float, float, float] | None = None
    # (west, south, east, north) in EPSG:4326 — only set when the source file
    # carried a CRS, so the Map view has something real to draw. Never
    # fabricated: a plain JPEG/PNG with no embedded geo-reference stays None.


def _percentile_stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Rescale a band to 0-255 using a percentile stretch — standard
    practice for visualizing satellite imagery, which is rarely already
    in a display-friendly 0-255 range."""
    lo, hi = np.percentile(band, [low, high])
    if hi <= lo:
        hi = lo + 1
    stretched = np.clip((band.astype("float32") - lo) / (hi - lo), 0, 1) * 255
    return stretched.astype("uint8")


def _load_with_rasterio(raw: bytes) -> LoadedImage:
    with rasterio.MemoryFile(raw) as memfile:
        with memfile.open() as src:
            band_count = src.count
            dtype = str(src.dtypes[0])
            crs = str(src.crs) if src.crs else None

            # Pick up to 3 bands to treat as RGB. Most multispectral products
            # put a usable true/false-color combo in the first three bands;
            # this is a reasonable default, not a spectral-index-aware choice
            # (revisit if a dataset needs a specific band combination).
            n_use = min(3, band_count)
            data = src.read(list(range(1, n_use + 1)))  # shape: (bands, H, W)

            needs_stretch = dtype != "uint8"
            channels = [
                _percentile_stretch(b) if needs_stretch else b.astype("uint8")
                for b in data
            ]

            if n_use == 1:
                arr = np.stack([channels[0]] * 3, axis=-1)  # grayscale -> RGB
            elif n_use == 2:
                arr = np.stack([channels[0], channels[1], channels[0]], axis=-1)
            else:
                arr = np.stack(channels, axis=-1)

            image = Image.fromarray(arr, mode="RGB")

            geo_bounds = None
            if src.crs and src.transform and not src.transform.is_identity:
                try:
                    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    geo_bounds = (round(west, 6), round(south, 6), round(east, 6), round(north, 6))
                except Exception:  # noqa: BLE001 — a bad/unusual CRS shouldn't break loading
                    logger.info("Could not reproject bounds to EPSG:4326 for this file.")

            return LoadedImage(
                image=image,
                source_bands=band_count,
                source_dtype=dtype,
                crs=crs,
                stretched=needs_stretch,
                geo_bounds=geo_bounds,
            )


def _load_with_pil(raw: bytes) -> LoadedImage:
    image = Image.open(io.BytesIO(raw))
    mode = image.mode
    image = image.convert("RGB")
    return LoadedImage(
        image=image,
        source_bands=3 if mode in ("RGB", "RGBA") else 1,
        source_dtype="uint8",
        crs=None,
        stretched=False,
    )


def load_image_any(raw: bytes) -> LoadedImage:
    """Load raw image bytes into a normalized 8-bit RGB PIL image.

    Tries rasterio first (handles GeoTIFFs, multi-band, non-8-bit data
    correctly); falls back to plain PIL for anything rasterio can't open.
    Raises ValueError if neither reader can make sense of the file.
    """
    if _HAS_RASTERIO:
        try:
            return _load_with_rasterio(raw)
        except Exception as e:  # noqa: BLE001 — deliberate: fall through to PIL
            logger.info("rasterio could not open the file (%s); falling back to PIL.", e)
    try:
        return _load_with_pil(raw)
    except Exception as e:
        raise ValueError(f"Could not read this file as an image: {e}") from e
