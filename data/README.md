# data/ — where your real satellite data lives

**First time? Read `DOWNLOAD_GUIDE.md` (what to download from the Copernicus Browser, file format, dates, places).**

Folders starting with `_` (like `aois/_template`) are ignored by the app.
Copy `aois/_template/aoi.json` into a new AOI folder and edit it.

## Layout

```
data/
├── inbox/                     raw downloads; `tools/ingest_scene.py` turns them into aois/ (below)
├── quicklooks/                single JPG / PNG / TIF for simple one-image demos
└── aois/
    └── <aoi_slug>/            lowercase, underscores: beira_flood_2019
        ├── aoi.json           title, place, event, notes
        ├── acquisitions/
        │   └── 2019-03-05/    ISO date, one folder per acquisition
        │       ├── optical/   B02.tif B03.tif B04.tif B08.tif B11.tif B12.tif SCL.tif
        │       └── sar/       VV.tif VH.tif
        ├── dem/dem.tif        elevation
        └── labels/            optional *.geojson (training samples, buildings)
```

## Getting real data in (the ingest tool)

Easiest route: drop your downloads into `inbox/` in any folder layout, then

```
python tools/ingest_scene.py beira_flood_2019 --dry-run    # shows the plan, writes nothing
python tools/ingest_scene.py beira_flood_2019               # does it
```

What it accepts (recognised from file names, so folder layout doesn't matter):

| You downloaded | It reads |
|---|---|
| An unzipped Sentinel-2 L2A `.SAFE` folder | `B01..B12, B8A, SCL` from `R10m/R20m/R60m`; the finest copy of each band wins; `TCI/AOT/WVP` are skipped and reported |
| Copernicus Browser / Sentinel Hub exports | one band per file, with the band name (`B04`, `VV`...) somewhere in the file name, e.g. `..._Sentinel-2_L2A_B04_(Raw).tiff`. If a name isn't recognised the report says so; just rename it |
| Loose files you renamed | `B04.tif`, `VV.tif`, `dem.tif`, inside a `YYYY-MM-DD` folder (or use `--date`) |
| A DEM | any name containing `DEM`, or any file inside a folder called `dem` |

What it does: puts every raster on **one grid for the whole AOI** (the finest optical
band, clipped to the area all dates cover), so later analyses can compare dates and
sensors pixel-for-pixel. 20 m and 60 m bands are upsampled (bilinear; `SCL` uses nearest so
class codes stay classes), SAR and the DEM are reprojected onto that grid, and everything is written
as compressed GeoTIFF. `ingest_manifest.json` in the AOI folder records the grid and, for
every file, its source, resampling, value range and nodata share. Data you ingest later is aligned to
the same stored grid. The inbox is never modified.

What it refuses or skips (and says why in the report): zip files (unzip first),
multi-band files, files with no georeferencing (raw Sentinel-1 GRD `measurement/*.tiff` are in radar
geometry and need terrain correction first), acquisitions over a different place, and
adding to an AOI that holds hand-placed data with no manifest.

Full Sentinel-2 tiles are ~110 km wide. Clip while ingesting with
`--bbox WEST SOUTH EAST NORTH` (degrees); grids over 25 M pixels are refused unless you
raise `--max-pixels`. Values are copied as delivered: no reflectance scaling, no dB conversion.

## Naming rules

- Date folders: `YYYY-MM-DD` exactly.
- Optical files: the Sentinel-2 band name, `.tif`. Recognised: B01 B02 B03 B04
  B05 B06 B07 B08 B8A B09 B11 B12 SCL. (B02/B03/B04/B08 are 10 m; B05-B07,
  B8A, B11, B12 and SCL are 20 m — mixed sizes are fine, ingest resamples.)
- SAR files: `VV.tif` and/or `VH.tif`.
- DEM: exactly `dem.tif`.
- Everything must be a georeferenced GeoTIFF. Anything else is listed as a
  warning and ignored.
- Clip each AOI to roughly 10 x 10 km to keep files small.

## What each file unlocks (the app tells you what's missing)

| Analysis | Needs |
|---|---|
| True-colour view / optical change detection | B02+B03+B04 (two dates for change) |
| NDVI, time series | B04+B08 (three or more dates for time series) |
| NDWI / MNDWI | B03+B08 / B03+B11 |
| NDBI, NBR, dNBR | B08+B11 / B08+B12 / B08+B12 on two dates |
| Cloud masking | SCL |
| SAR view, flood extent | VV or VH / VV on two dates |
| 3D terrain, hillshade, slope | dem/dem.tif |
| Supervised land cover | labels/*.geojson + true-colour |

Check what the app sees: `GET http://localhost:8000/scenes` or
`python tools/check_scenes.py`.

Sentinel and Copernicus DEM data are free to use, but keep the attribution
text from the source in `aoi.json` -> `notes`.
