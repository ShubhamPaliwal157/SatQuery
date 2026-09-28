# Download guide — India demo data from the Copernicus Browser

Time: about 1-1.5 hours for the first three areas. Disk: about 250 MB.
Cost: free (you need a free account). Site: https://browser.dataspace.copernicus.eu/

The menus on that site change now and then. If a label below doesn't match
what you see, take a screenshot and send it — don't guess.

## The 7 rules (read once)

1. **Download bands, not pictures.** Use the "Analytical" download (raw bands as
   separate files). Do NOT download the "true colour"/TCI picture — it is 8-bit,
   the app can't analyse it (the ingest tool skips it anyway).
2. **Format: TIFF, 32-bit float** (best). 16-bit is acceptable if that's all
   you're offered. **Never 8-bit** for anything except a pretty picture — it
   throws away the detail the indices need.
3. **Resolution: 10 m per pixel.**
4. **Coordinate system: pick the UTM option for your area** if it is offered
   (metre pixels -> exact areas in hectares). Use the SAME choice for every file
   of one area. If unsure, WGS84 also works; areas are then slightly approximate.
5. **Same rectangle for every download of one area.** Draw it once, keep it.
   About 10 x 10 km. Don't draw bigger (bigger = huge files and slow).
6. **Optical clouds:** choose scenes with cloud cover under 10 percent, and look at
   the preview over YOUR rectangle, not just the percentage (that's for the whole
   110 km tile).
7. **Every date gets its own folder** named YYYY-MM-DD (see "After downloading").

## One download, step by step (same pattern every time)

1. Log in (top right). Zoom to the place (search box), pick the data source on the
   left: **Sentinel-2 -> L2A**, set the date range and cloud slider, draw your
   rectangle with the area tool on the map, press **Search**.
2. Click a result, then **Visualize**. Check the preview: your rectangle, low cloud.
3. Open the **Download** (image) panel -> **Analytical** tab.
4. Format **TIFF (32-bit float)**, resolution 10 m, coordinate system as in rule 4,
   area = your drawn rectangle, and **tick only the bands listed for that area**.
5. Download. You get a .zip with one .tiff per band. **Unzip it.**

**Sentinel-1 (radar, for floods):** data source Sentinel-1 -> GRD. Mode **IW**,
polarisation **VV+VH**. Pick ONE orbit direction (ascending OR descending) and use
the same one for the before and after date. If you see options for
orthorectification or backscatter: orthorectification ON (Copernicus DEM),
sigma0 if asked, no speckle filter (the app does its own). Bands: **VV, VH**.
Raw GRD "measurement" files from the full-product download do NOT work — they are
in radar geometry; use the Analytical image download above.

**Elevation:** data source **Copernicus DEM** (30 m), band **DEM**. If it isn't in
the list, tell me — there is another free source.

## The areas (do them in this order)

| # | Folder name | Where (W S E N) | What | Bands / dates |
|---|---|---|---|---|
| 1 | `navi_mumbai_urban_growth` | 73.02 18.94 73.12 19.04 (new airport site, Navi Mumbai) | urban change before/after | Sentinel-2 L2A, TWO dates: one in Jan-Mar 2019, one in Jan-Mar 2025 (or 2026), both clear. Bands **B02 B03 B04 B08 B11 SCL** |
| 2 | `chennai_flood_2023` | 80.15 12.90 80.25 13.00 (south Chennai: Velachery, Pallikaranai, Adyar) | Cyclone Michaung flood | Sentinel-1 GRD VV+VH, SAME orbit: BEFORE = last pass before 3 Dec 2023, AFTER = first pass on/after 5 Dec 2023 (within ~10 days). **DEM** once. Optional: one clear Sentinel-2 date in Oct-Nov 2023, bands B02 B03 B04 B08 B11 SCL (skip if it's all cloud) |
| 3 | `punjab_crop_cycle` | 75.55 30.50 75.65 30.60 (farmland, Barnala-Sangrur area) | wheat season, NDVI over time | Sentinel-2 L2A, 5-6 dates roughly a month apart, Nov 2025 - Apr 2026, as clear as you can find (winter fog is common - take what is usable, minimum 3). Bands **B02 B03 B04 B08 SCL** |
| 4 (optional) | `munnar_terrain` | 77.01 10.04 77.11 10.14 (Munnar hills, Kerala) | 3D terrain | **DEM** + one clear Sentinel-2 date Jan-Mar 2026, bands B02 B03 B04 |

The boxes are approximate: after Visualize, make sure the place you want is in
the picture (airport construction, Chennai suburbs, farmland with no big city).
If Sentinel-1 has no usable before/after pair for Chennai, pick another Indian
flood (for example Assam during the monsoon) and tell me the place and dates.

Rough sizes: a 10 x 10 km band at 10 m is ~4 MB (32-bit float). Area 1 ~50 MB,
area 2 ~20-45 MB, area 3 ~100 MB, area 4 ~15 MB.

## After downloading

Make one folder per area inside `data/inbox/`, and inside it one folder per date
(the date the scene was taken, shown in the Browser), then put the unzipped
.tiff files in that date's folder:

```
data/inbox/navi_mumbai_urban_growth/2019-02-10/B02.tiff ... SCL.tiff
data/inbox/navi_mumbai_urban_growth/2025-02-14/B02.tiff ... SCL.tiff
data/inbox/chennai_flood_2023/2023-11-27/  (the VV and VH files)
data/inbox/chennai_flood_2023/2023-12-09/  (the VV and VH files)
data/inbox/chennai_flood_2023/dem/         (the DEM file)
```

Keep the downloaded file names as they are (they contain the band name); the
date folder is what tells the app the date. Then, one area at a time, from the
project folder:

```
python tools/ingest_scene.py navi_mumbai_urban_growth --inbox data/inbox/navi_mumbai_urban_growth --dry-run
```

Read the report (what it placed, skipped, and which analyses it unlocks). If it
looks right, run the same command without `--dry-run`. Then:

```
python tools/check_scenes.py
```

Put the Copernicus attribution into that area's `aoi.json` -> `notes`:
"Contains modified Copernicus Sentinel data [year]." (The DEM: "Copernicus DEM.")

## If something goes wrong

Send me: the exact ingest report text (or the error), and a screenshot of the
Browser's download panel if a menu didn't match. Don't delete anything from
`data/inbox/` until the ingest has succeeded — the tool never modifies it.
