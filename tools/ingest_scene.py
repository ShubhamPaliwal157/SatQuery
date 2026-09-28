"""Turn raw downloads in data/inbox/ into a normalised AOI bundle.
Run from the project root:

    python tools/ingest_scene.py beira_flood_2019 --dry-run     # look first
    python tools/ingest_scene.py beira_flood_2019               # then do it

Full behaviour and limits: the docstring at the top of backend/ingest.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import ingest  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Ingest raw Sentinel-2 / Sentinel-1 / DEM files from data/inbox/ into data/aois/<aoi>/.",
        epilog="Dates come from file names or folders (e.g. inbox/2019-03-05/B04.tif); --date is only a "
               "fallback for files that carry none. Re-running is safe: files already ingested are kept "
               "unless you pass --overwrite. The inbox is never modified.")
    ap.add_argument("aoi", help="AOI name: lowercase letters, digits, underscores (e.g. beira_flood_2019)")
    ap.add_argument("--inbox", type=Path, help="folder to read (default: data/inbox)")
    ap.add_argument("--date", metavar="YYYY-MM-DD", help="acquisition date for files that have none in their name/folders")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                    help="clip to this lon/lat box (use it for full Sentinel-2 tiles, which are ~110 km wide)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and report; write nothing")
    ap.add_argument("--overwrite", action="store_true", help="redo files that were already ingested")
    ap.add_argument("--max-pixels", type=int, default=ingest.DEFAULT_MAX_PIXELS,
                    help=f"refuse grids bigger than this (default {ingest.DEFAULT_MAX_PIXELS:,})")
    args = ap.parse_args(argv)

    try:  # never crash the report on a legacy Windows console
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    try:
        result = ingest.run(args.aoi, inbox=args.inbox, default_date=args.date, bbox=args.bbox,
                            dry_run=args.dry_run, overwrite=args.overwrite, max_pixels=args.max_pixels)
    except ImportError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(ingest.format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
