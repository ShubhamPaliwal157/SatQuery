"""Prints what the app sees under data/aois — no backend needed.
Run from the project root:  python tools/check_scenes.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import scenes  # noqa: E402

found = scenes.list_scenes()
if not found:
    print(f"No scenes under {scenes.DATA_ROOT / 'aois'} yet — see data/README.md.")
for s in found:
    print(f"\n== {s.id}: {s.title}")
    for a in s.acquisitions:
        print(f"   {a.date}  optical={a.optical or '-'}  sar={a.sar or '-'}")
    print("   DEM:", "yes" if s.has_dem else "no", "| labels:", s.label_files or "none")
    ok = [v["label"] for v in s.capabilities.values() if v["available"]]
    missing = [f"{v['label']} (needs {v['requires']})" for v in s.capabilities.values() if not v["available"]]
    print("   CAN DO:", ", ".join(ok) or "nothing yet")
    print("   MISSING:", "; ".join(missing) or "nothing")
    for w in s.warnings:
        print("   ! ", w)
