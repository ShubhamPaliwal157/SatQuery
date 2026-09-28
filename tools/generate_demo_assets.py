"""
Generates the bundled demo scenes shipped in `frontend/demo_assets/`.
=======================================================================
Everything here is procedurally drawn with numpy — no downloaded photos,
so there's no copyright question and no internet dependency at demo
time. The goal isn't photorealism, it's giving a judge (or you, mid-
hackathon) something to click that (a) looks like a plausible remote-
sensing scene, (b) has enough texture/edges for the *real* ORB+RANSAC
registration in change_detector.py to actually succeed, and (c) is
reproducible — rerun this any time the demo assets need regenerating.

Produces:
  frontend/demo_assets/farmland.tif        — georeferenced (EPSG:4326),
                                              patchwork fields + a pond.
                                              Drives the Analyze + Map demo.
  frontend/demo_assets/coastal_urban.png   — NOT georeferenced on purpose,
                                              so the Map tab's honest
                                              "no location data" state has
                                              something real to show.
  frontend/demo_assets/change_before.tif   — same vacant-lot scene,
  frontend/demo_assets/change_after.tif      + a new building complex,
                                              both georeferenced identically.
                                              Drives the Compare demo.

Run: python tools/generate_demo_assets.py   (needs numpy, pillow, rasterio)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import rasterio
from rasterio.transform import from_origin

rng = np.random.default_rng(7)
OUT = "frontend/demo_assets"
import os
os.makedirs(OUT, exist_ok=True)

SIZE = 512


def _speckle(arr: np.ndarray, amount: float = 10.0) -> np.ndarray:
    noise = rng.normal(0, amount, arr.shape)
    return np.clip(arr.astype("float32") + noise, 0, 255).astype("uint8")


def _write_geotiff(path: str, rgb: np.ndarray, origin_lon: float, origin_lat: float, deg_per_px: float = 0.00009):
    """rgb: (H, W, 3) uint8. Writes a 3-band GeoTIFF in EPSG:4326."""
    h, w, _ = rgb.shape
    transform = from_origin(origin_lon, origin_lat, deg_per_px, deg_per_px)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(rgb[:, :, b], b + 1)


# ---------------------------------------------------------------------------
# Scene 1: farmland patchwork (georeferenced — Analyze + Map demo)
# ---------------------------------------------------------------------------
def make_farmland() -> np.ndarray:
    img = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(img)
    palette = [(150, 168, 84), (196, 170, 92), (121, 148, 72), (176, 140, 78),
               (110, 132, 60), (205, 186, 120)]
    # irregular field grid via a jittered lattice
    xs = sorted(rng.integers(0, SIZE, 7).tolist() + [0, SIZE])
    ys = sorted(rng.integers(0, SIZE, 7).tolist() + [0, SIZE])
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            color = palette[(i + j) % len(palette)]
            jitter = rng.integers(-10, 10, 3)
            color = tuple(int(np.clip(c + j2, 0, 255)) for c, j2 in zip(color, jitter))
            draw.rectangle([xs[i], ys[j], xs[i + 1] - 1, ys[j + 1] - 1], fill=color)
            # furrow lines for texture (helps ORB keypoints + sells "farmland")
            step = 6
            line_col = tuple(int(c * 0.85) for c in color)
            for x in range(xs[i] + step, xs[i + 1], step):
                draw.line([(x, ys[j]), (x, ys[j + 1] - 1)], fill=line_col, width=1)
    # dirt road
    draw.line([(0, int(SIZE * 0.42)), (SIZE, int(SIZE * 0.58))], fill=(150, 130, 100), width=10)
    # pond
    draw.ellipse([SIZE * 0.68, SIZE * 0.12, SIZE * 0.85, SIZE * 0.26], fill=(58, 92, 128))
    arr = np.array(img)
    return _speckle(arr, 6)


# ---------------------------------------------------------------------------
# Scene 2: coastal urban strip (NOT georeferenced — honest Map fallback)
# ---------------------------------------------------------------------------
def make_coastal_urban() -> np.ndarray:
    img = Image.new("RGB", (SIZE, SIZE))
    draw = ImageDraw.Draw(img)
    # water gradient (top third)
    water_h = int(SIZE * 0.32)
    for y in range(water_h):
        t = y / water_h
        color = (int(30 + 20 * t), int(70 + 40 * t), int(110 + 60 * t))
        draw.line([(0, y), (SIZE, y)], fill=color)
    # sand strip
    draw.rectangle([0, water_h, SIZE, water_h + 14], fill=(210, 195, 150))
    # city grid below
    base_y = water_h + 14
    block = 34
    for gy in range(base_y, SIZE, block):
        for gx in range(0, SIZE, block):
            roof = rng.integers(60, 210)
            tint = rng.integers(-15, 15, 3)
            color = tuple(int(np.clip(roof + t, 20, 235)) for t in tint)
            pad = 3
            draw.rectangle([gx + pad, gy + pad, gx + block - pad, gy + block - pad], fill=color)
    # a couple of green park blocks
    for _ in range(4):
        gx = int(rng.integers(1, SIZE // block - 2) * block)
        gy = int(rng.integers((base_y // block) + 1, SIZE // block - 1) * block)
        draw.rectangle([gx, gy, gx + block * 2, gy + block * 2], fill=(70, 120, 60))
    # roads
    for rx in range(0, SIZE, block * 4):
        draw.line([(rx, base_y), (rx, SIZE)], fill=(90, 90, 95), width=4)
    for ry in range(base_y, SIZE, block * 4):
        draw.line([(0, ry), (SIZE, ry)], fill=(90, 90, 95), width=4)
    arr = np.array(img)
    return _speckle(arr, 5)


# ---------------------------------------------------------------------------
# Scene 3: before/after vacant-lot -> new construction (georeferenced pair)
# ---------------------------------------------------------------------------
def make_change_pair():
    base = Image.new("RGB", (SIZE, SIZE), (132, 150, 96))
    draw = ImageDraw.Draw(base)
    # scrubland texture
    for _ in range(900):
        x, y = rng.integers(0, SIZE, 2)
        r = rng.integers(2, 5)
        shade = rng.integers(-20, 20)
        c = tuple(int(np.clip(v + shade, 0, 255)) for v in (132, 150, 96))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    # a perimeter access road present in both frames (shared structure -> good ORB anchors)
    draw.line([(0, 40), (SIZE, 40)], fill=(140, 128, 108), width=8)
    draw.line([(40, 0), (40, SIZE)], fill=(140, 128, 108), width=8)

    before = np.array(base).copy()

    after_img = base.copy()
    d2 = ImageDraw.Draw(after_img)
    # new building complex
    bx0, by0, bx1, by1 = int(SIZE * 0.32), int(SIZE * 0.30), int(SIZE * 0.72), int(SIZE * 0.62)
    d2.rectangle([bx0, by0, bx1, by1], fill=(150, 150, 155))
    for rx in range(bx0 + 10, bx1 - 10, 22):
        d2.line([(rx, by0), (rx, by1)], fill=(110, 110, 115), width=2)
    for ry in range(by0 + 10, by1 - 10, 22):
        d2.line([(bx0, ry), (bx1, ry)], fill=(110, 110, 115), width=2)
    # roof edge / shadow for realism
    d2.rectangle([bx0, by1, bx1, by1 + 8], fill=(90, 90, 95))
    # a small paved lot + a few "vehicles"
    d2.rectangle([bx1 + 8, by0, bx1 + 60, by1], fill=(105, 105, 108))
    for _ in range(6):
        vx = rng.integers(bx1 + 14, bx1 + 52)
        vy = rng.integers(by0 + 6, by1 - 6)
        d2.rectangle([vx, vy, vx + 6, vy + 3], fill=(200, 40, 40) if rng.random() > 0.5 else (40, 60, 190))

    after = np.array(after_img)
    # simulate a slightly different pass: mild global brightness/contrast shift
    after = np.clip(after.astype("float32") * 1.05 + 4, 0, 255).astype("uint8")
    return _speckle(before, 5), _speckle(after, 5)


if __name__ == "__main__":
    farmland = make_farmland()
    _write_geotiff(f"{OUT}/farmland.tif", farmland, origin_lon=75.8410, origin_lat=30.9010)
    print(f"wrote {OUT}/farmland.tif", farmland.shape)

    coastal = make_coastal_urban()
    Image.fromarray(coastal).save(f"{OUT}/coastal_urban.png")
    print(f"wrote {OUT}/coastal_urban.png", coastal.shape)

    before, after = make_change_pair()
    _write_geotiff(f"{OUT}/change_before.tif", before, origin_lon=77.4210, origin_lat=12.8510)
    _write_geotiff(f"{OUT}/change_after.tif", after, origin_lon=77.4210, origin_lat=12.8510)
    print(f"wrote {OUT}/change_before.tif and change_after.tif", before.shape)

    print("\nDone. These are procedurally generated, not real satellite captures —")
    print("good enough to demo the pipeline end-to-end without needing a real dataset on hand.")
