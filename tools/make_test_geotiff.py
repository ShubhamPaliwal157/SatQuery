"""
Generates a small synthetic 16-bit, 4-band GeoTIFF for testing SatQuery AI's
image loader. Plain image viewers/PIL can't open this file directly — that's
the point, it's the exact case image_utils.py exists to handle.

Run from anywhere with `pip install numpy rasterio` available:
    python make_test_geotiff.py
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

h, w = 256, 256
data = (np.random.rand(4, h, w) * 60000).astype("uint16")
transform = from_origin(77.5, 12.9, 0.0001, 0.0001)  # arbitrary lon/lat near Bengaluru

with rasterio.open(
    "test_multiband.tif", "w",
    driver="GTiff", height=h, width=w, count=4, dtype="uint16",
    crs="EPSG:4326", transform=transform,
) as dst:
    dst.write(data)

print("Wrote test_multiband.tif — try uploading this to the Streamlit app.")
