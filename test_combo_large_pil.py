from pathlib import Path
import numpy as np
from PIL import Image

root = Path("D:/Nemesis/data/HOPS/RGBs/HOPS_99")

testimg = root / "test_HOPS_99_img_m1KsJ.JPEG"
newimg = root / "HOPS_99_img_m1KsJ.JPEG"

with Image.open(newimg) as img:
    d = np.array(img)
with Image.open(testimg) as img:
    d2 = np.array(img)

assert (d2 == d).all()

testimg = root / "test_HOPS_99_img_m1i3Ks.JPEG"
newimg = root / "HOPS_99_img_m1i3Ks.JPEG"

with Image.open(newimg) as img:
    d = np.array(img)
with Image.open(testimg) as img:
    d2 = np.array(img)

assert (d2 == d).all()

testimg = root / "test_HOPS_99_img_i4i3i2.JPEG"
newimg = root / "HOPS_99_img_i4i3i2.JPEG"

with Image.open(newimg) as img:
    d = np.array(img)
with Image.open(testimg) as img:
    d2 = np.array(img)

assert (d2 == d).all()

testimg = root / "test_HOPS_99_img_i3KsJ.JPEG"
newimg = root / "HOPS_99_img_i3KsJ.JPEG"

with Image.open(newimg) as img:
    d = np.array(img)
with Image.open(testimg) as img:
    d2 = np.array(img)

assert (d2 == d).all()
