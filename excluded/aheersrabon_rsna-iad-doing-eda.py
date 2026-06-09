%%capture output

!pip install dicom


!apt install ffmpeg


from pathlib import Path
import os
import random

from PIL import Image

from types import SimpleNamespace

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use('ggplot')

import pydicom as dicom
import pydicom


cfg = SimpleNamespace()
cfg.INPUT = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")


train = pl.read_csv(cfg.INPUT / "train.csv")
train_localizers = pl.read_csv(cfg.INPUT / "train_localizers.csv")


print(f"# of training data: {len(train)}")
print(f"# of train localizers: {len(train_localizers['SeriesInstanceUID'].unique())}")


absent, present = train.partition_by("Aneurysm Present")

fig, axes = plt.subplots(1,2, figsize=(8,4))
axes[0].hist(
    absent['PatientAge'],
    bins=30,
)
axes[0].set_title("Absent")
axes[1].hist(
    present['PatientAge'],
    bins=30,
)
axes[1].set_title("Present")
plt.show()


present.group_by("PatientSex").agg(count=pl.col('Aneurysm Present').count())

plt.bar(
    present.group_by("PatientSex").agg(count=pl.col('Aneurysm Present').count())['PatientSex'],
    present.group_by("PatientSex").agg(count=pl.col('Aneurysm Present').count())['count']
)
plt.title("Aneurysm Positive")
plt.show()


series_sop_df = train_localizers.group_by('SeriesInstanceUID').agg(
    SOPInstanceUIDs = pl.col('SOPInstanceUID'),
    SOPInstanceUIDCount = pl.col('SOPInstanceUID').count(),
    coordinates = pl.col('coordinates')
).sort('SOPInstanceUIDCount')

sop = series_sop_df.row(by_predicate=(
        pl.col('SeriesInstanceUID') == random.choice(series_sop_df['SeriesInstanceUID'])
))

fig, axes = plt.subplots(1, len(sop[1]), figsize=(8,8))

if len(sop[1]) > 1:
    for i, ax in enumerate(axes.ravel()):
        image = dicom.dcmread(cfg.INPUT / "series" / f"{sop[0]}" / f"{sop[1][i]}.dcm").pixel_array
        
        # there are some 3d data
        if(len(image.shape) > 2):
            image = image[1, :, :]
            
        ax.imshow(
            image,
            cmap=plt.cm.bone
        )
        ax.plot(
            eval(sop[3][i])['x'], eval(sop[3][i])['y'],
            marker='x',
            color='red'
        )
elif len(sop[1]) == 1:
    image = dicom.dcmread(cfg.INPUT / "series" / f"{sop[0]}" / f"{sop[1][0]}.dcm").pixel_array
    
    # there are some 3d data
    if(len(image.shape) > 2):
        image = image[1, :, :]
        
    axes.imshow(
        image,
        cmap=plt.cm.bone
    )
    axes.plot(
        eval(sop[3][0])['x'], eval(sop[3][0])['y'],
        marker='x',
        color='red'
    )


train_localizers = train_localizers.with_columns(
    pl.format(
        "/kaggle/input/rsna-intracranial-aneurysm-detection/series/{}/{}.dcm",
        pl.col("SeriesInstanceUID"),
        pl.col("SOPInstanceUID")
    ).alias("filename")
)


# params
paths = [p for p in train_localizers.get_column("filename").to_list() if p and os.path.exists(p)]
want = 5
out = "five_multiframe_sidebyside.gif"
fps = 8

def load_multiframe(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    # make sure frames dim exists
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # assume (F,H,W) if first dim >1 (multi-frame)
        if arr.shape[0] <= 4 and arr.shape[0] != arr.shape[1]:
            # sometimes (H,W,C) -> transpose to (H,W,C) handled later; treat as single frame
            arr = arr[np.newaxis, ...]
    # rescale
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    # photometric inversion for MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr
    return arr  # shape (F, H, W) ideally

def to_uint8(img2d, p1=1, p99=99):
    lo, hi = np.percentile(img2d, (p1, p99))
    if hi <= lo:
        lo, hi = img2d.min(), img2d.max()
    if hi == lo:
        return np.zeros_like(img2d, dtype=np.uint8)
    clipped = np.clip(img2d, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

def ensure_size(img, shape):
    # img: 2D uint8, shape: (H, W)
    if img.shape == shape:
        return img
    img_pil = Image.fromarray(img).resize((shape[1], shape[0]), Image.LANCZOS)
    return np.asarray(img_pil)

def make_rgb(img2d):
    return np.stack([img2d]*3, axis=-1)  # (H,W,3)

# find first 5 multi-frame files
multi = []
for p in paths:
    try:
        arr = pydicom.dcmread(p, force=True).pixel_array
        if getattr(arr, "ndim", 2) == 3 and arr.shape[0] > 1:
            multi.append(p)
    except Exception:
        continue
    if len(multi) >= want:
        break

if len(multi) < want:
    raise RuntimeError(f"Found only {len(multi)} multi-frame DICOM(s); need {want}.")

# load arrays for chosen files
arrs = [load_multiframe(p) for p in multi[:want]]
nframes = min(a.shape[0] for a in arrs)  # sync length by minimum

# choose reference shape from first file's single frame (H, W)
ref_shape = arrs[0].shape[1], arrs[0].shape[2]

# build first combined frame for fig setup
first_pieces = []
for a in arrs:
    img = to_uint8(a[0])
    img = ensure_size(img, ref_shape)
    first_pieces.append(make_rgb(img))
first_frame = np.concatenate(first_pieces, axis=1)

fig, ax = plt.subplots(figsize=(first_frame.shape[1]/100, first_frame.shape[0]/100), dpi=100)
ax.axis("off")
im = ax.imshow(first_frame, animated=True)

def gen():
    for i in range(nframes):
        pieces = []
        for a in arrs:
            img = a[i]
            # if (C,H,W) channel-first unlikely here; handle only (H,W)
            if img.ndim == 3 and img.shape[0] in (3,4):
                img = np.transpose(img, (1,2,0))  # to H,W,C
                # convert color -> grayscale by mean
                img = img.mean(axis=2)
            img8 = to_uint8(img)
            img8 = ensure_size(img8, ref_shape)
            pieces.append(make_rgb(img8))
        yield np.concatenate(pieces, axis=1)

def update(frame):
    im.set_array(frame)
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=gen(), interval=1000/fps, blit=True)
ani.save(out, writer="pillow", fps=fps)
plt.close(fig)
print("Saved:", out)
from IPython.display import HTML
HTML(ani.to_jshtml())


import os, math, ast
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

filenames = [f for f in train_localizers.get_column("filename").to_list() if f and os.path.exists(f)]
coords_raw = train_localizers.get_column("coordinates").to_list()
out_gif = "train_localizers_with_cross.gif"
fps = 8
cross_half = 6
cross_thickness = 2
sample_frames_for_window = 8

def parse_coord(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            import json
            return json.loads(raw)
        except Exception:
            return None

def read_pixel_array(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    # If multi-frame (F,H,W), pick middle frame (you can change to mean/max if desired)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    # rescale
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    # photometric inversion for MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr
    return arr

n = len(filenames)
indices = np.linspace(0, n-1, min(sample_frames_for_window, n), dtype=int)
samples = []
for i in indices:
    try:
        samples.append(read_pixel_array(filenames[i]).ravel())
    except Exception:
        continue
if not samples:
    raise RuntimeError("Failed to read sample frames for window estimation.")
all_sample_pixels = np.concatenate(samples)
p1, p99 = np.percentile(all_sample_pixels, (1, 99))
if p99 <= p1:
    p1, p99 = all_sample_pixels.min(), all_sample_pixels.max()
if p99 == p1:
    p99 = p1 + 1.0

def to_uint8_with_window(arr, lo=p1, hi=p99):
    arr = np.clip(arr, lo, hi)
    scaled = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return scaled

images = []
for i, path in enumerate(tqdm(filenames, desc="Preparing frames")):
    try:
        arr = read_pixel_array(path)
    except Exception:
        # fallback blank image
        arr = np.zeros_like(read_pixel_array(filenames[0]))
    arr8 = to_uint8_with_window(arr)            # 2D uint8
    # convert to RGB (H, W, 3)
    rgb = np.stack([arr8, arr8, arr8], axis=-1)
    # parse coord and draw cross by setting pixel blocks (fast)
    coord = parse_coord(coords_raw[i]) if i < len(coords_raw) else None
    if coord and "x" in coord and "y" in coord:
        try:
            x = int(round(float(coord["x"])))
            y = int(round(float(coord["y"])))
            H, W = arr8.shape
            # clamp center
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            # horizontal bar
            x0 = max(0, x - cross_half)
            x1 = min(W, x + cross_half + 1)
            y0 = max(0, y - cross_thickness // 2)
            y1 = min(H, y + (cross_thickness + 1)//2)
            rgb[y0:y1, x0:x1, 0] = 255  # R channel
            rgb[y0:y1, x0:x1, 1] = 0
            rgb[y0:y1, x0:x1, 2] = 0
            # vertical bar
            x0_v = max(0, x - cross_thickness // 2)
            x1_v = min(W, x + (cross_thickness + 1)//2)
            y0_v = max(0, y - cross_half)
            y1_v = min(H, y + cross_half + 1)
            rgb[y0_v:y1_v, x0_v:x1_v, 0] = 255
            rgb[y0_v:y1_v, x0_v:x1_v, 1] = 0
            rgb[y0_v:y1_v, x0_v:x1_v, 2] = 0
        except Exception:
            pass
    # convert to PIL Image and append
    images.append(Image.fromarray(rgb))

duration_ms = int(1000 / fps)
images[0].save(out_gif, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
print("Saved:", out_gif)


%%capture output

# requires ffmpeg installed and on PATH
!ffmpeg -y -i train_localizers_with_cross.gif -movflags +faststart -pix_fmt yuv420p -vcodec libx264 -crf 23 train_localizers_with_cross.mp4


from IPython.display import Video, display

# local file
display(Video("train_localizers_with_cross.mp4", embed=True, width=640, height=360))

