# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# === Paths (senden gelenler) ===
COW_MASK = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381_cowseg.nii"
VOL_NII  = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381.nii"
ONE_DICOM = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/1.2.826.0.1.3680043.8.498.10124807242473374136099471315028464450.dcm"

TRAIN_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
LOC_CSV   = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"

# === Imports ===
import os, json, numpy as np, nibabel as nib, SimpleITK as sitk, pydicom
import pandas as pd

WORKDIR = "/kaggle/working/check_case"
os.makedirs(WORKDIR, exist_ok=True)

def load_canonical_nii(path, dtype=None):
    """NIfTI'yi RAS kanonik yönüne çevir, isteğe bağlı dtype uygula."""
    img = nib.load(path)
    can = nib.as_closest_canonical(img)
    data = can.get_fdata()
    if dtype is not None:
        data = data.astype(dtype)
    return nib.Nifti1Image(data, can.affine)

def save_nii(img, out_path):
    nib.save(img, out_path)
    return out_path

def nii_info(img):
    """Şekil, voxel spacing (zooms), aralık (min/max)."""
    data = img.get_fdata()
    return {
        "shape": tuple(data.shape),
        "zooms": tuple(np.round(img.header.get_zooms(), 5)),
        "dtype": str(data.dtype),
        "minmax": (float(np.nanmin(data)), float(np.nanmax(data))),
    }

def resample_to_ref(moving_path, ref_path, out_path, is_label=False):
    """moving NIfTI'yi ref NIfTI'ye SimpleITK ile hizala (grid/spacinge)."""
    moving = sitk.ReadImage(moving_path)
    ref = sitk.ReadImage(ref_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    out = resampler.Execute(moving)
    sitk.WriteImage(out, out_path)
    return out_path

# 1) Hacmi ve CoW maskesini kanonikleştir
vol_can = load_canonical_nii(VOL_NII, np.float32)
msk_can = load_canonical_nii(COW_MASK, np.uint8)

vol_can_path = save_nii(vol_can, os.path.join(WORKDIR, "vol_canonical.nii.gz"))
msk_can_path = save_nii(msk_can, os.path.join(WORKDIR, "cow_canonical.nii.gz"))

print("Vol:", nii_info(vol_can))
print("Mask:", nii_info(msk_can))

# 2) Shape/affine kontrolü; farklıysa maske hacme yeniden örneklenir
same_shape = vol_can.shape == msk_can.shape
same_affine = np.allclose(vol_can.affine, msk_can.affine, atol=1e-3)

if not (same_shape and same_affine):
    print(">> UYARI: vol & mask grid/affine farklı. Maskeyi volume referansına yeniden örnekliyorum...")
    msk_res_path = os.path.join(WORKDIR, "cow_resampled_to_vol.nii.gz")
    resample_to_ref(msk_can_path, vol_can_path, msk_res_path, is_label=True)
    msk_can = nib.load(msk_res_path)
    print("Yeni Mask:", nii_info(msk_can))
else:
    print("Vol & Mask grid/affine uyumlu görünüyor.")

# 3) Maske etiketlerini listele (0..13 beklenir)
uniq = np.unique(msk_can.get_fdata()).astype(int)
print("Mask unique labels:", uniq[:20], "(toplam benzersiz:", len(uniq), ")")

# 4) Tek bir DICOM dosyasından temel meta (seri/örnek eşlemesi için)
ds = pydicom.dcmread(ONE_DICOM, stop_before_pixels=True)
print("SeriesInstanceUID:", str(ds.SeriesInstanceUID))
print("SOPInstanceUID   :", str(ds.SOPInstanceUID))
# Opsiyonel yararlı alanlar:
pixsp = getattr(ds, "PixelSpacing", None)
slth  = getattr(ds, "SliceThickness", None)
ipp   = getattr(ds, "ImagePositionPatient", None)
ior   = getattr(ds, "ImageOrientationPatient", None)
print("PixelSpacing:", pixsp, "| SliceThickness:", slth)
print("ImagePositionPatient:", ipp)
print("ImageOrientationPatient:", ior)

# 5) CSV'ler erişilebilir mi?
print("train.csv satır sayısı  :", sum(1 for _ in open(TRAIN_CSV)))
print("localizers.csv satır sayısı:", sum(1 for _ in open(LOC_CSV)))


import os, json, glob, numpy as np, nibabel as nib
from tqdm import tqdm

# === Giriş kökleri ===
SEGS_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations"

# === nnU-Net kökleri (Adım-1'de oluşturmadıysak) ===
os.environ['nnUNet_raw'] = os.getenv('nnUNet_raw', '/kaggle/working/nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.getenv('nnUNet_preprocessed', '/kaggle/working/nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.getenv('nnUNet_results', '/kaggle/working/nnUNet_results')

DATASET_ID = 100
NAME = f"Dataset{DATASET_ID}_RSNAIA_CoW"
RAW_BASE = f"{os.environ['nnUNet_raw']}/{NAME}"
paths = {k: os.path.join(RAW_BASE,k) for k in ['imagesTr','labelsTr']}
for p in paths.values(): os.makedirs(p, exist_ok=True)

# CoW sınıf isimleri (0 arka plan)
LABELS = {
    "0":"background",
    "1":"Other Posterior Circulation",
    "2":"Basilar Tip",
    "3":"Right Posterior Communicating Artery",
    "4":"Left Posterior Communicating Artery",
    "5":"Right Infraclinoid Internal Carotid Artery",
    "6":"Left Infraclinoid Internal Carotid Artery",
    "7":"Right Supraclinoid Internal Carotid Artery",
    "8":"Left Supraclinoid Internal Carotid Artery",
    "9":"Right Middle Cerebral Artery",
    "10":"Left Middle Cerebral Artery",
    "11":"Right Anterior Cerebral Artery",
    "12":"Left Anterior Cerebral Artery",
    "13":"Anterior Communicating Artery"
}

def as_canonical_nii(path, dtype=None):
    img = nib.load(path)
    can = nib.as_closest_canonical(img)
    arr = can.get_fdata()
    if dtype is not None:
        arr = arr.astype(dtype)
    return nib.Nifti1Image(arr, can.affine)

pairs = []
for fol in sorted(glob.glob(os.path.join(SEGS_ROOT, "*"))):
    vols = [v for v in glob.glob(os.path.join(fol, "*.nii")) if "_cowseg" not in v]
    if not vols:
        continue
    vol = vols[0]
    base = os.path.splitext(os.path.basename(vol))[0]
    msk = os.path.join(fol, base + "_cowseg.nii")
    if os.path.exists(msk):
        pairs.append((base, vol, msk))

print("Bulunan hacim+maske çifti:", len(pairs))

# Kopyala & kanonikleştir & doğru dtype
problem_cases = []
all_label_values = set()
for cid, v, m in tqdm(pairs):
    try:
        v_can = as_canonical_nii(v, np.float32)
        m_can = as_canonical_nii(m)  # etiket değerlerini koruyacağız
        # grid kontrolü: shape aynı değilse AFFINE/gridi korumak için etiketi yeniden kaydetmiyoruz;
        # burada genelde aynı zaten. Fark varsa yine de shape'leri kontrol edelim:
        if v_can.shape != m_can.shape:
            problem_cases.append((cid, "shape_mismatch", v_can.shape, m_can.shape))
            continue

        # maske uint8'e çevir (0..13 korunur)
        m_arr = m_can.get_fdata()
        all_label_values.update(np.unique(m_arr).astype(int).tolist())

        nib.save(nib.Nifti1Image(v_can.get_fdata().astype(np.float32), v_can.affine),
                 os.path.join(paths['imagesTr'], f"{cid}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(m_arr.astype(np.uint8), m_can.affine),
                 os.path.join(paths['labelsTr'], f"{cid}.nii.gz"))
    except Exception as e:
        problem_cases.append((cid, repr(e)))

print("Sorunlu vaka sayısı:", len(problem_cases))
if problem_cases[:5]: 
    print("Örnek sorunlar:", problem_cases[:5])

print("Tüm veride görülen etiket değerleri (toplu):", sorted(all_label_values))

# dataset.json
dataset_json = {
  "name": NAME,
  "description": "Circle of Willis multi-class segmentation to drive aneurysm classification",
  "reference": "RSNA Intracranial Aneurysm Detection",
  "licence": "academic",
  "release": "1.0",
  "modality": {"0": "CT"},  # veri MR içeriyorsa da sorun olmaz; bu alan bilgilendirici
  "labels": LABELS,
  "numTraining": len(os.listdir(paths['imagesTr'])),
  "numTest": 0,
  "training": [
      {
        "image": f"./imagesTr/{fn}",
        "label": f"./labelsTr/{fn.replace('_0000.nii.gz','.nii.gz')}"
      }
      for fn in sorted(os.listdir(paths['imagesTr'])) if fn.endswith("_0000.nii.gz")
  ],
  "test": []
}
with open(os.path.join(RAW_BASE, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print("nnU-Net ham veri hazır:", RAW_BASE)
print("imagesTr:", len(os.listdir(paths['imagesTr'])), "labelsTr:", len(os.listdir(paths['labelsTr'])))


import json, os

RAW_BASE = "/kaggle/working/nnUNet_raw/Dataset100_RSNAIA_CoW"
jpath = os.path.join(RAW_BASE, "dataset.json")

with open(jpath) as f:
    d = json.load(f)

old = d["labels"]                      # şu an {"0":"background","1":"..."} biçiminde
# yeni biçim: {"background":0, "Other Posterior Circulation":1, ...}
new_labels = {}
for k, v in old.items():
    try:
        i = int(k)                     # "0","1",...
        new_labels[v] = i              # "background":0, "Basilar Tip":2, ...
    except:
        pass

# zorunlu anahtarlar
d["labels"] = new_labels
d.setdefault("channel_names", {"0": "CT"})
d.setdefault("file_ending", ".nii.gz")

with open(jpath, "w") as f:
    json.dump(d, f, indent=2)

print("labels patched ->", d["labels"])
print("channel_names:", d["channel_names"], "| file_ending:", d["file_ending"])


import os, json
RAW_BASE = "/kaggle/working/nnUNet_raw/Dataset100_RSNAIA_CoW"
jpath = os.path.join(RAW_BASE, "dataset.json")

with open(jpath) as f:
    d = json.load(f)

# Zorunlu alanları ekle
d["channel_names"] = {"0": "CT/MRA"}       
d["file_ending"] = ".nii.gz"              # dosya uzantımız


with open(jpath, "w") as f:
    json.dump(d, f, indent=2)

print("Patched keys:", [k for k in ["labels","channel_names","numTraining","file_ending"] if k in d])
print("channel_names =", d["channel_names"], "| file_ending =", d["file_ending"])


!nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity -c 3d_fullres


%env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
%env OMP_NUM_THREADS=1
%env OPENBLAS_NUM_THREADS=1
%env MKL_NUM_THREADS=1
%env NUMEXPR_NUM_THREADS=1

!nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity -c 3d_fullres -np 2


# Preprocessed klasöründe kaç tane .npz var?
!ls /kaggle/working/nnUNet_preprocessed/Dataset100_RSNAIA_CoW | grep ".npz" | wc -l

# Raw dataset klasöründe kaç tane case var? (imagesTr)
!ls /kaggle/working/nnUNet_raw/Dataset100_RSNAIA_CoW/imagesTr | wc -l


!find /kaggle/working/nnUNet_preprocessed/Dataset100_RSNAIA_CoW/nnUNetPlans_3d_fullres -type f -name "*.npz" | wc -l


%%bash
set -e

# 1) Doğru symlink: /kaggle/working/nnUNet_raw -> /tmp/nnUNet_raw/nnUNet_raw
rm -f /kaggle/working/nnUNet_raw
ln -s /tmp/nnUNet_raw/nnUNet_raw /kaggle/working/nnUNet_raw

# 2) dataset.json gerçekten var mı? (yoksa dur)
test -f /kaggle/working/nnUNet_raw/Dataset100_RSNAIA_CoW/dataset.json \
  || { echo "ERROR: dataset.json bulunamadı. Symlink hedefini kontrol et."; ls -l /kaggle/working/nnUNet_raw; exit 1; }

# 3) Yer aç: 2D ve 3D lowres preprocessed klasörleri (lazım değilse)
rm -rf /kaggle/working/nnUNet_preprocessed/Dataset100_RSNAIA_CoW/nnUNetPlans_2d || true
rm -rf /kaggle/working/nnUNet_preprocessed/Dataset100_RSNAIA_CoW/nnUNetPlans_3d_lowres || true
rm -rf /kaggle/working/check_case || true

# 4) RAM dostu thread sınırları (bu hücre için geçerli)
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# 5) Kaldığı yerden 3D fullres preprocess (2 worker)
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity -c 3d_fullres -np 2


%%bash
set -e

# 0) Durum bilgisi (isteğe bağlı)
echo "Before:"; df -h /kaggle/working | tail -n1

# 1) nnUNet_preprocessed'i /tmp'ye taşı ve geri symlink koy
if [ -d /kaggle/working/nnUNet_preprocessed ]; then
  mv /kaggle/working/nnUNet_preprocessed /tmp/nnUNet_preprocessed
  ln -s /tmp/nnUNet_preprocessed /kaggle/working/nnUNet_preprocessed
fi

# 2) (Hatırlatma) raw zaten /tmp'de; symlink doğru mu?
ls -ld /kaggle/working/nnUNet_raw || true

# 3) nnU-Net ortam değişkenleri: sonuçlar da /tmp'ye gitsin
export nnUNet_raw=/kaggle/working/nnUNet_raw
export nnUNet_preprocessed=/tmp/nnUNet_preprocessed
export nnUNet_results=/tmp/nnUNet_results
mkdir -p "$nnUNet_results"

# 4) RAM dostu thread sınırları
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "After move:"; df -h /kaggle/working | tail -n1
echo "Ready to train. Results will go to: $nnUNet_results"


# Ortam değişkenleri (sen zaten ayarladın ama tam olsun)
%env nnUNet_raw=/kaggle/working/nnUNet_raw
%env nnUNet_preprocessed=/tmp/nnUNet_preprocessed
%env nnUNet_results=/kaggle/working/nnUNet_results

# SIRALAMA: [DATASET_ID] [CONFIG] [FOLD]
!nnUNetv2_train 100 3d_fullres 0


%%bash
set -e

echo "Before:"; df -h /kaggle/working | tail -n1

# 1) Çalışmada eski symlink/dizin varsa temizle
if [ -L /kaggle/working/nnUNet_preprocessed ]; then
  rm -f /kaggle/working/nnUNet_preprocessed
elif [ -d /kaggle/working/nnUNet_preprocessed ]; then
  echo "Uyarı: /kaggle/working/nnUNet_preprocessed zaten var, yedekliyorum."
  mv /kaggle/working/nnUNet_preprocessed /kaggle/working/nnUNet_preprocessed_backup_$(date +%H%M%S)
fi

# 2) /tmp'den working'e taşı
if [ -d /tmp/nnUNet_preprocessed ]; then
  mv /tmp/nnUNet_preprocessed /kaggle/working/nnUNet_preprocessed
else
  echo "HATA: /tmp/nnUNet_preprocessed bulunamadı!"; exit 1
fi

# 3) Hızlı doğrulamalar
echo "Preprocessed boyutu:"
du -sh /kaggle/working/nnUNet_preprocessed

echo "3d_fullres dosya sayısı (.b2nd):"
find /kaggle/working/nnUNet_preprocessed/Dataset100_RSNAIA_CoW/nnUNetPlans_3d_fullres -type f -name "*.b2nd" | wc -l

# 4) Sonuç klasörünü hazırla (çıkışlar burada kalsın)
mkdir -p /kaggle/working/nnUNet_results

echo "After:"; df -h /kaggle/working | tail -n1
echo "TAŞIMA TAMAM. Artık kernel reset/GPU açsan da preprocessed silinmez."


import os, torch
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"   # preprocessed bizde kalıcı
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"             # Output panelinde görünsün
os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

print("CUDA available:", torch.cuda.is_available())  # True olmalı


import pathlib, re

p = pathlib.Path("/usr/local/lib/python3.11/dist-packages/nnunetv2/run/run_training.py")
txt = p.read_text()
# Satırı string'e çevir: = 1  --> = "1"
fixed = re.sub(r"os\.environ\[\s*['\"]TORCHINDUCTOR_COMPILE_THREADS['\"]\s*\]\s*=\s*1",
               "os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = \"1\"", txt)
if fixed != txt:
    p.write_text(fixed)
    print("Patched run_training.py (TORCHINDUCTOR_COMPILE_THREADS now string).")
else:
    print("Patch not needed (already string or different version).")


!mkdir -p /kaggle/working/nnUNet_results /kaggle/working/nnUNet_raw
!ls -lah /kaggle/working


import os, sys, subprocess, shutil, inspect, re, time
from pathlib import Path

# ---------- 0) Hızlı teşhis ----------
print("== Teşhis ==")
try:
    import torch
    print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import error:", e)

try:
    import nnunetv2
    print("nnunetv2:", nnunetv2.__version__ if hasattr(nnunetv2, "__version__") else "ok")
    rt_path = Path(inspect.getfile(nnunetv2)).parent / "run" / "run_training.py"
    print("run_training.py:", rt_path)
except Exception as e:
    print("nnunetv2 import error:", e)

# ---------- 1) Patch (idempotent) ----------
try:
    txt = rt_path.read_text()
    fixed = re.sub(
        r"os\.environ\[\s*['\"]TORCHINDUCTOR_COMPILE_THREADS['\"]\s*\]\s*=\s*1",
        "os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = \"1\"",
        txt
    )
    if fixed != txt:
        rt_path.write_text(fixed)
        print("[PATCH] TORCHINDUCTOR_COMPILE_THREADS -> '1' (string) uygulandı")
    else:
        print("[PATCH] zaten doğru")
except Exception as e:
    print("[PATCH] hata:", e)

# ---------- 2) Yol ve dosyalar ----------
RAW = "/kaggle/working/nnUNet_raw"  # symlink olabilir
PRE = "/kaggle/working/nnUNet_preprocessed"
RES = "/kaggle/working/nnUNet_results"

print("\n== Yol kontrol ==")
print("PRE exists:", Path(PRE).exists())
print("RES exists:", Path(RES).exists())

split_file = Path(PRE) / "Dataset100_RSNAIA_CoW" / "splits_final.json"
print("splits_final.json:", split_file, "->", split_file.exists())

fold_dir = Path(RES) / "Dataset100_RSNAIA_CoW" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
fold_dir.mkdir(parents=True, exist_ok=True)
best = fold_dir/"checkpoint_best.pth"
latest = fold_dir/"checkpoint_latest.pth"

if (not latest.exists()) and best.exists():
    shutil.copy2(best, latest)
    print("[RESUME] checkpoint_latest.pth oluşturuldu (best'ten).")

print("latest.exists:", latest.exists(), "| best.exists:", best.exists())

# ---------- 3) Env ----------
os.environ.update({
    "nnUNet_raw": RAW,
    "nnUNet_preprocessed": PRE,
    "nnUNet_results": RES,
    "CUDA_VISIBLE_DEVICES": "0",   # tek GPU
    "PYTHONUNBUFFERED": "1",
    "TORCH_COMPILE_DISABLE": "1",  # compile kapalı (daha stabil)
    "nnUNet_compile": "0",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
})

# ---------- 4) Hızlı ön-test: help/versiyon, hata yakalamak için ----------
print("\n== Hızlı ön-test ==")
test_cmd = ["python","-c","import torch,nnunetv2;print('cuda',torch.cuda.is_available())"]
ret = subprocess.run(test_cmd, capture_output=True, text=True)
print("Pretest rc:", ret.returncode, "| out:", ret.stdout.strip(), "| err:", ret.stderr.strip())

# Eğer burada hata varsa, nnunetv2/torch ortamı sorunsuz çalışmıyor demektir.
# ---------- 5) Eğitim (filtreli, yalnızca önemli satırlar) ----------
cmd = ["python","-u","-m","nnunetv2.run.run_training","100","3d_fullres","0","--c"]
print("\n[RUN]", " ".join(cmd), "\n", flush=True)

want_prefixes = (
    "Using device:",
    "Epoch ",
    "Current learning rate:",
    "train_loss", "val_loss",
    "Pseudo dice [",
    "Yayy! New best EMA",
)

last_epoch = None
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for ln in p.stdout:
        s = ln.strip()
        if s.startswith("Using device:"):
            print(s); continue
        if s.startswith(want_prefixes):
            if s.startswith("Epoch "):
                try:
                    ep = int(s.split()[1])
                except: ep = None
                if ep is None or ep != last_epoch:
                    last_epoch = ep
                    print(s)
            else:
                print(s)
    rc = p.wait()
    print(f"\n[EXIT] return code: {rc}")
except KeyboardInterrupt:
    print("\n[STOP] kullanıcı durdurdu.")
    try: p.terminate()
    except: pass


!mkdir -p /kaggle/working/tmp_nifti

# DICOM -> NIfTI çevir
!dcm2niix -z y -o /kaggle/working/tmp_nifti /kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647


import os, shutil, time, subprocess, glob, pathlib

# --- Yollar ---
SERIES_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647"
NIFTI_DIR  = "/kaggle/working/tmp_nifti"
OUT_DIR    = "/kaggle/working/nnunet_predictions"

# nnU-Net ortam değişkenleri (emin olmak için tekrar ayarlıyoruz)
os.environ["nnUNet_raw"]          = "/kaggle/working/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnUNet_preprocessed"
os.environ["nnUNet_results"]      = "/kaggle/working/nnUNet_results"

# Klasörleri hazırla
os.makedirs(NIFTI_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("== DICOM -> NIfTI ==")
# DICOM -> NIfTI (sıkıştırılmış .nii.gz)
subprocess.run(
    ["dcm2niix", "-z", "y", "-o", NIFTI_DIR, SERIES_DIR],
    check=True
)

# Klasördeki NIfTI'leri bul
niis = sorted(glob.glob(os.path.join(NIFTI_DIR, "*.nii*")), key=os.path.getmtime)
if not niis:
    raise RuntimeError("NIfTI bulunamadı. dcm2niix çıktısını kontrol et.")
# En son oluşturulanı seç, diğerlerini temizle (karışıklık olmasın)
latest = niis[-1]
for f in niis[:-1]:
    try:
        os.remove(f)
    except:
        pass

# Tek NIfTI kaldığından emin ol
niis = glob.glob(os.path.join(NIFTI_DIR, "*.nii*"))
assert len(niis) == 1, f"Birden fazla NIfTI var: {niis}"
nii_path = niis[0]

# İsim sonunu _0000.nii.gz yap (tek kanal beklentisi)
base = os.path.basename(nii_path)
if not (base.endswith("_0000.nii") or base.endswith("_0000.nii.gz")):
    # .nii.gz mi .nii mi?
    if base.endswith(".nii.gz"):
        new_base = base[:-7] + "_0000.nii.gz"
    elif base.endswith(".nii"):
        new_base = base[:-4] + "_0000.nii"
    else:
        # olağan dışı uzantı
        stem = pathlib.Path(base).stem
        new_base = stem + "_0000.nii.gz"
    new_path = os.path.join(NIFTI_DIR, new_base)
    os.rename(nii_path, new_path)
    nii_path = new_path

print(f"Kullanılacak NIfTI: {nii_path}")

# Yan ürün JSON’ları (sidecar) varsa kalsın; nnUNet bunlara dokunmaz.
# Inference
print("\n== nnUNetv2_predict ==")
cmd = [
    "nnUNetv2_predict",
    "-i", NIFTI_DIR,
    "-o", OUT_DIR,
    "-d", "100",             # Dataset100_RSNAIA_CoW
    "-c", "3d_fullres",
    "-f", "0",
    "-tr", "nnUNetTrainer",
    "-chk", "checkpoint_best.pth",   # .pth yazma, sadece adı ver
    "--verbose"
]

t0 = time.time()
# Canlı çıktı için:
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
t1 = time.time()

print(f"\n✅ İnferans tamamlandı. Geçen süre: {t1 - t0:.2f} s, çıkış kodu: {rc}")

# Çıktıları listele
print("\n== Çıktı dosyaları ==")
for p in sorted(glob.glob(os.path.join(OUT_DIR, "*"))):
    print(" -", os.path.basename(p))


import nibabel as nib, numpy as np, os
p = "/kaggle/working/nnunet_predictions/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647_AX_COW_20230531221060_1.nii.gz"
lab = nib.load(p).get_fdata().astype(np.int16)
vals, cnt = np.unique(lab, return_counts=True)
print(dict(zip(vals.tolist(), cnt.tolist())))


# === 0) Kurulum & import
import os, glob, json, math
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 120

# === 1) Yollar
NIFTI_DIR = "/kaggle/working/tmp_nifti"
PRED_DIR  = "/kaggle/working/nnunet_predictions"
OUT_VIS   = "/kaggle/working/vis"
OUT_CROP  = "/kaggle/working/crops"
OUT_CSV   = "/kaggle/working/features.csv"

Path(OUT_VIS).mkdir(parents=True, exist_ok=True)
Path(OUT_CROP).mkdir(parents=True, exist_ok=True)

# === 2) Dosyaları bul
nifti_files = sorted(glob.glob(f"{NIFTI_DIR}/*.nii.gz"))
pred_files  = sorted([p for p in glob.glob(f"{PRED_DIR}/*.nii.gz")
                      if not Path(p).name.startswith(("dataset","plans","predict_from_raw"))])

assert len(nifti_files)>=1, f"NIfTI bulunamadı: {NIFTI_DIR}"
assert len(pred_files)>=1,  f"Tahmin NIfTI bulunamadı: {PRED_DIR}"

nii_path  = nifti_files[0]
pred_path = pred_files[0]
print("Görüntü:", Path(nii_path).name)
print("Tahmin :", Path(pred_path).name)

# === 3) Oku
img_nii  = nib.load(nii_path)
img      = img_nii.get_fdata().astype(np.float32)
aff      = img_nii.affine
vox_mm   = np.abs(np.linalg.det(aff[:3,:3]))  # voxel hacmi (mm^3)

pred_nii = nib.load(pred_path)
pred     = pred_nii.get_fdata().astype(np.int16)

# Şekiller uyuşmuyorsa (nadiren olur) yeniden örnekleme yapmayalım; uyarı verelim
if img.shape != pred.shape:
    print("UYARI: img ve pred şekilleri farklı:", img.shape, pred.shape)

# === 4) Sınıf sözlüğü (1..13)
id2name = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right PCom",
    4: "Left PCom",
    5: "Right Infraclinoid ICA",
    6: "Left Infraclinoid ICA",
    7: "Right Supraclinoid ICA",
    8: "Left Supraclinoid ICA",
    9: "Right MCA",
    10: "Left MCA",
    11: "Right ACA",
    12: "Left ACA",
    13: "ACom",
}
labels = sorted(id2name.keys())

# === 5) Basit normalizasyon (görselleştirme için)
p = np.percentile(img, (1, 99))
img_viz = np.clip((img - p[0]) / (p[1]-p[0] + 1e-6), 0, 1)

# === 6) MIP + overlay (3 eksen)
def save_mip_overlay(img01, lab, axis, out_png):
    # img01: [0,1] normalize
    mip_img = np.max(img01, axis=axis)
    mip_lab = np.max(lab,    axis=axis)  # sınıf idlerinin MIP'i (en büyük id)
    # Renk haritası: 0 şeffaf, >0 sabit renk (hızlıca)
    overlay = np.zeros((*mip_img.shape, 3), dtype=np.float32)
    overlay[...,0] = (mip_lab>0).astype(np.float32)  # kırmızı ton
    alpha = 0.35*(mip_lab>0)
    rgb = np.stack([mip_img]*3, axis=-1)
    out = (1-alpha[...,None])*rgb + alpha[...,None]*overlay

    plt.figure(figsize=(6,6))
    plt.imshow(out, cmap=None)
    plt.axis('off')
    plt.title(f"MIP axis={axis}")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

save_mip_overlay(img_viz, pred, axis=0, out_png=f"{OUT_VIS}/mip_ax0.png")
save_mip_overlay(img_viz, pred, axis=1, out_png=f"{OUT_VIS}/mip_ax1.png")
save_mip_overlay(img_viz, pred, axis=2, out_png=f"{OUT_VIS}/mip_ax2.png")
print("MIP görseller kaydedildi:", OUT_VIS)

# === 7) Sınıf bazlı istatistik + crop çıkarma
rows = []
for cls in labels:
    mask = (pred==cls)
    vox = int(mask.sum())
    if vox==0:
        rows.append({
            "label_id": cls,
            "label_name": id2name[cls],
            "voxels": 0, "mm3": 0.0,
            "bbox_zyx": None,
            "centroid_zyx": None,
            "crop_path": None
        })
        continue

    # bbox
    zyx = np.array(np.where(mask)).T
    zmin, ymin, xmin = zyx.min(axis=0)
    zmax, ymax, xmax = zyx.max(axis=0)

    # küçük bir margin ile crop
    m = 4
    z0, z1 = max(0, zmin-m), min(mask.shape[0], zmax+m+1)
    y0, y1 = max(0, ymin-m), min(mask.shape[1], ymax+m+1)
    x0, x1 = max(0, xmin-m), min(mask.shape[2], xmax+m+1)

    crop_img  = img[z0:z1, y0:y1, x0:x1]
    crop_mask = mask[z0:z1, y0:y1, x0:x1].astype(np.uint8)

    # crop NIfTI kaydet (img ve maskeyi ayrı)
    base = f"class{cls:02d}_{id2name[cls].replace(' ','_')}"
    img_nii_out  = os.path.join(OUT_CROP, base+"_img.nii.gz")
    mask_nii_out = os.path.join(OUT_CROP, base+"_mask.nii.gz")
    nib.save(nib.Nifti1Image(crop_img, aff),  img_nii_out)
    nib.save(nib.Nifti1Image(crop_mask, aff), mask_nii_out)

    # merkez (yoğunluk ağırlıksız)
    cz, cy, cx = zyx.mean(axis=0)

    rows.append({
        "label_id": cls,
        "label_name": id2name[cls],
        "voxels": int(vox),
        "mm3": float(vox * vox_mm),
        "bbox_zyx": [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
        "centroid_zyx": [float(cz), float(cy), float(cx)],
        "crop_path": [img_nii_out, mask_nii_out],
    })

df = pd.DataFrame(rows)
df = df.sort_values("label_id")
df.to_csv(OUT_CSV, index=False)
print("Özellik tablosu:", OUT_CSV)
df


# --- Yollar ---
IMG_PATH = "/kaggle/working/tmp_nifti/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647_AX_COW_20230531221060_1_0000.nii.gz"

import os, os.path as op, numpy as np, nibabel as nib, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# SEG yolunu IMG dosya adına göre türet ( _0000.nii.gz -> .nii.gz )
case_id = op.basename(IMG_PATH).replace("_0000.nii.gz", "")
SEG_PATH = f"/kaggle/working/nnunet_predictions/{case_id}.nii.gz"

print("IMG_PATH:", IMG_PATH)
print("SEG_PATH:", SEG_PATH)
assert op.exists(IMG_PATH), "IMG dosyası bulunamadı!"
assert op.exists(SEG_PATH), "SEG dosyası bulunamadı!"

# --- Yükle ---
img = nib.load(IMG_PATH);  img_data = img.get_fdata().astype(np.float32)
seg = nib.load(SEG_PATH);  seg_data = seg.get_fdata().astype(np.int16)

# Görüntüyü (sadece görselleştirme için) 1–99 persentil aralığında normalize et
p1, p99 = np.percentile(img_data, [1, 99])
img_disp = np.clip((img_data - p1) / max(p99 - p1, 1e-5), 0, 1)

# --- Renkler: 0 arka plan, 1..13 damar sınıfları ---
# (tab20’den seçilmiş 13 renk)
tab = plt.get_cmap("tab20").colors
lut = np.zeros((14, 4))   # RGBA
lut[1:14, :3] = np.array(tab[:13])
lut[:, 3] = [0.0] + [1.0]*13       # alfa = 1, overlay'de ayrıca şeffaflık vereceğiz
cmap = ListedColormap(lut)

# Orta dilimler
z, y, x = img_disp.shape
slices = (z//2, y//2, x//2)

def show_overlay(img3d, seg3d, slices, alpha=0.35, figsize=(15,5)):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    # axial (z)
    ax[0].imshow(img3d[slices[0], :, :], cmap="gray")
    ax[0].imshow(seg3d[slices[0], :, :], cmap=cmap, alpha=alpha, interpolation="nearest")
    ax[0].set_title(f"Axial z={slices[0]}")
    ax[0].axis("off")
    # coronal (y)
    ax[1].imshow(img3d[:, slices[1], :].T, cmap="gray", origin="lower")
    ax[1].imshow(seg3d[:, slices[1], :].T, cmap=cmap, alpha=alpha, origin="lower", interpolation="nearest")
    ax[1].set_title(f"Coronal y={slices[1]}")
    ax[1].axis("off")
    # sagittal (x)
    ax[2].imshow(img3d[:, :, slices[2]].T, cmap="gray", origin="lower")
    ax[2].imshow(seg3d[:, :, slices[2]].T, cmap=cmap, alpha=alpha, origin="lower", interpolation="nearest")
    ax[2].set_title(f"Sagittal x={slices[2]}")
    ax[2].axis("off")
    fig.tight_layout()
    return fig

fig = show_overlay(img_disp, seg_data, slices, alpha=0.35)
os.makedirs("/kaggle/working/vis", exist_ok=True)
out_png = "/kaggle/working/vis/overlay_preview.png"
fig.savefig(out_png, dpi=150)
plt.show()
print("Kaydedildi:", out_png)

# --- Sınıf başına voxel sayısı ---
labels = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right PCom",
    4: "Left PCom",
    5: "Right Infraclinoid ICA",
    6: "Left Infraclinoid ICA",
    7: "Right Supraclinoid ICA",
    8: "Left Supraclinoid ICA",
    9: "Right MCA",
    10: "Left MCA",
    11: "Right ACA",
    12: "Left ACA",
    13: "ACom",
}
counts = {k: int((seg_data == k).sum()) for k in range(0, 14)}
print("\nVoxeller:")
print(f"  0: background -> {counts[0]:,}")
for k in range(1, 14):
    print(f"{k:>3}: {labels[k]:35s} -> {counts[k]:,}")


# dcm2niix yüklü değilse kur (DICOM -> NIfTI çevirisi için)
!apt-get update -qq
!apt-get install -y -qq dcm2niix

# ======= Parametreler (sadece STUDY_UID'yi değiştirmen yeterli) =======
STUDY_UID = "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"  # GT olan hasta
DATA_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection"

# nnU-Net eğitim çıktıların
import os, shutil
os.environ["nnUNet_results"] = "/kaggle/working/nnUNet_results"

# Çalışma klasörleri
TMP_NIFTI = "/kaggle/working/tmp_eval_nifti"     # DICOM->NIfTI geçici
PRED_DIR  = "/kaggle/working/nnunet_eval_pred"   # nnU-Net tahminleri
for p in [TMP_NIFTI, PRED_DIR]:
    os.makedirs(p, exist_ok=True)


from pathlib import Path
import shutil, os

DATA_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection"
STUDY_UID = "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"
TMP_NIFTI = "/kaggle/working/tmp_eval_nifti"
os.makedirs(TMP_NIFTI, exist_ok=True)

SERIES_DIR = f"{DATA_ROOT}/series/{STUDY_UID}"
SEG_DIR    = f"{DATA_ROOT}/segmentations/{STUDY_UID}"

assert Path(SERIES_DIR).exists(), f"Series yok: {SERIES_DIR}"
assert Path(SEG_DIR).exists(),    f"Seg yok: {SEG_DIR}"

gt_files = sorted(Path(SEG_DIR).glob("*.nii*"))
assert gt_files, f"GT .nii bulunamadı: {SEG_DIR}"
GT_PATH = str(gt_files[0])
print("Seri klasörü:", SERIES_DIR)
print("GT maske   :", GT_PATH)

# geçici klasörü temizle
for p in Path(TMP_NIFTI).glob("*"):
    try: p.unlink()
    except: shutil.rmtree(p, ignore_errors=True)

# DICOM -> NIfTI
!dcm2niix -z y -o "{TMP_NIFTI}" "{SERIES_DIR}"

# nnU-Net tek kanal isimlendirmesi: *_0000.nii.gz
nii_list = sorted(Path(TMP_NIFTI).glob("*.nii.gz"))
assert nii_list, "dcm2niix çıktı üretmedi."
img0 = nii_list[0]
if not img0.name.endswith("_0000.nii.gz"):
    fixed = img0.with_name(img0.stem + "_0000.nii.gz")
    shutil.copy2(img0, fixed)
    IMG_PATH = str(fixed)
else:
    IMG_PATH = str(img0)
print("Girdi NIfTI :", IMG_PATH)


import os, time, subprocess, torch
from pathlib import Path

# --- 1) Yol tanımları (senin düzenin) ---
RAW = "/kaggle/working/nnUNet_raw"                 # symlink olabilir (varsa sorun değil)
PRE = "/kaggle/working/nnUNet_preprocessed"        # mevcut (18 GB)
RES = "/kaggle/working/nnUNet_results"             # eğitim çıktıları burada

MODEL_DIR = Path(RES) / "Dataset100_RSNAIA_CoW" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
CKPT = MODEL_DIR / "checkpoint_best.pth"

IN_DIR  = Path("/kaggle/working/eval_in")          # 0000.nii.gz burada
OUT_DIR = Path("/kaggle/working/eval_pred")        # tahminler buraya

# --- 2) Kontroller ---
assert Path(PRE).exists(), f"Bulunamadı: {PRE}"
assert Path(RES).exists(), f"Bulunamadı: {RES}"
assert MODEL_DIR.exists(), f"Model klasörü yok: {MODEL_DIR}"
assert CKPT.exists(), f"Checkpoint yok: {CKPT}"
assert IN_DIR.exists() and any(IN_DIR.glob("*.nii.gz")), f"Girdi NIfTI bulunamadı: {IN_DIR}"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 3) Ortam değişkenlerini KUR ---
os.environ["nnUNet_raw"] = RAW
os.environ["nnUNet_preprocessed"] = PRE
os.environ["nnUNet_results"] = RES
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["nnUNet_compile"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# --- 4) Cihaz seçimi (cuda varsa cuda, yoksa cpu) ---
device_flag = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device_flag}")

# --- 5) Tahmin komutu ---
cmd = [
    "nnUNetv2_predict",
    "-i", str(IN_DIR),
    "-o", str(OUT_DIR),
    # Dataset kimliği: adıyla vermek daha güvenli
    "-d", "Dataset100_RSNAIA_CoW",
    "-c", "3d_fullres",
    "-f", "0",
    "-tr", "nnUNetTrainer",
    "-chk", "checkpoint_best.pth",   # cwd=MODEL_DIR olduğu için sadece adı yeterli
    "-device", device_flag,
    "--verbose"
]

print("[RUN]", " ".join(cmd))
t0 = time.time()
rc = subprocess.run(cmd, cwd=str(MODEL_DIR)).returncode
dt = time.time() - t0
print(f"\n✅ İnferans bitti. rc={rc}, süre={dt:.1f}s")
print("Çıktılar:", [p.name for p in OUT_DIR.glob("*")])


# -- Hücre 1: Yükle & hizala (SimpleITK) --
import os
from pathlib import Path
import SimpleITK as sitk

# YOLLAR — bunları değiştirebilirsin
GT_PATH   = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381_cowseg.nii"
PRED_PATH = "/kaggle/working/eval_pred/case.nii.gz"

# ÇIKTI: pred'i GT uzayına getirilmiş hizalanmış label
ALIGNED_PRED_PATH = "/kaggle/working/eval_pred/case_aligned_to_gt.nii.gz"

# Oku
gt_img   = sitk.ReadImage(GT_PATH)
pred_img = sitk.ReadImage(PRED_PATH)

print("GT size / spacing :", gt_img.GetSize(), "/", gt_img.GetSpacing())
print("PR size / spacing :", pred_img.GetSize(), "/", pred_img.GetSpacing())

# Eğer boyut/spacing/affine farklıysa pred'i GT uzayına resample et
needs_resample = (gt_img.GetSize()!=pred_img.GetSize()) or (gt_img.GetSpacing()!=pred_img.GetSpacing()) or (gt_img.GetOrigin()!=pred_img.GetOrigin()) or (gt_img.GetDirection()!=pred_img.GetDirection())

if needs_resample:
    print("→ Boyut/uzay farklı: prediction GT uzayına yeniden örneklenecek (nearest).")
    # Nearest neighbor (etiket verisi!)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(gt_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt16)
    pred_aligned = resampler.Execute(pred_img)
    sitk.WriteImage(pred_aligned, ALIGNED_PRED_PATH)
    print("Kaydedildi:", ALIGNED_PRED_PATH)
    pred_img = pred_aligned
else:
    print("→ Uzaylar uyumlu. Hizalama gerekmiyor.")
    # Uyumluysa yine de tekilleştirelim
    ALIGNED_PRED_PATH = PRED_PATH


# -- Hücre 2: Dice metrikleri --
import numpy as np
import SimpleITK as sitk
import pandas as pd

gt = sitk.GetArrayFromImage(sitk.ReadImage(GT_PATH)).astype(np.int16)
pr = sitk.GetArrayFromImage(sitk.ReadImage(ALIGNED_PRED_PATH)).astype(np.int16)

# Dice hesaplayıcı
def dice_coef(y_true, y_pred, cls):
    y1 = (y_true==cls)
    y2 = (y_pred==cls)
    inter = (y1 & y2).sum()
    denom = y1.sum() + y2.sum()
    return (2.0*inter/denom) if denom>0 else np.nan

# Hangi etiketleri değerlendirelim?
# Senin kurulumuna göre 0: background, 1..13: damarsal segmentler
classes = list(range(1,14))

rows = []
for c in classes:
    d = dice_coef(gt, pr, c)
    vox_gt = int((gt==c).sum())
    vox_pr = int((pr==c).sum())
    rows.append({"class": c, "dice": d, "gt_vox": vox_gt, "pred_vox": vox_pr})

df = pd.DataFrame(rows).sort_values("class")
print(df.to_string(index=False))

# Ortalama (geçerli sınıflar üzerinden)
mean_dice = np.nanmean(df["dice"].values)
print(f"\nMean Dice (1..13): {mean_dice:.4f}")


# -- Hücre 3: Görselleştirme --
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

gt_img   = sitk.GetArrayFromImage(sitk.ReadImage(GT_PATH))
pr_img   = sitk.GetArrayFromImage(sitk.ReadImage(ALIGNED_PRED_PATH))

# Aynı klasörde dcm2niix ile üretilen intensity görüntüsünü de göstermek istersen:
# (Eğer varsa; yoksa sadece maskeleri gösterir)
maybe_intensity = list(Path("/kaggle/working/tmp_eval_nifti").glob("*_0000.nii.gz"))
img_arr = None
if maybe_intensity:
    try:
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(maybe_intensity[0]))).astype(np.float32)
        # normalize
        p1, p99 = np.percentile(img_arr, [1,99])
        img_arr = np.clip((img_arr - p1)/(p99-p1 + 1e-6), 0, 1)
    except:
        img_arr = None

Z = gt_img.shape[0]
slices = np.linspace(Z*0.2, Z*0.8, 6).astype(int)

plt.figure(figsize=(12,8))
for i, z in enumerate(slices, 1):
    plt.subplot(2,3,i)
    if img_arr is not None and z < img_arr.shape[0]:
        base = img_arr[z]
        plt.imshow(base, cmap='gray')
        # tahmini kontur
        plt.contour(pr_img[z] > 0, levels=[0.5], linewidths=0.6)
        # GT kontur
        plt.contour(gt_img[z] > 0, levels=[0.5], linewidths=0.6, linestyles='--')
        plt.title(f"z={z} | pred(—), gt(--)")
    else:
        # sadece maskeler
        overlay = np.zeros((*gt_img.shape[1:], 3), dtype=np.float32)
        overlay[..., 0] = (gt_img[z] > 0) * 1.0      # GT kırmızı
        overlay[..., 1] = (pr_img[z] > 0) * 1.0      # Pred yeşil
        plt.imshow(overlay)
        plt.title(f"z={z} | R=GT, G=Pred")
    plt.axis('off')

plt.tight_layout()
plt.show()


# ========= RSNA CoW: Tek-seri uçtan-uca pipeline =========
# Girdi: SeriesInstanceUID  (anevrizmalı olduğunu söylediğin örnek)
SERIES_ID = "1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317"

# --------- Sabitler / Yollar (eğittiğin nnU-Net'e göre) ---------
DATASET_NAME = "Dataset100_RSNAIA_CoW"
CFG          = "3d_fullres"
FOLD         = "0"
TRAINER      = "nnUNetTrainer"
CHKPT_NAME   = "checkpoint_best.pth"

SERIES_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
NNUNET_RES  = "/kaggle/working/nnUNet_results"          # eğitim çıktıların burada
NNUNET_PRE  = "/kaggle/working/nnUNet_preprocessed"     # preprocessed burada
NNUNET_RAW  = "/kaggle/working/nnUNet_raw"              # boş olabilir ama env ister

# CNN ağı (opsiyonel): burada bir .pt bulursan aday crop'larda var/yok olasılığı üretirim.
CNN_WEIGHTS = "/kaggle/working/aneurysm_cnn.pt"          # yoksa heuristik ile devam

# Çalışma klasörü
from pathlib import Path
BASE = Path(f"/kaggle/working/pipeline_out/{SERIES_ID}")
NIFTI_DIR = BASE/"nifti";  PRED_DIR = BASE/"pred";  VIZ_DIR = BASE/"viz";  CROP_DIR = BASE/"crops"
for d in [BASE, NIFTI_DIR, PRED_DIR, VIZ_DIR, CROP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------- Ortam ve bağımlılıklar ---------
import os, sys, json, time, shutil, math, subprocess, warnings
import numpy as np, pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

os.environ["nnUNet_results"]      = NNUNET_RES
os.environ["nnUNet_preprocessed"] = NNUNET_PRE
os.environ["nnUNet_raw"]          = NNUNET_RAW

# dcm2niix yoksa kur
if shutil.which("dcm2niix") is None:
    !apt-get -y update >/dev/null
    !apt-get -y install dcm2niix >/dev/null

# nnUNetv2_predict yoksa pip (genelde kurulu)
if shutil.which("nnUNetv2_predict") is None:
    try:
        import nnunetv2  # noqa
    except Exception:
        !pip -q install nnunetv2 >/dev/null

# --------- Yardımcılar ---------
LABELS = {
 1:"Other Posterior Circulation",
 2:"Basilar Tip",
 3:"Right Posterior Communicating Artery",
 4:"Left Posterior Communicating Artery",
 5:"Right Infraclinoid Internal Carotid Artery",
 6:"Left Infraclinoid Internal Carotid Artery",
 7:"Right Supraclinoid Internal Carotid Artery",
 8:"Left Supraclinoid Internal Carotid Artery",
 9:"Right Middle Cerebral Artery",
10:"Left Middle Cerebral Artery",
11:"Right Anterior Cerebral Artery",
12:"Left Anterior Cerebral Artery",
13:"Anterior Communicating Artery",
}

ANEURYSMY_LOCATIONS = {2,3,4,5,6,7,8,9,10,11,12,13}  # klinikte daha sık görülenler

def norm_clip(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    m, s = x.mean(), x.std() + 1e-6
    return (x - m) / s

def ensure_nii(series_id:str) -> Path:
    """DICOM → NIfTI; çıktı *_0000.nii.gz garantilenir."""
    out_dir = NIFTI_DIR
    out_dir.mkdir(exist_ok=True, parents=True)
    # zaten varsa alma
    cand = sorted(out_dir.glob("*_0000.nii.gz"))
    if cand:
        return cand[-1]
    src = Path(SERIES_ROOT)/series_id
    assert src.exists(), f"DICOM klasörü yok: {src}"
    cmd = ["dcm2niix", "-z", "y", "-o", str(out_dir), str(src)]
    print("[DICOM→NIfTI]", " ".join(cmd))
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    if rc != 0:
        raise RuntimeError("dcm2niix hata")
    nii = sorted(out_dir.glob("*.nii.gz"), key=lambda p:p.stat().st_mtime)[-1]
    if not nii.name.endswith("_0000.nii.gz"):
        newp = nii.with_name(nii.name.replace(".nii.gz","_0000.nii.gz"))
        nii.rename(newp); nii = newp
    return nii

def run_nnunet(nii_path:Path) -> Path:
    """nnU-Net tahmini; çıktı PRED_DIR/case.nii.gz"""
    out_dir = PRED_DIR; out_dir.mkdir(exist_ok=True)
    seg_path = out_dir/"case.nii.gz"
    if seg_path.exists():
        return seg_path
    # nnU-Net klasör inputu
    tmp_in = out_dir/"in"; 
    if tmp_in.exists(): shutil.rmtree(tmp_in)
    tmp_in.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nii_path, tmp_in/"case_0000.nii.gz")

    device = "cuda" if (shutil.which("nvidia-smi") or os.environ.get("CUDA_VISIBLE_DEVICES")) else "cpu"
    print("Device:", device)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(tmp_in),
        "-o", str(out_dir),
        "-d", DATASET_NAME,
        "-c", CFG,
        "-f", FOLD,
        "-tr", TRAINER,
        "-chk", CHKPT_NAME,
        "-device", device,
        "--verbose"
    ]
    print("[nnUNet]", " ".join(cmd))
    t0=time.time()
    rc = subprocess.run(cmd).returncode
    print(f"[nnUNet] rc={rc}, süre={time.time()-t0:.1f}s")
    if rc!=0: raise RuntimeError("nnUNet tahmin hatası")
    assert seg_path.exists(), "Tahmin çıktısı yok"
    return seg_path

def connected_components_by_class(seg):
    """Her sınıf için bağlı bileşen listesi: (label, vox_count, bbox, centroid)"""
    out=[]
    for lab in range(1,14):
        mask = (seg==lab)
        if mask.sum()==0: 
            continue
        lab_img, n = ndi.label(mask)
        for cc in range(1, n+1):
            comp = (lab_img==cc)
            v = int(comp.sum())
            z,y,x = np.where(comp)
            z0,z1 = int(z.min()), int(z.max())+1
            y0,y1 = int(y.min()), int(y.max())+1
            x0,x1 = int(x.min()), int(x.max())+1
            cz,cy,cx = int(z.mean()), int(y.mean()), int(x.mean())
            out.append({
                "label": lab,
                "voxels": v,
                "bbox": (z0,y0,x0,z1,y1,x1),
                "centroid": (cz,cy,cx),
                "dims": (z1-z0, y1-y0, x1-x0)
            })
    return out

def blob_score(voxels, dims):
    """Saccular 'şişkinlik' sezgisi: hacim & izotropiye yakınlık."""
    dz,dy,dx = dims
    # izotropi ~ min/max oranı
    sphericity = min(dz,dy,dx) / (max(dz,dy,dx)+1e-6)
    # boyut önceliği (çok küçük gürültüyü ve çok büyük trunk'ı azalt)
    # 80..8000 aralığını tercih
    v = voxels
    size_pref = np.exp(-((np.log1p(v) - np.log(800))**2) / (2*(np.log(6)**2)))
    return float(0.6*sphericity + 0.4*size_pref)

def small_3d_cnn(in_ch=1, nclass=2):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv3d(in_ch,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
        nn.Conv3d(16,32,3,padding=1),    nn.ReLU(), nn.MaxPool3d(2),
        nn.Conv3d(32,64,3,padding=1),    nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        nn.Flatten(), nn.Linear(64, nclass)
    )

def crop_around(img, center, size=(48,96,96)):
    Z,Y,X = img.shape
    dz,dy,dx = size; cz,cy,cx = center
    z0=max(0,cz-dz//2); z1=min(Z,z0+dz)
    y0=max(0,cy-dy//2); y1=min(Y,y0+dy)
    x0=max(0,cx-dx//2); x1=min(X,x0+dx)
    crop = img[z0:z1,y0:y1,x0:x1]
    # pad gerekirse
    pad = [(0,0),(0,0),(0,0)]
    if crop.shape!=(dz,dy,dx):
        pad = [(0, dz-crop.shape[0]), (0, dy-crop.shape[1]), (0, dx-crop.shape[2])]
        crop = np.pad(crop, ((0,pad[0][1]),(0,pad[1][1]),(0,pad[2][1])), mode="edge")
    return crop

# --------- 1) DICOM → NIfTI ---------
nii_path = ensure_nii(SERIES_ID)
print("NIfTI:", nii_path)

# --------- 2) nnU-Net tahmini ---------
seg_path = run_nnunet(nii_path)
print("Seg:", seg_path)

# --------- 3) Aday çıkarımı & (opsiyonel) CNN skoru ---------
img = nib.load(nii_path).get_fdata().astype(np.float32)
seg = nib.load(seg_path).get_fdata().astype(np.int16)

# Normalle
img_n = norm_clip(img)

# Sınıf başına bileşenleri topla
comps = connected_components_by_class(seg)
df = pd.DataFrame(comps)
if df.empty:
    raise RuntimeError("Hiç segment bulunamadı.")

# Heuristik skor hesapla
df["blob_score"] = df.apply(lambda r: blob_score(r["voxels"], r["dims"]), axis=1)
# Klinik açıdan daha anlamlı sınıflara küçük bonus
df["loc_bonus"] = df["label"].apply(lambda l: 0.15 if l in ANEURYSMY_LOCATIONS else 0.0)
df["heuristic_score"] = df["blob_score"] + df["loc_bonus"]

# Adayları heuristik ile sırala (en çok 20)
cands = df.sort_values("heuristic_score", ascending=False).head(20).reset_index(drop=True)

# (Opsiyonel) CNN varsa aday crop'ları puanla
cnn_used = False
cnn_probs = None
if Path(CNN_WEIGHTS).exists():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = small_3d_cnn().to(device)
        net.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))
        net.eval()
        crops=[]
        for r in cands.itertuples():
            cz,cy,cx = r.centroid
            crop = crop_around(img_n, (int(cz),int(cy),int(cx)))
            crops.append(crop[None,None,...])           # (1,1,D,H,W)
        X = torch.from_numpy(np.concatenate(crops,0)).float().to(device)
        with torch.no_grad():
            logits = net(X); prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        cands["cnn_prob"] = prob
        cnn_probs = prob
        cnn_used = True
    except Exception as e:
        print("[CNN] kullanılamadı:", e)

# Nihai skor: (CNN varsa) 0.7*cnn + 0.3*heuristic, yoksa heuristic
if cnn_used:
    cands["final_score"] = 0.7*cands["cnn_prob"] + 0.3*cands["heuristic_score"]
else:
    cands["final_score"] = cands["heuristic_score"]

# En iyi aday
best = cands.iloc[0]
pred_has_aneurysm = bool(best["final_score"] >= (0.6 if cnn_used else 0.75))
pred_label_id = int(best["label"])
pred_label_name = LABELS.get(pred_label_id, f"Class {pred_label_id}")

# --------- 4) Görseller & Rapor ---------
# 3 dilim (axial, coronal, sagittal) üstüne en iyi adayın konturunu çizelim
cz,cy,cx = map(int, best["centroid"])
mask_best = (seg==pred_label_id).astype(np.uint8)
# sadece en yakın küçük çevre
lab,n = ndi.label(mask_best)
lbl_at_centroid = lab[max(cz,0), max(cy,0), max(cx,0)]
mask_best = (lab==lbl_at_centroid)

def overlay_slice(ax, base, mask, title):
    ax.imshow(base, cmap="gray")
    ax.contour(mask.astype(float), levels=[0.5], colors="magenta", linewidths=1.0)
    ax.set_title(title); ax.axis("off")

fig = plt.figure(figsize=(14,4.5))
ax1 = fig.add_subplot(1,3,1)
overlay_slice(ax1, img_n[cz,:,:], mask_best[cz,:,:], f"Axial z={cz}")
ax2 = fig.add_subplot(1,3,2)
overlay_slice(ax2, img_n[:,cy,:], mask_best[:,cy,:], f"Coronal y={cy}")
ax3 = fig.add_subplot(1,3,3)
overlay_slice(ax3, img_n[:,:,cx], mask_best[:,:,cx], f"Sagittal x={cx}")
viz_path = VIZ_DIR/"overlay.png"; fig.tight_layout(); fig.savefig(viz_path, dpi=200); plt.close(fig)

# MIP'ler
for axis, name in [(0,"ax0"),(1,"ax1"),(2,"ax2")]:
    mip = img_n.max(axis=axis)
    mask_mip = mask_best.max(axis=axis)
    plt.figure(figsize=(6,5))
    plt.imshow(mip, cmap="gray"); plt.contour(mask_mip.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    plt.title(f"MIP axis={axis}"); plt.axis("off")
    plt.savefig(VIZ_DIR/f"mip_{name}.png", dpi=180); plt.close()

# Özet CSV/JSON
summary = {
    "series_id": SERIES_ID,
    "nii": str(nii_path),
    "seg": str(seg_path),
    "pred_has_aneurysm": pred_has_aneurysm,
    "pred_vessel_label_id": pred_label_id,
    "pred_vessel_label_name": pred_label_name,
    "final_score": float(best["final_score"]),
    "cnn_used": cnn_used,
}
(pd.DataFrame(comps)
   .assign(label_name=lambda d: d["label"].map(LABELS))
   .to_csv(BASE/"components.csv", index=False))
with open(BASE/"summary.json","w") as f: json.dump(summary, f, indent=2)

print("\n==== SONUÇ ====")
print("Aneurizma var mı?:", "EVET" if pred_has_aneurysm else "HAYIR")
print(f"Olası damar: {pred_label_id} - {pred_label_name}")
print(f"Skor: {summary['final_score']:.3f} | CNN kullanıldı mı?: {cnn_used}")
print("Görseller:", viz_path, "| MIP'ler:", list((VIZ_DIR).glob("mip_*.png")))
print("Ayrıntılar:", BASE/"summary.json", "| Bileşenler:", BASE/"components.csv")


# DICOM -> NIfTI -> nnU-Net inference (Python API) — MODEL_DIR düzeltildi

import os, subprocess, time, shutil
from pathlib import Path
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# --- Girdiler ---
SERIES_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"
MODEL_DIR  = "/kaggle/working/nnUNet_results/Dataset100_RSNAIA_CoW/nnUNetTrainer__nnUNetPlans__3d_fullres"  # <- fold_0 DEĞİL!
FOLDS      = (0,)
CKPT_NAME  = "checkpoint_best.pth"

TMP_NIFTI = Path("/kaggle/working/tmp_eval_nifti"); TMP_NIFTI.mkdir(parents=True, exist_ok=True)
EVAL_IN   = Path("/kaggle/working/eval_in");        EVAL_IN.mkdir(parents=True, exist_ok=True)
OUT_DIR   = Path("/kaggle/working/eval_pred");      OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --- 1) DICOM -> NIfTI ---
print("\n== DICOM -> NIfTI ==")
subprocess.run(["dcm2niix","-z","y","-o",str(TMP_NIFTI),SERIES_DIR], check=True)
nii_list = sorted(TMP_NIFTI.glob("*.nii.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
assert nii_list, "dcm2niix çıktı üretmedi."
src = nii_list[0]
dst = EVAL_IN / (src.stem + "_0000.nii.gz")
shutil.copy2(src, dst)
print("NIfTI:", src)
print("Inference girişi:", dst)

# --- 2) nnU-Net inference (Python API) ---
print("\n== nnU-Net inference (Python API) ==")
# Kök klasörde dataset.json / plans.json olmalı:
assert (Path(MODEL_DIR)/"dataset.json").exists(), "dataset.json kök klasörde bulunamadı!"
assert (Path(MODEL_DIR)/"plans.json").exists(), "plans.json kök klasörde bulunamadı!"
# fold_0 içinde checkpoint olmalı:
assert (Path(MODEL_DIR)/"fold_0"/CKPT_NAME).exists(), f"{CKPT_NAME} bulunamadı!"

pred = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,                 # TTA açık
    perform_everything_on_device=True,
    device=DEVICE,
    verbose=True
)

pred.initialize_from_trained_model_folder(
    model_training_output_dir=MODEL_DIR,
    use_folds=FOLDS,
    checkpoint_name=CKPT_NAME
)

cases = [[str(dst)]]
t0 = time.time()
pred.predict_from_files(
    list_of_lists_or_source_folder=cases,
    output_folder_or_list_of_truncated_output_files=str(OUT_DIR),
    save_probabilities=False,
    num_processes_segmentation_export=1
)
dt = time.time() - t0
print(f"\n✅ Tahmin tamamlandı. Süre: {dt:.1f} s")
print("Çıktılar:", [p.name for p in sorted(OUT_DIR.iterdir())])


# === Yan yana GT vs. Prediction (tek hücre) ===
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# ---------- KULLANICI AYARLARI ----------
# nnU-Net tahmin çıktısı (genelde /kaggle/working/eval_pred/case.nii.gz)
pred_path = "/kaggle/working/eval_pred/case.nii.gz"

# GT maske (cowseg) - bu YOLU kendi vakana göre değiştir
gt_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381_cowseg.nii"

# Görselde gösterilecek dilim (z-ekseni). None => otomatik (maskenin en dolu olduğu dilim)
z = None

# İsteğe bağlı: gri-tonlu arkaplan hacim (otomatik arama)
# /kaggle/working/eval_in içine dcm2niix -> _0000.nii.gz oluşturuyoruz
img_candidates = sorted(Path("/kaggle/working/eval_in").glob("*.nii*"), key=lambda p: p.stat().st_mtime, reverse=True)
img_path = str(img_candidates[0]) if img_candidates else None  # bulunamazsa None kalır

# ---------- YARDIMCILAR ----------
def load_nifti(path, dtype=None):
    ni = nib.load(path)
    arr = ni.get_fdata()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr, ni.affine

def ensure_channel0(img_path):
    """nnU-Net girişi gibi _0000 son ekli tek-kanal adı üretmek için değil,
    burada sadece dosya mevcut mu diye kontrol için kullanıyoruz."""
    return img_path

def resample_labels_to_shape(lbl, target_shape):
    """ Sadece şekle uydurmak için en basit yaklaşım: nearest-neighbor zoom.
        Affine farklılıklarını dikkate almaz. (Genelde gerekmez; dcm2niix -> nnUNet çıktısı aynı uzama dönüyor.)
    """
    if tuple(lbl.shape) == tuple(target_shape):
        return lbl
    scale = np.array(target_shape) / np.array(lbl.shape)
    return zoom(lbl, zoom=scale, order=0)

def dice_score(gt, pr, cls):
    gt_bin = (gt == cls)
    pr_bin = (pr == cls)
    inter = (gt_bin & pr_bin).sum()
    denom = gt_bin.sum() + pr_bin.sum()
    if denom == 0:
        return np.nan  # o sınıf yoksa NaN
    return 2.0 * inter / denom

# ---------- YÜKLE ----------
pred, aff_pred = load_nifti(pred_path, dtype=np.int16)
gt,   aff_gt   = load_nifti(gt_path,   dtype=np.int16)

# Gerekirse pred'i GT şekline uydur (etiket olduğu için NN)
if pred.shape != gt.shape:
    pred_rs = resample_labels_to_shape(pred, gt.shape)
else:
    pred_rs = pred

# Arkaplan görüntü (varsa)
img = None
if img_path and Path(img_path).exists():
    # img NIfTI tek kanallı hacim
    img, aff_img = load_nifti(img_path, dtype=np.float32)
    # img'i de GT şekline uydur (görüntü olduğu için lineer; ama scipy zoom sadece order=0/1/3... -> 1 kullanalım)
    if img.shape != gt.shape:
        scale = np.array(gt.shape) / np.array(img.shape)
        img = zoom(img, zoom=scale, order=1)

# ---------- HANGİ DİLİM? ----------
if z is None:
    # GT veya pred üzerinde en yoğun sınıf (arka plan hariç) piksellerinin en çok olduğu dilimi bul
    nonbg_gt = (gt > 0).sum(axis=(1,2))
    nonbg_pr = (pred_rs > 0).sum(axis=(1,2))
    z = int(np.argmax(nonbg_gt + nonbg_pr))

z = int(np.clip(z, 0, gt.shape[0]-1))

# ---------- DICE HESAPLARI ----------
classes = sorted(list(set(np.unique(gt)) | set(np.unique(pred_rs))))
if 0 in classes:
    classes.remove(0)  # arkaplan hariç
per_class = {c: dice_score(gt, pred_rs, c) for c in classes}
valid_dice = [v for v in per_class.values() if not np.isnan(v)]
macro_dice = float(np.mean(valid_dice)) if valid_dice else np.nan

print("Hacim şekli (Z,Y,X):", gt.shape)
print("Seçilen z:", z)
print("Sınıf sayısı (arka plan hariç):", len(classes))
print("Macro Dice (GT vs Pred): {:.4f}".format(macro_dice))
print("Per-class Dice:")
for c in classes:
    print("  {:2d}: {}".format(c, "NaN" if np.isnan(per_class[c]) else f"{per_class[c]:.4f}"))

# ---------- GÖRSELLEŞTİRME ----------
# Basit bir etiket renk haritası (0..13 arası için yeterli)
import matplotlib
cmap = matplotlib.cm.get_cmap('tab20', 14)  # 14 farklı renk

def show_overlay(ax, base, mask, title):
    if base is not None:
        v = np.percentile(base, [1, 99])
        ax.imshow(base, cmap='gray', vmin=v[0], vmax=v[1])
        # maskeyi yarı saydam bindir
        m = mask.copy()
        m[m==0] = -1  # arkaplan görünmesin
        im = ax.imshow(m, cmap=cmap, alpha=0.45, vmin=-1, vmax=13)
    else:
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=13)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    return im

fig, axs = plt.subplots(1, 3 if img is not None else 2, figsize=(15, 5))
if img is not None:
    show_overlay(axs[0], img[z], np.zeros_like(gt[z]), "Görüntü (z={})".format(z))
    show_overlay(axs[1], img[z], gt[z], "GT maske")
    im = show_overlay(axs[2], img[z], pred_rs[z], "Tahmin maske")
else:
    show_overlay(axs[0], None, gt[z], "GT maske (z={})".format(z))
    im = show_overlay(axs[1], None, pred_rs[z], "Tahmin maske")

plt.tight_layout()
plt.show()


# ===== Ortam & Path Setup =====
import os
from pathlib import Path
import inspect, re
import nnunetv2, torch

# ---- nnU-Net pathleri
RAW = "/kaggle/working/nnUNet_raw"
PRE = "/kaggle/working/nnUNet_preprocessed"
RES = "/kaggle/working/nnUNet_results"
for p in (RAW, PRE, RES):
    Path(p).mkdir(parents=True, exist_ok=True)

os.environ.update({
    "nnUNet_raw": RAW,
    "nnUNet_preprocessed": PRE,
    "nnUNet_results": RES,
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONUNBUFFERED": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "nnUNet_compile": "0",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
})

# ---- Patch: TORCHINDUCTOR_COMPILE_THREADS string fix
rt = Path(inspect.getfile(nnunetv2)).parent / "run" / "run_training.py"
txt = rt.read_text()
fixed = re.sub(
    r"os\.environ\[\s*['\"]TORCHINDUCTOR_COMPILE_THREADS['\"]\s*\]\s*=\s*1",
    "os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = \"1\"",
    txt
)
if fixed != txt:
    rt.write_text(fixed)
    print("[PATCH] TORCHINDUCTOR_COMPILE_THREADS -> '1'")

print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | dev={(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')}")
print("nnunetv2:", getattr(nnunetv2, "__version__", "ok"))


# ===== Fold-1 training + checkpoint temizleyici (best & latest hariç sil) =====
import os, time, threading, subprocess
from pathlib import Path

# ---- Sabitler (gerekirse değiştir)
DATASET_ID   = "100"
CONFIG       = "3d_fullres"
FOLD         = "1"                      # fold_1
TRAINER      = "nnUNetTrainer"
RES          = Path(os.environ["nnUNet_results"])
MODEL_DIR    = RES / "Dataset100_RSNAIA_CoW" / f"{TRAINER}__nnUNetPlans__{CONFIG}"
FOLD_DIR     = MODEL_DIR / f"fold_{FOLD}"

FOLD_DIR.mkdir(parents=True, exist_ok=True)
print(f"Model klasörü: {FOLD_DIR}")

# ---- Eğitim komutu
cmd = [
    "python","-u","-m","nnunetv2.run.run_training",
    DATASET_ID, CONFIG, FOLD
]
print("[RUN]", " ".join(cmd))

# ---- Checkpoint temizleyici: best + latest harici .pth'leri sil
def cleaner(stop_event):
    kept = {"checkpoint_best.pth", "checkpoint_latest.pth"}
    size_cache = {}
    while not stop_event.is_set():
        try:
            if FOLD_DIR.exists():
                # Aşırı agresif olmasın: yalnızca 2 dakikadan eski ve boyutu değişmeyen dosyaları sil
                for p in FOLD_DIR.glob("checkpoint_*.pth"):
                    if p.name in kept: 
                        continue
                    age = time.time() - p.stat().st_mtime
                    size_prev = size_cache.get(p, None)
                    size_now  = p.stat().st_size
                    size_cache[p] = size_now
                    if age > 120 and size_prev == size_now:
                        try:
                            p.unlink()
                        except Exception:
                            pass
        except Exception:
            pass
        stop_event.wait(30)  # 30 sn'de bir kontrol

# ---- Eğitim sürecini başlat + eşzamanlı temizleyici
stop_evt = threading.Event()
t = threading.Thread(target=cleaner, args=(stop_evt,), daemon=True)
t.start()

want_prefixes = ("Epoch ", "Current learning rate:", "train_loss", "val_loss", "Yayy!", "Using device:")
last_epoch = None
t0 = time.time()

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in proc.stdout:
        s = line.strip()
        if s.startswith("Epoch "):
            try:
                ep = int(s.split()[1])
            except:
                ep = None
            if ep is None or ep != last_epoch:
                last_epoch = ep
                print(s)
        elif s.startswith(want_prefixes):
            print(s)
except KeyboardInterrupt:
    print("\n[STOP] kullanıcı durdurdu, süreç sonlandırılıyor…")
    try: proc.terminate()
    except: pass

rc = proc.wait()
stop_evt.set()
t.join(timeout=2)

print(f"\n[EXIT] rc={rc} | elapsed={(time.time()-t0)/60:.1f} min")
print("Kalan checkpoint'ler:", [p.name for p in sorted(FOLD_DIR.glob('checkpoint_*.pth'))])




