# --- Ayarla: DICOM dosya yolu ---
DICOM_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/1.2.826.0.1.3680043.8.498.11145423894464257824946219093727029191.dcm"

# Kesit seçimi için z değişkeni
z = 5   # istediğin slice index'i buradan değiştir (0 = ilk slice)

import numpy as np
import pydicom
import sys

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def get_image_array(ds, z_index=0):
    """DICOM'dan piksel matrisini okur, rescale ve window uygular (varsa).
       Multi-frame ise z_index ile belirlenen slice alınır.
    """
    arr = ds.pixel_array.astype(np.float32)

    # Multi-frame (stack) ise
    if arr.ndim == 3 and arr.shape[0] > 1 and (arr.shape[-1] != 3 and arr.shape[-1] != 4):
        # güvenlik için index clamp
        z_index = max(0, min(z_index, arr.shape[0]-1))
        arr = arr[z_index]

    # RGB (SamplesPerPixel=3) ise griye çevir
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.float32)

    # Rescale Slope/Intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Windowing (mevcutsa)
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]

    wc, ww = to_float(wc), to_float(ww)
    if wc is not None and ww not in (None, 0):
        low, high = wc - ww/2.0, wc + ww/2.0
    else:
        low, high = np.percentile(arr, 1), np.percentile(arr, 99)

    arr = np.clip((arr - low) / (high - low + 1e-6), 0, 1)
    return arr


def main():
    try:
        ds = pydicom.dcmread(DICOM_PATH)
    except Exception as e:
        print("DICOM okunamadı. Sıkıştırılmış olabilir:", file=sys.stderr)
        print("- 'pylibjpeg' veya 'gdcm' kurmanız gerekebilir.", file=sys.stderr)
        raise

    img = get_image_array(ds, z_index=z)

    if _HAS_MPL:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(f"DICOM Kesit z={z}")
        plt.axis("off")
        plt.show()

    rows = getattr(ds, "Rows", "?")
    cols = getattr(ds, "Columns", "?")
    frames = getattr(ds, "NumberOfFrames", 1)
    print(f"\nBoyut: {rows} x {cols} | Frame sayısı: {frames}")
    print(f"Gösterilen kesit: {z}")
    if hasattr(ds, "Modality"):
        print(f"Modality: {ds.Modality}")


if __name__ == "__main__":
    main()


# --- Ayarla: NIfTI dosya yolu ---
NIFTI_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381_cowseg.nii"

# slice seçmek için
z = 160   # burayı değiştirerek farklı kesitlere bakabilirsin

import numpy as np
import nibabel as nib
import sys

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def main():
    # NIfTI oku
    nii = nib.load(NIFTI_PATH)
    data = nii.get_fdata()  # float64 array döner
    print("Hacim shape:", data.shape)

    if data.ndim == 3:
        z_index = max(0, min(z, data.shape[2]-1))
        slice2d = data[:, :, z_index]
    elif data.ndim == 4:
        # örn: fMRI gibi (X,Y,Z,T). İlk zaman noktası alınır
        z_index = max(0, min(z, data.shape[2]-1))
        slice2d = data[:, :, z_index, 0]
    else:
        print("Desteklenmeyen boyut:", data.shape, file=sys.stderr)
        return

 
    # (isteğe bağlı) Matplotlib göster
    if _HAS_MPL:
        plt.figure()
        plt.imshow(slice2d.T, cmap="gray", origin="lower")
        plt.title(f"NIfTI slice z={z_index}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

