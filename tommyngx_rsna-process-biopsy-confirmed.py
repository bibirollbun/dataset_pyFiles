# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session






import pandas as pd
import os

def process_rsna(link, root):
    """
    Load dataframe từ file CSV và thêm cột link = patient_id + "_" + image_id
    
    Parameters:
        link (str): đường dẫn file CSV
        root (str): thư mục gốc (nếu cần để join path)
    
    Returns:
        pd.DataFrame: dataframe đã thêm cột link
    """
    # Load CSV
    file_path = os.path.join(root, link)
    df = pd.read_csv(link)
    df = df[df['biopsy'] == 1].copy()
    
    # Tạo cột link = patient_id + "_" + image_id
    df['name']  = df['patient_id'].astype(str) + "_" + df['image_id'].astype(str)
    df['link']  = df['patient_id'].astype(str) + "/" + df['image_id'].astype(str)
    df['link1'] =  "/train_images/"+ df['link'] + ".dcm"
    df['link2'] =  "/kaggle/working/RSNA/"+ df['patient_id'].astype(str)+ "/" + df['name'] + "_" + df['view'].astype(str)+ "_" + df['laterality'].astype(str) + ".png"
    return df

df = process_rsna(link="/kaggle/input/rsna-breast-cancer-detection/train.csv", 
                  root= "/kaggle/input/rsna-breast-cancer-detection")
df 


pip install -q "pydicom>=2.3" "pylibjpeg>=2.0" "pylibjpeg-libjpeg>=2.1" "pylibjpeg-openjpeg" "python-gdcm>=3.0.10"


import os
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----- utils giữ nguyên ý tưởng trước -----
def _to_uint8(imgf32, mono1=False):
    imgf32 = imgf32 - np.nanmin(imgf32)
    maxv = np.nanmax(imgf32)
    if maxv > 0:
        imgf32 = imgf32 / maxv * 255.0
    img = np.clip(imgf32, 0, 255).astype(np.uint8)
    if mono1:
        img = 255 - img
    return img

def _dicom_to_uint8(ds):
    arr = ds.pixel_array.astype(np.float32)  # cần pylibjpeg/gdcm cho ảnh nén
    try:  arr = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:  pass
    try:  arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:  pass
    mono1 = str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1"
    return _to_uint8(arr, mono1)

def _resize_keep_w(img_u8: np.ndarray, target_w=2048) -> Image.Image:
    h, w = img_u8.shape[:2]
    if w == target_w: 
        return Image.fromarray(img_u8)
    target_h = int(round(h * (target_w / float(w))))
    return Image.fromarray(img_u8).resize(
        (target_w, target_h), resample=Image.Resampling.LANCZOS
    )

# ----- worker chạy cho từng dòng -----
def _process_one(root, link1, link2, target_w=2048):
    link1 = str(link1).lstrip(os.sep)
    link2 = str(link2).lstrip(os.sep)
    in_path  = os.path.join(root, link1)
    out_path = str(Path(link2).with_suffix(".png"))  # link2 là đường dẫn tương đối/đích

    if not os.path.exists(in_path):
        return ("MISSING", in_path)

    try:
        ds  = pydicom.dcmread(in_path, force=True)
        img = _dicom_to_uint8(ds)
        pil = _resize_keep_w(img, target_w=target_w)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
        return ("OK", out_path)
    except Exception as e:
        return ("ERROR", f"{in_path} :: {e}")

def Process_RSNA_parallel(df, root="", max_workers=None, target_w=2048):
    """
    df: có cột link1 (.dcm trong 'root') và link2 (đường đích png)
    """
    # nơi đặt metadata.csv: thư mục cấp 1 của link2
    first_folder = Path(str(df.iloc[0]["link2"])).parts[0] if len(df) else ""
    meta_base = os.path.join(root, first_folder) if first_folder else root
    Path(meta_base).mkdir(parents=True, exist_ok=True)

    tasks = []
    results = {"OK":0, "MISSING":0, "ERROR":0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in df.itertuples(index=False):
            fut = ex.submit(_process_one, root, getattr(row,"link1"), getattr(row,"link2"), target_w)
            tasks.append(fut)

        for fut in tqdm(as_completed(tasks), total=len(tasks)):
            status, info = fut.result()
            results[status] += 1
            if status != "OK":
                print(status, "->", info)

    # lưu metadata sau khi chạy
    pd.DataFrame(df).to_csv(os.path.join(meta_base, "metadata.csv"), index=False)
    print("Summary:", results)
    return results           
#/kaggle/input/rsna-breast-cancer-detection/train_images
#/kaggle/input/rsna-breast-cancer-detection/train_images/10011/1031443799.dcm
Process_RSNA_parallel(df, root="/kaggle/input/rsna-breast-cancer-detection")


df


import pandas as pd

#df = pd.DataFrame({"a": [1,2], "b": [3,4]})
link = "/kaggle/working/kaggle/working/metadata.csv"

# make sure parent folder exists
import os
from pathlib import Path
#Path(link).parent.mkdir(parents=True, exist_ok=True)

df.to_csv(link, index=False)
print("Saved:", link)


import shutil
import os

%cd /kaggle/working/kaggle

folder_to_zip = "working"
output_zip_file = "RSNA_working.zip"

shutil.make_archive(output_zip_file.replace(".zip", ""), 'zip', folder_to_zip)







