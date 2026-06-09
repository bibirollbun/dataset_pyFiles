import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def find_competition_path():
    candidates = [
        '/kaggle/input/rsna-2022-cervical-spine-fracture-detection',
        '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/data'
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for p in glob.glob('/kaggle/input/*'):
        if 'rsna' in os.path.basename(p).lower():
            return p


COMP_PATH = find_competition_path()
print('Використовується шлях:', COMP_PATH)


import os
import pandas as pd

csv_files = [f for f in os.listdir(COMP_PATH) if f.endswith('.csv')]
dfs = {}

print(f"Знайдено CSV: {len(csv_files)}")
for f in csv_files:
    fpath = os.path.join(COMP_PATH, f)
    try:
        df = pd.read_csv(fpath)
        dfs[f.replace('.csv', '')] = df
        print(f" {f} — зчитано ({df.shape[0]} рядків, {df.shape[1]} колонок)")
    except Exception as e:
        print(f"Помилка при зчитуванні {f}: {e}")

for name, df in dfs.items():
    print(f'\n--- {name} ---')
    display(df.head())
    print('Колонки:', df.columns.tolist())
    print('Відсутні значення:')
    print(df.isnull().sum().sort_values(ascending=False).head(10))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def find_label_columns(df):
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ['label', 'fracture', 'target', 'class']):
            candidates.append(c)
        elif df[c].nunique(dropna=True) <= 3 and pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
    return list(set(candidates))


label_summary = {}

for name, df in dfs.items():
    labels = find_label_columns(df)
    for lab in labels:
        counts = df[lab].value_counts(dropna=False)
        pct = (counts / len(df) * 100).round(2)
        res = pd.DataFrame({'Кількість': counts, 'Відсоток': pct})
        label_summary[(name, lab)] = res

        display(res)

        plt.figure(figsize=(6,4))
        sns.barplot(
            x=res.index.astype(str),
            y='Кількість',
            data=res.reset_index().rename(columns={'index': lab})
        )
        plt.title(f'Розподіл класів для {name}.{lab}')
        plt.xlabel('Клас')
        plt.ylabel('Кількість')
        plt.tight_layout()
        plt.savefig(f'/kaggle/working/balance_{name}_{lab}.png')
        plt.show()


import os
import glob
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut

train_path = os.path.join(COMP_PATH, "train_images")
test_path = os.path.join(COMP_PATH, "test_images")

subdirs = sorted([os.path.join(train_path, d) for d in os.listdir(train_path)])[:3]
train_dicom_files = []
for d in subdirs:
    train_dicom_files.extend(glob.glob(os.path.join(d, '*.dcm')))
train_dicom_files = train_dicom_files[:5] 

for f in train_dicom_files:
    print(" -", os.path.relpath(f, COMP_PATH))

def read_dicom_image(path, voi_lut=True):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    if voi_lut:
        try:
            img = apply_voi_lut(img, ds)
        except Exception:
            pass
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8), ds

if train_dicom_files:
    sample_path = train_dicom_files[0]
    img, ds = read_dicom_image(sample_path)
    print(f"\nПриклад DICOM-зображення:\n{sample_path}")
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Приклад DICOM')
    plt.show()
else:
    print('Не знайдено DICOM-файлів')


for (dfname, lab), tab in label_summary.items():
    out = f'/kaggle/working/class_balance_{dfname}_{lab}.csv'
    tab.to_csv(out)
    print('Збережено:', out)

