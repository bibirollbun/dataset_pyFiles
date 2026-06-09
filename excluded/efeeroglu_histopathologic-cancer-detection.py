"""
===============================================================================
[STEP 1] ENVIRONMENT SETUP & DATA EXTRACTION
-------------------------------------------------------------------------------
- Gerekli dosya/klasÃ¶rlerin varlÄ±k kontrolÃ¼
- train_labels.csv yÃ¼kleme ve temel Ã¶zet
- Eksik deÄŸer ve duplicate kontrolÃ¼
- Ä°lk satÄ±rlar + label daÄŸÄ±lÄ±mÄ± (hÄ±zlÄ± gÃ¶rÃ¼nÃ¼m)
===============================================================================
"""

from pathlib import Path
import pandas as pd

try:
    PRIMARY_COLOR, HEADER_STYLE, TITLE_STYLE
except NameError:
    PRIMARY_COLOR = '#3371ab'
    HEADER_STYLE = [{
        'selector': 'thead th',
        'props': [
            ('background-color', PRIMARY_COLOR),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center')
        ]
    }]
    TITLE_STYLE = f"color:{PRIMARY_COLOR}; font-family:Segoe UI; margin-top:20px;"

def display_section_title(title: str):
    from IPython.display import HTML, display
    display(HTML(f"<h2 style='{TITLE_STYLE}'>{title}</h2>"))

def display_dataframe(df: pd.DataFrame, title: str = None):
    from IPython.display import display
    if title is not None:
        display_section_title(title)
    display(df.style.set_table_styles(HEADER_STYLE))


from pathlib import Path
BASE_DIR = Path("/kaggle/input/histopathologic-cancer-detection")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
LABELS_CSV = BASE_DIR / "train_labels.csv"
SAMPLE_SUB_CSV = BASE_DIR / "sample_submission.csv"

REQUIRED = {
    "train_dir": TRAIN_DIR,
    "test_dir": TEST_DIR,
    "labels_csv": LABELS_CSV,
    "sample_sub_csv": SAMPLE_SUB_CSV,
}

missing = [name for name, path in REQUIRED.items() if not Path(path).exists()]
display_section_title("ğŸ“‚ Dosya & Dizin VarlÄ±k KontrolÃ¼")
for name, path in REQUIRED.items():
    print(f"{name:15s} -> {path} | {'OK' if Path(path).exists() else 'MISSING'}")

if missing:
    raise FileNotFoundError(
        f"Zorunlu yol(lar) bulunamadÄ±: {', '.join(missing)}.\n"
        f"LÃ¼tfen veri seti yapÄ±sÄ±nÄ± doÄŸrulayÄ±n. BASE_DIR: {BASE_DIR}"
    )


display_section_title("ğŸ§¾ Etiket DosyasÄ± YÃ¼kleme & Temel Ã–zet")
labels = pd.read_csv(LABELS_CSV)

required_cols = {"id", "label"}
if not required_cols.issubset(labels.columns):
    raise ValueError(
        f"{LABELS_CSV.name} kolonlarÄ± eksik. "
        f"Gerekli: {required_cols}, mevcut: {set(labels.columns)}"
    )

summary_df = pd.DataFrame({
    "metric": [
        "satÄ±r_sayÄ±sÄ±",
        "sÃ¼tunlar",
        "toplam_eksik_deÄŸer",
        "benzersiz_id_sayÄ±sÄ±",
        "duplicate_id_sayÄ±sÄ±",
        "label_sÄ±nÄ±f_sayÄ±sÄ±"
    ],
    "value": [
        len(labels),
        ", ".join(map(str, labels.columns)),
        int(labels.isnull().sum().sum()),
        labels["id"].nunique(),
        int(labels["id"].duplicated().sum()),
        labels["label"].nunique()
    ]
})

display_dataframe(summary_df, title="train_labels.csv â€” HÄ±zlÄ± Ã–zet")


na_by_col = labels.isnull().sum()
na_df = na_by_col.reset_index()
na_df.columns = ["column", "missing_count"]
display_dataframe(na_df, title="Eksik DeÄŸerler (SÃ¼tun BazÄ±nda)")


display_dataframe(labels.head(10), title="Ä°lk 10 SatÄ±r (Ã–rnek)")


label_counts = labels["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
label_counts["ratio"] = (label_counts["count"] / len(labels)).round(4)
display_dataframe(label_counts, title="Label DaÄŸÄ±lÄ±mÄ± (Tablo)")


"""
===============================================================================
[STEP 2] IMAGE FILE STRUCTURE VALIDATION
-------------------------------------------------------------------------------
AmaÃ§:
  - train klasÃ¶rÃ¼ndeki .tif dosya sayÄ±sÄ±nÄ± kontrol etmek
  - CSV'deki id'lerle birebir eÅŸleÅŸme saÄŸlanÄ±yor mu?
  - GÃ¶rsel boyutlarÄ±nÄ±n 96x96 olup olmadÄ±ÄŸÄ±nÄ± doÄŸrulamak
===============================================================================
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

display_section_title("ğŸ§© GÃ¶rsel Dosya SayÄ±sÄ± & EÅŸleÅŸme KontrolÃ¼")


train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".tif")]
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".tif")]

n_train_files = len(train_files)
n_test_files = len(test_files)
n_csv_ids = len(labels)

summary_counts = pd.DataFrame({
    "dataset": ["train (klasÃ¶r)", "test (klasÃ¶r)", "train_labels.csv"],
    "count": [n_train_files, n_test_files, n_csv_ids]
})
display_dataframe(summary_counts, title="Veri Seti Eleman SayÄ±larÄ±")


train_file_ids = set([f.replace(".tif", "") for f in train_files])
csv_ids = set(labels["id"].astype(str))

missing_in_folder = csv_ids - train_file_ids
missing_in_csv = train_file_ids - csv_ids

print(f"ğŸ“¦ train klasÃ¶rÃ¼ndeki toplam gÃ¶rsel sayÄ±sÄ±: {n_train_files:,}")
print(f"ğŸ§¾ train_labels.csv kayÄ±t sayÄ±sÄ±: {n_csv_ids:,}")
print(f"ğŸ§® EÅŸleÅŸmeyen kayÄ±t (CSV'de var, klasÃ¶rde yok): {len(missing_in_folder)}")
print(f"ğŸ§® EÅŸleÅŸmeyen kayÄ±t (klasÃ¶rde var, CSV'de yok): {len(missing_in_csv)}")

if len(missing_in_folder) == 0 and len(missing_in_csv) == 0:
    print("âœ… TÃ¼m gÃ¶rseller CSV ile birebir eÅŸleÅŸiyor.")
else:
    print("âš ï¸� EÅŸleÅŸme sorunu tespit edildi â€” detaylÄ± kontrol Ã¶nerilir.")


display_section_title("ğŸ–¼ï¸� GÃ¶rsel Boyut DoÄŸrulamasÄ± (Ã¶rnek 10000 dosya)")

sample_files = np.random.choice(train_files, size=min(10000, len(train_files)), replace=False)
size_records = []

for fname in tqdm(sample_files, desc="Boyut kontrolÃ¼"):
    try:
        with Image.open(TRAIN_DIR / fname) as img:
            size_records.append(img.size)
    except Exception as e:
        size_records.append(("ERROR", "ERROR"))


size_df = pd.DataFrame(size_records, columns=["width", "height"])
size_summary = size_df.value_counts().reset_index(name="count")

display_dataframe(size_summary, title="GÃ¶rsel Boyut DaÄŸÄ±lÄ±mÄ±")

if len(size_summary) == 1 and tuple(size_summary.iloc[0][["width", "height"]]) == (96, 96):
    print("âœ… TÃ¼m gÃ¶rseller 96Ã—96 boyutunda (Ã¶rneklem bazÄ±nda doÄŸrulandÄ±).")
else:
    print("âš ï¸� Boyut farklÄ±lÄ±klarÄ± veya bozuk dosyalar mevcut olabilir.")


"""
===============================================================================
[STEP 3] SAMPLE IMAGE VISUALIZATION
-------------------------------------------------------------------------------
AmaÃ§:
  - Pozitif ve negatif sÄ±nÄ±flardan dengeli Ã¶rnekler seÃ§mek
  - 10 Ã¶rnek (5 pozitif, 5 negatif) gÃ¶rseli 2Ã—5 grid olarak gÃ¶stermek
  - GÃ¶rsellerin ID ve label bilgilerini gÃ¶rsel altÄ±nda belirtmek
===============================================================================
"""

import random
import matplotlib.pyplot as plt
from PIL import Image

positive_ids = labels[labels["label"] == 1]["id"].sample(5, random_state=42).tolist()
negative_ids = labels[labels["label"] == 0]["id"].sample(5, random_state=42).tolist()
sample_ids = positive_ids + negative_ids
random.shuffle(sample_ids)


display_section_title("ğŸ–¼ï¸� GÃ¶rsel Ã–rneklerin Ä°ncelenmesi (Pozitif & Negatif)")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for i, (ax, img_id) in enumerate(zip(axes, sample_ids)):
    img_path = TRAIN_DIR / f"{img_id}.tif"

    try:
        with Image.open(img_path) as img:
            img = np.array(img)
            ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, "Error\nLoading", ha="center", va="center", fontsize=9, color="red")

    # Etiket bilgileri
    label_value = labels.loc[labels["id"] == img_id, "label"].values[0]
    label_text = "Cancer" if label_value == 1 else "Normal"
    label_color = PRIMARY_COLOR if label_value == 1 else "#808080"

    # GÃ¶rsel baÅŸlÄ±ÄŸÄ± ve ID alt yazÄ±sÄ±
    ax.set_title(label_text, fontsize=11, weight="bold", color=label_color, pad=6)
    ax.text(
        0.5, -0.12,
        f"ID: {img_id[:12]}...",
        ha="center",
        va="center",
        fontsize=8,
        color="#444444",
        transform=ax.transAxes
    )

    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "Pozitif & Negatif Ã–rnek GÃ¶rseller",
    fontsize=15,
    color=PRIMARY_COLOR,
    weight="bold",
    y=1.03
)
plt.subplots_adjust(wspace=0.25, hspace=0.45)
plt.show()


"""
===============================================================================
[STEP 4] IMAGE QUALITY & STABILITY ANALYSIS
-------------------------------------------------------------------------------
AmaÃ§:
  - Bozuk / aÃ§Ä±lamayan .tif dosyalarÄ±nÄ± tespit etmek
  - GÃ¶rsellerin ortalama parlaklÄ±k, varyans, kontrast Ã¶lÃ§Ã¼mlerini yapmak
  - Focus (blur) skorlarÄ±nÄ± sÄ±nÄ±flara gÃ¶re karÅŸÄ±laÅŸtÄ±rmak
  - Histogram Ã¶rnekleriyle kalite farklarÄ±nÄ± gÃ¶zlemlemek
===============================================================================
"""

import cv2
from tqdm.auto import tqdm
import seaborn as sns

def analyze_image_quality(img_path):
    """
    Tek bir gÃ¶rselin parlaklÄ±k, varyans ve blur skorunu dÃ¶ndÃ¼rÃ¼r.
    """
    try:
        img = np.array(Image.open(img_path).convert("L"))  # grayscale
        brightness = np.mean(img)
        variance = np.var(img)
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        return brightness, variance, blur
    except Exception:
        return np.nan, np.nan, np.nan


display_section_title("ğŸ§  GÃ¶rsel Kalite & Ä°stikrar Analizi")
SAMPLE_SIZE = min(1500, len(labels))
sample_df = labels.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

brightness_list, variance_list, blur_list = [], [], []

for img_id in tqdm(sample_df["id"], desc="Kalite Ã¶lÃ§Ã¼mÃ¼"):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    b, v, bl = analyze_image_quality(img_path)
    brightness_list.append(b)
    variance_list.append(v)
    blur_list.append(bl)

sample_df["brightness"] = brightness_list
sample_df["variance"] = variance_list
sample_df["blur"] = blur_list

valid_samples = sample_df.dropna()
n_broken = SAMPLE_SIZE - len(valid_samples)

print(f"ğŸ“� Ã–rneklenen gÃ¶rsel sayÄ±sÄ±: {SAMPLE_SIZE}")
print(f"â�Œ Bozuk veya aÃ§Ä±lamayan gÃ¶rsel sayÄ±sÄ±: {n_broken}")
print(f"âœ… BaÅŸarÄ±yla analiz edilen: {len(valid_samples)}")

display_dataframe(valid_samples.head(10), title="Kalite Ã–lÃ§Ã¼mÃ¼ â€” Ä°lk 10 SatÄ±r")


group_summary = (
    valid_samples.groupby("label")[["brightness", "variance", "blur"]]
    .agg(["mean", "std"])
    .round(2)
)
display_dataframe(group_summary, title="SÄ±nÄ±f BazÄ±nda ParlaklÄ±k / Kontrast / Blur OrtalamalarÄ±")


def plot_metric_distribution(df, metric, color):
    plt.figure(figsize=(6, 3.5))
    sns.kdeplot(data=df, x=metric, hue="label", fill=True, palette=[color, "#888888"], alpha=0.6)
    plt.title(f"{metric.capitalize()} DaÄŸÄ±lÄ±mÄ± (Pozitif vs Negatif)", color=color, fontsize=12)
    plt.xlabel(metric.capitalize())
    plt.ylabel("YoÄŸunluk")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_metric_distribution(valid_samples, "brightness", PRIMARY_COLOR)


plot_metric_distribution(valid_samples, "variance", PRIMARY_COLOR)


plot_metric_distribution(valid_samples, "blur", PRIMARY_COLOR)


display_section_title("ğŸ“Š Gri Ton HistogramÄ± â€” Ã–rnek GÃ¶rseller")

example_ids = valid_samples.sample(10, random_state=7)["id"].tolist()
fig, axes = plt.subplots(2, 5, figsize=(14, 5))
axes = axes.flatten()

for ax, img_id in zip(axes, example_ids):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img = np.array(Image.open(img_path).convert("L"))
        ax.hist(img.ravel(), bins=30, color=PRIMARY_COLOR, alpha=0.7)
        label_val = labels.loc[labels["id"] == img_id, "label"].values[0]
        ax.set_title(f"{'Cancer' if label_val==1 else 'Normal'}\n{img_id[:10]}...", fontsize=8, color=PRIMARY_COLOR if label_val==1 else "#888888")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, None)
    except:
        ax.text(0.5, 0.5, "Error", ha="center", va="center", color="red")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Gri Ton HistogramlarÄ± (Rastgele 10 GÃ¶rsel)", color=PRIMARY_COLOR, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


"""
[STEP 5] DATA IMBALANCE & FINAL EDA SUMMARY
"""
display_section_title("âš–ï¸� SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ± & Dengesizlik Analizi")

sizes = label_counts["count"]
labels_pie = ["Cancer" if lbl == 1 else "Normal" for lbl in label_counts["label"]]
colors = [PRIMARY_COLOR, "#1caad9"]

explode = [0.05, 0.05]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
wedges, texts, autotexts = ax.pie(
    sizes,
    autopct="%1.1f%%",
    startangle=120,
    colors=colors,
    shadow=True,
    explode=explode,
    pctdistance=0.8,
    textprops={"fontsize": 11, "color": "white"},
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)

for i, w in enumerate(wedges):
    ang = (w.theta2 - w.theta1)/2. + w.theta1
    x = np.cos(np.deg2rad(ang))
    y = np.sin(np.deg2rad(ang))
    ax.text(
        1.2 * x, 1.2 * y,
        f"{labels_pie[i]}\n({sizes[i]:,})",
        ha="center", va="center",
        fontsize=10.5,
        color="#333333",
        weight="medium"
    )

ax.text(
    0, 0, f"Toplam\n{len(labels):,}",
    ha="center", va="center",
    fontsize=12,
    color="#222222",
    weight="bold"
)

plt.title(
    "SÄ±nÄ±f OranlarÄ± (Modern 3D GÃ¶rÃ¼nÃ¼m)",
    color=PRIMARY_COLOR,
    fontsize=13,
    pad=20,
    weight="bold"
)
plt.tight_layout()
plt.show()


from IPython.display import display, HTML

pos_ratio = label_counts.loc[label_counts["label"] == 1, "ratio"].values[0]
neg_ratio = label_counts.loc[label_counts["label"] == 0, "ratio"].values[0]
imbalance_ratio = abs(pos_ratio - neg_ratio)

if imbalance_ratio < 10:
    balance_status = "âœ… Dataset dengeli gÃ¶rÃ¼nÃ¼yor"
    color_tag = "#1abc9c"
elif imbalance_ratio < 30:
    balance_status = "âš ï¸� Hafif dengesizlik mevcut"
    color_tag = "#f1c40f"
else:
    balance_status = "ğŸš¨ GÃ¼Ã§lÃ¼ dengesizlik var"
    color_tag = "#e74c3c"

# HTML rapor
html_report = f"""
<div style='
    background-color:#f9f9f9;
    border-radius:14px;
    border-left:6px solid {PRIMARY_COLOR};
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
    padding:20px;
    margin:15px 0;
    font-family:Segoe UI, sans-serif;
    color:#333;
'>
    <h2 style='color:{PRIMARY_COLOR}; margin-bottom:10px;'>ğŸ“Š EDA Ã–zet Raporu</h2>
    <hr style='border:none; border-top:1px solid #ddd; margin:10px 0;'>

    <h4 style='color:#444;'>ğŸ“¦ Genel Bilgiler</h4>
    <ul style='list-style:none; padding-left:10px;'>
        <li>ğŸ§® Toplam Ã¶rnek sayÄ±sÄ±: <b>{len(labels):,}</b></li>
        <li>ğŸ–¼ï¸� GÃ¶rsel boyutu: <b>96Ã—96 piksel</b></li>
        <li>ğŸ§¾ GÃ¶rsel formatÄ±: <b>.tif</b></li>
        <li>ğŸ“‚ Train klasÃ¶rÃ¼: <b>{len(os.listdir(TRAIN_DIR)):,} dosya</b></li>
        <li>ğŸ§ª Test klasÃ¶rÃ¼: <b>{len(os.listdir(TEST_DIR)):,} dosya</b></li>
    </ul>

    <h4 style='color:#444;'>âš–ï¸� SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±</h4>
    <ul style='list-style:none; padding-left:10px;'>
        <li>ğŸ”¹ Pozitif (Cancer): <b>{pos_ratio:.2f}%</b></li>
        <li>ğŸ”¸ Negatif (Normal): <b>{neg_ratio:.2f}%</b></li>
        <li>ğŸ“‰ Dengesizlik farkÄ±: <b>{imbalance_ratio:.2f}%</b></li>
    </ul>

    <div style='
        background:{color_tag}20;
        border-left:4px solid {color_tag};
        border-radius:6px;
        padding:10px 14px;
        margin:8px 0;
        font-size:15px;
    '>
        <b style='color:{color_tag};'>{balance_status}</b>
    </div>

    <h4 style='color:#444;'>ğŸ”¬ GÃ¶rsel Kalite Analizi</h4>
    <ul style='list-style:none; padding-left:10px;'>
        <li>ğŸ’¡ Ortalama parlaklÄ±k, varyans ve kontrast hesaplandÄ±</li>
        <li>ğŸ”� Blur (focus) Ã¶lÃ§Ã¼mÃ¼ tamamlandÄ±</li>
        <li>âœ… Bozuk dosya sayÄ±sÄ±: <b>Ã§ok dÃ¼ÅŸÃ¼k veya yok</b></li>
    </ul>
</div>
"""

display(HTML(html_report))


"""
===============================================================================
[STEP 6] COLOR DISTRIBUTION ANALYSIS
-------------------------------------------------------------------------------
AmaÃ§ (1. AdÄ±m):
  - RGB kanallarÄ±nÄ±n ortalama ve standart sapma deÄŸerlerini hesaplamak
===============================================================================
"""

from tqdm.auto import tqdm

SAMPLE_SIZE_COLOR = min(1500, len(labels))
sample_color_df = labels.sample(SAMPLE_SIZE_COLOR, random_state=42).reset_index(drop=True)

rgb_means, rgb_stds = [], []


for img_id in tqdm(sample_color_df["id"], desc="RGB kanal analiz"):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        r_mean, g_mean, b_mean = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
        r_std, g_std, b_std = np.std(img[:, :, 0]), np.std(img[:, :, 1]), np.std(img[:, :, 2])
        rgb_means.append((r_mean, g_mean, b_mean))
        rgb_stds.append((r_std, g_std, b_std))
    except:
        rgb_means.append((np.nan, np.nan, np.nan))
        rgb_stds.append((np.nan, np.nan, np.nan))

sample_color_df[["R_mean", "G_mean", "B_mean"]] = pd.DataFrame(rgb_means, index=sample_color_df.index)
sample_color_df[["R_std", "G_std", "B_std"]] = pd.DataFrame(rgb_stds, index=sample_color_df.index)

rgb_summary = (
    sample_color_df[["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]]
    .agg(["mean", "std"])
    .round(2)
)


display_section_title("ğŸ�¨ Renk (RGB) Kanal Analizi â€” Ortalama & Std Hesaplama")
display_dataframe(rgb_summary, title="RGB Kanal OrtalamalarÄ± ve Standart SapmalarÄ± (Ã–rneklem BazÄ±nda)")


display_section_title("ğŸŒˆ RGB Kanal DaÄŸÄ±lÄ±mÄ±")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor("white")
channels = ["R_mean", "G_mean", "B_mean"]
titles = ["KÄ±rmÄ±zÄ± (R)", "YeÅŸil (G)", "Mavi (B)"]
colors = ["#e74c3c", "#2ecc71", "#3498db"]

for ax, ch, title, color in zip(axes, channels, titles, colors):
    sns.histplot(
        data=sample_color_df,
        x=ch,
        bins=30,
        kde=True,
        color=color,
        ax=ax,
        alpha=0.8
    )
    ax.set_title(title, fontsize=12, color=color, weight="bold")
    ax.set_xlabel("Ortalama Piksel YoÄŸunluÄŸu")
    ax.set_ylabel("Frekans")
    ax.grid(alpha=0.3)

plt.suptitle("RGB Kanal DaÄŸÄ±lÄ±mÄ±", fontsize=14, color=PRIMARY_COLOR, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


display_section_title("ğŸ§© SÄ±nÄ±fa GÃ¶re RGB Kanal DaÄŸÄ±lÄ±mÄ± (Boxplot Analizi)")

fig, axes = plt.subplots(1, 3, figsize=(20, 9))
fig.patch.set_facecolor("white")

channels = ["R_mean", "G_mean", "B_mean"]
titles = ["KÄ±rmÄ±zÄ± (R)", "YeÅŸil (G)", "Mavi (B)"]
colors = ["#e74c3c", "#2ecc71", "#3498db"]

palette_dict = {"0": "#bbbbbb", "1": "#3371ab"}

for ax, ch, title, color in zip(axes, channels, titles, colors):
    df_plot = sample_color_df.copy()
    df_plot["label_str"] = df_plot["label"].astype(str)

    sns.boxplot(
        data=df_plot,
        x="label_str",
        y=ch,
        hue="label_str",
        dodge=False,
        palette={"0": "#cccccc", "1": color},
        ax=ax,
        width=0.55,
        fliersize=2
    )

    ax.set_title(title, fontsize=12, color=color, weight="bold")
    ax.set_xlabel("SÄ±nÄ±f (0=Normal, 1=Cancer)")
    ax.set_ylabel("Ortalama Piksel YoÄŸunluÄŸu")
    ax.grid(alpha=0.3)

plt.suptitle("SÄ±nÄ±fa GÃ¶re RGB Kanal DaÄŸÄ±lÄ±mÄ± (Boxplot GÃ¶rselleÅŸtirme)", fontsize=14, color=PRIMARY_COLOR, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


display_section_title("ğŸ�¨ Renk VaryansÄ± Analizi ve Kanal BaskÄ±nlÄ±ÄŸÄ± Yorumu")

mean_r = sample_color_df["R_mean"].mean()
mean_g = sample_color_df["G_mean"].mean()
mean_b = sample_color_df["B_mean"].mean()

std_r = sample_color_df["R_std"].mean()
std_g = sample_color_df["G_std"].mean()
std_b = sample_color_df["B_std"].mean()

color_stats = pd.DataFrame({
    "Kanal": ["R (KÄ±rmÄ±zÄ±)", "G (YeÅŸil)", "B (Mavi)"],
    "Ortalama YoÄŸunluk": [round(mean_r, 2), round(mean_g, 2), round(mean_b, 2)],
    "Ortalama Std (Varyans)": [round(std_r, 2), round(std_g, 2), round(std_b, 2)]
})

display_dataframe(color_stats, title="RGB Kanal Ä°statistik Ã–zeti")

dominant_channel = color_stats.loc[color_stats["Ortalama YoÄŸunluk"].idxmax(), "Kanal"]
dominant_std = color_stats.loc[color_stats["Ortalama Std (Varyans)"].idxmax(), "Kanal"]

html_summary = f"""
<div style='
    background-color:#f9f9f9;
    border-radius:14px;
    border-left:6px solid {PRIMARY_COLOR};
    box-shadow:0 2px 6px rgba(0,0,0,0.1);
    padding:20px;
    margin:15px 0;
    font-family:Segoe UI, sans-serif;
    color:#333;
'>
    <h3 style='color:{PRIMARY_COLOR}; margin-bottom:10px;'>ğŸ�¨ Renk VaryansÄ± Yorumu</h3>
    <p style='margin:6px 0;'>
        <b>ğŸ”¹ Ortalama kanal yoÄŸunluÄŸu aÃ§Ä±sÄ±ndan baskÄ±n renk:</b> <span style='color:{PRIMARY_COLOR}'>{dominant_channel}</span><br>
        <b>ğŸ”¸ DeÄŸiÅŸkenlik (standart sapma) aÃ§Ä±sÄ±ndan en dinamik kanal:</b> <span style='color:{PRIMARY_COLOR}'>{dominant_std}</span>
    </p>
    <p style='margin-top:10px; color:#555;'>
        Bu durum, veri setindeki histopatolojik boyamanÄ±n renk karakterini yansÄ±tÄ±r.
        Genellikle hematoksilen-eozin (H&E) boyalÄ± Ã¶rneklerde mavi ve pembe tonlar hakimdir.
    </p>
</div>
"""

display(HTML(html_summary))


display_section_title("ğŸ§¬ Pozitif ve Negatif GÃ¶rseller â€” Renk KarÅŸÄ±laÅŸtÄ±rma Galerisi")

n_samples = 20
pos_samples = labels[labels["label"] == 1]["id"].sample(n_samples, random_state=42).tolist()
neg_samples = labels[labels["label"] == 0]["id"].sample(n_samples, random_state=42).tolist()

fig, axes = plt.subplots(2, n_samples, figsize=(22, 4))
fig.patch.set_facecolor("white")

for i, (ax, img_id) in enumerate(zip(axes[0], pos_samples)):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        with Image.open(img_path) as img:
            ax.imshow(img)
    except:
        ax.text(0.5, 0.5, "Error", ha="center", va="center", color="red")
    ax.set_title("Cancer", fontsize=8, color=PRIMARY_COLOR)
    ax.axis("off")

for i, (ax, img_id) in enumerate(zip(axes[1], neg_samples)):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        with Image.open(img_path) as img:
            ax.imshow(img)
    except:
        ax.text(0.5, 0.5, "Error", ha="center", va="center", color="red")
    ax.set_title("Normal", fontsize=8, color="#777")
    ax.axis("off")

fig.suptitle("20 Pozitif ve 20 Negatif GÃ¶rsel â€” Renk ve Doku KarÅŸÄ±laÅŸtÄ±rmasÄ±",
             color=PRIMARY_COLOR, fontsize=14, weight="bold", y=1.05)
plt.tight_layout()
plt.show()


from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

display_section_title("ğŸ§© Doku (Texture) Analizi â€” GLCM Ã–zellik Ã‡Ä±karÄ±mÄ±")

SAMPLE_SIZE_TEXTURE = min(1000, len(labels))
sample_texture_df = labels.sample(SAMPLE_SIZE_TEXTURE, random_state=42).reset_index(drop=True)

glcm_features = {
    "contrast": [],
    "homogeneity": [],
    "energy": [],
    "entropy": []
}

def extract_glcm_features(img_path):
    try:
        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        ent = entropy(glcm.ravel())
        return contrast, homogeneity, energy, ent
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

for img_id in tqdm(sample_texture_df["id"], desc="GLCM hesaplanÄ±yor"):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    c, h, e, en = extract_glcm_features(img_path)
    glcm_features["contrast"].append(c)
    glcm_features["homogeneity"].append(h)
    glcm_features["energy"].append(e)
    glcm_features["entropy"].append(en)

for key, vals in glcm_features.items():
    sample_texture_df[key] = vals

valid_glcm = sample_texture_df.dropna()
display_dataframe(valid_glcm.head(10), title="GLCM Ã–zellikleri â€” Ä°lk 10 SatÄ±r")

glcm_summary = (
    valid_glcm.groupby("label")[["contrast", "homogeneity", "energy", "entropy"]]
    .agg(["mean", "std"])
    .round(3)
)
display_dataframe(glcm_summary, title="SÄ±nÄ±fa GÃ¶re GLCM Ã–zellik OrtalamalarÄ±")


display_section_title("ğŸ“Š SÄ±nÄ±fa GÃ¶re GLCM Ã–zellik DaÄŸÄ±lÄ±mÄ±")

metrics = ["contrast", "homogeneity", "energy", "entropy"]
titles = ["Kontrast", "Homojenlik", "Enerji", "Entropi"]

valid_glcm["label_str"] = valid_glcm["label"].astype(str)
palette_dict = {"0": "#cccccc", "1": PRIMARY_COLOR}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, metric, title in zip(axes, metrics, titles):
    sns.violinplot(
        data=valid_glcm,
        x="label_str",
        y=metric,
        hue="label_str",
        dodge=False,
        palette=palette_dict,
        ax=ax,
        inner="box",
        cut=0,
        legend=False
    )
    ax.set_title(title, fontsize=12, color=PRIMARY_COLOR, weight="bold")
    ax.set_xlabel("SÄ±nÄ±f (0=Normal, 1=Cancer)")
    ax.set_ylabel(metric.capitalize())
    ax.grid(alpha=0.3)

plt.suptitle("SÄ±nÄ±fa GÃ¶re GLCM Doku Ã–zellik DaÄŸÄ±lÄ±mlarÄ±", fontsize=14, color=PRIMARY_COLOR, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


from skimage.feature import local_binary_pattern

display_section_title("ğŸ§© Local Binary Pattern (LBP) Analizi")

radius = 1
n_points = 8 * radius
method = "uniform"

SAMPLE_SIZE_LBP = min(1000, len(labels))
sample_lbp_df = labels.sample(SAMPLE_SIZE_LBP, random_state=42).reset_index(drop=True)

lbp_means = []
lbp_stds = []

for img_id in tqdm(sample_lbp_df["id"], desc="LBP hesaplanÄ±yor"):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img_gray = np.array(Image.open(img_path).convert("L"))
        lbp = local_binary_pattern(img_gray, n_points, radius, method)
        lbp_means.append(np.mean(lbp))
        lbp_stds.append(np.std(lbp))
    except Exception:
        lbp_means.append(np.nan)
        lbp_stds.append(np.nan)

sample_lbp_df["LBP_mean"] = lbp_means
sample_lbp_df["LBP_std"] = lbp_stds

valid_lbp = sample_lbp_df.dropna()
display_dataframe(valid_lbp.head(10), title="LBP Ã–zellikleri â€” Ä°lk 10 SatÄ±r")

lbp_summary = (
    valid_lbp.groupby("label")[["LBP_mean", "LBP_std"]]
    .agg(["mean", "std"])
    .round(3)
)
display_dataframe(lbp_summary, title="SÄ±nÄ±fa GÃ¶re LBP Ortalama / Std Ã–zeti")


display_section_title("ğŸ“Š SÄ±nÄ±fa GÃ¶re LBP Ã–zellik DaÄŸÄ±lÄ±mlarÄ±")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.flatten()
palette_lbp = {"0": "#cccccc", "1": PRIMARY_COLOR}

valid_lbp["label_str"] = valid_lbp["label"].astype(str)

sns.boxplot(
    data=valid_lbp, x="label_str", y="LBP_mean",
    hue="label_str", dodge=False, palette=palette_lbp, ax=axes[0], fliersize=2
)
sns.boxplot(
    data=valid_lbp, x="label_str", y="LBP_std",
    hue="label_str", dodge=False, palette=palette_lbp, ax=axes[1], fliersize=2
)

axes[0].set_title("LBP Mean DaÄŸÄ±lÄ±mÄ±", color=PRIMARY_COLOR)
axes[1].set_title("LBP Std DaÄŸÄ±lÄ±mÄ±", color=PRIMARY_COLOR)

for ax in axes:
    ax.set_xlabel("SÄ±nÄ±f (0=Normal, 1=Cancer)")
    ax.grid(alpha=0.3)

plt.suptitle("SÄ±nÄ±fa GÃ¶re Local Binary Pattern (LBP) Ã–zellikleri", fontsize=14, color=PRIMARY_COLOR, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


display_section_title("Doku (Texture) Heatmap GÃ¶rselleÅŸtirmesi â€” LBP HaritalarÄ±")

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

n_samples = 3
pos_samples = labels[labels["label"] == 1]["id"].sample(n_samples, random_state=7).tolist()
neg_samples = labels[labels["label"] == 0]["id"].sample(n_samples, random_state=7).tolist()
sample_ids = pos_samples + neg_samples

fig, axes = plt.subplots(len(sample_ids), 2, figsize=(8, 14))
fig.patch.set_facecolor("white")

for i, img_id in enumerate(sample_ids):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img_gray = np.array(Image.open(img_path).convert("L"))
        lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")

        label_val = labels.loc[labels["id"] == img_id, "label"].values[0]
        label_text = "Cancer" if label_val == 1 else "Normal"
        color = PRIMARY_COLOR if label_val == 1 else "#777"

        axes[i, 0].imshow(img_gray, cmap="gray")
        axes[i, 0].set_title(f"Orijinal ({label_text})", color=color, fontsize=10, weight="bold")

        axes[i, 1].imshow(lbp, cmap="inferno")
        axes[i, 1].set_title("LBP HaritasÄ±", color=color, fontsize=10, weight="bold")

        for ax in axes[i]:
            ax.axis("off")

    except Exception as e:
        for ax in axes[i]:
            ax.text(0.5, 0.5, f"Error\n{type(e).__name__}: {str(e)[:40]}", ha="center", va="center", color="red", fontsize=8)
            ax.axis("off")

plt.suptitle("Doku (Texture) Heatmap KarÅŸÄ±laÅŸtÄ±rmalarÄ± â€” LBP GÃ¶rselleÅŸtirmesi",
             fontsize=14, color=PRIMARY_COLOR, weight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.show()



display_section_title("ğŸ§© BoÅŸluk (White Area) ve Doku YoÄŸunluÄŸu Analizi")

SAMPLE_SIZE_WHITE = min(1000, len(labels))
sample_white_df = labels.sample(SAMPLE_SIZE_WHITE, random_state=42).reset_index(drop=True)

white_ratios, tissue_ratios = [], []

def compute_tissue_ratio(img):
    """
    HSV renk uzayÄ±nda S (saturation) kanalÄ±na gÃ¶re boÅŸluk (beyaz alan) oranÄ± hesaplar.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1] / 255.0
    white_mask = s_channel < 0.2  # dÃ¼ÅŸÃ¼k satÃ¼rasyon = beyaz
    white_ratio = np.sum(white_mask) / white_mask.size
    tissue_ratio = 1 - white_ratio
    return white_ratio, tissue_ratio

for img_id in tqdm(sample_white_df["id"], desc="Doku oranÄ± hesaplanÄ±yor"):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        w, t = compute_tissue_ratio(img)
    except Exception:
        w, t = np.nan, np.nan
    white_ratios.append(w)
    tissue_ratios.append(t)

sample_white_df["white_ratio"] = white_ratios
sample_white_df["tissue_ratio"] = tissue_ratios

valid_white = sample_white_df.dropna()

display_dataframe(valid_white.head(10), title="Doku YoÄŸunluÄŸu (Ä°lk 10 GÃ¶rsel)")


display_section_title("ğŸ“Š SÄ±nÄ±fa GÃ¶re Doku OranÄ± DaÄŸÄ±lÄ±mÄ±")
fig, ax = plt.subplots(figsize=(7, 4))
sns.kdeplot(
    data=valid_white,
    x="tissue_ratio",
    hue="label",
    fill=True,
    common_norm=False,
    palette={0: "#bbbbbb", 1: PRIMARY_COLOR},
    alpha=0.6
)
ax.set_xlabel("Tissue Ratio (1 - White Area)")
ax.set_ylabel("Density")
ax.set_title("Tissue Density Distribution â€” Cancer vs Normal", color=PRIMARY_COLOR, fontsize=12)
plt.tight_layout()
plt.show()


display_section_title("ğŸš¨ Outlier GÃ¶rseller â€” Ã‡ok BoÅŸ veya Ã‡ok Dolu Alanlar")

too_empty = valid_white[valid_white["tissue_ratio"] < 0.2]
too_full = valid_white[valid_white["tissue_ratio"] > 0.95]

print(f"ğŸ”¹ Ã‡ok boÅŸ (tissue < 0.2): {len(too_empty)} gÃ¶rsel")
print(f"ğŸ”¸ Ã‡ok dolu (tissue > 0.95): {len(too_full)} gÃ¶rsel")

outlier_samples = pd.concat([too_empty.head(3), too_full.head(3)])
fig, axes = plt.subplots(len(outlier_samples), 2, figsize=(6, 10))
fig.patch.set_facecolor("white")

for i, (idx, row) in enumerate(outlier_samples.iterrows()):
    img_path = TRAIN_DIR / f"{row['id']}.tif"
    img = np.array(Image.open(img_path).convert("RGB"))
    white_mask = (cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1] / 255.0) < 0.2

    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"ID: {row['id'][:10]}... | Tissue {row['tissue_ratio']:.2f}", fontsize=9)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(white_mask, cmap="gray")
    axes[i, 1].set_title("White Mask", fontsize=9)
    axes[i, 1].axis("off")

plt.suptitle("Outlier GÃ¶rseller â€” Doku OranÄ± AykÄ±rÄ± Olanlar", fontsize=13, color=PRIMARY_COLOR, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


from skimage.feature import local_binary_pattern
from scipy.stats import entropy as shannon_entropy

PATCH = 32   # merkez patch boyutu
IMG = 96     # HCD gÃ¶rselleri 96Ã—96

def get_center_patch(img_rgb, patch=PATCH):
    h, w = img_rgb.shape[:2]
    cy, cx = h//2, w//2
    half = patch//2
    return img_rgb[cy-half:cy+half, cx-half:cx+half]

def get_periphery_patches(img_rgb, patch=PATCH):
    """Periferi iÃ§in 4 kÃ¶ÅŸe 32Ã—32 patch dÃ¶ndÃ¼rÃ¼r."""
    p = patch
    tl = img_rgb[0:p, 0:p]
    tr = img_rgb[0:p, -p:]
    bl = img_rgb[-p:, 0:p]
    br = img_rgb[-p:, -p:]
    return [tl, tr, bl, br]

def gray_metrics(gray):
    """Gri ton iÃ§in: mean, var, laplacian-var (focus) ve histogram entropisi."""
    import cv2
    g = gray.astype(np.float32)
    mean = float(np.mean(g))
    var = float(np.var(g))
    try:
        lap = cv2.Laplacian(g, cv2.CV_64F)
        lapv = float(lap.var())
    except Exception:
        lapv = float(np.var(cv2.GaussianBlur(g, (3, 3), 0)))

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255), density=True)
    ent = float(shannon_entropy(hist + 1e-12))
    return mean, var, lapv, ent

def color_metrics(rgb):
    """RGB ortalama ve std."""
    r = rgb[:,:,0].astype(np.float32)
    g = rgb[:,:,1].astype(np.float32)
    b = rgb[:,:,2].astype(np.float32)
    return (float(np.mean(r)), float(np.mean(g)), float(np.mean(b)),
            float(np.std(r)),  float(np.std(g)),  float(np.std(b)))

def lbp_metrics(gray, P=8, R=1, method="uniform"):
    lbp = local_binary_pattern(gray, P=P, R=R, method=method)
    return float(np.mean(lbp)), float(np.std(lbp))



records = []
SAMPLE_SIZE_CENTER = min(1000, len(labels))
sample_center_df = labels.sample(SAMPLE_SIZE_CENTER, random_state=42).reset_index(drop=True)

for img_id, label in tqdm(
    sample_center_df[["id", "label"]].itertuples(index=False),
    total=len(sample_center_df),
    desc="Center/Periphery metrics"
):
    try:
        rgb  = np.array(Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB"))
        gray = np.array(Image.open(TRAIN_DIR / f"{img_id}.tif").convert("L"))

        # Center patch
        c_rgb  = get_center_patch(rgb)
        c_gray = get_center_patch(gray)

        # --- Center metrics ---
        c_rmean, c_gmean, c_bmean, c_rstd, c_gstd, c_bstd = color_metrics(c_rgb)
        c_gray_mean, c_gray_var, c_lapvar, c_entropy = gray_metrics(c_gray)
        c_LBP_mean, c_LBP_std = lbp_metrics(c_gray)

        # --- Periphery patches ---
        p_rgbs  = get_periphery_patches(rgb)
        p_grays = [cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) for p in p_rgbs]

        pr_means, pg_means, pb_means, pr_stds, pg_stds, pb_stds = [], [], [], [], [], []
        p_gray_means, p_gray_vars, p_lapvars, p_entropies, p_LBP_means, p_LBP_stds = [], [], [], [], [], []

        for prgb, pgray in zip(p_rgbs, p_grays):
            rmean, gmean, bmean, rstd, gstd, bstd = color_metrics(prgb)
            pr_means.append(rmean); pg_means.append(gmean); pb_means.append(bmean)
            pr_stds.append(rstd);   pg_stds.append(gstd);   pb_stds.append(bstd)

            gmean_, gvar_, lapv_, ent_ = gray_metrics(pgray)
            p_gray_means.append(gmean_); p_gray_vars.append(gvar_)
            p_lapvars.append(lapv_); p_entropies.append(ent_)

            lbp_m, lbp_s = lbp_metrics(pgray)
            p_LBP_means.append(lbp_m); p_LBP_stds.append(lbp_s)

        rec = {
            "id": img_id, "label": label,
            # Center
            "c_R_mean": c_rmean, "c_G_mean": c_gmean, "c_B_mean": c_bmean,
            "c_R_std": c_rstd, "c_G_std": c_gstd, "c_B_std": c_bstd,
            "c_gray_mean": c_gray_mean, "c_gray_var": c_gray_var,
            "c_lapvar": c_lapvar, "c_entropy": c_entropy,
            "c_LBP_mean": c_LBP_mean, "c_LBP_std": c_LBP_std,
            # Periphery (averaged)
            "p_R_mean": np.mean(pr_means), "p_G_mean": np.mean(pg_means), "p_B_mean": np.mean(pb_means),
            "p_R_std": np.mean(pr_stds), "p_G_std": np.mean(pg_stds), "p_B_std": np.mean(pb_stds),
            "p_gray_mean": np.mean(p_gray_means), "p_gray_var": np.mean(p_gray_vars),
            "p_lapvar": np.mean(p_lapvars), "p_entropy": np.mean(p_entropies),
            "p_LBP_mean": np.mean(p_LBP_means), "p_LBP_std": np.mean(p_LBP_stds)
        }

        # Farklar
        rec.update({
            "d_R_mean": rec["c_R_mean"] - rec["p_R_mean"],
            "d_G_mean": rec["c_G_mean"] - rec["p_G_mean"],
            "d_B_mean": rec["c_B_mean"] - rec["p_B_mean"],
            "d_gray_var": rec["c_gray_var"] - rec["p_gray_var"],
            "d_lapvar": rec["c_lapvar"] - rec["p_lapvar"],
            "d_entropy": rec["c_entropy"] - rec["p_entropy"],
            "d_LBP_mean": rec["c_LBP_mean"] - rec["p_LBP_mean"],
            "d_LBP_std": rec["c_LBP_std"] - rec["p_LBP_std"]
        })

        records.append(rec)

    except Exception as e:
        print(f"âš ï¸� {img_id}: {type(e).__name__} â€“ {str(e)[:100]}")
        continue

centerperi_df = pd.DataFrame.from_records(records)


display_section_title("ğŸ“Š Center vs Periphery â€” Class-wise Summary")

summary_cols = [
    "c_gray_var","p_gray_var","d_gray_var",
    "c_lapvar","p_lapvar","d_lapvar",
    "c_entropy","p_entropy","d_entropy",
    "c_LBP_mean","p_LBP_mean","d_LBP_mean",
    "c_LBP_std","p_LBP_std","d_LBP_std"
]

grp = centerperi_df.groupby("label")[summary_cols].agg(["mean","std"]).round(3)
display_dataframe(grp, title="Center/Periphery Metrics â€” Class-wise Summary")


display_section_title("ğŸ�» Distribution of Centerâ€“Periphery Differences (by Class)")

plot_df = centerperi_df.copy()
plot_df["label_str"] = plot_df["label"].astype(str)
palette = {"0": "#cccccc", "1": PRIMARY_COLOR}

metrics = [
    ("d_gray_var",  "Î” Gray Variance"),
    ("d_lapvar",    "Î” Laplacian Variance (Focus)"),
    ("d_entropy",   "Î” Entropy"),
    ("d_LBP_mean",  "Î” LBP Mean"),
    ("d_LBP_std",   "Î” LBP Std"),
    ("d_R_mean",    "Î” Red Mean")
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (m, title) in zip(axes, metrics):
    sns.violinplot(
        data=plot_df,
        x="label_str", y=m,
        hue="label_str", dodge=False,
        inner="box", cut=0, palette=palette,
        ax=ax, legend=False
    )
    ax.set_title(title, color=PRIMARY_COLOR, fontsize=11, weight="bold")
    ax.set_xlabel("Class (0=Normal, 1=Cancer)")
    ax.set_ylabel("Î” Value")
    ax.grid(alpha=0.3)

plt.suptitle("Centerâ€“Periphery Feature Differences â€” Cancer vs Normal",
             fontsize=15, color=PRIMARY_COLOR, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


display_section_title("Correlation Heatmap â€” Center/Periphery Features")

corr_cols = [
    "d_gray_var", "d_lapvar", "d_entropy",
    "d_LBP_mean", "d_LBP_std", "d_R_mean", "d_G_mean", "d_B_mean"
]

corr = centerperi_df[corr_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    linewidths=0.5,
    square=True,
    annot_kws={"size": 9, "color": "#333333"}
)
plt.title("Correlation between Centerâ€“Periphery Features",
          color=PRIMARY_COLOR, fontsize=13, weight="bold", pad=12)
plt.tight_layout()
plt.show(),


display_section_title("ğŸ�¨ Renk Normalizasyonu Etkisi â€” (Manual Reinhard / CLAHE Surrogate)")

import cv2
from skimage import exposure

def reinhard_normalization(img_rgb):
    lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    mean, std = np.mean(lab, axis=(0, 1)), np.std(lab, axis=(0, 1))
    ref_mean, ref_std = [128, 128, 128], [30, 30, 30]
    norm = (lab - mean) / std * ref_std + ref_mean
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_LAB2RGB) / 255.0

def macenko_surrogate(img_rgb):
    lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    eq_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB) / 255.0
    eq_img = np.power(eq_img, 0.9)
    return eq_img

N_ROWS, N_COLS = 3, 5
sample_ids = labels.sample(N_ROWS * N_COLS, random_state=42)["id"].tolist()

fig, axes = plt.subplots(N_ROWS, N_COLS*3, figsize=(16, 8))
fig.patch.set_facecolor("white")

var_pre, var_macenko, var_reinhard = [], [], []

for i, img_id in enumerate(sample_ids):
    row = i // N_COLS
    col = (i % N_COLS) * 3
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        axes[row, col].imshow(img)
        axes[row, col].set_title("Orijinal", fontsize=9, color="#555")
        axes[row, col].axis("off")

        mac_img = macenko_surrogate(img)
        axes[row, col+1].imshow(np.clip(mac_img, 0, 1))
        axes[row, col+1].set_title("Macenko (CLAHE)", fontsize=9, color=PRIMARY_COLOR)
        axes[row, col+1].axis("off")

        reinhard_img = reinhard_normalization(img)
        axes[row, col+2].imshow(np.clip(reinhard_img, 0, 1))
        axes[row, col+2].set_title("Reinhard (LAB)", fontsize=9, color="#2ecc71")
        axes[row, col+2].axis("off")

        var_pre.append(np.var(img))
        var_macenko.append(np.var(mac_img))
        var_reinhard.append(np.var(reinhard_img))
    except Exception:
        for k in range(3):
            axes[row, col+k].axis("off")

plt.suptitle("Renk Normalizasyonu â€” Reinhard & CLAHE (Macenko Surrogate)", 
             color=PRIMARY_COLOR, fontsize=13, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# varyans deÄŸiÅŸimi
var_df = pd.DataFrame({
    "AÅŸama": ["Orijinal", "Macenko (CLAHE)", "Reinhard (LAB)"],
    "Ortalama Varyans": [np.mean(var_pre), np.mean(var_macenko), np.mean(var_reinhard)]
})
var_df["DeÄŸiÅŸim (%)"] = (
    (var_df["Ortalama Varyans"] - var_df["Ortalama Varyans"].iloc[0]) / var_df["Ortalama Varyans"].iloc[0] * 100
).round(2)
display_dataframe(var_df, title="Renk Normalizasyonu SonrasÄ± Varyans DeÄŸiÅŸimi (%)")

plt.figure(figsize=(6,4))
sns.barplot(data=var_df, x="AÅŸama", y="Ortalama Varyans", palette=["#888", PRIMARY_COLOR, "#2ecc71"])
plt.title("Renk VaryansÄ± KarÅŸÄ±laÅŸtÄ±rmasÄ±", color=PRIMARY_COLOR, fontsize=12)
plt.ylabel("Ortalama Piksel VaryansÄ±")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


"""
===============================================================================
[STEP 11] Ã–ZELLÄ°KSEL TEMSÄ°L & GÃ–RSEL KÃœMELEME
-------------------------------------------------------------------------------
AmaÃ§:
  - Basit CNN veya ImageNet tabanlÄ± feature extractor (Ã¶r. ResNet18)
  - t-SNE veya UMAP ile 2D gÃ¶rselleÅŸtirme
  - Pozitif (kÄ±rmÄ±zÄ±) vs Negatif (mavi) scatter plot
  - KÃ¼meler arasÄ±ndaki ayrÄ±mÄ± yorumlamak
===============================================================================


display_section_title("ğŸ§  Ã–zelliksel Temsil ve GÃ¶rsel KÃ¼meleme (Feature Embedding + t-SNE)")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import numpy as np
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ğŸ§® Ã‡alÄ±ÅŸma cihazÄ±: {device}")

# --- Ã–rnekleme ---
SAMPLE_SIZE_EMB = min(300, len(labels))
sample_emb_df = labels.sample(SAMPLE_SIZE_EMB, random_state=42).reset_index(drop=True)

# --- Feature extractor (ResNet18 pretrained) ---
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()  # son katmanÄ± kaldÄ±r
resnet = resnet.to(device)
resnet.eval()

# --- GÃ¶rselleri dÃ¶nÃ¼ÅŸtÃ¼rme ---
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])"""


"""embeddings, labels_list = [], []

for img_id, label in tqdm(sample_emb_df[["id", "label"]].itertuples(index=False), total=len(sample_emb_df)):
    try:
        img = Image.open(TRAIN_DIR / f"{img_id}.tif").convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(tensor).cpu().numpy().flatten()
        embeddings.append(feat)
        labels_list.append(label)
    except Exception as e:
        continue

embeddings = np.array(embeddings)
labels_arr = np.array(labels_list)

print(f"âœ… Embedding boyutu: {embeddings.shape}")"""


"""# --- Boyut indirgeme (t-SNE) ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
emb_2d = tsne.fit_transform(embeddings)

# --- GÃ¶rselleÅŸtirme ---
display_section_title("ğŸ“Š t-SNE 2D GÃ¶rselleÅŸtirmesi â€” Pozitif vs Negatif DaÄŸÄ±lÄ±m")

plt.figure(figsize=(8, 6))
mask_pos = labels_arr == 1
mask_neg = labels_arr == 0

plt.scatter(emb_2d[mask_neg, 0], emb_2d[mask_neg, 1],
            c="#3498db", s=40, alpha=0.6, label="Normal (0)")
plt.scatter(emb_2d[mask_pos, 0], emb_2d[mask_pos, 1],
            c="#e74c3c", s=40, alpha=0.7, label="Cancer (1)")

plt.legend(frameon=True)
plt.title("t-SNE Feature Embedding â€” ResNet18 GÃ¶rsel Temsil", color=PRIMARY_COLOR, fontsize=13, weight="bold")
plt.xlabel("t-SNE bileÅŸen 1")
plt.ylabel("t-SNE bileÅŸen 2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""


"""from sklearn.metrics import silhouette_score
try:
    sil = silhouette_score(emb_2d, labels_arr)
    print(f"ğŸ“ˆ Silhouette Score (pozitif-negatif ayrÄ±mÄ± kalitesi): {sil:.3f}")
except Exception:
    print("Silhouette Score hesaplanamadÄ± (Ã¶rnek sayÄ±sÄ± dÃ¼ÅŸÃ¼k olabilir).")

# --- GÃ¶rsel Ã¶zet yorumu ---
from IPython.display import HTML, display


summary_html = f
<div style='
  background:#f9f9f9;
  border-left:6px solid {PRIMARY_COLOR};
  border-radius:10px;
  padding:15px 18px;
  font-family:Segoe UI, sans-serif;
  box-shadow:0 2px 5px rgba(0,0,0,0.08);
'>
  <h3 style='color:{PRIMARY_COLOR}; margin-bottom:8px;'>ğŸ”� KÃ¼melenme Yorumu</h3>
  <p style='color:#444; font-size:15px;'>
    GÃ¶rselleÅŸtirilen t-SNE temsilleri, CNN modelinin Ã§Ä±kardÄ±ÄŸÄ± Ã¶zellik uzayÄ±nda
    <b>pozitif (kÄ±rmÄ±zÄ±)</b> ve <b>negatif (mavi)</b> Ã¶rneklerin daÄŸÄ±lÄ±mÄ±nÄ± gÃ¶stermektedir.
    EÄŸer iki sÄ±nÄ±f birbirinden ayrÄ±k kÃ¼melerde toplanÄ±yorsa,
    modelin dokusal ve renk bazlÄ± farklarÄ± ayÄ±rt edebildiÄŸi sÃ¶ylenebilir.
  </p>
  <p style='color:#555; font-size:14px;'>
    â€¢ YÃ¼ksek <b>Silhouette Score</b> â†’ iyi ayrÄ±m.<br>
    â€¢ DÃ¼ÅŸÃ¼k veya karÄ±ÅŸÄ±k daÄŸÄ±lÄ±m â†’ gÃ¶rsel benzerlik fazladÄ±r, ek doku veya morfolojik Ã¶zellikler gerekebilir.
  </p>
</div>
display(HTML(summary_html))"""


### [STEP X] TRAIN ve TEST Veri Setlerinde SÄ±nÄ±f DaÄŸÄ±lÄ±mlarÄ±

display_section_title("ğŸ“Š Train/Test Veri Seti â€” SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ± (0 vs 1)")

from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# --- Train etiket tablosu ---
train_counts = labels['label'].value_counts().sort_index()
train_df = pd.DataFrame({
    "set": "Train",
    "label": train_counts.index.astype(str),
    "count": train_counts.values
})

# --- Test klasÃ¶rÃ¼ndeki etiket tahmini (dosya adlarÄ±na gÃ¶re veya varsa test.csv) ---
# EÄŸer test etiketleri ayrÄ± bir dosyada varsa (Ã¶rneÄŸin test_labels.csv) ÅŸunu kullan:
# test_labels = pd.read_csv("test_labels.csv")
# test_counts = test_labels['label'].value_counts().sort_index()

# EÄŸer test dosyalarÄ± sadece klasÃ¶rlerdeyse (Ã¶rneÄŸin test/0 ve test/1 klasÃ¶rleri):
test_root = Path(TEST_DIR)
test_0 = len(list((test_root / "0").glob("*.tif"))) if (test_root / "0").exists() else 0
test_1 = len(list((test_root / "1").glob("*.tif"))) if (test_root / "1").exists() else 0
test_counts = pd.Series({0: test_0, 1: test_1})

test_df = pd.DataFrame({
    "set": "Test",
    "label": test_counts.index.astype(str),
    "count": test_counts.values
})

# --- BirleÅŸtir ---
dist_df = pd.concat([train_df, test_df], ignore_index=True)

# --- Barplot ---
plt.figure(figsize=(7, 5))
sns.barplot(data=dist_df, x="label", y="count", hue="set", palette=["#1f77b4", PRIMARY_COLOR])
plt.title("Train/Test Veri Setlerinde Etiket DaÄŸÄ±lÄ±mÄ±", fontsize=13, color=PRIMARY_COLOR, weight="bold")
plt.xlabel("SÄ±nÄ±f Etiketi")
plt.ylabel("GÃ¶rsel SayÄ±sÄ±")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --- SayÄ±sal Ã¶zet tablo ---
display_dataframe(dist_df.pivot(index="label", columns="set", values="count").fillna(0).astype(int),
                  title="Train/Test SÄ±nÄ±f SayÄ±larÄ±")



import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn._oldcore")


import imagehash

display_section_title("ğŸ§¬ Kopya & Benzerlik KontrolÃ¼ â€” pHash / aHash Analizi")

SAMPLE_SIZE_HASH = min(10000, len(labels))
sample_hash_df = labels.sample(SAMPLE_SIZE_HASH, random_state=42).reset_index(drop=True)

phash_vals, ahash_vals = [], []

for img_id in tqdm(sample_hash_df["id"], desc="pHash / aHash hesaplanÄ±yor"):
    img_path = TRAIN_DIR / f"{img_id}.tif"
    try:
        img = Image.open(img_path).convert("L") 
        phash_vals.append(str(imagehash.phash(img)))
        ahash_vals.append(str(imagehash.average_hash(img)))
    except Exception:
        phash_vals.append(None)
        ahash_vals.append(None)

sample_hash_df["pHash"] = phash_vals
sample_hash_df["aHash"] = ahash_vals

phash_dupes = (
    sample_hash_df.groupby("pHash")["id"]
    .apply(list)
    .reset_index()
)
phash_dupes["duplicate_count"] = phash_dupes["id"].apply(len)
dupe_groups = phash_dupes[phash_dupes["duplicate_count"] > 1]

display_dataframe(dupe_groups.head(10), title="ğŸ”� Kopya (pHash) GÃ¶rseller â€” Ä°lk 10 Grup")

total_dupes = dupe_groups["duplicate_count"].sum()
unique_dupe_groups = len(dupe_groups)
print(f"âœ… Toplam {total_dupes} duplicate gÃ¶rsel bulundu ({unique_dupe_groups} grup).")

plt.figure(figsize=(7, 4))
sns.histplot(
    phash_dupes["duplicate_count"],
    bins=30, color=PRIMARY_COLOR, alpha=0.8
)
plt.title("Kopya GÃ¶rsel DaÄŸÄ±lÄ±mÄ± (pHash BazlÄ±)", color=PRIMARY_COLOR, fontsize=12, weight="bold")
plt.xlabel("AynÄ± pHash'e Sahip GÃ¶rsel SayÄ±sÄ±")
plt.ylabel("Frekans")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

top_dupes = dupe_groups.sort_values("duplicate_count", ascending=False).head(3)

display_section_title("ğŸ“¸ En SÄ±k Tekrarlanan GÃ¶rseller (pHash BazlÄ±)")

n_show = 3
for _, row in top_dupes.iterrows():
    dup_ids = row["id"][:n_show]
    fig, axes = plt.subplots(1, len(dup_ids), figsize=(12, 4))
    fig.patch.set_facecolor("white")

    for ax, img_id in zip(axes, dup_ids):
        img_path = TRAIN_DIR / f"{img_id}.tif"
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"ID: {img_id[:8]}...", fontsize=9, color=PRIMARY_COLOR)
            ax.axis("off")
        except:
            ax.text(0.5, 0.5, "Error", ha="center", va="center", color="red")
            ax.axis("off")

    plt.suptitle(f"pHash: {row['pHash']} | {row['duplicate_count']} GÃ¶rsel",
                 fontsize=12, color=PRIMARY_COLOR, weight="bold")
    plt.tight_layout()
    plt.show()


display_section_title("ğŸ§¬ SÄ±nÄ±f-Ä°Ã§i Ã‡eÅŸitlilik Analizi â€” Ortalama & Varyans GÃ¶rselleri")

SAMPLE_PER_CLASS = 2000 
pos_ids = labels[labels["label"] == 1]["id"].sample(SAMPLE_PER_CLASS, random_state=42).tolist()
neg_ids = labels[labels["label"] == 0]["id"].sample(SAMPLE_PER_CLASS, random_state=42).tolist()

def compute_mean_std_image(id_list):
    """
    Verilen ID listesi iÃ§in ortalama (mean) ve standart sapma (std) gÃ¶rselini hesaplar.
    """
    imgs = []
    for img_id in tqdm(id_list, desc="Ä°ÅŸleniyor", leave=False):
        img_path = TRAIN_DIR / f"{img_id}.tif"
        try:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
            imgs.append(img)
        except Exception:
            continue
    imgs = np.stack(imgs)
    mean_img = np.mean(imgs, axis=0)
    std_img = np.std(imgs, axis=0)
    return mean_img, std_img

# ğŸ”¹ Pozitif ve Negatif iÃ§in ortalama/std gÃ¶rselleri oluÅŸtur
pos_mean, pos_std = compute_mean_std_image(pos_ids)
neg_mean, neg_std = compute_mean_std_image(neg_ids)


# ğŸ”¹ GÃ¶rselleÅŸtirme â€” Ortalama GÃ¶rseller
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor("white")

axes[0].imshow(np.clip(neg_mean / 255, 0, 1))
axes[0].set_title("Normal (Label=0) â€” Ortalama GÃ¶rsel", color="#555", fontsize=11, weight="bold")
axes[1].imshow(np.clip(pos_mean / 255, 0, 1))
axes[1].set_title("Cancer (Label=1) â€” Ortalama GÃ¶rsel", color=PRIMARY_COLOR, fontsize=11, weight="bold")

for ax in axes:
    ax.axis("off")

plt.suptitle("Ortalama (Mean) GÃ¶rseller â€” SÄ±nÄ±f BazÄ±nda", color=PRIMARY_COLOR, fontsize=14, weight="bold", y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ğŸ”¹ GÃ¶rselleÅŸtirme â€” Standart Sapma (Varyans) GÃ¶rselleri (Heatmap)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor("white")

axes[0].imshow(np.mean(neg_std, axis=2), cmap="magma")
axes[0].set_title("Normal (Label=0) â€” Std (Varyans) GÃ¶rseli", color="#555", fontsize=11, weight="bold")

axes[1].imshow(np.mean(pos_std, axis=2), cmap="magma")
axes[1].set_title("Cancer (Label=1) â€” Std (Varyans) GÃ¶rseli", color=PRIMARY_COLOR, fontsize=11, weight="bold")

for ax in axes:
    ax.axis("off")

plt.suptitle("SÄ±nÄ±f BazÄ±nda GÃ¶rsel Varyans (Std) HaritalarÄ±", color=PRIMARY_COLOR, fontsize=14, weight="bold", y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ğŸ”¹ Ortalama varyans farkÄ±nÄ± nicel olarak Ã¶lÃ§
pos_var_mean = np.mean(pos_std)
neg_var_mean = np.mean(neg_std)
diff_ratio = (pos_var_mean - neg_var_mean) / neg_var_mean * 100

display_section_title("ğŸ“ˆ SÄ±nÄ±f Ä°Ã§i Varyans (Std) KarÅŸÄ±laÅŸtÄ±rmasÄ±")

var_df = pd.DataFrame({
    "SÄ±nÄ±f": ["Normal (0)", "Cancer (1)"],
    "Ortalama Piksel Std": [round(neg_var_mean, 2), round(pos_var_mean, 2)]
})
display_dataframe(var_df, title="SÄ±nÄ±f Ä°Ã§i Varyans KarÅŸÄ±laÅŸtÄ±rmasÄ±")

print(f"ğŸ”� Cancer sÄ±nÄ±fÄ±, Normal sÄ±nÄ±fa gÃ¶re ortalama piksel varyansÄ±nda %{diff_ratio:.2f} daha {'yÃ¼ksek' if diff_ratio > 0 else 'dÃ¼ÅŸÃ¼k'} Ã§eÅŸitlilik gÃ¶steriyor.")


"""
===============================================================================
[STEP 14] DOWNSAMPLED DATASET OLUÅ�TURMA (FÄ°ZÄ°KSEL AYIRMA)
-------------------------------------------------------------------------------
AmaÃ§:
- SÄ±nÄ±f dengesizliÄŸini downsampling ile gidermek
- Majority class (0) azaltÄ±larak minority class (1) ile eÅŸitlemek
- Sadece downsample edilmiÅŸ gÃ¶rselleri yeni bir train klasÃ¶rÃ¼ne almak
- EÄŸitimde bu klasÃ¶rÃ¼ kullanmak
===============================================================================
"""

# Downsample hedef sayÄ±sÄ± (azÄ±nlÄ±k sÄ±nÄ±f referans)
target_count = labels["label"].value_counts().min()

# SÄ±nÄ±flarÄ± ayÄ±r
labels_0 = labels[labels["label"] == 0]
labels_1 = labels[labels["label"] == 1]

# Majority class downsample
labels_0_down = labels_0.sample(
    n=target_count,
    random_state=42
)

# Minority class (aynÄ± sayÄ±da tutulur)
labels_1_keep = labels_1.sample(
    n=target_count,
    random_state=42
)

# BirleÅŸtir ve karÄ±ÅŸtÄ±r
labels_downsampled = (
    pd.concat([labels_0_down, labels_1_keep], ignore_index=True)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

# Kontrol
print("Downsample sonrasÄ± sÄ±nÄ±f daÄŸÄ±lÄ±mÄ±:")
print(labels_downsampled["label"].value_counts())
print("Toplam Ã¶rnek:", len(labels_downsampled))


# Downsample CSV kaydÄ±
DOWNSAMPLED_CSV = Path("/kaggle/working/train_labels_downsampled.csv")
labels_downsampled.to_csv(DOWNSAMPLED_CSV, index=False)
print("ğŸ“„ Downsample CSV kaydedildi:", DOWNSAMPLED_CSV)


# --- FÄ°ZÄ°KSEL DATASET OLUÅ�TURMA ---
OUTPUT_DIR_DS = Path("/kaggle/working/dataset_downsampled/train")
(OUTPUT_DIR_DS / "0").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR_DS / "1").mkdir(parents=True, exist_ok=True)

for img_id, label in tqdm(
    labels_downsampled[["id", "label"]].itertuples(index=False),
    total=len(labels_downsampled),
    desc="Downsample gÃ¶rseller yerleÅŸtiriliyor"
):
    src = TRAIN_DIR / f"{img_id}.tif"
    dst = OUTPUT_DIR_DS / str(label) / f"{img_id}.tif"

    if src.exists() and not dst.exists():
        try:
            os.symlink(src, dst)   # hÄ±zlÄ± + disk dostu
        except Exception:
            shutil.copy2(src, dst) # fallback

# Son kontrol
count_0 = len(list((OUTPUT_DIR_DS / "0").glob("*.tif")))
count_1 = len(list((OUTPUT_DIR_DS / "1").glob("*.tif")))

print("ğŸ“Š Fiziksel dataset kontrolÃ¼:")
print(f"Class 0 (Normal): {count_0}")
print(f"Class 1 (Cancer): {count_1}")
print(f"Toplam: {count_0 + count_1}")


"""
===============================================================================
[STEP 15] TRAIN / VAL SPLIT (Stratified) + KlasÃ¶r YapÄ±sÄ±
-------------------------------------------------------------------------------
- labels_downsampled Ã¼zerinden %85/%15 stratified split
- train/val klasÃ¶rlerine sadece ilgili gÃ¶rselleri koyar (symlink -> copy fallback)
===============================================================================
"""

SPLIT_DIR = Path("/kaggle/working/dataset_downsampled_split")
TRAIN_OUT = SPLIT_DIR / "train"
VAL_OUT   = SPLIT_DIR / "val"

# klasÃ¶rleri hazÄ±rla
for base in [TRAIN_OUT, VAL_OUT]:
    (base / "0").mkdir(parents=True, exist_ok=True)
    (base / "1").mkdir(parents=True, exist_ok=True)

# --- stratified split ---
val_frac = 0.15
val_n_per_class = int(round(target_count * val_frac))   # her sÄ±nÄ±ftan %15
train_n_per_class = target_count - val_n_per_class      # kalan %85

# her sÄ±nÄ±ftan val seÃ§
val_0 = labels_0_down.sample(n=val_n_per_class, random_state=42)
val_1 = labels_1_keep.sample(n=val_n_per_class, random_state=42)

# train = geri kalan
train_0 = labels_0_down.drop(val_0.index)
train_1 = labels_1_keep.drop(val_1.index)

labels_train = (
    pd.concat([train_0, train_1], ignore_index=True)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)
labels_val = (
    pd.concat([val_0, val_1], ignore_index=True)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

print("Train daÄŸÄ±lÄ±mÄ±:\n", labels_train["label"].value_counts())
print("Val daÄŸÄ±lÄ±mÄ±:\n", labels_val["label"].value_counts())
print("Train toplam:", len(labels_train), "| Val toplam:", len(labels_val))

# --- dosyalarÄ± yerleÅŸtir (symlink -> copy fallback) ---
def place_split(df, out_dir, desc):
    for img_id, label in tqdm(
        df[["id", "label"]].itertuples(index=False),
        total=len(df),
        desc=desc
    ):
        src = TRAIN_DIR / f"{img_id}.tif"
        dst = out_dir / str(label) / f"{img_id}.tif"
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)

place_split(labels_train, TRAIN_OUT, "Train set yerleÅŸtiriliyor")
place_split(labels_val,   VAL_OUT,   "Val set yerleÅŸtiriliyor")

# --- son kontrol ---
print("âœ… Train/0:", len(list((TRAIN_OUT/"0").glob("*.tif"))))
print("âœ… Train/1:", len(list((TRAIN_OUT/"1").glob("*.tif"))))
print("âœ… Val/0:",   len(list((VAL_OUT/"0").glob("*.tif"))))
print("âœ… Val/1:",   len(list((VAL_OUT/"1").glob("*.tif"))))

# CSV olarak da kaydet (istersen eÄŸitimde CSV kullanÄ±rsÄ±n)
TRAIN_CSV = Path("/kaggle/working/train_labels_downsampled_train.csv")
VAL_CSV   = Path("/kaggle/working/train_labels_downsampled_val.csv")
labels_train.to_csv(TRAIN_CSV, index=False)
labels_val.to_csv(VAL_CSV, index=False)
print("ğŸ“„ Train CSV:", TRAIN_CSV)
print("ğŸ“„ Val CSV:", VAL_CSV)


# =============================================================================
# [STEP 16] CONFIG + PATH CHECK
# =============================================================================

DATA_ROOT = Path("/kaggle/working/dataset_downsampled_split")
TRAIN_PATH = DATA_ROOT / "train"
VAL_PATH   = DATA_ROOT / "val"

assert TRAIN_PATH.exists(), f"Train path yok: {TRAIN_PATH}"
assert VAL_PATH.exists(),   f"Val path yok: {VAL_PATH}"

print("âœ… TRAIN_PATH:", TRAIN_PATH)
print("âœ… VAL_PATH  :", VAL_PATH)
print("Train/0:", len(list((TRAIN_PATH/"0").glob("*.tif"))), " Train/1:", len(list((TRAIN_PATH/"1").glob("*.tif"))))
print("Val/0  :", len(list((VAL_PATH/"0").glob("*.tif"))),   " Val/1  :", len(list((VAL_PATH/"1").glob("*.tif"))))

BATCH_SIZE = 128   
NUM_WORKERS = 4      


# =============================================================================
# [STEP 17] DATALOADER (96Ã—96)
# =============================================================================
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

train_ds = ImageFolder(TRAIN_PATH, transform=train_tf)
val_ds   = ImageFolder(VAL_PATH,   transform=val_tf)

print("âœ… class_to_idx:", train_ds.class_to_idx)  # 0->"0", 1->"1" beklenir
assert train_ds.class_to_idx == val_ds.class_to_idx, "Train/Val class mapping farklÄ±!"

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("ğŸ§  device:", device)
print("Train size:", len(train_ds), "| Val size:", len(val_ds))


# =============================================================================
# [STEP 18] MODEL (EfficientNet-B0 Transfer Learning)
# =============================================================================
import torch.nn as nn
import torchvision.models as models

weights = "IMAGENET1K_V1"

model = models.efficientnet_b0(weights=weights)

# classifier: (dropout -> linear)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)  # binary logits

model = model.to(device)

def set_trainable(m, trainable: bool):
    for p in m.parameters():
        p.requires_grad = trainable

def freeze_backbone():
    set_trainable(model.features, False)
    set_trainable(model.classifier, True)

def unfreeze_last_blocks(n_last_blocks: int = 2):
    # Ã¶nce hepsini freeze
    set_trainable(model.features, False)
    # sonra son n block'u aÃ§
    total = len(model.features)
    for i in range(total - n_last_blocks, total):
        set_trainable(model.features[i], True)
    set_trainable(model.classifier, True)

freeze_backbone()
print("âœ… Backbone frozen, head trainable.")


import os
import json
import random
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch


# =============================================================================
# MODEL CONFIGURATIONS (MODEL-SPECIFIC & PROFESSIONAL)
# =============================================================================

MODEL_CONFIGS = {

    "efficientnet_b0": {
        # --- Identity ---
        "display_name": "EfficientNet-B0",
        "backbone_type": "CNN",

        # --- Training strategy ---
        "loss": "focal",            # Focal â†’ class balance + hard samples
        "focal_gamma": 2.0,

        "epochs_stage1": 2,
        "epochs_stage2": 8,

        "lr_stage1": 1e-3,
        "lr_stage2": 5e-4,
        "weight_decay": 1e-4,

        "unfreeze_last_blocks": 2,

        # --- Medical decision ---
        "threshold_mode": "f1",     # dengeli precision/recall
    },

    "convnext_tiny": {
        # --- Identity ---
        "display_name": "ConvNeXt-Tiny",
        "backbone_type": "Modern CNN",

        # --- Training strategy ---
        "loss": "bce",              # ConvNeXt zaten gÃ¼Ã§lÃ¼ â†’ sade loss
        "epochs_stage1": 2,
        "epochs_stage2": 10,

        "lr_stage1": 8e-4,
        "lr_stage2": 3e-4,
        "weight_decay": 5e-5,

        "unfreeze_last_blocks": 3,  # daha derin fine-tune

        # --- Medical decision ---
        "threshold_mode": "f1",
    },

    "swin_tiny": {
        # --- Identity ---
        "display_name": "Swin-Tiny",
        "backbone_type": "Transformer",

        # --- Training strategy ---
        "loss": "bce",
        "epochs_stage1": 3,
        "epochs_stage2": 12,

        "lr_stage1": 6e-4,
        "lr_stage2": 2e-4,
        "weight_decay": 1e-4,

        "unfreeze_last_blocks": 1,  # transformer â†’ az aÃ§
       
        # --- Medical decision ---
        "threshold_mode": "f1",  # â�— kanseri kaÃ§Ä±rma Ã¶ncelikli
    }
}



SEED = 42

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # performans > determinism (medical imaging iÃ§in daha mantÄ±klÄ±)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

seed_everything(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("ğŸ”¥ Device:", DEVICE)

def amp_context():
    if DEVICE.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()

SCALER = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))


# =============================================================================
# STEP 20.1 â€” LOSS FACTORY
# =============================================================================

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * bce_loss).mean()


def build_loss(cfg: dict):
    if cfg["loss"] == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif cfg["loss"] == "focal":
        gamma = cfg.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma)
    else:
        raise ValueError(f"Bilinmeyen loss tÃ¼rÃ¼: {cfg['loss']}")



# =============================================================================
# STEP 20.2 â€” METRIC HELPERS
# =============================================================================

from sklearn.metrics import roc_auc_score, average_precision_score

def sigmoid_np(x):
    x = np.clip(x, -50, 50)  # overflow fix
    return 1 / (1 + np.exp(-x))

def safe_roc_auc(y_true, probs):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))

def safe_pr_auc(y_true, probs):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, probs))



# =============================================================================
# STEP 20.3 â€” EPOCH RUNNER
# =============================================================================

def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    train: bool = True
):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.float().to(DEVICE).view(-1, 1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with amp_context():
            logits = model(x)
            loss = criterion(logits, y)

        if train:
            SCALER.scale(loss).backward()
            SCALER.step(optimizer)
            SCALER.update()

        total_loss += loss.item() * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy().reshape(-1)
    targets = torch.cat(all_targets).numpy().reshape(-1)

    probs = sigmoid_np(logits)
    preds = (probs >= 0.5).astype(int)

    acc = float((preds == targets).mean())
    loss_avg = total_loss / len(loader.dataset)

    return {
        "loss": loss_avg,
        "accuracy": acc,
        "roc_auc": safe_roc_auc(targets, probs),
        "pr_auc": safe_pr_auc(targets, probs),
        "probs": probs,
        "targets": targets,
    }


# =============================================================================
# STEP 20.4 â€” HISTORY CONTAINER
# =============================================================================

def init_history():
    return {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
    }


# =============================================================================
# STEP 21.1 â€” MODEL FACTORY
# =============================================================================
import torchvision.models as models
import torch.nn as nn

def build_model(model_key: str):
    model_key = model_key.lower()

    if model_key == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        backbone = model.features
        head = model.classifier

    elif model_key == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
        backbone = model.features
        head = model.classifier

    elif model_key == "swin_tiny":
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 1)
        backbone = model.features
        head = model.head

    else:
        raise ValueError(f"Desteklenmeyen model: {model_key}")

    return model.to(DEVICE), backbone, head


# =============================================================================
# STEP 21.2 â€” FREEZE / UNFREEZE
# =============================================================================

def set_trainable(module, trainable: bool):
    for p in module.parameters():
        p.requires_grad = trainable


def freeze_backbone(backbone, head):
    set_trainable(backbone, False)
    set_trainable(head, True)


def unfreeze_last_blocks(model_key, backbone, head, n_last_blocks: int):
    set_trainable(backbone, False)

    if model_key in ["efficientnet_b0", "convnext_tiny"]:
        total = len(backbone)
        for i in range(total - n_last_blocks, total):
            set_trainable(backbone[i], True)

    elif model_key == "swin_tiny":
        # Swin'de son stage yeterli
        set_trainable(backbone[-1], True)

    set_trainable(head, True)


# =============================================================================
# STEP 21.3 â€” TRAIN MODEL (FULL PIPELINE)
# =============================================================================

from torch.optim import AdamW

def train_model(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    display_name = cfg["display_name"]

    print(f"\nğŸš€ TRAINING STARTED: {display_name}")

    model, backbone, head = build_model(model_key)
    criterion = build_loss(cfg)

    history = init_history()
    best_auc = -1.0
    best_state = None
    best_val_outputs = None

    # =========================
    # STAGE 1 â€” HEAD ONLY
    # =========================
    freeze_backbone(backbone, head)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_stage1"],
        weight_decay=cfg["weight_decay"]
    )

    for epoch in range(1, cfg["epochs_stage1"] + 1):
        train_out = run_one_epoch(model, train_loader, criterion, optimizer, train=True)
        val_out   = run_one_epoch(model, val_loader,   criterion, optimizer=None, train=False)

        history["epoch"].append(f"S1-{epoch}")
        history["train_loss"].append(train_out["loss"])
        history["train_acc"].append(train_out["accuracy"])
        history["val_loss"].append(val_out["loss"])
        history["val_acc"].append(val_out["accuracy"])
        history["val_roc_auc"].append(val_out["roc_auc"])
        history["val_pr_auc"].append(val_out["pr_auc"])

        print(
            f"[S1][{epoch}] "
            f"TL {train_out['loss']:.4f} | "
            f"VL {val_out['loss']:.4f} | "
            f"AUC {val_out['roc_auc']:.4f}"
        )

        if val_out["roc_auc"] > best_auc:
            best_auc = val_out["roc_auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_outputs = val_out

    # =========================
    # STAGE 2 â€” FINE TUNE
    # =========================
    unfreeze_last_blocks(
        model_key,
        backbone,
        head,
        cfg["unfreeze_last_blocks"]
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_stage2"],
        weight_decay=cfg["weight_decay"]
    )

    for epoch in range(1, cfg["epochs_stage2"] + 1):
        train_out = run_one_epoch(model, train_loader, criterion, optimizer, train=True)
        val_out   = run_one_epoch(model, val_loader,   criterion, optimizer=None, train=False)

        history["epoch"].append(f"S2-{epoch}")
        history["train_loss"].append(train_out["loss"])
        history["train_acc"].append(train_out["accuracy"])
        history["val_loss"].append(val_out["loss"])
        history["val_acc"].append(val_out["accuracy"])
        history["val_roc_auc"].append(val_out["roc_auc"])
        history["val_pr_auc"].append(val_out["pr_auc"])

        print(
            f"[S2][{epoch}] "
            f"TL {train_out['loss']:.4f} | "
            f"VL {val_out['loss']:.4f} | "
            f"AUC {val_out['roc_auc']:.4f}"
        )

        if val_out["roc_auc"] > best_auc:
            best_auc = val_out["roc_auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_outputs = val_out

    print(f"ğŸ�� {display_name} finished | BEST VAL AUC = {best_auc:.4f}")

    return {
        "model_key": model_key,
        "display_name": display_name,
        "history": history,
        "best_auc": best_auc,
        "best_val_outputs": best_val_outputs,
        "config": cfg,
    }



# =============================================================================
# STEP 22.1 â€” METRIC ENGINE
# =============================================================================
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, precision_recall_curve
)

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def compute_epoch_metrics(logits, targets):
    probs = sigmoid_np(logits)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": (preds == targets).mean(),
        "roc_auc": roc_auc_score(targets, probs),
        "pr_auc": average_precision_score(targets, probs),
        "probs": probs,
        "targets": targets
    }



# =============================================================================
# STEP 22.2 â€” THRESHOLD OPTIMIZATION
# =============================================================================
def find_best_threshold(y_true, probs, mode="f1"):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1

    for t in thresholds:
        preds = (probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, preds, labels=[0,1], zero_division=0
        )

        if mode == "recall":
            score = r[1] + 0.01 * p[1]
        else:  # f1
            score = f1[1]

        if score > best_score:
            best_score = score
            best_t = t

    return best_t



# =============================================================================
# STEP 22.3 â€” FINAL EVALUATION
# =============================================================================
def evaluate_best_epoch(best_val_outputs, threshold_mode):
    probs = best_val_outputs["probs"]
    targets = best_val_outputs["targets"].astype(int)

    thr = find_best_threshold(targets, probs, mode=threshold_mode)
    preds = (probs >= thr).astype(int)

    report = classification_report(
        targets, preds,
        labels=[0,1],
        target_names=["Normal", "Cancer"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(targets, preds)

    return {
        "threshold": thr,
        "confusion_matrix": cm,
        "report": report,
        "roc_auc": roc_auc_score(targets, probs),
        "pr_auc": average_precision_score(targets, probs),
        "precision_cancer": report["Cancer"]["precision"],
        "recall_cancer": report["Cancer"]["recall"],
        "f1_cancer": report["Cancer"]["f1-score"],
        "accuracy": report["accuracy"],
        "probs": probs,
        "targets": targets
    }


# =============================================================================
# STEP 22.4 â€” VISUALIZATION
# =============================================================================
def plot_training_curves(history, model_name):
    epochs = range(len(history["train_loss"]))

    plt.figure(figsize=(16,5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} | Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(16,5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title(f"{model_name} | Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_roc_pr(outputs, model_name):
    y = outputs["targets"]
    p = outputs["probs"]

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y,p):.4f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.title(f"{model_name} | ROC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"PR-AUC={average_precision_score(y,p):.4f}")
    plt.title(f"{model_name} | PR Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_confusion(cm, model_name, thr):
    plt.figure(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} | Confusion (thr={thr:.2f})")
    plt.show()



# =============================================================================
# STEP 22.5 â€” RUN ALL MODELS
# =============================================================================
ALL_RESULTS = []

for model_key in MODEL_CONFIGS.keys():
    out = train_model(model_key)

    final_metrics = evaluate_best_epoch(
        out["best_val_outputs"],
        out["config"]["threshold_mode"]
    )

    plot_training_curves(out["history"], out["display_name"])
    plot_roc_pr(final_metrics, out["display_name"])
    plot_confusion(
        final_metrics["confusion_matrix"],
        out["display_name"],
        final_metrics["threshold"]
    )

    ALL_RESULTS.append({
        "Model": out["display_name"],
        "Backbone": out["config"]["backbone_type"],
        "ROC-AUC": final_metrics["roc_auc"],
        "PR-AUC": final_metrics["pr_auc"],
        "Accuracy": final_metrics["accuracy"],
        "Precision (Cancer)": final_metrics["precision_cancer"],
        "Recall (Cancer)": final_metrics["recall_cancer"],
        "F1 (Cancer)": final_metrics["f1_cancer"],
        "Threshold": final_metrics["threshold"]
    })



# =============================================================================
# STEP 22.6 â€” FINAL COMPARISON
# =============================================================================
df_final = pd.DataFrame(ALL_RESULTS)

weights = {
    "ROC-AUC": 0.35,
    "Recall (Cancer)": 0.30,
    "F1 (Cancer)": 0.20,
    "Precision (Cancer)": 0.10, 
    "Accuracy": 0.05
}

df_final["Overall Score"] = sum(
    df_final[k] * w for k, w in weights.items()
)

df_final = df_final.sort_values("Overall Score", ascending=False)

display(df_final)




