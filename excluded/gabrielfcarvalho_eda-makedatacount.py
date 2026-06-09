from pathlib import Path
import re, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, PyPDF2
from tqdm.auto import tqdm

sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", None)

DATA_DIR      = Path("/kaggle/input/make-data-count-finding-data-references")
TRAIN_PDF_DIR = DATA_DIR / "train" / "PDF"
TRAIN_XML_DIR = DATA_DIR / "train" / "XML"
TEST_PDF_DIR  = DATA_DIR / "test"  / "PDF"
TEST_XML_DIR  = DATA_DIR / "test"  / "XML"


def describe_files(folder: Path, suffix: str):
    files = sorted(folder.glob(f"*{suffix}"))
    sizes = [f.stat().st_size for f in files]
    return {
        "num_files": len(files),
        "median_size_KB": int(np.median(sizes) / 1024) if sizes else 0,
        "min_size_KB": int(np.min(sizes) / 1024) if sizes else 0,
        "max_size_KB": int(np.max(sizes) / 1024) if sizes else 0,
    }

layout_df = pd.DataFrame({
    "train_PDF": describe_files(TRAIN_PDF_DIR, ".pdf"),
    "train_XML": describe_files(TRAIN_XML_DIR, ".xml"),
    "test_PDF" : describe_files(TEST_PDF_DIR , ".pdf"),
    "test_XML" : describe_files(TEST_XML_DIR , ".xml"),
}).T
layout_df.style


labels = pd.read_csv(DATA_DIR / "train_labels.csv")

label_counts = labels["type"].value_counts().rename_axis("class").reset_index(name="count")
display(label_counts)

sns.barplot(data=label_counts, x="class", y="count")
plt.title("Class distribution (train)")
plt.tight_layout()
plt.show()

per_paper = labels.groupby("article_id")["type"].count()
print(per_paper.describe())


doi_mask = labels.dataset_id.str.startswith("https://doi", na=False)
accession_mask = ~doi_mask & labels.dataset_id.str.contains(r'\d', na=False)

taxonomy = pd.Series({
    "DOI": doi_mask.sum(),
    "Accession / misc.": accession_mask.sum(),
    "Missing label": (labels.type == "Missing").sum()
})
taxonomy

plt.pie(taxonomy, labels=taxonomy.index, autopct="%1.0f%%", counterclock=False)
plt.title("Which ID types appear in labels?")
plt.tight_layout()
plt.show()


pdf_set = {p.stem for p in TRAIN_PDF_DIR.glob("*.pdf")}
xml_set = {x.stem for x in TRAIN_XML_DIR.glob("*.xml")}

avail = (
    pd.DataFrame({"article_id": sorted(pdf_set | xml_set)})
      .assign(PDF=lambda d: d.article_id.isin(pdf_set).astype(int),
              XML=lambda d: d.article_id.isin(xml_set).astype(int))
      .set_index("article_id")
)

# 1ï¸�âƒ£ sort columns: those without XML first
avail = avail.sort_values("XML")

matrix = avail.T.values  # 2 x N numeric

fig, ax = plt.subplots(figsize=(12, 3))
ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

# 2ï¸�âƒ£ xâ€‘ticks every 25 articles (no labels)
step = 25
ax.set_xticks(np.arange(0, matrix.shape[1], step))
ax.set_xticklabels([f"{i}" for i in range(0, matrix.shape[1], step)])
ax.set_xlabel("Articles (sorted â€“ missing XML at left)")

# yâ€‘labels
ax.set_yticks([0,1])
ax.set_yticklabels(["PDF", "XML"])

# 4ï¸�âƒ£ annotation
xml_cnt = avail["XML"].sum()
ax.text(0.99, 1.05, f"{xml_cnt} / {avail.shape[0]} train articles have XML",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=9)

ax.set_title("File availability heatâ€‘map  (blueÂ =Â file present)")
plt.tight_layout()
plt.show()



SAMPLE = 3
sample_ids = np.random.choice(sorted(pdf_set), SAMPLE, replace=False)
for aid in sample_ids:
    pdf_path = TRAIN_PDF_DIR / f"{aid}.pdf"
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f, strict=False)
        pages = len(reader.pages)
        snippet = (reader.pages[0].extract_text() or "").replace("\n", " ")[:300]
    print(f"\nâ€” {aid}.pdf ({pages}â€¯pages) â€”")
    print(snippet, "â€¦")

    xml_path = TRAIN_XML_DIR / f"{aid}.xml"
    if xml_path.exists():
        tags = [el.tag for el in ET.parse(xml_path).getroot().iter()][0:12]
        print("XML tags:", tags)
    else:
        print("XML: â�Œ  missing")


def pdf_word_count(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f, strict=False)
            return sum(len((p.extract_text() or "").split()) for p in reader.pages)
    except Exception:
        return np.nan

wc_series = pd.Series({
    aid: pdf_word_count(TRAIN_PDF_DIR / f"{aid}.pdf")
    for aid in tqdm(pdf_set, desc="wordâ€‘count")
}).dropna()

# keep only articles that also have labels
common_idx = wc_series.index.intersection(per_paper.index)
sns.scatterplot(x=wc_series.loc[common_idx],
                y=per_paper.loc[common_idx])
plt.xscale("log")
plt.xlabel("Word count (log scale)")
plt.ylabel("#labels in paper")
plt.title("Longer papers â†” more dataset mentions")
plt.tight_layout()
plt.show()



# compute Pearson r (logâ€‘wordâ€‘count vs label count)
from scipy.stats import pearsonr

log_wc = np.log10(wc_series.loc[common_idx])
lab_cnt = per_paper.loc[common_idx]

r, p = pearsonr(log_wc, lab_cnt)
print(f"Pearson r = {r:.2f}  (p = {p:.2g})")


from scipy.stats import spearmanr

# (a) Pearson without log
r_lin, p_lin = pearsonr(wc_series.loc[common_idx], lab_cnt)
print(f"Pearson raw  = {r_lin:.2f}  (p={p_lin:.2g})")

# (b) Rank correlation â€“ more robust to outliers / nonâ€‘linearity
rho, p_rho = spearmanr(wc_series.loc[common_idx], lab_cnt)
print(f"Spearman rho = {rho:.2f}  (p={p_rho:.2g})")


sample = labels.sample(1).iloc[0]
aid, dsid = sample.article_id, sample.dataset_id
pdf_path = TRAIN_PDF_DIR / f"{aid}.pdf"

with open(pdf_path, "rb") as f:
    full = " ".join((p.extract_text() or "") for p in PyPDF2.PdfReader(f, strict=False).pages)
hit = re.search(re.escape(dsid), full, re.I)
print(f"{dsid} found?" , bool(hit))


phrases = [
    r"data (?:are|is) available (?:at|from)",
    r"deposited (?:at|in)",
    r"accession (?:number|code|id)",
    r"downloaded from",
    r"publicly available"
]

counts = {}
for pat in phrases:
    regex = re.compile(pat, re.I)
    hit_sum = 0
    for pdf_path in TRAIN_PDF_DIR.glob("*.pdf"):
        try:
            with open(pdf_path, "rb") as f:
                first_page = PyPDF2.PdfReader(f, strict=False).pages[0]
                txt = (first_page.extract_text() or "")[:8000]  # 1stÂ page slice
            if regex.search(txt):
                hit_sum += 1
        except Exception:
            pass
    counts[pat] = hit_sum

(pd.Series(counts)
   .sort_values()
   .plot.barh(figsize=(6,3), color="steelblue"))
plt.title("How many PDFs contain the phrase (first page)")
plt.tight_layout()
plt.show()

