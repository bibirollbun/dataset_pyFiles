import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
print("Shape:", df.shape)


df.info()


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



with open('/kaggle/input/DontGetKicked/Carvana_Data_Dictionary.txt', 'r') as f:
    dictionary_text = f.read()

print(dictionary_text[:1000])



!pip install ydata_profiling


!pip install ipywidgets==7.7.1
!jupyter nbextension enable --py widgetsnbextension



from ydata_profiling import ProfileReport

# Ø§ÛŒØ¬Ø§Ø¯ Ú¯Ø²Ø§Ø±Ø´ Ø¨Ø§ Ù…Ø¹Ø±Ù�ÛŒ Ù…ØªØºÛŒØ± Ù‡Ø¯Ù�
profile = ProfileReport(
    df,
    title="DontGetKicked Training Data EDA",
    explorative=True,
    type_schema={
        "IsBadBuy": "categorical"  # Ù…Ø¹Ø±Ù�ÛŒ Ù�ÛŒÙ„Ø¯ Ù‡Ø¯Ù�
    },
    sensitive=False  # ØºÛŒØ±Ø¶Ø±ÙˆØ±ÛŒ ÙˆÙ„ÛŒ Ø®ÙˆØ¨Ù‡ Ø¨Ø±Ø§ÛŒ Ú¯Ø²Ø§Ø±Ø´ Ø³Ø§Ø¯Ù‡â€ŒØªØ±
)

# Ø°Ø®ÛŒØ±Ù‡â€ŒÛŒ Ú¯Ø²Ø§Ø±Ø´ Ø¨Ù‡ ØµÙˆØ±Øª Ù�Ø§ÛŒÙ„ HTML
profile.to_file("dontgetkicked_training_profile_report.html")



from ydata_profiling import ProfileReport

# Ø¬Ø¯Ø§ Ú©Ø±Ø¯Ù† Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø¨Ø± Ø§Ø³Ø§Ø³ Ù…Ù‚Ø¯Ø§Ø± Ù‡Ø¯Ù�
df_target_0 = df[df["IsBadBuy"] == 0]   # ÛŒØ§ "0" Ø§Ú¯Ø± Ø±Ø´ØªÙ‡ Ø§Ø³Øª
df_target_1 = df[df["IsBadBuy"] == 1]   # ÛŒØ§ "1"

# ØªÙˆÙ„ÛŒØ¯ Ú¯Ø²Ø§Ø±Ø´ Ù¾Ø±ÙˆÙ�Ø§ÛŒÙ„ Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ú¯Ø±ÙˆÙ‡
profile_0 = ProfileReport(
    df_target_0,
    title="EDA - IsBadBuy = 0",
    minimal=True,
    type_schema={"IsBadBuy": "categorical"}
)

profile_1 = ProfileReport(
    df_target_1,
    title="EDA - IsBadBuy = 1",
    minimal=True,
    type_schema={"IsBadBuy": "categorical"}
)

# Ù…Ù‚Ø§ÛŒØ³Ù‡â€ŒÛŒ Ø¯Ùˆ Ú¯Ø²Ø§Ø±Ø´ Ø¨Ø±Ø§ÛŒ Ø¯ÛŒØ¯Ù† ØªÙ�Ø§ÙˆØªâ€ŒÙ‡Ø§ Ø¨ÛŒÙ† Ú©Ù„Ø§Ø³ 0 Ùˆ 1
comparison_report = profile_0.compare(profile_1)

# Ø°Ø®ÛŒØ±Ù‡ Ø¯Ø± Ù�Ø§ÛŒÙ„ HTML
comparison_report.to_file("comparison_IsBadBuy.html")



from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Carvana Training Data EDA", minimal=False)
profile.to_file("Carvana_training_EDA.html")



# Robust EDA + plots embedded into single HTML (heatmap + scatter plots + ydata_profiling)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from pathlib import Path
import base64
import io


target_col = "IsBadBuy"        # <- set your actual target column name here
output_html = "full_EDA_with_plots.html"

if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataframe. Available columns: {list(df.columns[:50])}")

# ---------- 2. Create ydata_profiling report (with target) ----------
profile = ProfileReport(
    df,
    title="EDA Report (with target)",
    explorative=True,
    type_schema={target_col: "categorical"}  # ensure profiler treats target as categorical
)
# save intermediate profile HTML (we'll embed it)
profile_path = Path("ydata_profile.html")
profile.to_file(profile_path)

# ---------- 3. Numeric heatmap ----------
numeric_df = df.select_dtypes(include=[np.number])
# ensure target included if numeric (we keep it)
# compute correlation matrix (Spearman is more robust for non-linear monotonic)
corr = numeric_df.corr(method="spearman")

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="vlag", center=0, square=False, linewidths=.25, cbar_kws={"shrink":.6})
plt.title("Spearman Correlation Heatmap (numeric features)")
plt.tight_layout()
heatmap_path = Path("heatmap.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()

# ---------- 4. Scatter plots of important features vs target ----------
# Choose a short list of important numeric columns to plot (price-related + odo)
candidates = [
    "MMRCurrentRetailCleanPrice",
    "MMRCurrentAuctionCleanPrice",
    "MMRAcquisitionRetailCleanPrice",
    "MMRAcquisitionAuctionCleanPrice",
    "VehOdo",
    "VehBCost",
    "WarrantyCost"
]
# Filter existing columns
plot_cols = [c for c in candidates if c in df.columns]

scatter_paths = []
for col in plot_cols:
    plt.figure(figsize=(8,4))
    # jitter target for visibility if it's binary
    if set(df[target_col].dropna().unique()) <= {0,1}:
        y = df[target_col] + np.random.normal(0, 0.02, size=len(df))  # small jitter
        sns.scatterplot(x=df[col], y=y, alpha=0.25, s=10)
        plt.ylabel(target_col + " (jittered)")
    else:
        sns.scatterplot(x=df[col], y=df[target_col], alpha=0.6, s=10)
    plt.xlabel(col)
    plt.title(f"{col} vs {target_col}")
    plt.tight_layout()
    p = Path(f"scatter_{col}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    scatter_paths.append(p)

# ---------- 5. Helper to embed image as base64 ---------- 
def img_to_base64_str(img_path: Path):
    with img_path.open("rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    mime = "image/png"
    return f"data:{mime};base64,{encoded}"

# ---------- 6. Build final single HTML by embedding profile + images ----------
profile_html = profile_path.read_text(encoding="utf-8")

# We will insert our extra visualizations before </body>
extra_parts = []
extra_parts.append("<h1>Additional Visualizations</h1>")

# embed heatmap
heatmap_b64 = img_to_base64_str(heatmap_path)
extra_parts.append("<h2>Correlation Heatmap (Spearman)</h2>")
extra_parts.append(f"<img src='{heatmap_b64}' style='max-width:100%;height:auto;border:1px solid #ddd;'>")

# embed scatter plots
extra_parts.append("<h2>Scatter plots vs Target</h2>")
for p in scatter_paths:
    b64 = img_to_base64_str(p)
    extra_parts.append(f"<h3>{p.stem.replace('scatter_','')}</h3>")
    extra_parts.append(f"<img src='{b64}' style='max-width:100%;height:auto;border:1px solid #ddd;margin-bottom:20px;'>")

insertion_html = "\n".join(extra_parts)

# Insert before closing </body> tag (robust)
if "</body>" in profile_html:
    final_html = profile_html.replace("</body>", insertion_html + "\n</body>")
else:
    final_html = profile_html + insertion_html

# Save final combined HTML
Path(output_html).write_text(final_html, encoding="utf-8")
print("âœ… Done. Output HTML:", output_html)


