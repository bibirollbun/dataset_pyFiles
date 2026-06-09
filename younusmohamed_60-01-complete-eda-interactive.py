import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import seaborn as sns

import gc, os, random, warnings

plt.style.use("ggplot")
sns.set(font_scale=1.1)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


PATH = "/kaggle/input/playground-series-s5e6"
train = pd.read_csv(f"{PATH}/train.csv")
test  = pd.read_csv(f"{PATH}/test.csv")

print("Train shape :", train.shape)
print("Test  shape :", test.shape)
display(train.head())


display(train.info())
display(train.describe(include="all").T.sort_index())


plt.figure(figsize=(10,3))
sns.heatmap(train.isna(), cbar=False)
plt.title("Missing‐value pattern"); plt.show()


target_col = "Fertilizer Name"
vc = train[target_col].value_counts().sort_values(ascending=False)
vc_pct = vc / len(train) * 100

fig, ax = plt.subplots(1,2,figsize=(14,4))
vc.plot.bar(ax=ax[0])
ax[0].set_title("Absolute counts"); ax[0].set_ylabel("records")

vc_pct.plot.bar(ax=ax[1])
ax[1].set_title("Percentage share"); ax[1].set_ylabel("% of train")
plt.suptitle("Target distribution"); plt.show()


cat_cols = ["Soil Type", "Crop Type"]
for col in cat_cols:
    plt.figure(figsize=(8,3))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index)
    plt.title(f"{col} distribution"); plt.xticks(rotation=45); plt.show()


num_cols = train.select_dtypes(include="number").columns.drop("id")
ncols = 3
nrows = int(np.ceil(len(num_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    sns.histplot(train[col], kde=True, ax=ax)
    ax.set_title(col)

plt.tight_layout(); plt.show()


for col in num_cols:
    plt.figure(figsize=(14,6))
    sns.boxplot(
        data=train,
        x=target_col, y=col,
        order=vc.index
    )
    plt.title(f"{col} by Fertilizer")
    plt.xticks(rotation=45)
    plt.show()


corr = train[num_cols].corr(method="spearman")
plt.figure(figsize=(9,7))
sns.heatmap(corr, cmap="coolwarm", annot=True, square=True)
plt.title("Spearman correlation")
plt.show()


sns.pairplot(
    train[num_cols.union([target_col])],
    hue=target_col, corner=True, diag_kind="kde",
    height=1.5, plot_kws=dict(alpha=.3, linewidth=0)
)
plt.suptitle("Pairwise relationships (sample)", y=1.02)
plt.show()


ct = pd.crosstab(
    [train["Soil Type"], train["Crop Type"]],
    train[target_col],
    normalize="index"
)
plt.figure(figsize=(14,6))
sns.heatmap(ct, cmap="YlGnBu", linewidths=.3)
plt.title("Fertilizer share by (Soil, Crop)")
plt.ylabel("(Soil, Crop)")
plt.show()


pio.renderers.default = "kaggle"


# for col in num_cols:
#     fig = px.histogram(train, x=col, nbins=50, color=target_col,
#                        title=f"{col} distribution by fertilizer", opacity=0.7)
#     fig.update_layout(bargap=0.05)
#     fig.show()


# fig = px.scatter_matrix(
#     train, dimensions=num_cols, color=target_col,
#     title="Scatter-matrix", height=900, width=900
# )
# fig.update_traces(diagonal_visible=False, showupperhalf=False)
# fig.show()


# fig = px.sunburst(
#     train,
#     path=[target_col, "Soil Type", "Crop Type"],
#     title="Nested proportion of Fertilizer / Soil / Crop",
# )
# fig.show()


# fig = ff.create_annotated_heatmap(
#     z=np.around(corr.values,3),
#     x=corr.columns.tolist(), y=corr.index.tolist(),
#     colorscale="Viridis", showscale=True, hoverinfo="z"
# )
# fig.update_layout(title_text="Spearman correlation (interactive)")
# fig.show()


# fig = px.scatter_3d(
#     train, x="Temparature", y="Humidity", z="Moisture",
#     color=target_col, opacity=0.5,
#     title="3-D micro-climate vs. fertilizer (sample)"
# )
# fig.show()




