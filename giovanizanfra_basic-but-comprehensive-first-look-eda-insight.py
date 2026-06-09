import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

NUMERICAL_COLS = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]
CATEGORICAL_COLS = ["Soil Type", "Crop Type"]
TARGET_COL = "Fertilizer Name"
train.head()


missing_counts = train.isnull().sum()
print(missing_counts)


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(NUMERICAL_COLS):
    axes[idx].boxplot(train[col].dropna(), vert=True, patch_artist=True)
    axes[idx].set_title(f"{col} Boxplot")
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(36, 20))
axes = axes.flatten()

for idx, col in enumerate(NUMERICAL_COLS):
    train[col].dropna().plot.kde(ax=axes[idx], label="Train", linestyle="-")
    test[col].dropna().plot.kde(ax=axes[idx], label="Test", linestyle="--")

    axes[idx].set_title(f"{col} Distribution")
    axes[idx].set_xlabel(col)
    axes[idx].legend()

plt.tight_layout()
plt.show()


num_plots = len(CATEGORICAL_COLS) + 1
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))

for idx, col in enumerate(CATEGORICAL_COLS):
    train_counts = train[col].value_counts().sort_index()
    test_counts = test[col].value_counts().sort_index()

    counts_df = pd.DataFrame({"Train": train_counts, "Test": test_counts}).fillna(0)

    counts_df.plot(kind="bar", ax=axes[idx])
    axes[idx].set_title(f"{col} count (Train vs Test)")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Count")
    axes[idx].legend()

train_fert_counts = train[TARGET_COL].value_counts().sort_index()

train_fert_counts.plot(kind="bar", ax=axes[-1], color="skyblue")
axes[-1].set_title(f"{TARGET_COL} Distribution Train")
axes[-1].set_xlabel(TARGET_COL)
axes[-1].set_ylabel("Count")

plt.tight_layout()
plt.show()

fert_balance = train_fert_counts.to_frame(name="count")
fert_balance["percent"] = (fert_balance["count"] / len(train) * 100).round(2)
print("Fertilizer Name Distribution (Train):")
print(fert_balance.head(10)) 


grouped_stats = train.groupby(TARGET_COL)[NUMERICAL_COLS].agg(["mean", "std"])
grouped_stats.columns = ["_".join(col) for col in grouped_stats.columns]

grouped_stats


crosstab_soil = pd.crosstab(train["Soil Type"], train["Fertilizer Name"])
crosstab_soil_norm_row = crosstab_soil.div(crosstab_soil.sum(axis=1), axis=0)
crosstab_soil_norm_col = crosstab_soil.div(crosstab_soil.sum(axis=0), axis=1)
crosstab_crop = pd.crosstab(train["Crop Type"], train["Fertilizer Name"])
crosstab_crop_norm_row = crosstab_crop.div(crosstab_crop.sum(axis=1), axis=0)
crosstab_crop_norm_col = crosstab_crop.div(crosstab_crop.sum(axis=0), axis=1)

print("Crosstab Soil Type x Fertilizer Name (Normalized by Feature - Rows):")
display(crosstab_soil_norm_row.round(3))

print("\nCrosstab Soil Type x Fertilizer Name (Normalized by Target - Columns):")
display(crosstab_soil_norm_col.round(3))

print("\nCrosstab Crop Type x Fertilizer Name (Normalized by Feature - Rows):")
display(crosstab_crop_norm_row.round(3))

print("\nCrosstab Crop Type x Fertilizer Name (Normalized by Target - Columns):")
display(crosstab_crop_norm_col.round(3))


train_numeric = train[NUMERICAL_COLS]

corr_matrix = train_numeric.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix between numerical features")
plt.tight_layout()
plt.show()

