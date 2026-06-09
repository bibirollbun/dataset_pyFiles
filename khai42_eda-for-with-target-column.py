import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import cudf


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


print('Train Shape: ', train.shape)
train.head(3)


print('Test Shape: ', test.shape)
test.head(3)


print('Train Null Values: ',train.isnull().sum().sum())
print('Test Null Values: ',test.isnull().sum().sum())


train.dtypes


train_no_id = train.drop(columns=["id", 'accident_risk'])

numerical_cols = train_no_id.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_no_id.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numerical Columns:")
print(numerical_cols)
print("\nCategorical Columns:")
print(categorical_cols)


n_num = len(numerical_cols)
rows_num = (n_num + 2) // 3
plt.figure(figsize=(15, 4*rows_num))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(rows_num, 2, i)
    plt.hist(train_no_id[col].dropna().values)
    plt.title(f"{col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 12))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols)//3 + 1, 3, i)
    train_no_id[col].value_counts(dropna=False).plot(kind="bar")
    plt.title(col)
    plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


train_no_id = train.drop(columns=["id"])
numerical_cols = train_no_id.select_dtypes(include=["number"]).columns.tolist()
numerical_cols.remove("accident_risk")
for col in numerical_cols:
    grouped = train_no_id.groupby(col)["accident_risk"].mean()
    plt.figure(figsize=(8,4))
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.title(f"Accident Risk vs {col}")
    plt.xlabel(col)
    plt.ylabel("Average Accident Risk")
    plt.tight_layout()
    plt.show()


train_no_id = train.drop(columns=["id"])
plt.figure(figsize=(6,4))
lighting_grouped = train_no_id.groupby("lighting")["accident_risk"].mean().sort_values()
lighting_grouped.plot(kind="bar", color="skyblue")
plt.title("Accident Risk vs Lighting")
plt.xlabel("Lighting Condition")
plt.ylabel("Average Accident Risk")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
weather_grouped = train_no_id.groupby("weather")["accident_risk"].mean().sort_values()
weather_grouped.plot(kind="bar", color="orange")
plt.title("Accident Risk vs Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Average Accident Risk")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


train_no_id = train.drop(columns=["id"])
plt.figure(figsize=(6,4))
road_grouped = train_no_id.groupby("public_road")["accident_risk"].mean().sort_values()
road_grouped.plot(kind="bar", color="skyblue")
plt.title("Accident Risk vs public_road")
plt.xlabel("public_road Condition")
plt.ylabel("Average Accident Risk")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
holiday_grouped = train_no_id.groupby("holiday")["accident_risk"].mean().sort_values()
holiday_grouped.plot(kind="bar", color="orange")
plt.title("Accident Risk vs holiday")
plt.xlabel("holiday Condition")
plt.ylabel("Average Accident Risk")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


time_grouped = train_no_id.groupby("time_of_day")["accident_risk"].mean()
plt.figure(figsize=(6,4))
plt.plot(time_grouped.index, time_grouped.values, marker="o", linestyle="-", color="purple")
plt.title("Average Accident Risk by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Average Accident Risk")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
sns.barplot(data=train_no_id, x="weather", y="accident_risk", hue="time_of_day", ci=None)
plt.title("Accident Risk by Weather and Time of Day")
plt.show()


plt.figure(figsize=(10,6))
sns.barplot(data=train_no_id, x="road_type", y="accident_risk", hue="weather", ci=None)
plt.title("Accident Risk by Road Type and Weather")
plt.xlabel("Road Type")
plt.ylabel("Average Accident Risk")
plt.legend(title="Weather")
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,6))
sns.barplot(data=train_no_id, x="road_type", y="accident_risk", hue="weather", ci=None)
plt.title("Accident Risk by Road Type and Weather")
plt.xlabel("Road Type")
plt.ylabel("Average Accident Risk")
plt.legend(title="Weather")
plt.tight_layout()
plt.show()


df = train.drop(columns=["id"])

def numeric_categorical_plot(num_col, cat_col, agg="mean", n_bins=6):
    bins = pd.qcut(df[num_col], q=n_bins, duplicates="drop")
    temp = df.groupby([bins, cat_col])["accident_risk"].agg(agg).unstack(cat_col)

    temp.plot(kind="bar", figsize=(10,5))
    plt.title(f"Accident Risk by {num_col} (binned) × {cat_col} [{agg}]")
    plt.xlabel(f"{num_col} bins")
    plt.ylabel(f"{agg.capitalize()} Accident Risk")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    temp.plot(marker="o", figsize=(10,5))
    plt.title(f"Accident Risk by {num_col} (binned) × {cat_col} [{agg}]")
    plt.xlabel(f"{num_col} bins")
    plt.ylabel(f"{agg.capitalize()} Accident Risk")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

numeric_categorical_plot("speed_limit", "weather", agg="mean")
numeric_categorical_plot("curvature", "road_type", agg="median")
numeric_categorical_plot("num_lanes", "lighting", agg="mean")
numeric_categorical_plot("num_reported_accidents", "time_of_day", agg="max")


df = train.drop(columns=["id"])
n_bins_speed = 10
n_bins_curve = 10
speed_bins = pd.qcut(df["speed_limit"], q=n_bins_speed, duplicates="drop")
curve_bins = pd.qcut(df["curvature"], q=n_bins_curve, duplicates="drop")

pivot = (
    df.assign(speed_bin=speed_bins, curve_bin=curve_bins)
      .groupby(["curve_bin", "speed_bin"])["accident_risk"]
      .mean()
      .unstack("speed_bin")
)

y_labels = [str(idx) for idx in pivot.index]
x_labels = [str(col) for col in pivot.columns]
plt.figure(figsize=(12, 6))
im = plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.colorbar(im, label="Mean Accident Risk")
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

plt.xlabel("Speed Limit (binned)")
plt.ylabel("Curvature (binned)")
plt.title("Mean Accident Risk Heatmap: Curvature × Speed Limit")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
hb = plt.hexbin(
    df["curvature"].values,
    df["speed_limit"].values,
    C=df["accident_risk"].values,
    gridsize=40,
    reduce_C_function=np.mean,
    mincnt=20 
)
plt.colorbar(hb, label="Mean Accident Risk")
plt.xlabel("Curvature")
plt.ylabel("Speed Limit")
plt.title("Hexbin: Mean Accident Risk over Curvature × Speed Limit")
plt.tight_layout()
plt.show()


df["curvature_bin"] = pd.qcut(df["curvature"], q=5, duplicates="drop")

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="curvature_bin", y="accident_risk", hue="lighting")
plt.title("Accident Risk by Curvature (binned) and Lighting Conditions")
plt.xlabel("Curvature Bins")
plt.ylabel("Accident Risk")
plt.legend(title="Lighting")
plt.tight_layout()
plt.show()


df["curvature_bin"] = pd.qcut(df["curvature"], q=6, duplicates="drop")
grouped = df.groupby(["curvature_bin", "lighting"])["accident_risk"].mean().unstack("lighting")

plt.figure(figsize=(8,5))
for col in grouped.columns:
    plt.plot(grouped.index.astype(str), grouped[col], marker="o", label=col)

plt.title("Average Accident Risk by Curvature (binned) and Lighting")
plt.xlabel("Curvature Bins")
plt.ylabel("Average Accident Risk")
plt.legend(title="Lighting")
plt.tight_layout()
plt.show()


pivot = df.pivot_table(index="road_type", columns="weather", values="accident_risk", aggfunc="mean")
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Mean Accident Risk by Road Type × Weather")
plt.tight_layout()
plt.show()


import squarify 
df = train.drop(columns=["id"])
grouped = (
    df.groupby(["road_type", "weather"])["accident_risk"]
      .mean()
      .reset_index()
)

labels = grouped["road_type"] + " | " + grouped["weather"] + "\n" + grouped["accident_risk"].round(2).astype(str)
sizes = grouped["accident_risk"].values

plt.figure(figsize=(12,8))
squarify.plot(sizes=sizes, label=labels, alpha=0.8)
plt.axis("off")
plt.title("Treemap of Accident Risk by Road Type × Weather")
plt.show()


import plotly.express as px
agg = (
    df.groupby(["road_type", "weather", "time_of_day"], as_index=False)
      .agg(mean_risk=("accident_risk", "mean"),
           count=("accident_risk", "size"))
)

fig = px.sunburst(
    agg,
    path=["road_type", "weather", "time_of_day"],
    values="count",
    color="mean_risk",
    color_continuous_scale="RdYlGn_r",
    hover_data={"mean_risk":":.3f", "count": True},
    title="Sunburst: Road Type → Weather → Time of Day (color = mean accident_risk)"
)
fig.update_layout(margin=dict(t=60, l=0, r=0, b=0))
fig.show()


from pandas.plotting import parallel_coordinates
df["risk_level"] = pd.qcut(df["accident_risk"], q=3, labels=["Low", "Medium", "High"])
features = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents", "risk_level"]
sample_df = df[features].sample(2000, random_state=42)

plt.figure(figsize=(12,6))
parallel_coordinates(sample_df, "risk_level", colormap="coolwarm", alpha=0.5)
plt.title("Parallel Coordinates: Numeric Features vs Accident Risk Level")
plt.ylabel("Scaled Feature Values")
plt.tight_layout()
plt.show()


num_features = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]
group_means = df.groupby("road_type")[num_features].mean()

z_df = df.copy()
z_df[num_features] = (z_df[num_features] - z_df[num_features].mean()) / z_df[num_features].std(ddof=0)
z_means = z_df.groupby("road_type")[num_features].mean()

categories = num_features
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close loop

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for road_type, row in z_means.iterrows():
    values = row.values.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=str(road_type))
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_rlabel_position(0)
ax.set_title("Radar Chart: Z-scored Numeric Profiles by Road Type", pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.show()


def plot_ecdf_by_category(df, value_col, cat_col, ncols=3):
    categories = df[cat_col].dropna().unique().tolist()
    n = len(categories)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(5*ncols, 3.5*nrows))

    for i, cat in enumerate(sorted(categories), 1):
        vals = df.loc[df[cat_col] == cat, value_col].dropna().values
        if len(vals) == 0:
            continue
        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)

        ax = plt.subplot(nrows, ncols, i)
        ax.plot(x, y, marker="", linestyle="-")
        ax.set_title(f"{cat_col}: {cat}")
        ax.set_xlabel(value_col)
        ax.set_ylabel("ECDF")
        

    plt.suptitle(f"ECDF of {value_col} by {cat_col}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

plot_ecdf_by_category(df, value_col="accident_risk", cat_col="weather", ncols=3)
plot_ecdf_by_category(df, value_col="accident_risk", cat_col="lighting", ncols=3)
plot_ecdf_by_category(df, value_col="accident_risk", cat_col="road_type", ncols=3)




