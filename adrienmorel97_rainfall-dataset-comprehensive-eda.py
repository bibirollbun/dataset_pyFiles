# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# =========================
# âš™ï¸� Standard Libraries
# =========================
import math
import warnings

# =========================
# ğŸ“Š Data Manipulation
# =========================
import numpy as np
import pandas as pd

# =========================
# ğŸ“ˆ Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Set seaborn style globally
sns.set(style="whitegrid", font_scale=1.1)

# =========================
# ğŸ§ª Scientific Computing
# =========================
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import boxcox
from scipy.stats import ks_2samp

# =========================
# ğŸ¤– Machine Learning - Core
# =========================
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# =========================
# ğŸ¤– Machine Learning - Models
# =========================
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering

# =========================
# ğŸ“Š Evaluation Metrics
# =========================
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    silhouette_score
)

# =========================
# âš ï¸� Warning Handling
# =========================
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# Check data types and missing values
print(train_df.info())


# Save 'id' column for submission
test_ids = test_df['id']

# Define the target column
target_column = "rainfall"

# Select categorical and numerical columns (initial)
numerical_columns = [col for col in train_df.columns if col != "rainfall"]

# Print out column information
print("Target Column:", target_column)
print("\nNumerical Columns:", numerical_columns)


# Check dataset shape and first rows
print(f"Dataset contains {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
train_df.head()


# Get statistical summary
train_df.describe()


# Compute summary stats
skew_values = train_df[numerical_columns].skew(numeric_only=True)
mean_values = train_df[numerical_columns].mean(numeric_only=True)
std_values = train_df[numerical_columns].std(numeric_only=True)

# Create DataFrame
stat_df = pd.DataFrame({
    'Feature': skew_values.index,
    'Mean': mean_values.values.round(3),
    'Std Dev': std_values.values.round(3),
    'Skewness': skew_values.values.round(3)
}).sort_values(by='Mean', key=abs, ascending=False).reset_index(drop=True)

# Color rules
def highlight_stats(val, column):
    if column == 'Skewness' and abs(val) > 1:
        return 'color: red; font-weight: bold'
    if column == 'Std Dev' and val > stat_df['Std Dev'].median() + stat_df['Std Dev'].std():
        return 'color: orange; font-weight: bold'
    if column == 'Mean' and val > stat_df['Mean'].median() + stat_df['Mean'].std():
        return 'color: blue; font-weight: bold'
    return ''

# Apply styling
def highlight_df(df):
    styled_df = df.style.applymap(lambda v: highlight_stats(v, 'Mean'), subset=['Mean']) \
                        .applymap(lambda v: highlight_stats(v, 'Std Dev'), subset=['Std Dev']) \
                        .applymap(lambda v: highlight_stats(v, 'Skewness'), subset=['Skewness']) \
                        .set_properties(**{'background-color': 'white'}, subset=pd.IndexSlice[:, :]) \
                        .set_table_styles([{'selector': 'th', 'props': [('background-color', '#afe1f0'), ('font-weight', 'bold')]}])
    return styled_df

# Display
highlight_df(stat_df)



# Target variable distribution
target_counts = train_df[target_column].value_counts()
labels = ['No Rain (0)', 'Rain (1)']
counts = target_counts.values
total = counts.sum()
ratios = [f'{(count / total) * 100:.1f}%' for count in counts]

# Car counter-style visualization
plt.figure(figsize=(7, 4))
bars = sns.barplot(x=labels, y=counts, palette="coolwarm")

# Annotate percentage above each bar
for bar, ratio in zip(bars.patches, ratios):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 20,
             ratio, ha='center', fontsize=12, fontweight='bold')

plt.title('Rainfall Distribution (Binary Target)', fontsize=14, weight='bold')
plt.ylabel('Number of Days')
plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.show()


# Dynamically calculate number of rows & columns
num_vars = len(numerical_columns)
num_cols = 2  # Keep 2 columns for readability
num_rows = math.ceil(num_vars / num_cols)  # Calculate rows dynamically

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4.5))
axes = axes.flatten()

# Color palette (color-blind friendly)
palette = sns.color_palette("crest", n_colors=num_vars)

# Plotting each variable
for i, var in enumerate(numerical_columns):
    sns.histplot(
        data=train_df,
        x=var,
        kde=True,
        color=palette[i],
        bins=30,
        edgecolor="white",
        linewidth=1.3,
        ax=axes[i]
    )
    axes[i].set_title(f"Distribution of {var}", fontsize=14, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis='x', labelrotation=15)

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()


# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot violin plots
for i, var in enumerate(numerical_columns):
    sns.violinplot(x=train_df[target_column], y=train_df[var], palette="coolwarm", inner="quartile", ax=axes[i])
    axes[i].set_title(f"{var} Distribution on Rainy vs. Non-Rainy Days")

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])  

plt.tight_layout()
plt.show()


# Grille des scatterplots deux Ã  deux pour visualiser les corrÃ©lations
pd.plotting.scatter_matrix(train_df[numerical_columns], figsize=(12, 12), diagonal='kde', alpha=0.7)

# Ajouter un titre gÃ©nÃ©ral Ã  la grille
plt.suptitle("Grille des scatterplots deux Ã  deux", fontsize=16, y=1.02)

# Afficher le graphique
plt.tight_layout()
plt.show()


df_eda = train_df.copy()
cols_to_use = [col for col in numerical_columns if col not in ['id', 'day']]

# Clean copy for analysis
X_eda = df_eda[cols_to_use]

# Standardize meteorological features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_eda)

# PCA on 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Insert PCA results into the copy
df_eda["PC1"] = X_pca[:, 0]
df_eda["PC2"] = X_pca[:, 1]


# Visualization in PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_eda, x="PC1", y="PC2", hue="rainfall", palette="coolwarm", alpha=0.7)
plt.title("PCA of Weather Conditions (colored by Rainfall)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Distribution of PCA components over the year
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_eda, x="day", y="PC1", alpha=0.5, label="PC1")
sns.scatterplot(data=df_eda, x="day", y="PC2", alpha=0.5, label="PC2")
plt.title("Evolution of Weather Components Over the Year")
plt.xlabel("Day of Year (1â€“365)")
plt.ylabel("PCA Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Retrieve from previous PCA
components = pca.components_
explained_var = pca.explained_variance_ratio_
feature_names = cols_to_use

# Plot
plt.figure(figsize=(8, 8))
plt.axhline(0, color='lightgray', linewidth=1)
plt.axvline(0, color='lightgray', linewidth=1)

# Unit circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1)
plt.gca().add_artist(circle)

# Arrows and angle-based label positioning
for i, var in enumerate(feature_names):
    x, y = components[0, i], components[1, i]

    # Draw arrow
    plt.arrow(0, 0, x, y, color='royalblue', alpha=0.8, head_width=0.02, length_includes_head=True)

    # Compute angle to spread labels naturally
    angle = np.arctan2(y, x)
    radius = 1.05  # label distance

    label_x = radius * np.cos(angle)
    label_y = radius * np.sin(angle)

    align_h = 'left' if label_x > 0 else 'right'
    align_v = 'bottom' if label_y > 0 else 'top'

    plt.text(label_x, label_y, var, fontsize=11, ha=align_h, va=align_v)

# Layout
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f} %)", fontsize=12)
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f} %)", fontsize=12)
plt.title("Correlation circle (PCA)", fontsize=14, weight='bold')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Data from PCA
X_pca = df_eda[["PC1", "PC2"]].values

# Range of clusters
k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    clustering = AgglomerativeClustering(n_clusters=k)
    labels = clustering.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters (Agglomerative Clustering)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.tight_layout()
plt.show()



# 1. Run Agglomerative Clustering (k=4)
agglo = AgglomerativeClustering(n_clusters=4)
clusters_agglo = agglo.fit_predict(X_pca)

# 2. True labels
y_true = df_eda[target_column].values
y_pred = clusters_agglo

# 3. Handle potential label inversion
acc1 = accuracy_score(y_true, y_pred)
acc2 = accuracy_score(y_true, 1 - y_pred)
if acc2 > acc1:
    y_pred = 1 - y_pred

# 4. Metrics
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)

# 5. Add to df_eda
df_eda["cluster_agglo"] = y_pred

# 6. Plot PCA with color=cluster, marker=rainfall
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_eda, x="PC1", y="PC2",
                hue="cluster_agglo", style=target_column,
                palette="Set2", alpha=0.7, s=60)
plt.title("Agglomerative Clustering vs Rainfall")
plt.legend(title="Cluster / Rainfall")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Print metrics
print(f" Agglomerative Clustering - Accuracy: {accuracy:.4f}")
print(f" Agglomerative Clustering - ROC AUC: {auc:.4f}")


# Ensure cluster_agglo and rainfall are in your df
df_clustered = df_eda.copy()

# Rainfall rate per cluster
plt.figure(figsize=(6, 4))
rain_perc = df_clustered.groupby("cluster_agglo")[target_column].mean()
sns.barplot(x=rain_perc.index, y=rain_perc.values, palette="Set2")
plt.title("Rainfall rate per cluster")
plt.ylabel("Mean rainfall (0â€“1)")
plt.xlabel("Cluster")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Sort clusters for consistency
df_clustered["cluster_agglo"] = df_clustered["cluster_agglo"].astype(int)

# Cluster-wise means for each feature (raw values)
mean_df = df_clustered.groupby("cluster_agglo")[cols_to_use].mean().T  # shape = (features, clusters)

# Apply MinMaxScaler row-wise (each feature independently)
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(mean_df.values.T).T,  # transpose twice to scale rows
    index=mean_df.index,
    columns=mean_df.columns
)
# 1. Radar chart setup
labels = normalized.index.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the circle

# 2. Create 2x2 grid for clusters
fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(12, 10))
axs = axs.flatten()  # Flatten to loop easily

for i, cluster in enumerate(normalized.columns):
    values = normalized[cluster].tolist()
    values += values[:1]  # close the shape

    ax = axs[i]
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_title(f"Cluster {cluster} - Normalized Feature Profile", fontsize=13, pad=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.grid(True)

# Layout
plt.suptitle("Radar Plot of Normalized Weather Features per Cluster", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Base DataFrame
df_plot = df_clustered.copy()

# Plot: temperature vs day of year
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_plot,
                x="id",
                y="temparature",
                hue="cluster_agglo",
                palette="Set2",
                alpha=0.7,
                s=40)

plt.title("Temperature evolution over the year (colored by cluster)")
plt.xlabel("Day of year")
plt.ylabel("Temperature (Â°C)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()


# Plot: humidity vs day of year
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_plot,
                x="id",
                y="cloud",
                hue="cluster_agglo",
                palette="Set2",
                alpha=0.7,
                s=40)

plt.title("Cloud evolution over the year (colored by cluster)")
plt.xlabel("Day of year")
plt.ylabel("Cloud")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()



def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Count outliers for each numerical feature
outlier_counts = {var: len(detect_outliers_iqr(train_df, var)) for var in numerical_columns}

# Create barplot
plt.figure(figsize=(10, 5))
sns.barplot(x=list(outlier_counts.keys()), y=list(outlier_counts.values()), color='coral')
plt.xticks(rotation=45)
plt.title("Number of Outliers per Numerical Feature (IQR Method)", fontsize=14, weight='bold')
plt.ylabel("Number of Outliers")
plt.xlabel("Feature")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Estimate an automatic contamination level using percentiles
estimated_contamination = np.mean([
    len(detect_outliers_iqr(train_df, col)) / len(train_df) for col in numerical_columns
])

print(f"Estimated Contamination Rate: {estimated_contamination:.4f}")


# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
train_df["anomaly"] = iso_forest.fit_predict(train_df[numerical_columns])

# Step 1: Detect IQR-based outliers across all numerical features
iqr_outliers = set()
for var in numerical_columns:
    iqr_outliers.update(detect_outliers_iqr(train_df, var).index)

# Step 2: Detect Isolation Forest anomalies
train_df["anomaly"] = iso_forest.fit_predict(train_df[numerical_columns])

# Step 3: Compare IQR outliers and Isolation Forest anomalies
iso_outliers = set(train_df[train_df["anomaly"] == -1].index)

# Step 4: Find common and unique outliers
common_outliers = iqr_outliers.intersection(iso_outliers)
only_iqr_outliers = iqr_outliers - iso_outliers
only_iso_outliers = iso_outliers - iqr_outliers

# Step 5: Print results
print(f"Total IQR Outliers: {len(iqr_outliers)}")
print(f"Total Isolation Forest Outliers: {len(iso_outliers)}")
print(f"Common Outliers: {len(common_outliers)}")
print(f"Outliers detected only by IQR: {len(only_iqr_outliers)}")
print(f"Outliers detected only by Isolation Forest: {len(only_iso_outliers)}")


plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=train_df["temparature"], y=train_df["humidity"], 
    hue=train_df.index.map(lambda idx: 
        "Both" if idx in common_outliers else 
        "Only IQR" if idx in only_iqr_outliers else 
        "Only IF" if idx in only_iso_outliers else "Normal"),
    palette={"Both": "red", "Only IQR": "blue", "Only IF": "purple", "Normal": "gray"},
    alpha=0.7
)
plt.title("Comparison of IQR & Isolation Forest Outliers")
plt.legend(title="Outlier Type")
plt.show()


# Compute overall rainfall rate
global_rainfall_rate = train_df["rainfall"].mean() * 100  # Convert to percentage

# Compute rainfall rate among anomalies
anomaly_rainfall_rate = train_df[train_df["anomaly"] == -1]["rainfall"].mean() * 100  # Convert to percentage

# Create a DataFrame for visualization
rainfall_comparison = pd.DataFrame({
    "Category": ["Overall Rainfall Rate", "Anomalous Data Rainfall Rate"],
    "Rainfall Rate (%)": [global_rainfall_rate, anomaly_rainfall_rate]
})

# Plot the comparison
plt.figure(figsize=(8, 6))
sns.barplot(data=rainfall_comparison, x="Category", y="Rainfall Rate (%)", palette=["lightblue", "red"])


# Formatting
plt.title("Comparison of Rainfall Rate: Overall vs. Among Anomalies", fontsize=14)
plt.ylabel("Rainfall Rate (%)", fontsize=12)
plt.xlabel("")
plt.ylim(0, 100)  # Ensure the scale is percentage-based
plt.grid(axis="y", alpha=0.3)

# Show plot
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Shared scatterplot parameters
scatter_kwargs = dict(
    palette={1: "blue", -1: "red"},
    s=60,
    alpha=0.7,
    legend=False
)

# Plot 1: Pressure vs. Humidity
sns.scatterplot(
    ax=axes[0, 0],
    x=train_df["pressure"], 
    y=train_df["humidity"], 
    hue=train_df["anomaly"], 
    style=train_df["rainfall"],
    **scatter_kwargs
)
axes[0, 0].set_title("Outlier Detection: Pressure vs. Humidity")

# Plot 2: Temperature vs. Cloud
sns.scatterplot(
    ax=axes[0, 1],
    x=train_df["temparature"], 
    y=train_df["cloud"], 
    hue=train_df["anomaly"], 
    style=train_df["rainfall"],
    **scatter_kwargs
)
axes[0, 1].set_title("Outlier Detection: Temperature vs. Cloud")

# Plot 3: Temperature vs. Dewpoint
sns.scatterplot(
    ax=axes[1, 0],
    x=train_df["temparature"], 
    y=train_df["dewpoint"], 
    hue=train_df["anomaly"], 
    style=train_df["rainfall"],
    **scatter_kwargs
)
axes[1, 0].set_title("Outlier Detection: Temperature vs. Dewpoint")

# Plot 4: ID vs. Temperature
sns.scatterplot(
    ax=axes[1, 1],
    x=train_df["id"], 
    y=train_df["temparature"], 
    hue=train_df["anomaly"], 
    style=train_df["rainfall"],
    **scatter_kwargs
)
axes[1, 1].set_title("Outlier Detection: Observation ID vs. Temperature")

# Custom Legend
anomaly_legend = [
    mpatches.Patch(color='red', label='Anomaly (-1)'),
    mpatches.Patch(color='blue', label='Normal (1)')
]

rainfall_legend = [
    mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='No Rain (0)'),
    mlines.Line2D([], [], color='black', marker='X', linestyle='None', label='Rain (1)')
]

custom_legend = anomaly_legend + rainfall_legend
fig.legend(handles=custom_legend, loc='upper center', ncol=4, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


# Columns to include (exclude 'id' and 'day')
features = [col for col in numerical_columns if col not in ['id', 'day']]

# Normalize all selected features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(train_df[features])
df_scaled = pd.DataFrame(scaled, columns=features, index=train_df.index)

# Add anomaly column back for grouping
df_scaled["anomaly"] = train_df["anomaly"]

# Compute normalized mean profiles
anomaly_profile = df_scaled[df_scaled["anomaly"] == -1][features].mean()
normal_profile = df_scaled[df_scaled["anomaly"] == 1][features].mean()

# Build comparison DataFrame
comparison = pd.DataFrame({
    "Anomalies": anomaly_profile,
    "Normal": normal_profile
})

# Plot
comparison.plot(kind="bar", figsize=(14, 6), alpha=0.8)
plt.title("Normalized Feature Profiles: Anomalies vs. Normal Observations")
plt.ylabel("Normalized Mean Value (0â€“1)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# Compare rainy days and suspicious dry anomalies
rainy_days = train_df[train_df["rainfall"] == 1]
suspicious_dry_days = train_df[(train_df["anomaly"] == -1) & (train_df["rainfall"] == 0)]

# Run KS-test for each numerical variable
ks_results = {}
for var in numerical_columns:
    stat, p_value = ks_2samp(rainy_days[var], suspicious_dry_days[var])
    ks_results[var] = {"KS Statistic": stat, "p-value": p_value}

# Create sorted DataFrame of results
ks_df = pd.DataFrame(ks_results).T.sort_values("p-value", ascending=False)

# Display as table
plt.figure(figsize=(10, len(ks_df)*0.4))
sns.heatmap(
    ks_df[["p-value"]].T,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    cbar=False
)
plt.title("KS Test p-values: Rainy Days vs. Suspicious Dry Anomalies")
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 

