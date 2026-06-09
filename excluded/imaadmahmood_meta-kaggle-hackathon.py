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


# ========== Part 1: Environment Setup & Data Loading ==========

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# Use an available visual style
plt.style.use('seaborn-whitegrid')
sns.set_theme(style="whitegrid")

# Increase display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# === Load Key Meta Kaggle Files ===
print("Loading dataset files...")

kernels = pd.read_csv("/kaggle/input/meta-kaggle/Kernels.csv", parse_dates=["CreationDate", "EvaluationDate", "MadePublicDate", "MedalAwardDate"])
kernel_versions = pd.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv", parse_dates=["CreationDate", "EvaluationDate"])
users = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv", parse_dates=["RegisterDate"])

# === Quick Summary ===
print(f"Kernels: {kernels.shape}")
print(f"Kernel Versions: {kernel_versions.shape}")
print(f"Users: {users.shape}")

# === Preview Samples ===
print("\nSample Kernels Data:")
display(kernels.head(3))

print("\nSample KernelVersions Data:")
display(kernel_versions.head(3))

print("\nSample Users Data:")
display(users.head(3))

# === Visual: Kernels over time ===
plt.figure(figsize=(12, 5))
sns.histplot(kernels["CreationDate"].dt.year.dropna(), bins=15, color="#228B22", kde=False)
plt.title("Distribution of Kernel Creation Over the Years", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Number of Kernels")
plt.tight_layout()
plt.show()

# === Visual: Users by Performance Tier ===
plt.figure(figsize=(10, 4))
sns.countplot(data=users, x="PerformanceTier", palette="Set2")
plt.title("User Distribution by Performance Tier", fontsize=16)
plt.xlabel("Performance Tier")
plt.ylabel("User Count")
plt.tight_layout()
plt.show()



# ===== Part 2: Merging Metadata & Feature Engineering =====

# 1) Consistent, collision‑free column names
kernels = kernels.rename(columns={
    "Id": "KernelId",
    "AuthorUserId": "KernelAuthorId"
})

kernel_versions = kernel_versions.rename(columns={
    "Id": "VersionId",
    "ScriptId": "KernelId"
})

users = users.rename(columns={"Id": "UserId"})

# 2) Merge Kernels ⟷ KernelVersions (use CurrentKernelVersionId)
merged = pd.merge(
    kernels,
    kernel_versions,
    left_on="CurrentKernelVersionId",
    right_on="VersionId",
    how="left",
    suffixes=("_kernel", "_version")
)

# 3) Merge with Users
merged = pd.merge(
    merged,
    users[["UserId", "UserName", "DisplayName", "PerformanceTier", "Country"]],
    left_on="KernelAuthorId",
    right_on="UserId",
    how="left"
)

# 4) Feature Engineering
merged["Score"] = (
    merged["TotalVotes_kernel"].fillna(0) * 1.5 +
    merged["TotalViews"].fillna(0) * 0.05
)

merged["LinesChangedRatio"] = (
    merged["LinesChangedFromPrevious"] / merged["TotalLines"]
)

merged["CodeComplexity"] = (
    merged["LinesChangedFromPrevious"].fillna(0) +
    merged["TotalLines"].fillna(0)
)

merged["CreationDate"] = pd.to_datetime(merged["CreationDate_kernel"], errors="coerce")
merged["Year"]  = merged["CreationDate"].dt.year
merged["Month"] = merged["CreationDate"].dt.to_period("M")

# 5) Inspect the enriched data
print("✅ Merged shape:", merged.shape)
display(merged[[
    "Title", "UserName", "Score", "TotalVotes_kernel",
    "TotalViews", "CodeComplexity", "PerformanceTier", "Country"
]].head(10))

# 6) Visual 1 – Score Distribution
plt.figure(figsize=(12, 5))
sns.histplot(merged["Score"], bins=50, color="#33658a", kde=True)
plt.title("Kernel Engagement Score Distribution", fontsize=16)
plt.xlabel("Score")
plt.ylabel("Kernel Count")
plt.tight_layout()
plt.show()

# 7) Visual 2 – Top Author Countries
top_countries = (merged["Country"]
                 .value_counts()
                 .head(10)
                 .rename_axis("Country")
                 .reset_index(name="KernelCount"))

plt.figure(figsize=(10, 4))
sns.barplot(data=top_countries, x="Country", y="KernelCount", palette="icefire")
plt.title("Top 10 Author Countries by Kernel Count", fontsize=16)
plt.ylabel("Kernel Count")
plt.xlabel("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Use modern style
plt.style.use("seaborn-whitegrid")
sns.set_theme(style="whitegrid")

# Ensure datetime already handled earlier: merged["CreationDate"] = pd.to_datetime(...)
# Monthly group analysis
monthly_trends = (
    merged.groupby("Month")
          .agg(Kernels=("KernelId_kernel", "count"), Views=("TotalViews", "sum"))
          .reset_index()
)

# Convert Month back to timestamp for plotting
monthly_trends["Month"] = monthly_trends["Month"].dt.to_timestamp()

# Plot 1: Monthly Kernel Count
plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_trends, x="Month", y="Kernels", linewidth=2.5, marker="o", color="#2980b9")
plt.title("Monthly Kaggle Kernel Submissions", fontsize=16, weight='bold')
plt.xlabel("Month", fontsize=12)
plt.ylabel("Kernel Count", fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Monthly Views (recent 24 months)
recent = monthly_trends.sort_values("Month").tail(24)
plt.figure(figsize=(14, 5))
sns.barplot(data=recent, x="Month", y="Views", color="#27ae60")
plt.title("Total Monthly Views (Last 24 Months)", fontsize=16, weight='bold')
plt.xlabel("Month", fontsize=12)
plt.ylabel("Views", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: PerformanceTier-based monthly kernel count
if "PerformanceTier" in merged.columns:
    tier_monthly = (
        merged.dropna(subset=["PerformanceTier"])
              .groupby(["Month", "PerformanceTier"])
              .agg(Kernels=("KernelId_kernel", "count"))
              .reset_index()
    )
    tier_monthly["Month"] = tier_monthly["Month"].dt.to_timestamp()
    tier_labels = {1: "Novice", 2: "Contributor", 3: "Expert", 4: "Master", 5: "Grandmaster"}

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=tier_monthly, x="Month", y="Kernels", hue="PerformanceTier",
        palette="Set2", linewidth=2
    )
    plt.title("Monthly Kernel Submissions by Performance Tier", fontsize=16, weight='bold')
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Kernel Count", fontsize=12)
    plt.legend(
        title="Tier", 
        labels=[tier_labels.get(t, f"Tier {t}") for t in sorted(tier_monthly["PerformanceTier"].unique())]
    )
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# Top authors by total score (views × votes)
author_scores = (
    merged.groupby(["UserId", "UserName", "DisplayName", "Country"])
          .agg(Kernels=("KernelId_kernel", "count"),
               TotalViews=("TotalViews", "sum"),
               TotalVotes=("TotalVotes_kernel", "sum"),
               Score=("Score", "sum"))
          .reset_index()
          .sort_values("Score", ascending=False)
)

# Top 15 authors
top_authors = author_scores.head(15)

# Plot 1: Top authors by score
plt.figure(figsize=(12, 6))
sns.barplot(data=top_authors, y="DisplayName", x="Score", palette="viridis")
plt.title("Top 15 Kaggle Authors by Total Kernel Score", fontsize=16, weight='bold')
plt.xlabel("Score (Views × Votes Normalized)", fontsize=12)
plt.ylabel("Author", fontsize=12)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Country-wise total kernels
country_counts = (
    merged.dropna(subset=["Country"])
          .groupby("Country")
          .agg(Kernels=("KernelId_kernel", "count"))
          .sort_values("Kernels", ascending=False)
          .reset_index()
)

top_countries = country_counts.head(15)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_countries, x="Kernels", y="Country", palette="magma")
plt.title("Top 15 Countries by Kernel Submissions", fontsize=16, weight='bold')
plt.xlabel("Kernel Count", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()



# Optional : Geographic Map Plot


# Extended bar plot
top30 = country_counts.head(30)
plt.figure(figsize=(14, 6))
sns.barplot(data=top30, x="Country", y="Kernels", palette="coolwarm")
plt.title("Top 30 Countries by Kernel Contributions", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Load Tag Mappings
kernel_tags = pd.read_csv("/kaggle/input/meta-kaggle/KernelTags.csv")
tags = pd.read_csv("/kaggle/input/meta-kaggle/Tags.csv")

# Confirm column names
# print(kernel_tags.columns)  # ['Id', 'KernelId', 'TagId']
# print(tags.columns)         # ['Id', 'ParentTagId', 'Name', 'Slug', ..., 'KernelCount']

# Merge tags with names
merged_tags = pd.merge(kernel_tags, tags, left_on="TagId", right_on="Id", how="left")

# Count most common tag names
tag_counts = merged_tags["Name"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# Top 20 tags
top_tags = tag_counts.head(20)

# Visualize with Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=top_tags, x="Count", y="Tag", palette="magma", edgecolor="black")
plt.title("Top 20 Most Common Tags in Kaggle Kernels", fontsize=18, fontweight='bold')
plt.xlabel("Usage Count")
plt.ylabel("Tag Name")
plt.grid(axis="x", linestyle="--", alpha=0.4)
sns.despine()
plt.tight_layout()
plt.show()



from wordcloud import WordCloud, STOPWORDS

# Concatenate all titles
all_titles = merged["Title"].dropna().astype(str).str.cat(sep=" ")

# Define stopwords
stopwords = set(STOPWORDS)

# Generate word cloud
wordcloud = WordCloud(
    width=1400,
    height=700,
    background_color="white",
    stopwords=stopwords,
    max_words=200,
    colormap="inferno",
    contour_color="black",
    contour_width=1.5
).generate(all_titles)

# Show WordCloud
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Words in Kernel Titles", fontsize=20, weight="bold")
plt.tight_layout()
plt.show()



import matplotlib.dates as mdates
from prophet import Prophet

# Ensure 'CreationDate_version' is datetime
merged["CreationDate_version"] = pd.to_datetime(merged["CreationDate_version"])

# Create 'YearMonth' column
merged["YearMonth"] = merged["CreationDate_version"].dt.to_period("M").astype(str)

# Monthly aggregations
monthly_data = (
    merged.groupby("YearMonth")
    .agg(Kernels=("VersionId", "count"), Views=("TotalViews", "sum"))
    .reset_index()
)

# Convert to datetime
monthly_data["YearMonth"] = pd.to_datetime(monthly_data["YearMonth"])

# Plot trends
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_data, x="YearMonth", y="Kernels", linewidth=2.5, marker="o", label="Kernels Created")
sns.lineplot(data=monthly_data, x="YearMonth", y="Views", linewidth=2.5, label="Total Views", color="orange")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.title("Monthly Trend of Kernel Creations and Views", fontsize=18, weight="bold")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()



# Clean and prepare data
df_prophet = (
    monthly_data[["YearMonth", "Kernels"]]
    .rename(columns={"YearMonth": "ds", "Kernels": "y"})
)

# Drop rows with NaN in 'ds' or 'y'
df_prophet = df_prophet.dropna(subset=["ds", "y"])

# Prophet Model
from prophet import Prophet
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

# Fit the model
model.fit(df_prophet)

# Forecast next 12 months
future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Forecast of Kaggle Kernel Creations (Next 12 Months)", fontsize=18, weight="bold")
plt.xlabel("Date")
plt.ylabel("Predicted Kernels")
plt.tight_layout()
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()

# Plot forecast components (trend + seasonality)
fig2 = model.plot_components(forecast)
plt.tight_layout()
plt.show()



# Select relevant columns for export
export_columns = [
    "Title", "UserName", "DisplayName", "Country", "Score",
    "TotalViews", "TotalVotes_kernel", "TotalVotes_version", "TotalComments",
    "LinesChangedFromPrevious", "LinesChangedRatio", "CodeComplexity",
    "Medal", "CreationDate"
]

# Safely drop rows with missing key values
top_kernels_export = merged[export_columns].dropna(subset=["Title", "UserName", "Score"])

# Sort by Score and export top 100
top_kernels_export = top_kernels_export.sort_values("Score", ascending=False).head(100)

# Save to CSV
top_kernels_export.to_csv("top_kaggle_kernels.csv", index=False)
print("✅ Exported top kernels to 'top_kaggle_kernels.csv'")


