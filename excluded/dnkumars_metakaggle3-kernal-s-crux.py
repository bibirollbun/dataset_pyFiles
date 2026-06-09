import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.patches import Rectangle
from IPython.display import IFrame, display
import kagglehub

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("âœ… Downloaded Meta-Kaggle data.")
print("ğŸ“‚ MK_PATH =", MK_PATH)
print("ğŸ“‚ MKC_PATH =", MKC_PATH)


Kernels = pl.read_csv("/kaggle/input/meta-kaggle/Kernels.csv")
print(Kernels.columns)
print(Kernels.shape)
Kernels.head()


KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
print(KernelVersions.columns)
print(KernelVersions.shape)
KernelVersions.head()


KernelVersions = KernelVersions.with_columns(
    pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").dt.year().alias("Year")
)
yearly_counts = KernelVersions.group_by("Year").agg(
    KernelCount=pl.col("Id").count()
).sort("Year")


yearly_counts_pd = yearly_counts.to_pandas()
fig = px.line(
    yearly_counts_pd,
    x="Year",
    y="KernelCount",
    title="Number of Kernels by Year",
    labels={"Year": "Year", "KernelCount": "Number of Kernels"},
    markers=True
)
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Kernels",
    xaxis=dict(tickmode="linear"),
    showlegend=True
)
fig.write_html("yearly_kernels.html")
print("Displaying Number of Kernels by Year:")
display(IFrame("yearly_kernels.html", width=1000, height=800))


KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
KernelAcceleratorTypes = pl.read_csv("/kaggle/input/meta-kaggle/KernelAcceleratorTypes.csv")
print(KernelAcceleratorTypes.columns)
print(KernelAcceleratorTypes.shape)
KernelAcceleratorTypes.head()


KernelAcceleratorTypes.to_pandas()


merged_df = (
    Kernels
    .join(KernelVersions, left_on="CurrentKernelVersionId", right_on="Id", how="left")
    .join(KernelAcceleratorTypes, left_on="AcceleratorTypeId", right_on="Id", how="left", suffix="_accel")
)
merged_df = merged_df.with_columns(
    pl.col("Label").fill_null("None").alias("Label")
)
accelerator_usage = (
    merged_df
    .group_by("Label")
    .agg(count=pl.col("Id").count())
    .sort("count", descending=True)
)


print("Number of kernels using each accelerator type:")
accelerator_usage.to_pandas()


merged_df = merged_df.with_columns(
    pl.col("CreationDate")
    .str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False)
    .dt.year()
    .alias("Year")
)
merged_df = merged_df.with_columns(
    pl.col("Year").fill_null("Unknown").alias("Year")
)

yearly_accelerator_usage = (
    merged_df
    .group_by(["Year", "Label"])
    .agg(count=pl.col("Id").count())
    .sort(["Year", "count"], descending=[False, True])
)


print("\nRise of accelerators by year:")
yearly_accelerator_usage.to_pandas()


accelerator_usage_pd = accelerator_usage.to_pandas()
yearly_accelerator_usage_pd = yearly_accelerator_usage.to_pandas()


fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=accelerator_usage_pd["Label"],
        y=accelerator_usage_pd["count"],
        name="Total Kernels",
        marker_color="blue",
        text=accelerator_usage_pd["count"],
        textposition="auto"
    )
)
fig1.update_layout(
    title_text="Total Kernels by Accelerator Type",
    xaxis_title="Accelerator Type",
    yaxis_title="Number of Kernels",
    xaxis_tickangle=45,
    template="plotly_white",
    height=600,
    width=1000
)
fig1.write_html("total_kernels.html")
print("Displaying Total Kernels by Accelerator Type:")
display(IFrame("total_kernels.html", width=1000, height=600))


fig2 = go.Figure()
pivot_df = yearly_accelerator_usage_pd.pivot(index="Year", columns="Label", values="count").fillna(0)
years = sorted(pivot_df.index, key=lambda x: "0" if x == "Unknown" else x)
colors = px.colors.qualitative.Plotly  

for i, label in enumerate(pivot_df.columns):
    fig2.add_trace(
        go.Bar(
            x=years,
            y=pivot_df[label],
            name=label,
            text=[int(x) if x > 0 else "" for x in pivot_df[label]],
            textposition="inside",
            marker_color=colors[i % len(colors)]
        )
    )
fig2.update_layout(
    title_text="Accelerator Usage by Year",
    xaxis_title="Year",
    yaxis_title="Number of Kernels",
    xaxis_tickangle=45,
    barmode="group", 
    template="plotly_white",
    height=600,
    width=1000,
    legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
)
fig2.write_html("category_trends_grouped.html")
print("Displaying Accelerator Usage by Year:")
display(IFrame("category_trends_grouped.html", width=1000, height=600))


Kernels = pl.read_csv("/kaggle/input/meta-kaggle/Kernels.csv", columns=["Id", "CurrentKernelVersionId", "CreationDate"])
KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv", columns=["Id", "AcceleratorTypeId", "TotalLines", "RunningTimeInMilliseconds"])
KernelAcceleratorTypes = pl.read_csv("/kaggle/input/meta-kaggle/KernelAcceleratorTypes.csv", columns=["Id", "Label"])


merged_df = (
    Kernels
    .join(KernelVersions, left_on="CurrentKernelVersionId", right_on="Id", how="left")
    .join(KernelAcceleratorTypes, left_on="AcceleratorTypeId", right_on="Id", how="left", suffix="_accel")
    .with_columns(
        pl.col("Label").fill_null("None").alias("Label"),
        pl.col("CreationDate")
        .str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False)
        .dt.year()
        .fill_null("Unknown")
        .alias("Year")
    )
    .select(["TotalLines", "RunningTimeInMilliseconds", "Label", "Year"]) 
    .filter(pl.col("TotalLines").is_not_null() & pl.col("RunningTimeInMilliseconds").is_not_null())  # Drop nulls early
)


print(merged_df.shape)
merged_df.sample(10)


df_pandas = merged_df.to_pandas()
df_pandas['Year'] = df_pandas['Year'].astype(int)
# Convert RunningTimeInMilliseconds to numeric, handling any non-numeric values
df_pandas['RunningTimeInMilliseconds'] = pd.to_numeric(df_pandas['RunningTimeInMilliseconds'], errors='coerce')
df_pandas['RunningTimeInSeconds'] = df_pandas['RunningTimeInMilliseconds'] / 1000


df_pandas.sample(10)


# 1. Box Plot - Running Time by Label
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=df_pandas, x='Label', y='RunningTimeInSeconds')
medians = df_pandas.groupby('Label')['RunningTimeInSeconds'].median()
for tick, label in enumerate(ax.get_xticklabels()):
    label_text = label.get_text()
    median_val = medians[label_text]
    ax.text(
        tick, median_val + 200,
        
        f'{median_val:.0f}', 
        horizontalalignment='center', size='small', color='black', weight='semibold'
    )

plt.title('Box Plot: Running Time by Accelerator Types', fontsize=16, fontweight='bold')
plt.xlabel('Label', fontsize=12)
plt.ylabel('Running Time (seconds)', fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 25000)  # Zoom in
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 3. Bar Plot - Average Running Time by Year
plt.figure(figsize=(10, 6))
yearly_avg = df_pandas.groupby('Year')['RunningTimeInSeconds'].mean()
bars = plt.bar(yearly_avg.index, yearly_avg.values, color='skyblue', edgecolor='navy', linewidth=1.5)
plt.title('Bar Plot: Average Running Time by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Running Time (seconds)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()


# 10. Pie Chart - Distribution of Labels
plt.figure(figsize=(10, 8))

label_counts = df_pandas['Label'].value_counts()
total = label_counts.sum()
percentages = label_counts / total * 100
labels_with_percent = [f'{label} ({pct:.1f}%)' for label, pct in zip(label_counts.index, percentages)]

colors_pie = sns.color_palette('pastel', len(label_counts))
explode = [0.05] * len(label_counts)
wedges, _, autotexts = plt.pie(
    label_counts.values,
    labels=None,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    explode=explode,
    shadow=True
)
plt.legend(wedges, labels_with_percent, title="Labels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.title('Pie Chart: Distribution of Labels', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
KernelLanguages = pl.read_csv("/kaggle/input/meta-kaggle/KernelLanguages.csv")
print(KernelLanguages.columns)
print(KernelLanguages.shape)
KernelLanguages.head()


KernelLanguages.to_pandas()


joined_df = KernelVersions.join(
    KernelLanguages,
    left_on="ScriptLanguageId",
    right_on="Id",
    how="left"
)
language_counts = joined_df.group_by("DisplayName").agg(
    count=pl.col("Id").count()
).sort("count", descending=True)
language_counts.to_pandas()


fig = px.bar(
    language_counts,
    x="DisplayName",
    y="count",
    title="Number of Kernel Versions per Language",
    labels={"DisplayName": "Programming Language", "count": "Number of Kernel Versions"},
    color="DisplayName",
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig.update_layout(
    xaxis_title="Programming Language",
    yaxis_title="Number of Kernel Versions",
    showlegend=False,
    title_x=0.5
)
fig.write_html("kernel_language_counts.html")
display(IFrame("kernel_language_counts.html", width=1200, height=800))


import gc
gc.collect()


import sys
for name, size in sorted(((name, sys.getsizeof(obj)) for name, obj in globals().items()), key=lambda x: -x[1])[:10]:
    print(f"{name}: {size/1e6:.2f} MB")


for name in dir():
    if not name.startswith('_'):
        del globals()[name]
import gc
gc.collect()


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import pandas as pd


# 6. KernelVersions.csv
KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
print(KernelVersions.columns)
print(KernelVersions.shape)
KernelVersions.head()


# 3. KernelVersionDatasetSources.csv
KernelVersionDatasetSources = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersionDatasetSources.csv")
print(KernelVersionDatasetSources.columns)
print(KernelVersionDatasetSources.shape)
KernelVersionDatasetSources.head()


DatasetVersions = pl.read_csv("/kaggle/input/meta-kaggle/DatasetVersions.csv")
print(DatasetVersions.columns)
print(DatasetVersions.shape)
DatasetVersions.head()


kernel_dataset_trends = (
    KernelVersionDatasetSources
    .join(KernelVersions, left_on="KernelVersionId", right_on="Id", suffix="_kernel")
    .join(DatasetVersions, left_on="SourceDatasetVersionId", right_on="Id", suffix="_dataset")
    .with_columns([
        pl.col("CreationDate").str.strptime(pl.Date, "%m/%d/%Y %H:%M:%S").alias("kernel_date"),
        pl.col("CreationDate_dataset").str.strptime(pl.Date, "%m/%d/%Y %H:%M:%S").alias("dataset_date")
    ])
    .with_columns([
        pl.col("kernel_date").dt.year().alias("year"),
        pl.col("kernel_date").dt.month().alias("month"),
        pl.col("kernel_date").dt.strftime("%Y-%m").alias("year_month")
    ])
)


monthly_usage = (
    kernel_dataset_trends
    .group_by("year_month")
    .agg([
        pl.count().alias("total_usages"),
        pl.col("SourceDatasetVersionId").n_unique().alias("unique_datasets_used")
    ])
    .sort("year_month")
)


monthly_usage_pd = monthly_usage.to_pandas()
monthly_usage_pd['year_month'] = pd.to_datetime(monthly_usage_pd['year_month'])
plt.figure(figsize=(15, 8))
plt.plot(monthly_usage_pd['year_month'], monthly_usage_pd['total_usages'], 
         marker='o', linewidth=2, markersize=4)
plt.title('Dataset Usage in Kernels Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Number of Dataset Usages', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


top_datasets = (
    kernel_dataset_trends
    .group_by("SourceDatasetVersionId")
    .agg([
        pl.count().alias("total_usage"),
        pl.col("Title_dataset").first().alias("dataset_title")
    ])
    .sort("total_usage", descending=True)
    .head(10)
)
top_dataset_ids = top_datasets.select("SourceDatasetVersionId").to_series().to_list()


dataset_evolution = (
    kernel_dataset_trends
    .filter(pl.col("SourceDatasetVersionId").is_in(top_dataset_ids))
    .group_by(["year_month", "SourceDatasetVersionId", "Title_dataset"])
    .agg(pl.count().alias("monthly_usage"))
    .sort(["year_month", "monthly_usage"], descending=[False, True])
)


dataset_evolution_pd = dataset_evolution.to_pandas()
dataset_evolution_pd['year_month'] = pd.to_datetime(dataset_evolution_pd['year_month'])

plt.figure(figsize=(15, 8))
for dataset_id in top_dataset_ids[:5]:  # Top 5 datasets
    data = dataset_evolution_pd[dataset_evolution_pd['SourceDatasetVersionId'] == dataset_id]
    if len(data) > 0:
        plt.plot(data['year_month'], data['monthly_usage'], 
                marker='o', linewidth=2, label=data['Title_dataset'].iloc[0][:30] + "...")
        
plt.title('Top 5 Datasets Usage Evolution Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Monthly Usage Count', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


dataset_age_usage = (
    kernel_dataset_trends
    .with_columns([
        (pl.col("kernel_date") - pl.col("dataset_date")).dt.total_days().alias("days_since_dataset_creation")
    ])
    .filter(pl.col("days_since_dataset_creation") >= 0)  # Only positive values
    .with_columns([
        (pl.col("days_since_dataset_creation") / 30).floor().alias("months_since_creation")
    ])
    .filter(pl.col("months_since_creation") <= 60)  # Focus on first 5 years
)

age_usage_summary = (
    dataset_age_usage
    .group_by("months_since_creation")
    .agg([
        pl.count().alias("usage_count"),
        pl.col("SourceDatasetVersionId").n_unique().alias("unique_datasets")
    ])
    .sort("months_since_creation")
)

age_usage_pd = age_usage_summary.to_pandas()


plt.figure(figsize=(15, 8))
plt.bar(age_usage_pd['months_since_creation'], age_usage_pd['usage_count'], 
        alpha=0.7, color='skyblue', edgecolor='navy')
plt.title('Dataset Usage by Age (Months Since Dataset Creation)', fontsize=16, fontweight='bold')
plt.xlabel('Months Since Dataset Creation', fontsize=12)
plt.ylabel('Number of Usages', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


language_trends = (
    kernel_dataset_trends
    .group_by(["year_month", "ScriptLanguageId"])
    .agg([
        pl.count().alias("usage_count"),
        pl.col("SourceDatasetVersionId").n_unique().alias("unique_datasets_used")
    ])
    .sort(["year_month", "usage_count"], descending=[False, True])
)


top_languages = (
    language_trends
    .group_by("ScriptLanguageId")
    .agg(pl.col("usage_count").sum().alias("total_usage"))
    .sort("total_usage", descending=True)
    .head(5)
    .select("ScriptLanguageId")
    .to_series()
    .to_list()
)


language_trends_filtered = language_trends.filter(
    pl.col("ScriptLanguageId").is_in(top_languages)
)

language_trends_pd = language_trends_filtered.to_pandas()
language_trends_pd['year_month'] = pd.to_datetime(language_trends_pd['year_month'])


plt.figure(figsize=(15, 8))
for lang_id in top_languages:
    data = language_trends_pd[language_trends_pd['ScriptLanguageId'] == lang_id]
    plt.plot(data['year_month'], data['usage_count'], 
             marker='o', linewidth=2, label=f'Language {lang_id}')

plt.title('Dataset Usage by Programming Language Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Number of Dataset Usages', fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


reuse_analysis = (
    kernel_dataset_trends
    .group_by(["year_month", "SourceDatasetVersionId"])
    .agg(pl.count().alias("monthly_usage_per_dataset"))
    .group_by("year_month")
    .agg([
        pl.count().alias("total_datasets_used"),
        pl.col("monthly_usage_per_dataset").filter(pl.col("monthly_usage_per_dataset") > 1).count().alias("reused_datasets"),
        pl.col("monthly_usage_per_dataset").filter(pl.col("monthly_usage_per_dataset") == 1).count().alias("single_use_datasets")
    ])
    .with_columns([
        (pl.col("reused_datasets") / pl.col("total_datasets_used") * 100).alias("reuse_percentage")
    ])
    .sort("year_month")
)


reuse_pd = reuse_analysis.to_pandas()
reuse_pd['year_month'] = pd.to_datetime(reuse_pd['year_month'])

plt.figure(figsize=(15, 8))
plt.plot(reuse_pd['year_month'], reuse_pd['reuse_percentage'], 
         marker='o', linewidth=2, color='red', markersize=4)
plt.title('Dataset Reuse Percentage Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Reuse Percentage (%)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


seasonal_analysis = (
    kernel_dataset_trends
    .with_columns([
        pl.col("kernel_date").dt.month().alias("month"),
        pl.col("kernel_date").dt.quarter().alias("quarter")
    ])
    .group_by("month")
    .agg([
        pl.count().alias("total_usage"),
        pl.col("SourceDatasetVersionId").n_unique().alias("unique_datasets")
    ])
    .sort("month")
)

seasonal_pd = seasonal_analysis.to_pandas()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


plt.figure(figsize=(12, 6))
plt.bar(range(1, 13), seasonal_pd['total_usage'], 
        alpha=0.7, color='lightgreen', edgecolor='darkgreen')
plt.title('Seasonal Patterns in Dataset Usage', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Dataset Usage', fontsize=12)
plt.xticks(range(1, 13), month_names)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


print(f"Total dataset-kernel relationships: {len(kernel_dataset_trends)}")
print(f"Date range: {kernel_dataset_trends.select(pl.col('kernel_date').min())} to {kernel_dataset_trends.select(pl.col('kernel_date').max())}")
print(f"Unique datasets used: {kernel_dataset_trends.select(pl.col('SourceDatasetVersionId').n_unique()).item()}")
print(f"Unique kernels: {kernel_dataset_trends.select(pl.col('KernelVersionId').n_unique()).item()}")
print("\nTop 10 Most Used Datasets:")
top_datasets.to_pandas()


print("\nMonthly Usage Statistics:")
monthly_usage.tail().to_pandas()


# 5. KernelVersionModelSources.csv
KernelVersionModelSources = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersionModelSources.csv")
print(KernelVersionModelSources.columns)
print(KernelVersionModelSources.shape)
KernelVersionModelSources.head()


# 5. KernelVersionModelSources.csv
ModelVariationVersions = pl.read_csv("/kaggle/input/meta-kaggle/ModelVariationVersions.csv")
print(ModelVariationVersions.columns)
print(ModelVariationVersions.shape)
ModelVariationVersions.head()


ModelVariations = pl.read_csv("/kaggle/input/meta-kaggle/ModelVariations.csv")
print(ModelVariations.columns)
print(ModelVariations.shape)
ModelVariations.head()


kernel_model_trends = (
    KernelVersionModelSources
    .join(KernelVersions, left_on="KernelVersionId", right_on="Id", how="inner")
    .join(ModelVariationVersions, left_on="SourceModelVariationVersionId", right_on="Id", how="inner")
    .join(ModelVariations, left_on="SourceModelVariationId", right_on="Id", how="inner")
    .select([
        "CreationDate",
        "ModelFramework", 
        "LicenseName",
        "SourceOrganizationName",
        "CurrentVariationSlug",
        "FineTunable"
    ])
    .with_columns([
        pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("parsed_date"),
        pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").dt.year().alias("year"),
        pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").dt.month().alias("month")
    ])
    .filter(pl.col("parsed_date").is_not_null())
)
df = kernel_model_trends.to_pandas()
df.head()


plt.figure(figsize=(14, 8))
framework_trends = df.groupby(['year', 'ModelFramework']).size().reset_index(name='count')
framework_pivot = framework_trends.pivot(index='year', columns='ModelFramework', values='count').fillna(0)

for framework in framework_pivot.columns:
    plt.plot(framework_pivot.index, framework_pivot[framework], marker='o', linewidth=2, label=framework)

plt.title('Model Framework Adoption Trends in Kernels Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Kernel-Model Connections', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 8))
org_trends = df.groupby(['year', 'SourceOrganizationName']).size().reset_index(name='count')
top_orgs = org_trends.groupby('SourceOrganizationName')['count'].sum().nlargest(3).index

org_filtered = org_trends[org_trends['SourceOrganizationName'].isin(top_orgs)]
org_pivot = org_filtered.pivot(index='year', columns='SourceOrganizationName', values='count').fillna(0)

for org in org_pivot.columns:
    plt.plot(org_pivot.index, org_pivot[org], marker='s', linewidth=2, label=org)

plt.title('Organizations: Model Usage in Kernels Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Kernel-Model Connections', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


org_pivot


# plt.figure(figsize=(12, 8))
# finetune_trends = df.groupby(['year', 'FineTunable']).size().reset_index(name='count')
# finetune_pivot = finetune_trends.pivot(index='year', columns='FineTunable', values='count').fillna(0)

# plt.plot(finetune_pivot.index, finetune_pivot.get('True', 0), marker='o', linewidth=3, label='Fine-tunable', color='green')
# plt.plot(finetune_pivot.index, finetune_pivot.get('False', 0), marker='s', linewidth=3, label='Non-Fine-tunable', color='red')

# plt.title('Fine-tunable vs Non-Fine-tunable Model Usage Trends', fontsize=16, fontweight='bold')
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Number of Kernel-Model Connections', fontsize=12)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(14, 8))
# recent_years = df[df['year'] >= 2023].copy()
# recent_years['year_month'] = recent_years['parsed_date'].dt.to_period('M')

# monthly_counts = recent_years.groupby('year_month').size().reset_index(name='count')
# monthly_counts['year_month_str'] = monthly_counts['year_month'].astype(str)

# plt.plot(range(len(monthly_counts)), monthly_counts['count'], marker='o', linewidth=2, color='purple')
# plt.title('Monthly Model Usage in Kernels', fontsize=16, fontweight='bold')
# plt.xlabel('Month', fontsize=12)
# plt.ylabel('Number of Kernel-Model Connections', fontsize=12)
# plt.xticks(range(0, len(monthly_counts), 3), monthly_counts['year_month_str'].iloc[::3], rotation=45)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(14, 8))
license_trends = df.groupby(['year', 'LicenseName']).size().reset_index(name='count')
top_licenses = license_trends.groupby('LicenseName')['count'].sum().nlargest(5).index

license_filtered = license_trends[license_trends['LicenseName'].isin(top_licenses)]
license_pivot = license_filtered.pivot(index='year', columns='LicenseName', values='count').fillna(0)

for license_type in license_pivot.columns:
    plt.plot(license_pivot.index, license_pivot[license_type], marker='d', linewidth=2, label=license_type)

plt.title('Model License Type Usage Trends in Kernels', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Kernel-Model Connections', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print(f"Total kernel-model connections: {len(df):,}")
print(f"Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")
print(f"Number of unique frameworks: {df['ModelFramework'].nunique()}")
print(f"Number of unique organizations: {df['SourceOrganizationName'].nunique()}")
print(f"Most popular framework: {df['ModelFramework'].value_counts().index[0]}")
print(f"Most active organization: {df['SourceOrganizationName'].value_counts().index[0]}")
print(f"Peak year for model usage: {df['year'].value_counts().index[0]}")


import gc
gc.collect()


import sys
for name, size in sorted(((name, sys.getsizeof(obj)) for name, obj in globals().items()), key=lambda x: -x[1])[:10]:
    print(f"{name}: {size/1e6:.2f} MB")


for name in dir():
    if not name.startswith('_'):
        del globals()[name]
import gc
gc.collect()


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


# 4. KernelVersionKernelSources.csv
KernelVersionKernelSources = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersionKernelSources.csv")
print(KernelVersionKernelSources.columns)
print(KernelVersionKernelSources.shape)
KernelVersionKernelSources.head()


# 6. KernelVersions.csv
KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
print(KernelVersions.columns)
print(KernelVersions.shape)
KernelVersions.head()


# 7. Kernels.csv
Kernels = pl.read_csv("/kaggle/input/meta-kaggle/Kernels.csv")
print(Kernels.columns)
print(Kernels.shape)
Kernels.head()


joined_data = KernelVersionKernelSources.join(
    KernelVersions, 
    left_on="KernelVersionId", 
    right_on="Id",
    how="inner"
)
joined_data = joined_data.with_columns([
    pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("creation_datetime")
])
joined_data = joined_data.with_columns([
    pl.col("creation_datetime").dt.year().alias("year"),
    pl.col("creation_datetime").dt.month().alias("month"),
    pl.col("creation_datetime").dt.strftime("%Y-%m").alias("year_month")
])
joined_data.head()


monthly_forking = joined_data.group_by("year_month").agg([
    pl.count().alias("fork_count"),
    pl.col("KernelVersionId").n_unique().alias("unique_kernels_forked")
])

monthly_forking = monthly_forking.sort("year_month")
print("Monthly Kernel Forking Trends:")
print(monthly_forking.head(10))

plt.figure(figsize=(12, 6))
plt.plot(monthly_forking["year_month"], monthly_forking["fork_count"], marker='o')
plt.title("Monthly Kernel Forking Activity Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Number of Forks")
plt.xticks(rotation=90, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


yearly_forking = joined_data.group_by("year").agg([
    pl.count().alias("fork_count"),
    pl.col("KernelVersionId").n_unique().alias("unique_kernels_forked"),
    pl.col("SourceKernelVersionId").n_unique().alias("unique_source_kernels")
])

yearly_forking = yearly_forking.sort("year")
print("\nYearly Kernel Forking Trends:")
print(yearly_forking)

plt.figure(figsize=(10, 6))
plt.bar(yearly_forking["year"], yearly_forking["fork_count"], alpha=0.7)
plt.title("Yearly Kernel Forking Activity")
plt.xlabel("Year")
plt.ylabel("Number of Forks")
plt.grid(True, alpha=0.3)
plt.show()


most_forked = joined_data.group_by("SourceKernelVersionId").agg([
    pl.count().alias("times_forked"),
    pl.col("Title").first().alias("kernel_title"),
    pl.col("creation_datetime").min().alias("first_fork_date"),
    pl.col("creation_datetime").max().alias("last_fork_date")
]).sort("times_forked", descending=True)

print("\nMost Forked Kernels:")
most_forked.head(10)


language_trends = joined_data.group_by(["year_month", "ScriptLanguageId"]).agg([
    pl.count().alias("fork_count")
]).sort(["year_month", "ScriptLanguageId"])

print("\nForking Activity by Language Over Time (sample):")
language_trends.head()


forking_intensity = joined_data.group_by("year_month").agg([
    pl.count().alias("total_forks"),
    pl.col("SourceKernelVersionId").n_unique().alias("unique_source_kernels")
]).with_columns([
    (pl.col("total_forks") / pl.col("unique_source_kernels")).alias("forks_per_kernel")
]).sort("year_month")

print("\nForking Intensity Over Time:")
print(forking_intensity.head(10))

plt.figure(figsize=(12, 6))
plt.plot(forking_intensity["year_month"], forking_intensity["forks_per_kernel"], marker='o', color='red')
plt.title("Average Forks per Kernel Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Forks per Kernel")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


growth_data = monthly_forking.with_columns([
    pl.col("fork_count").pct_change().alias("fork_growth_rate")
]).sort("year_month")

print("\nMonthly Growth Rate in Forking Activity:")
print(growth_data.head(10))
growth_data_trimmed = growth_data.slice(2)
plt.figure(figsize=(12, 6))
plt.plot(
    growth_data_trimmed["year_month"], 
    growth_data_trimmed["fork_growth_rate"], 
    marker='o', 
    color='green'
)
plt.title("Monthly Growth Rate in Forking Activity")
plt.xlabel("Year-Month")
plt.ylabel("Growth Rate")
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



import gc
gc.collect()


import sys

for name, size in sorted(((name, sys.getsizeof(obj)) for name, obj in globals().items()), key=lambda x: -x[1])[:10]:
    print(f"{name}: {size/1e6:.2f} MB")


for name in dir():
    if not name.startswith('_'):
        del globals()[name]
import gc
gc.collect()


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd


# 6. KernelVersions.csv
KernelVersions = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersions.csv")
print(KernelVersions.columns)
print(KernelVersions.shape)
KernelVersions.head()


# 6. Kernels.csv
Kernels = pl.read_csv("/kaggle/input/meta-kaggle/Kernels.csv")
print(Kernels.columns)
print(Kernels.shape)
Kernels.head()


# 2. KernelVersionCompetitionSources.csv
KernelVersionCompetitionSources = pl.read_csv("/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv")
print(KernelVersionCompetitionSources.columns)
print(KernelVersionCompetitionSources.shape)
KernelVersionCompetitionSources.head()


# 9. Competitions.csv
Competitions = pl.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
print(Competitions.columns)
print(Competitions.shape)
Competitions.head()


merged_data = (
    KernelVersionCompetitionSources
    .join(KernelVersions, left_on="KernelVersionId", right_on="Id", how="left")
    .join(Competitions, left_on="SourceCompetitionId", right_on="Id", how="left")
    .with_columns([
        pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("kernel_creation_date"),
        pl.col("EnabledDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("competition_enabled_date"),
        pl.col("DeadlineDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("competition_deadline_date")
    ])
    .with_columns([
        pl.col("kernel_creation_date").dt.year().alias("kernel_year"),
        pl.col("kernel_creation_date").dt.month().alias("kernel_month"),
        pl.col("competition_enabled_date").dt.year().alias("competition_year")
    ])
    .filter(pl.col("kernel_creation_date").is_not_null())
)


monthly_kernels = (
    merged_data
    .with_columns(pl.col("kernel_creation_date").dt.truncate("1mo").alias("month"))
    .group_by("month")
    .agg(pl.count().alias("kernel_count"))
    .sort("month")
)
months = monthly_kernels.select(pl.col("month")).to_series().to_list()
counts = monthly_kernels.select(pl.col("kernel_count")).to_series().to_list()

plt.figure(figsize=(15, 8))
plt.plot(months, counts, linewidth=2, color='steelblue')
plt.title('Monthly Kernel Creation Activity Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('Number of Kernels Created', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


reward_trends = (
    merged_data
    .with_columns(pl.col("kernel_creation_date").dt.truncate("1q").alias("quarter"))
    .group_by(["quarter", "RewardType"])
    .agg(pl.count().alias("kernel_count"))
    .sort(["quarter", "RewardType"])
)
quarters = sorted(reward_trends.select("quarter").unique().to_series().to_list())
reward_types = sorted(reward_trends.select("RewardType").unique().to_series().to_list())

plt.figure(figsize=(15, 8))
for reward_type in reward_types:
    reward_data = reward_trends.filter(pl.col("RewardType") == reward_type)
    quarters_data = reward_data.select("quarter").to_series().to_list()
    counts_data = reward_data.select("kernel_count").to_series().to_list()
    plt.plot(quarters_data, counts_data, linewidth=2, marker='o', markersize=4, label=reward_type)

plt.title('Quarterly Kernel Activity by Competition Reward Type', fontsize=16, fontweight='bold')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Number of Kernels', fontsize=12)
plt.legend(title='Reward Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


lines_vs_age = (
    merged_data
    .with_columns([
        (pl.col("kernel_creation_date") - pl.col("competition_enabled_date")).dt.total_days().alias("competition_age_days"),
        pl.col("TotalLines").cast(pl.Float64, strict=False).alias("total_lines_numeric")
    ])
    .filter(pl.col("competition_age_days").is_not_null() & pl.col("total_lines_numeric").is_not_null())
    .with_columns((pl.col("competition_age_days") / 30).round(0).alias("age_bins"))  # 30-day bins
    .group_by("age_bins")
    .agg(pl.col("total_lines_numeric").mean().alias("avg_lines"))
    .sort("age_bins")
)
age_bins = lines_vs_age.select("age_bins").to_series().to_list()
avg_lines = lines_vs_age.select("avg_lines").to_series().to_list()
plt.figure(figsize=(15, 8))
plt.plot(age_bins, avg_lines, linewidth=3, color='darkgreen', marker='s', markersize=6)
plt.title('Average Kernel Length vs Competition Age', fontsize=16, fontweight='bold')
plt.xlabel('Competition Age (Months since enabled)', fontsize=12)
plt.ylabel('Average Total Lines in Kernels', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


deadline_activity = (
    merged_data
    .with_columns(
        (pl.col("competition_deadline_date") - pl.col("kernel_creation_date")).dt.total_days().alias("days_to_deadline")
    )
    .filter(pl.col("days_to_deadline").is_between(-30, 30))
    .group_by("days_to_deadline")
    .agg(pl.count().alias("kernel_count"))
    .sort("days_to_deadline")
)
days = deadline_activity.select("days_to_deadline").to_series().to_list()
counts = deadline_activity.select("kernel_count").to_series().to_list()
plt.figure(figsize=(15, 8))
plt.bar(days, counts, color='coral', alpha=0.8)
plt.title('Kernel Creation Activity Around Competition Deadlines', fontsize=16, fontweight='bold')
plt.xlabel('Days to Competition Deadline (Negative = After Deadline)', fontsize=12)
plt.ylabel('Number of Kernels Created', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Deadline')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


yearly_connections = (
    merged_data
    .group_by("kernel_year")
    .agg([
        pl.count().alias("total_kernels"),
        pl.col("SourceCompetitionId").n_unique().alias("unique_competitions")
    ])
    .sort("kernel_year")
)

years = yearly_connections.select("kernel_year").to_series().to_list()
total_kernels = yearly_connections.select("total_kernels").to_series().to_list()
unique_comps = yearly_connections.select("unique_competitions").to_series().to_list()

plt.figure(figsize=(15, 8))
ax1 = plt.gca()
ax1.plot(years, total_kernels, 'b-o', linewidth=3, markersize=8, label='Total Kernels')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Kernels', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(years, unique_comps, 'r-s', linewidth=3, markersize=8, label='Unique Competitions')
ax2.set_ylabel('Number of Competitions', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Yearly Growth: Kernels vs Competitions Involved', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# yearly_connections.to_pandas()


Kernels = Kernels.with_columns(
    pl.col("CreationDate").str.strptime(pl.Date, "%m/%d/%Y %H:%M:%S", strict=False).dt.year().alias("kernel_year")
)
Kernels = Kernels.filter(pl.col("kernel_year").is_not_null())
merged_versions = KernelVersions.join(
    KernelVersionCompetitionSources,
    left_on="Id",
    right_on="KernelVersionId",
    how="inner",
    suffix="_comp" 
)
merged_data_k = Kernels.join(
    merged_versions,
    left_on="Id",
    right_on="ScriptId",
    how="inner",
    suffix="_version"
)
yearly_connections = (
    merged_data_k
    .group_by("kernel_year")
    .agg([
        pl.col("Id").n_unique().alias("total_kernels"),  
        pl.col("SourceCompetitionId").n_unique().alias("unique_competitions") 
    ])
    .sort("kernel_year")
)
years = yearly_connections.select("kernel_year").to_series().to_list()
total_kernels = yearly_connections.select("total_kernels").to_series().to_list()
unique_comps = yearly_connections.select("unique_competitions").to_series().to_list()
plt.figure(figsize=(15, 8))
ax1 = plt.gca()
ax1.plot(years, total_kernels, 'b-o', linewidth=3, markersize=8, label='Total Kernels')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Kernels', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(years, unique_comps, 'r-s', linewidth=3, markersize=8, label='Unique Competitions')
ax2.set_ylabel('Number of Competitions', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Yearly Growth: Unique Kernels vs Competitions Involved', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# yearly_connections.to_pandas()


seasonal_data = (
    merged_data
    .with_columns(pl.col("kernel_creation_date").dt.month().alias("month"))
    .group_by(["kernel_year", "month"])
    .agg(pl.count().alias("kernel_count"))
    .sort(["kernel_year", "month"])
)
years = sorted(seasonal_data.select("kernel_year").unique().to_series().to_list())
months = list(range(1, 13))
heatmap_data = []
for year in years:
    year_row = []
    for month in months:
        count = seasonal_data.filter(
            (pl.col("kernel_year") == year) & (pl.col("month") == month)
        ).select("kernel_count").to_series()
        year_row.append(count[0] if len(count) > 0 else 0)
    heatmap_data.append(year_row)

plt.figure(figsize=(15, 8))
plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
plt.colorbar(label='Number of Kernels')
plt.title('Seasonal Heatmap: Kernel Creation by Month and Year', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.xticks(range(12), months)
plt.yticks(range(len(years)), years)
plt.tight_layout()
plt.show()


host_trends = (
    merged_data
    .group_by(["kernel_year", "HostSegmentTitle"])
    .agg(pl.count().alias("kernel_count"))
    .sort(["kernel_year", "HostSegmentTitle"])
)
years = sorted(host_trends.select("kernel_year").unique().to_series().to_list())
host_segments = sorted(host_trends.select("HostSegmentTitle").unique().to_series().to_list())

plt.figure(figsize=(15, 8))
for host_segment in host_segments:
    host_data = host_trends.filter(pl.col("HostSegmentTitle") == host_segment)
    years_data = host_data.select("kernel_year").to_series().to_list()
    counts_data = host_data.select("kernel_count").to_series().to_list()
    plt.plot(years_data, counts_data, linewidth=2, marker='o', markersize=5, label=host_segment)

plt.title('Annual Kernel Activity by Competition Host Segment', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Kernels', fontsize=12)
plt.legend(title='Host Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


total_connections = merged_data.height
date_range = merged_data.select([
    pl.col("kernel_creation_date").min().alias("min_date"),
    pl.col("kernel_creation_date").max().alias("max_date")
])
unique_competitions = merged_data.select(pl.col("SourceCompetitionId").n_unique()).item()
unique_kernels = merged_data.select(pl.col("KernelVersionId").n_unique()).item()


print(f"Total kernel-competition connections: {total_connections:,}")
print(f"Date range: {date_range.item(0, 0)} to {date_range.item(0, 1)}")
print(f"Number of unique competitions: {unique_competitions:,}")
print(f"Number of unique kernels: {unique_kernels:,}")
print(f"Average kernels per competition: {total_connections / unique_competitions:.2f}")

