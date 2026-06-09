import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import IFrame
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import kagglehub
import os
pio.templates.default = "plotly_white"


MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")
print("âœ… Downloaded Meta-Kaggle data.")
print("ğŸ“‚ MK_PATH =", MK_PATH)
print("ğŸ“‚ MKC_PATH =", MKC_PATH)


# import polars as pl
# import os
# csv_files = [file for file in os.listdir(MK_PATH) if file.endswith(".csv")]
# for file in csv_files:
#     print(f"\nğŸ“„ Reading: {file}")
#     file_path = os.path.join(MK_PATH, file)
#     # try:
#         df = pl.read_csv(file_path)
#         print("ğŸ§© Columns and Dtypes:")
#         for col, dtype in zip(df.columns, df.dtypes):
#             print(f"   - {col}: {dtype}")
#         print("ğŸ”¢ Shape:", df.shape)
#     except Exception as e:
#         print(f"â�Œ Error reading {file}: {e}")


Forums = pl.read_csv("/kaggle/input/meta-kaggle/Forums.csv")
print(Forums.columns)
print(Forums.shape)
Forums.head()


ForumTopics = pl.read_csv("/kaggle/input/meta-kaggle/ForumTopics.csv")
print(ForumTopics.columns)
print(ForumTopics.shape)
ForumTopics.head()


ForumTopics = ForumTopics.with_columns(
    pl.col("CreationDate").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S")
)
forums_per_year = (
    ForumTopics.with_columns(pl.col("CreationDate").dt.year().alias("Year"))
    .group_by("Year")
    .agg(pl.len().alias("NumTopics"))
    .sort("Year")
)
forums_per_year.head()


forums_per_year_pd = forums_per_year.to_pandas()
fig = px.line(
    forums_per_year_pd,
    x="Year",
    y="NumTopics",
    title="Number of Forum Topics Created per Year",
    labels={"Year": "Year", "NumTopics": "Number of Topics"},
    markers=True,
    text="NumTopics"
)
fig.update_traces(textposition="top center")
fig.update_layout(
    xaxis=dict(tickmode="linear"),
    yaxis_title="Number of Topics",
    xaxis_title="Year",
    template="plotly_white"
)
fig.write_html("forum_year_wise.html", include_plotlyjs="cdn")
display(IFrame("forum_year_wise.html", width=1200, height=700))


unique_Forumsoc = ForumTopics["Score"].unique().sort()
print(unique_Forumsoc)


ForumMessages = pl.read_csv("/kaggle/input/meta-kaggle/ForumMessages.csv")
print(ForumMessages.columns)
print(ForumMessages.shape)
ForumMessages.head()


max_topic_row = ForumTopics.sort("Score", descending=True).select(["Id", "ForumId", "Score"]).row(0)
min_topic_row = ForumTopics.sort("Score").select(["Id", "ForumId", "Score"]).row(0)


max_topic_id, max_forum_id, max_score = max_topic_row
min_topic_id, min_forum_id, min_score = min_topic_row


max_message = ForumMessages.filter(pl.col("ForumTopicId") == max_topic_id).select("Message").item(0, 0)
min_message = ForumMessages.filter(pl.col("ForumTopicId") == min_topic_id).select("Message").item(0, 0)


max_forum_title = Forums.filter(pl.col("Id") == int(max_forum_id)).select("Title").item(0, 0)
min_forum_title = Forums.filter(pl.col("Id") == int(min_forum_id)).select("Title").item(0, 0)


print("Forum with Highest Score:")
print(f"Forum ID: {max_forum_id}, Score: {max_score}, Title: {max_forum_title}")
print("Message:\n", max_message)


print("\nForum with Lowest Score:")
print(f"Forum ID: {min_forum_id}, Score: {min_score}, Title: {min_forum_title}")
print("Message:\n", min_message)


ForumMessageVotes = pl.read_csv("/kaggle/input/meta-kaggle/ForumMessageVotes.csv")
print(ForumMessageVotes.columns)
print(ForumMessageVotes.shape)
ForumMessageVotes.head()


ForumMessageReactions = pl.read_csv("/kaggle/input/meta-kaggle/ForumMessageReactions.csv")
print(ForumMessageReactions.columns)
print(ForumMessageReactions.shape)
ForumMessageReactions.head()


reaction_counts = (
    ForumMessageReactions
    .group_by("ReactionType")
    .agg(pl.len().alias("TotalCount"))
    .sort("TotalCount", descending=True)
)

print("ğŸ”� Reaction Types and Total Usage:")
reaction_counts.to_pandas()


ForumMessageReactions = ForumMessageReactions.with_columns(
    pl.col("ReactionDate").str.strptime(pl.Date, "%m/%d/%Y").alias("ParsedDate")
)

ForumMessageReactions = ForumMessageReactions.with_columns(
    pl.col("ParsedDate").dt.year().alias("Year")
)


reaction_yearly_counts = (
    ForumMessageReactions
    .group_by(["Year", "ReactionType"])
    .agg(pl.len().alias("ReactionCount"))
    .sort(["Year", "ReactionType"])
)

print("ğŸ“† Year-wise Reaction Counts by Type:")
reaction_yearly_counts.to_pandas()


reaction_year_counts = (
    ForumMessageReactions
    .group_by(["Year", "ReactionType"])
    .agg(pl.len().alias("ReactionCount"))
    .sort(["Year", "ReactionType"])
)
reaction_df = reaction_year_counts.to_pandas()
pivot = reaction_df.pivot(index="Year", columns="ReactionType", values="ReactionCount").fillna(0)
cumsum_df = pivot.cumsum().reset_index()
long_df = cumsum_df.melt(id_vars="Year", var_name="ReactionType", value_name="CumulativeCount")
fig_bar = px.bar(
    long_df,
    x="CumulativeCount",
    y="ReactionType",
    orientation="h",
    animation_frame="Year",
    color="ReactionType",
    title="Cumulative Reactions per Type Over Years",
    range_x=[0, long_df["CumulativeCount"].max() * 1.1],
    height=600
)
fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
fig_bar.write_html("cumulative_reactions_by_type.html", include_plotlyjs="cdn")
fig_bar.show(renderer="iframe")


Tags = pl.read_csv("/kaggle/input/meta-kaggle/Tags.csv")
print(Tags.columns)
print(Tags.shape)
Tags.head()


KernelTags = pl.read_csv("/kaggle/input/meta-kaggle/KernelTags.csv")
print(KernelTags.columns)
print(KernelTags.shape)
KernelTags.head()


ModelTags = pl.read_csv("/kaggle/input/meta-kaggle/ModelTags.csv")
print(ModelTags.columns)
print(ModelTags.shape)
ModelTags.head()


DatasetTags = pl.read_csv("/kaggle/input/meta-kaggle/DatasetTags.csv")
print(DatasetTags.columns)
print(DatasetTags.shape)
DatasetTags.head()


CompetitionTags = pl.read_csv("/kaggle/input/meta-kaggle/CompetitionTags.csv")
print(CompetitionTags.columns)
print(CompetitionTags.shape)
CompetitionTags.head()


def top_tags_by_usage(tag_df, context_name):
    return (
        tag_df
        .group_by("TagId")
        .agg(pl.len().alias("UsageCount"))
        .join(Tags.select(["Id", "Name"]), left_on="TagId", right_on="Id", how="left")
        .sort("UsageCount", descending=True)
        .select(["Name", "UsageCount"])
        .head(5)
        .with_columns(pl.lit(context_name).alias("Context"))
    )


top_kernel_tags = top_tags_by_usage(KernelTags, "Kernel")
top_model_tags = top_tags_by_usage(ModelTags, "Model")
top_dataset_tags = top_tags_by_usage(DatasetTags, "Dataset")
top_competition_tags = top_tags_by_usage(CompetitionTags, "Competition")


top_tags_all = pl.concat([
    top_kernel_tags,
    top_model_tags,
    top_dataset_tags,
    top_competition_tags
])
top_tags_all.to_pandas()


Tags = pl.read_csv("/kaggle/input/meta-kaggle/Tags.csv")
KernelTags = pl.read_csv("/kaggle/input/meta-kaggle/KernelTags.csv")
Kernels = pl.read_csv("/kaggle/input/meta-kaggle/Kernels.csv")


Kernels = Kernels.with_columns(
    pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S", strict=False).alias("ParsedDate")
).with_columns(
    pl.col("ParsedDate").dt.year().alias("Year")
)
kernel_tags_with_year = KernelTags.join(Kernels.select(["Id", "Year"]), left_on="KernelId", right_on="Id", how="inner")



tag_counts = (
    kernel_tags_with_year
    .group_by(["TagId", "Year"])
    .agg(pl.len().alias("Count"))
    .join(Tags.select(["Id", "Name"]), left_on="TagId", right_on="Id", how="left")
    .select(["Name", "Year", "Count"])
)
tag_counts.head()


top_tags = (
    tag_counts.group_by("Name").agg(pl.sum("Count").alias("TotalCount"))
    .sort("TotalCount", descending=True).head(10).select("Name")
)


filtered = tag_counts.join(top_tags, on="Name", how="inner").sort(["Year", "Count"], descending=False)
df = filtered.to_pandas()


fig = px.bar(
    df, x="Count", y="Name", color="Name", animation_frame="Year",
    orientation="h", title="Top Kernel Tags Over Time",
    range_x=[0, df["Count"].max() * 1.2], height=600
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.write_html("kerneltagsevol.html", include_plotlyjs="cdn")
display(IFrame("kerneltagsevol.html", width=1200, height=700))


import pandas as pd
import matplotlib.pyplot as plt
dv = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Kernals/Data Visualization.csv", skiprows=1)
eda = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Kernals/Exploratory Data Analysis.csv", skiprows=1)
beginner = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Kernals/Beginner.csv", skiprows=1)
classification = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Kernals/Classification.csv", skiprows=1)
pandas_ = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Kernals/Pandas.csv", skiprows=1)
numpy_ = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Kernals/NumPy.csv", skiprows=1)
def total_interest(df):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    df_2024 = df[df['Date'].dt.year == 2024]
    return df_2024[df.columns[1]].sum()
kaggle_counts = {
    'Data Visualization': 32,
    'Exploratory Data Analysis': 19,
    'Beginner': 10,
    'Classification': 5,
    'Pandas': 5,
    'NumPy': 5
}
total_kaggle = sum(kaggle_counts.values())
kaggle_share = {k: round((v / total_kaggle) * 100, 2) for k, v in kaggle_counts.items()}
google_vals = {
    'Data Visualization': total_interest(dv),
    'Exploratory Data Analysis': total_interest(eda),
    'Beginner': total_interest(beginner),
    'Classification': total_interest(classification),
    'Pandas': total_interest(pandas_),
    'NumPy': total_interest(numpy_)
}
df = pd.DataFrame.from_dict(google_vals, orient='index', columns=['GoogleTrend'])
df['GoogleShare%'] = df['GoogleTrend'] / df['GoogleTrend'].sum() * 100
df['KaggleShare%'] = df.index.map(kaggle_share)
df = df.round(2)
print(df)
df[['GoogleShare%', 'KaggleShare%']].plot(
    kind='bar', figsize=(10, 6), rot=45, title='Kernel Tags: Kaggle vs Google Trends (2024)', colormap='Set3'
)
plt.ylabel('Share (%)')
plt.grid(True)
plt.tight_layout()
plt.show()



DatasetTags = pl.read_csv("/kaggle/input/meta-kaggle/DatasetTags.csv")
Datasets = pl.read_csv("/kaggle/input/meta-kaggle/Datasets.csv")


Datasets = Datasets.with_columns(
    pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("ParsedDate")
).with_columns(
    pl.col("ParsedDate").dt.year().alias("Year")
)


dataset_tags_with_year = DatasetTags.join(Datasets.select(["Id", "Year"]), left_on="DatasetId", right_on="Id", how="inner")
tag_counts = (
    dataset_tags_with_year
    .group_by(["TagId", "Year"])
    .agg(pl.len().alias("Count"))
    .join(Tags.select(["Id", "Name"]), left_on="TagId", right_on="Id", how="left")
    .select(["Name", "Year", "Count"])
)
tag_counts.head()


top_tags = (
    tag_counts.group_by("Name").agg(pl.sum("Count").alias("TotalCount"))
    .sort("TotalCount", descending=True).head(10).select("Name")
)
filtered = tag_counts.join(top_tags, on="Name", how="inner").sort(["Year", "Count"], descending=False)
df = filtered.to_pandas()


fig = px.bar(
    df, x="Count", y="Name", color="Name", animation_frame="Year",
    orientation="h", title="Top Dataset Tags Over Time",
    range_x=[0, df["Count"].max() * 1.2], height=600
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.write_html("datasettagsevol.html", include_plotlyjs="cdn")
display(IFrame("datasettagsevol.html", width=1200, height=700))


import pandas as pd
import matplotlib.pyplot as plt

business = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Datasets/Business.csv", skiprows=1)
earth = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Datasets/Earth and Nature.csv", skiprows=1)
tabular = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Datasets/Tabular.csv", skiprows=1)
cs = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Datasets/Computer Science.csv", skiprows=1)
edu = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Datasets/Education.csv", skiprows=1)

def total_interest(df):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    df_2024 = df[df['Date'].dt.year == 2024]
    return df_2024[df.columns[1]].sum()

kaggle_counts = {
    'Business': 32,
    'Earth and Nature': 19,
    'Tabular': 19,
    'Computer Science': 15,
    'Education': 12
}

total_kaggle = sum(kaggle_counts.values())
kaggle_share = {k: round((v / total_kaggle) * 100, 2) for k, v in kaggle_counts.items()}

google_vals = {
    'Business': total_interest(business),
    'Earth and Nature': total_interest(earth),
    'Tabular': total_interest(tabular),
    'Computer Science': total_interest(cs),
    'Education': total_interest(edu)
}

df = pd.DataFrame.from_dict(google_vals, orient='index', columns=['GoogleTrend'])
df['GoogleShare%'] = df['GoogleTrend'] / df['GoogleTrend'].sum() * 100
df['KaggleShare%'] = df.index.map(kaggle_share)
df = df.round(2)
print(df)

df[['GoogleShare%', 'KaggleShare%']].plot(
    kind='bar', figsize=(10, 6), rot=45, title='Dataset Tags: Kaggle vs Google Trends (2024)', colormap='Set2'
)
plt.ylabel('Share (%)')
plt.grid(True)
plt.tight_layout()
plt.show()


CompetitionTags = pl.read_csv("/kaggle/input/meta-kaggle/CompetitionTags.csv")
Competitions = pl.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")


Competitions = Competitions.with_columns(
    pl.col("EnabledDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("ParsedDate")
).with_columns(
    pl.col("ParsedDate").dt.year().alias("Year")
)
competition_tags_with_year = CompetitionTags.join(
    Competitions.select(["Id", "Year"]), left_on="CompetitionId", right_on="Id", how="inner"
)


tag_counts = (
    competition_tags_with_year
    .group_by(["TagId", "Year"])
    .agg(pl.len().alias("Count"))
    .join(Tags.select(["Id", "Name"]), left_on="TagId", right_on="Id", how="left")
    .select(["Name", "Year", "Count"])
)
tag_counts.head()


top_tags = (
    tag_counts.group_by("Name").agg(pl.sum("Count").alias("TotalCount"))
    .sort("TotalCount", descending=True).head(10).select("Name")
)
filtered = tag_counts.join(top_tags, on="Name", how="inner").sort(["Year", "Count"], descending=False)
df = filtered.to_pandas()


fig = px.bar(
    df, x="Count", y="Name", color="Name", animation_frame="Year",
    orientation="h", title="Top Competition Tags Over Time",
    range_x=[0, df["Count"].max() * 1.2], height=600
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.write_html("competitiontagsevol.html", include_plotlyjs="cdn")
display(IFrame("competitiontagsevol.html", width=1200, height=700))


import pandas as pd
import matplotlib.pyplot as plt

tabular = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Competitions/Tabular.csv", skiprows=1)
image = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Competitions/Image.csv", skiprows=1)
text = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Competitions/Text.csv", skiprows=1)

def total_interest(df):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    df_2024 = df[df['Date'].dt.year == 2024]
    return df_2024[df.columns[1]].sum()

kaggle_counts = {
    'Tabular': 32,
    'Image': 19,
    'Text': 5
}

total_kaggle = sum(kaggle_counts.values())
kaggle_share = {k: round((v / total_kaggle) * 100, 2) for k, v in kaggle_counts.items()}

google_vals = {
    'Tabular': total_interest(tabular),
    'Image': total_interest(image),
    'Text': total_interest(text)
}

df = pd.DataFrame.from_dict(google_vals, orient='index', columns=['GoogleTrend'])
df['GoogleShare%'] = df['GoogleTrend'] / df['GoogleTrend'].sum() * 100
df['KaggleShare%'] = df.index.map(kaggle_share)
df = df.round(2)
print(df)

df[['GoogleShare%', 'KaggleShare%']].plot(
    kind='bar', figsize=(8, 6), rot=45, title='Competition Tags: Kaggle vs Google Trends (2024)', colormap='Accent'
)
plt.ylabel('Share (%)')
plt.grid(True)
plt.tight_layout()
plt.show()


Tags = pl.read_csv("/kaggle/input/meta-kaggle/Tags.csv")
ModelTags = pl.read_csv("/kaggle/input/meta-kaggle/ModelTags.csv")
ModelVersions = pl.read_csv("/kaggle/input/meta-kaggle/ModelVersions.csv")


ModelVersions = ModelVersions.with_columns(
    pl.col("CreationDate").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S").alias("ParsedDate")
).with_columns(
    pl.col("ParsedDate").dt.year().alias("Year")
)
model_tags_with_year = (
    ModelTags.join(ModelVersions.select(["ModelId", "Year"]), on="ModelId", how="inner")
)


tag_counts = (
    model_tags_with_year
    .group_by(["TagId", "Year"])
    .agg(pl.len().alias("Count"))
    .join(Tags.select(["Id", "Name"]), left_on="TagId", right_on="Id", how="left")
    .select(["Name", "Year", "Count"])
)
tag_counts.head()


top_tags = (
    tag_counts
    .group_by("Name")
    .agg(pl.sum("Count").alias("TotalCount"))
    .sort("TotalCount", descending=True)
    .head(10)
    .select("Name")
)
filtered = tag_counts.join(top_tags, on="Name", how="inner").sort(["Year", "Count"], descending=False)
df = filtered.to_pandas()


fig = px.bar(
    df,
    x="Count",
    y="Name",
    color="Name",
    animation_frame="Year",
    orientation="h",
    title="Top Model Tags Over Time",
    range_x=[0, df["Count"].max() * 1.2],
    height=600
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.write_html("modeltagsevol.html", include_plotlyjs="cdn")
display(IFrame("modeltagsevol.html", width=1200, height=700))


import pandas as pd
import matplotlib.pyplot as plt

english = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/English.csv", skiprows=1)
image = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Image.csv", skiprows=1)
text = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Text.csv", skiprows=1)
pytorch = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Pytorch.csv", skiprows=1)
dl = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Deep Learning.csv", skiprows=1)
cv = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Computer Vision.csv", skiprows=1)
nlp = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/NLP.csv", skiprows=1)
tg = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Text Generation.csv", skiprows=1)
ic = pd.read_csv("/kaggle/input/metakaggle5tags/Google Trend Tags/Models/Image Classification.csv", skiprows=1)

def total_interest(df):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df[df.columns[0]], errors='coerce')
    df_2024 = df[df['Date'].dt.year == 2024]
    return df_2024[df.columns[1]].sum()

kaggle_counts = {
    'English': 23,
    'Image': 14,
    'Text': 14,
    'Pytorch': 11,
    'Deep Learning': 9,
    'Computer Vision': 7,
    'NLP': 7,
    'Text Generation': 11,
    'Image Classification': 6
}

total_kaggle = sum(kaggle_counts.values())
kaggle_share = {k: round((v / total_kaggle) * 100, 2) for k, v in kaggle_counts.items()}

google_vals = {
    'English': total_interest(english),
    'Image': total_interest(image),
    'Text': total_interest(text),
    'Pytorch': total_interest(pytorch),
    'Deep Learning': total_interest(dl),
    'Computer Vision': total_interest(cv),
    'NLP': total_interest(nlp),
    'Text Generation': total_interest(tg),
    'Image Classification': total_interest(ic)
}

df = pd.DataFrame.from_dict(google_vals, orient='index', columns=['GoogleTrend'])
df['GoogleShare%'] = df['GoogleTrend'] / df['GoogleTrend'].sum() * 100
df['KaggleShare%'] = df.index.map(kaggle_share)
df = df.round(2)
print(df)

df[['GoogleShare%', 'KaggleShare%']].plot(
    kind='bar', figsize=(12, 6), rot=45, title='Model Tags: Kaggle vs Google Trends (2024)', colormap='Set3'
)
plt.ylabel('Share (%)')
plt.grid(True)
plt.tight_layout()
plt.show()



Organizations = pl.read_csv("/kaggle/input/meta-kaggle/Organizations.csv")
print(Organizations.columns)
print(Organizations.shape)
Organizations.head()


Organizations = Organizations.with_columns([
    pl.col("CreationDate").str.strptime(pl.Date, format="%m/%d/%Y").alias("ParsedDate"),
    pl.col("CreationDate").str.strptime(pl.Date, format="%m/%d/%Y").dt.year().alias("Year")
])


yearly_counts = Organizations.group_by("Year").agg([
    pl.len().alias("OrganizationCount")
]).sort("Year")


df = yearly_counts.to_pandas()
fig = px.line(df, x="Year", y="OrganizationCount",
             title="Number of Organizations Created Each Year",
             labels={"OrganizationCount": "Number of Organizations"},
             template="plotly_dark")

fig.write_html("Organizationsbyear.html", include_plotlyjs="cdn")
display(IFrame("Organizationsbyear.html", width=1200, height=700))


text = " ".join(Organizations["Name"].to_list())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Organization Names Word Cloud")
plt.show()


df = Organizations.select(["Name", "ParsedDate", "Year"]).to_pandas()
fig = px.treemap(df, path=["Year", "Name"],title="Organizations by Year and Name")
fig.write_html("Treemaporg.html", include_plotlyjs="cdn")
display(IFrame("Treemaporg.html", width=1200, height=700))


Models = pl.read_csv("/kaggle/input/meta-kaggle/Models.csv")
print(Models.columns)
print(Models.shape)
Models.head()


ModelVersions = pl.read_csv("/kaggle/input/meta-kaggle/ModelVersions.csv")
print(ModelVersions.columns)
print(ModelVersions.shape)
ModelVersions.head()


ModelOverall = Models.join(
    ModelVersions,
    left_on="Id",
    right_on="ModelId",
    how="inner"
)
print(ModelOverall.shape)
ModelOverall.head()


ModelVariations = pl.read_csv("/kaggle/input/meta-kaggle/ModelVariations.csv")
print(ModelVariations.columns)
print(ModelVariations.shape)
ModelVariations.head()


unique_ModelFramework = ModelVariations["ModelFramework"].unique().sort()
unique_ModelFramework.to_pandas()


framework_counts = (
    ModelVariations
    .group_by("ModelFramework")
    .agg(pl.len().alias("Count"))
    .sort("Count", descending=True)
)
framework_counts.to_pandas()


Models = Models.rename({"Id": "ModelId_Real"})
joined = ModelVariations.join(
    Models.select(["ModelId_Real", "CreationDate"]),
    left_on="ModelId",
    right_on="ModelId_Real",
    how="inner"
)


joined = joined.with_columns([
    pl.col("CreationDate")
    .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False)
    .dt.year()
    .alias("Year")
])
framework_trend = (
    joined
    .group_by(["Year", "ModelFramework"])
    .agg(pl.len().alias("Count"))
    .sort(["Year", "Count"], descending=False)
)
df = framework_trend.to_pandas()


latest_year = df["Year"].max()
fig = px.treemap(
    df[df["Year"] == latest_year],
    path=["ModelFramework"],
    values="Count",
    title=f"Framework Distribution in {latest_year}"
)
fig.write_html("Modelframeworktree.html", include_plotlyjs="cdn")
display(IFrame("Modelframeworktree.html", width=1200, height=700))


fig = px.treemap(
    df[df["Year"] == 2024],
    path=["ModelFramework"],
    values="Count",
    title=f"Framework Distribution in 2024"
)
fig.write_html("Modelframeworktree2024.html", include_plotlyjs="cdn")
display(IFrame("Modelframeworktree2024.html", width=1200, height=700))


last_year = df["Year"].min()
fig = px.treemap(
    df[df["Year"] == last_year],
    path=["ModelFramework"],
    values="Count",
    title=f"Framework Distribution in {last_year}"
)
fig.write_html("Modelframeworktree2023.html", include_plotlyjs="cdn")
display(IFrame("Modelframeworktree2023.html", width=1200, height=700))


framework_counts = (
    joined
    .group_by(["Year", "ModelFramework"])
    .agg(Count=pl.count())  
    .sort(["Year", "ModelFramework"])
)
framework_counts_pd = framework_counts.to_pandas()
latest_year = framework_counts_pd["Year"].max()
earliest_year = framework_counts_pd["Year"].min()
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        f"Framework Distribution in {earliest_year}",
        f"Framework Distribution in 2024",
        f"Framework Distribution in {latest_year}"
    ),
    specs=[[{"type": "treemap"}, {"type": "treemap"}, {"type": "treemap"}]]
)
df_earliest = framework_counts_pd[framework_counts_pd["Year"] == earliest_year]
treemap_earliest = px.treemap(
    df_earliest,
    path=["ModelFramework"],
    values="Count",
    color_discrete_sequence=px.colors.qualitative.Set3
)
for trace in treemap_earliest.data:
    fig.add_trace(trace, row=1, col=1)
df_2024 = framework_counts_pd[framework_counts_pd["Year"] == 2024]
treemap_2024 = px.treemap(
    df_2024,
    path=["ModelFramework"],
    values="Count",
    color_discrete_sequence=px.colors.qualitative.Set3
)
for trace in treemap_2024.data:
    fig.add_trace(trace, row=1, col=2)
df_latest = framework_counts_pd[framework_counts_pd["Year"] == latest_year]
treemap_latest = px.treemap(
    df_latest,
    path=["ModelFramework"],
    values="Count",
    color_discrete_sequence=px.colors.qualitative.Set3
)
for trace in treemap_latest.data:
    fig.add_trace(trace, row=1, col=3)
fig.update_layout(
    title_text="Model Framework Distribution Across Years",
    title_x=0.5,
    showlegend=True, 
    height=700,
    width=1200,
    margin=dict(t=100, l=10, r=10, b=10)
)
fig.write_html("frameworktreemaps.html", include_plotlyjs="cdn")
display(IFrame("frameworktreemaps.html", width=1250, height=800))


fig = px.line(
    df,
    x="Year",
    y="Count",
    color="ModelFramework",
    title="Rise of Model Frameworks Over the Years",
    markers=True
)
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Models",
    legend_title="Framework"
)
fig.write_html("Modelframework.html", include_plotlyjs="cdn")
display(IFrame("Modelframework.html", width=1200, height=700))


ModelVariationVersions = pl.read_csv("/kaggle/input/meta-kaggle/ModelVariationVersions.csv")
print(ModelVariationVersions.columns)
print(ModelVariationVersions.shape)
ModelVariationVersions.head()


models_df = pl.read_csv('/kaggle/input/meta-kaggle/Models.csv')
model_versions_df = pl.read_csv('/kaggle/input/meta-kaggle/ModelVersions.csv')


models_df = models_df.with_columns(
    pl.col("CreationDate").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False)
)
model_versions_df = model_versions_df.with_columns(
    pl.col("CreationDate").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False)
)


analysis_df = models_df.join(
    model_versions_df, left_on="CurrentModelVersionId", right_on="Id", suffix="_version"
)
analysis_df = analysis_df.with_columns(
    pl.col("CreationDate").dt.strftime("%Y-%m").alias("YearMonth")
)


monthly_stats = (
    analysis_df
    .group_by("YearMonth")
    .agg(pl.col("Id").len().alias("ModelCount"))
    .sort("YearMonth")
)
monthly_stats_pd = monthly_stats.to_pandas()
monthly_stats_pd['YearMonth'] = pd.to_datetime(monthly_stats_pd['YearMonth'])


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly_stats_pd['YearMonth'],
    y=monthly_stats_pd['ModelCount'],
    mode='lines+markers',
    name='Models Created',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=6)
))
fig.update_layout(
    title="Models Created Over Time",
    xaxis_title="Date",
    yaxis_title="Number of Models",
    height=500,
    width=800,
    title_x=0.5
)
fig.write_html("models_created.html", include_plotlyjs="cdn")
print("Displaying Models Created Over Time:")
display(IFrame("models_created.html", width=1000, height=600))

