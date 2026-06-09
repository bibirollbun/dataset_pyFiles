import warnings
warnings.filterwarnings("ignore")


from IPython.display import display
from matplotlib import pyplot as plt


import numpy as np
import pandas as pd
import seaborn as sns


import glob
import os
import re
import tqdm


data_home = "/kaggle/input/make-data-count-finding-data-references/"
train_labels_path = os.path.join(data_home, "train_labels.csv")

train_data_home = os.path.join(data_home, "train")
train_pdf_path = os.path.join(train_data_home, "PDF")
train_xml_path = os.path.join(train_data_home, "XML")

test_data_home = os.path.join(data_home, "test")
test_pdf_path = os.path.join(test_data_home, "PDF")
test_xml_path = os.path.join(test_data_home, "XML")


train_xml_files = glob.glob(os.path.join(train_xml_path, "*.xml"))
train_pdf_files = glob.glob(os.path.join(train_pdf_path, "*.pdf"))

print(
    f"Found {len(train_xml_files)} XML files and {len(train_pdf_files)} PDF files in the training set."
)


test_xml_files = glob.glob(os.path.join(test_xml_path, "*.xml"))
test_pdf_files = glob.glob(os.path.join(test_pdf_path, "*.pdf"))

print(
    f"Found {len(test_xml_files)} XML files and {len(test_pdf_files)} PDF files in the test set."
)


def text_on_bars(ax):
    """
    Add text labels on top of the bars in a bar plot.
    """
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                x=p.get_x() + p.get_width() / 2,
                y=p.get_y() + height + 5,
                s=f"{int(height)}",
                ha="center",
                fontsize=10,
            )


train_labels_df = pd.read_csv(filepath_or_buffer=train_labels_path)
display(train_labels_df.head())
display(train_labels_df.tail())


def flag_pdf_xml_article_id(pdf_files, xml_files, df):
    """
    Flag the article_id based on the presence of PDF and XML files.
    """
    pdf_ids = set(
        os.path.basename(p=f.replace(".pdf", ""))
        for f in pdf_files
        if f.endswith(".pdf")
    )
    xml_ids = set(
        os.path.basename(p=f.replace(".xml", ""))
        for f in xml_files
        if f.endswith(".xml")
    )
    df["pdf"] = df["article_id"].astype(str).isin(values=pdf_ids)
    df["xml"] = df["article_id"].astype(str).isin(values=xml_ids)
    return df


train_labels_flag_df = flag_pdf_xml_article_id(
    pdf_files=train_pdf_files, xml_files=train_xml_files, df=train_labels_df
)


display(train_labels_flag_df.describe())


file_avail_df = train_labels_flag_df[["pdf", "xml"]].apply(pd.Series.value_counts)
display(file_avail_df)

ax = file_avail_df.plot(figsize=(8, 6), kind="bar")
text_on_bars(ax=ax)
plt.title(label="File Availability (PDF and XML)")
plt.ylabel(ylabel="Number of Entries")
plt.xlabel(xlabel="File Presence")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


type_counts = train_labels_flag_df["type"].value_counts()
type_counts_df = type_counts.reset_index()
type_counts_df.columns = ["citation_type", "count"]
display(type_counts_df)

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=type_counts_df, x="citation_type", y="count", hue="citation_type")
text_on_bars(ax=ax)
plt.title(label="Citation Type Distribution")
plt.xlabel(xlabel="Citation Type")
plt.ylabel(ylabel="Count")
plt.grid(linestyle="dotted")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
type_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title(label="Citation Type Proportion")
plt.ylabel("")
plt.tight_layout()
plt.show()


type_xml_df = train_labels_flag_df.groupby(["type", "xml"]).size().unstack(fill_value=0)
display(type_xml_df)

ax = type_xml_df.plot(
    figsize=(8, 6), kind="bar", stacked=True, color=["#ebcec1", "#aed6f1"]
)
text_on_bars(ax=ax)
plt.title("Citation Type vs. XML File Availability")
plt.xlabel("Citation Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Has XML")
plt.tight_layout()
plt.show()


article_citation_counts_df = (
    train_labels_flag_df["article_id"].value_counts().reset_index()
)
article_citation_counts_df.columns = ["article_id", "count"]

multi_cited_articles_df = article_citation_counts_df[
    article_citation_counts_df["count"] > 1
]
display(multi_cited_articles_df.head(n=10))

print(
    f"Number of articles with multiple data citations: {len(multi_cited_articles_df)}"
)


citation_count_distribution = (
    article_citation_counts_df["count"]
    .value_counts()
    .sort_index()
    .rename("frequency")
    .reset_index()
    .rename(columns={"count": "citation_count"})
)

display(citation_count_distribution)

plt.figure(figsize=(15, 8))
ax = sns.barplot(data=citation_count_distribution, x="citation_count", y="frequency")
text_on_bars(ax=ax)
plt.title(label="Citation Count Distribution")
plt.xlabel(xlabel="Citation per Article(s)")
plt.ylabel(ylabel="Number of Articles (Frequency)")
plt.grid(linestyle="dotted")
plt.tight_layout()
plt.show()


non_miss_dataset_df = train_labels_flag_df[
    train_labels_flag_df["dataset_id"] != "Missing"
]

no_miss_dataset_counts = non_miss_dataset_df["dataset_id"].value_counts()
no_miss_dataset_counts_df = no_miss_dataset_counts.reset_index()
no_miss_dataset_counts_df.columns = ["dataset_id", "count"]

print(
    f"Number of unique datasets (excluding 'Missing'): {len(no_miss_dataset_counts_df)}"
)

multi_datasets_df = no_miss_dataset_counts_df[no_miss_dataset_counts_df["count"] > 1]
print(f"Number of datasets cited more than once: {len(multi_datasets_df)}")


article_types = train_labels_flag_df.groupby("article_id")["type"].apply(
    lambda x: set(x)
)

article_citation_types = article_types.apply(
    lambda x: (
        "Primary only"
        if x == {"Primary"}
        else (
            "Secondary only"
            if x == {"Secondary"}
            else "Mixed" if x == {"Primary", "Secondary"} else "Missing"
        )
    )
)

article_citation_types_df = article_citation_types.reset_index()
article_citation_types_df.columns = ["article_id", "citation_type"]
articles_with_mixed_citations_df = article_citation_types_df[
    article_citation_types_df["citation_type"] == "Mixed"
]
display(articles_with_mixed_citations_df)

article_citation_types_counts_df = article_citation_types.value_counts().reset_index()
article_citation_types_counts_df.columns = ["citation_type", "count"]
display(article_citation_types_counts_df)

plt.figure(figsize=(8, 6))
ax = sns.barplot(
    data=article_citation_types_counts_df,
    x="citation_type",
    y="count",
    hue="citation_type",
)
text_on_bars(ax=ax)
plt.title(label="Articles by Citation Type Category")
plt.xlabel(xlabel="Article Citation Type")
plt.ylabel(ylabel="Number of Articles")
plt.grid(linestyle="dotted")
plt.tight_layout()
plt.show()




