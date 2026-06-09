import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


taxonomy = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv').merge(
    taxonomy[['primary_label', 'class_name']], 
    on='primary_label',
    how='inner',
    validate="m:1",
).merge(
    pd.read_parquet('/kaggle/input/bc25-audio-stats/train_metadata_stats.parquet'), 
    on='filename',
    how='inner',
    validate="1:1"
).merge(
    pd.read_parquet('/kaggle/input/bc25-spec-stats/quantize_params.parquet')\
    .rename(columns={'min_value': 'min_spec_value', 'max_value': 'max_spec_value'}),
    on='filename',
    how='inner',
    validate="1:1"
).merge(
    pd.read_parquet('/kaggle/input/bc25-folds/folds.parquet'),
    on='filename',
    how='inner',
    validate="1:1"
).merge(
    pd.read_parquet('/kaggle/input/analyze-voice-crops/train_voice_data.parquet'),
    on='filename',
    how='inner',
    validate="1:1"
).merge(
    pd.read_parquet('/kaggle/input/bc24-train-embedding-and-umap/train_embedding.parquet')[[
        'filename',
        'umap_component_1',
        'umap_component_2',
    ]],
    on='filename',
    how='inner',
    validate="1:1"
)
df.head()


list(df)


(df.numel / 32000).describe()


df.primary_label.nunique()


category_counts = df['class_name'].value_counts()
print('distinct classes:', len(category_counts))
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.show()


fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='class_name',
                     hover_name='primary_label', projection="natural earth",
                     title='Locations by class')
fig.show()


df[['class_name', 'min', 'mean', 'max']].groupby('class_name').agg('mean').style.bar(color='lightblue')


category_counts = df['collection'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.show()


fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='collection',
                     hover_name='primary_label', projection="natural earth",
                     title='Locations by collection')
fig.show()


grouped = df.groupby(['class_name', 'collection']).size().unstack(fill_value=0)
display(grouped)
grouped.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.xlabel("class_name")
plt.ylabel("count")
plt.title("class_name and collection")
plt.legend(title="collection")
plt.show()


df.groupby(['class_name', 'collection'])['std'].agg('mean')


has_secondary_labels = (df.secondary_labels != "['']") & (df.secondary_labels != "[]")
df.groupby(['class_name', 'collection', has_secondary_labels]).size().unstack(fill_value=0)


secondary_labels = (
    df
    .secondary_labels[has_secondary_labels]
    .str.extractall(r"'(?P<secondary_label>[^']+)'")
    .reset_index()
    .merge(taxonomy[['primary_label', 'class_name']], left_on='secondary_label', right_on="primary_label")
    .drop(columns=["primary_label", "match"])
    .merge(df[['primary_label', 'class_name']].reset_index(), left_on='level_0', right_on="index", suffixes=("", "_primary"))
    .drop(columns=["level_0"])
    .rename(columns={"class_name": "secondary_class_name", "class_name_primary": "primary_class_name"})
)
secondary_labels.groupby(['primary_class_name', 'secondary_class_name']).size().unstack(fill_value=0)


print('recordings with voice', df.has_voice.sum())
print()

all_labels = df.primary_label.value_counts()
for collection in df.collection.unique():
    df_csa = df[df.collection == collection]
    print(f'% of {collection} recordings with voice:', round(sum(df_csa.voice_time > 0) / len(df_csa) * 100, 3))
    print(f'% of {collection} recording samples with voice:', round(100 * df_csa.voice_time.sum() / (df_csa.numel.sum() / 32000), 3))
    lost_recs = ((df_csa.numel / 32000 - df_csa.voice_time) < 5) * (df_csa.voice_time > 0)
    print(f'# of {collection} recordings < 5 sec remaining:', sum(lost_recs))
    lost_labels = df_csa[lost_recs].primary_label.value_counts()
    num_lost = [label for label, c in lost_labels.items() if all_labels[label] - c <= 0]
    print(f'# of lost primary labels:', num_lost) 


def f(df, kind):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    collections = df.index
    x = range(len(collections))
    ax1.bar(x, df['mean'] * 100, width=0.4, align='center', color="#1f77b4")
    ax1.set_ylabel(f'% of {kind} in collection have voice', color="#1f77b4")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.4 for i in x], df['sum'], width=0.4, align='center', color="#ff7f0e")
    ax2.set_ylabel(f'# of {kind} with voice', color="#ff7f0e")
    ax1.set_xticks([i + 0.2 for i in x])
    ax1.set_xticklabels(collections)
    fig.suptitle(f'{kind} with human voice per collection')
    plt.show()
    display(df)

f(df.groupby(['collection']).has_voice.agg(['mean', 'sum']), 'recordings')

voice_samples = df.groupby('collection').voice_time.sum()
f(pd.DataFrame(dict(
    mean=voice_samples / (df.groupby(['collection']).numel.sum() / 32000),
    sum=voice_samples, 
), index=df.collection.unique()), "samples")


voice_authors = df.groupby(['author', 'collection']).has_voice.agg(['mean', 'sum'])
# take all authors that have more than 2 recordings and > 50% recordings with voice
voice_authors = voice_authors[(voice_authors['sum'] > 2) * (voice_authors['mean'] > .5)]
df['filter_voice'] = df.has_voice * (
    (df.collection == 'CSA') | # all CSA recordings
    df.author.isin({a for a, _ in voice_authors.index}) # filtered authors from the rest
)
voice_authors


df.has_voice.agg(['mean', 'sum']), df[df.filter_voice].has_voice.agg(['mean', 'sum'])


100 * df[df.filter_voice].voice_time.sum() / df.voice_time.sum()


!pip install -Uqq "vegafusion[embed]>=1.5.0"
!pip install -Uqq "vl-convert-python>=1.6.0"


import altair as alt
alt.data_transformers.enable("vegafusion");

def create_scatter_plot(df, color="common_name", dot_size=50, title="Species Embeddings"):
    selection = alt.selection_point(fields=[color], bind='legend')
    return alt.Chart(df).mark_circle(
        size=dot_size,
        opacity=0.7
    ).encode(
        x='umap_component_1',
        y='umap_component_2',
        color=alt.Color(color, legend=alt.Legend(
            title='Species',
            orient='right',
            symbolLimit=50
        )),
        tooltip=[
            alt.Tooltip('common_name', title='Common'),
            alt.Tooltip('scientific_name', title='Scientific'),
            alt.Tooltip('filename', title='File')
        ],
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.01))
    ).add_params(
        selection
    ).properties(
        width=600,
        height=600,
        title=title
    ).interactive()


create_scatter_plot(df)


df.to_parquet('train_metadata_joined.parquet')

