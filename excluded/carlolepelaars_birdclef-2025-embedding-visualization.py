!pip install -Uqq umap-learn
!pip install -Uqq "vegafusion[embed]>=1.5.0"
!pip install -Uqq "vl-convert-python>=1.6.0"


import numpy as np
import pandas as pd
from umap import UMAP
from ast import literal_eval

import altair as alt
alt.data_transformers.enable("vegafusion");


df = pd.read_csv(f'/kaggle/input/birdclef-2025-perch-embeddings/train_with_emb.csv')
tx = pd.read_csv(f'/kaggle/input/birdclef-2025/taxonomy.csv')


df['class_name'] = df['primary_label'].map(dict(zip(tx['primary_label'], tx['class_name'])))
df[['primary_label', 'common_name', 'emb', 'class_name']].head(2)


series = df['emb'].apply(literal_eval).tolist()


reducer = UMAP(
    n_components=2,
    n_neighbors=15, 
    min_dist=0.1
)
emb = reducer.fit_transform(series)


df['UMAP Component 1'] = emb[:, 0]
df['UMAP Component 2'] = emb[:, 1]


def create_scatter_plot(df, color="class_name", dot_size=50, title="Species Embeddings"):
    selection = alt.selection_point(fields=[color], bind='legend')
    return alt.Chart(df).mark_circle(
        size=dot_size,
        opacity=0.7
    ).encode(
        x='UMAP Component 1',
        y='UMAP Component 2',
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


create_scatter_plot(df, dot_size=20)


create_scatter_plot(df[df['class_name'] == 'Mammalia'], color="common_name", dot_size=50, title="Mammals")


create_scatter_plot(df[df['class_name'] == 'Insecta'], color="common_name", dot_size=60, title="Insects")


create_scatter_plot(df[df['class_name'] == 'Amphibia'], color="common_name", dot_size=50, title="Amphibians")


create_scatter_plot(df[df['class_name'] == 'Aves'], color="common_name", dot_size=20, title="Birds")

