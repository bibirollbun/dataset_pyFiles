!pip install biopython logomaker
!sudo apt-get install mafft


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
        #print(os.path.join(dirname, filename))
        pass
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess
import logomaker
import matplotlib.pyplot as plt

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_datasets = os.listdir("/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets")
train_datasets


os.listdir("/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/train_dataset_1")[:5]


train_datasets_path = "/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/"
plt.figure(figsize=(6,10))
for i, dataset in enumerate(os.listdir(train_datasets_path)):
    metadata = pd.read_csv(f"{train_datasets_path}/{dataset}/metadata.csv")
    plt.subplot(4,2,i+1)
    sns.countplot(metadata, x = "label_positive")
    plt.title(dataset)
plt.tight_layout()
plt.show()


train_datasets_path = "/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/"
for i, dataset in enumerate(os.listdir(train_datasets_path)):
    metadata = pd.read_csv(f"{train_datasets_path}/{dataset}/metadata.csv")
    print(dataset)
    display(metadata.head())


example = pd.read_csv("/kaggle/input/adaptive-immune-profiling-challenge-2025/train_datasets/train_datasets/train_dataset_1/2c65ef1f4fd1744fc8177cc73f63f589.tsv", sep="\t")
example


print("j_call unique", len(example.j_call.unique()))
print("v_call unique", len(example.v_call.unique()))


plt.figure(figsize=(10,6))
sns.countplot(example, x= "j_call")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(example, x= "v_call")
plt.xticks(rotation=90)
plt.show()


sns.histplot(example.junction_aa.str.len())
plt.xlabel("Sequence length")


example


example.junction_aa.value_counts().value_counts()


example = example.merge(example.v_call.value_counts(), on="v_call")
example = example.rename(columns={"count": "v_count"})
example = example.merge(example.j_call.value_counts(), on="j_call")
example = example.rename(columns={"count": "j_count"})


example


def series_to_fasta(series, fasta_path="sequences.fasta"):
    records = []
    for i, seq in enumerate(series):
        records.append(SeqRecord(Seq(str(seq)), id=f"seq{i}", description=""))
    SeqIO.write(records, fasta_path, "fasta")
    return fasta_path


def align_fasta_mafft(input_fasta, output_fasta="aligned.fasta"):
    cmd = ["mafft", "--auto", input_fasta]
    with open(output_fasta, "w") as out:
        subprocess.run(cmd, stdout=out, check=True, stderr=subprocess.DEVNULL)
    return output_fasta


def make_logo_from_alignment(aligned_fasta, title, output_png="logo.png"):
    alignment = [str(rec.seq) for rec in SeqIO.parse(aligned_fasta, "fasta")]
    matrix = logomaker.alignment_to_matrix(alignment, to_type='information')
    plt.figure(figsize=(12, 3))
    logo = logomaker.Logo(matrix,
                          shade_below=0.5,
                          fade_below=0.5,
                          font_name='DejaVu Sans Mono')

    logo.style_xticks(anchor=0)
    logo.ax.set_ylabel("Information (bits)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig(output_png)
    return output_png


def make_logo_from_series(series, title):
    fasta = series_to_fasta(series, fasta_path = f"{title}.fasta")
    aln = align_fasta_mafft(fasta, output_fasta = f"{title}_aligned.fasta")
    logo = make_logo_from_alignment(aln, title, output_png = f"{title}.png")
    return logo



for j in example.j_call.unique()[:5]: #Drop [:5] to obtain all Logos
    j_df = example[example.j_call == j]
    make_logo_from_series(j_df.junction_aa, j)


for v in example.v_call.unique()[:5]: #Drop [:5] to obtain all Logos
    v_df = example[example.v_call == v]
    make_logo_from_series(v_df.junction_aa, v)


concordance = example[["v_call", "j_call"]].value_counts().reset_index().pivot(columns="j_call", index="v_call").fillna(0)
plt.figure(figsize=(15,20))
sns.heatmap(concordance)


#Normalize by j_call
plt.figure(figsize=(15,25))
sns.heatmap(concordance/concordance.sum(axis = 0) * 100)


flows = example[["v_call", "j_call"]].value_counts().reset_index()
flows = flows.rename(columns={"v_call": "source", "j_call": "target", "count": "value"}).sort_values(by="source")
sns.histplot(flows, x = "value")
plt.title("Combinations of V and J")
plt.show()


def sankey(flows_sampled):
    # Nodos únicos
    sources = flows_sampled['source'].unique()
    targets = flows_sampled['target'].unique()
    nodes = list(sources) + list(targets)
    
    # Mapear cada nodo a un índice
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Crear listas para sankey
    flows_sampled['source_id'] = flows_sampled['source'].map(node_index)
    flows_sampled['target_id'] = flows_sampled['target'].map(node_index)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
        ),
        link=dict(
            source=flows_sampled['source_id'],
            target=flows_sampled['target_id'],
            value=flows_sampled['value']
        )
    )])
    
    fig.update_layout(
        title_text="Sankey plot",
        font_size=12,
        height=900
    )
    fig.show()


threshold = 100 #Limit this to noise reduce
flows_sampled = flows[flows["value"] > threshold]
sankey(flows_sampled)

