import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.io as pio
pio.renderers.default = "notebook"

import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)


# Path
COMPETITION_NAME = "stanford-rna-3d-folding"
ROOT = Path(".").resolve().parent
INPUT_ROOT = ROOT / "input"
RAW_DATA = INPUT_ROOT / COMPETITION_NAME

# read_csv
train_labels    = pl.read_csv(RAW_DATA / "train_labels.csv")
train_sequences = pl.read_csv(RAW_DATA / "train_sequences.csv")
valid_labels      = pl.read_csv(RAW_DATA / "validation_labels.csv") 
valid_sequences   = pl.read_csv(RAW_DATA / "validation_sequences.csv")


# train data
train_labels = train_labels.with_columns(
    ('train_' + pl.col("ID").str.replace(r"_[^_]*$", "")).alias("target_id") # Remove suffix
)
train_sequences = train_sequences.with_columns(
    ('train_' + pl.col("target_id")).alias('target_id') # Remove suffix
)

valid_labels = valid_labels.with_columns(
    pl.col("ID").str.replace(r"_[^_]*$", "").alias("target_id") # Remove suffix
)

# validation data
dfs = []
for target_id in valid_sequences.select('target_id').to_series():
    for i in range(1,41):
        dfs.append(valid_labels
            .filter(pl.col('target_id')==target_id)
            .select(
                pl.col('ID'),
                pl.col('resname'),
                pl.col('resid'),
                pl.col(f'x_{i}').alias('x_1'),
                pl.col(f'y_{i}').alias('y_1'),
                pl.col(f'z_{i}').alias('z_1'),
                pl.col('target_id'),
                ('valid_' + pl.col('target_id') + f'_{i}').alias('target_id_n')
            )
        )

valid_labels = pl.concat(dfs).filter(pl.col('x_1')>-1.0e18)

valid_sequences = valid_sequences.join(
    valid_labels.select(['target_id', 'target_id_n']).unique(),
    on='target_id',
    how='left')

valid_sequences = valid_sequences.with_columns(pl.col('target_id_n').alias('target_id')).sort('target_id_n')
valid_labels = valid_labels.with_columns(pl.col('target_id_n').alias('target_id')).sort('target_id_n')

cols = ['target_id','sequence']
train_sequences = train_sequences.select(pl.col(cols))
valid_sequences = valid_sequences.select(pl.col(cols))
sequences = pl.concat([
    train_sequences,
    valid_sequences,
    ])
cols = ['target_id', 'resname', 'resid', 'x_1', 'y_1', 'z_1']
train_labels = train_labels.select(pl.col(cols))
valid_labels = valid_labels.select(pl.col(cols))
labels = pl.concat([
    train_labels,
    valid_labels,
    ])


def plotlyRNAStructure(df_labels, target_ids):
    """
    Plots the 3D structure of RNA molecules for the given target IDs.
    This function visualizes the RNA structure in a 3D space using Plotly.
    Each nucleotide (A, C, G, U, X, and -) is represented by a different color.
    The function retrieves structural data from `train_labels` and sequence data from `train_sequences`,
    then creates a subplot for each target ID.

    Parameters:
        df_labels: [train/validation]_labels.
        target_ids (list of str): A list of target IDs for which the RNA structures will be plotted.
    """
    sequence = sequences.filter(pl.col('target_id')==target_ids[0]).select('sequence').to_series()[0]
    print('sequence:', sequence)
    
    color_map = {
        'A': 'red',
        'C': 'blue',
        'G': 'green',
        'U': 'orange',
        'X': 'black',
        '-': 'gray'
    }
    column_num = len(target_ids)
    
    fig = make_subplots(
        rows=1, cols=column_num,
        specs=[[{'type': 'scene'}]*column_num],
        subplot_titles=target_ids
    )

    # dummy plot for legend
    for base, color in color_map.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                opacity=0.8
            ),
            name=base
        ))

    # plot RNA structure
    for i in range(column_num):
        df = df_labels.filter(pl.col('target_id')==target_ids[i])
        colors = [color_map[base] for base in df['resname']]
        hover_texts = [f'{base}{id}' for base, id in zip(df['resname'], df['resid'])]
        
        # fig = go.Figure()
    
        fig.add_trace(
            go.Scatter3d(
                x=df['x_1'],
                y=df['y_1'],
                z=df['z_1'],
                mode='markers+text',
                marker=dict(
                    size=6,
                    color=colors,
                    opacity=0.8,
                    symbol='circle',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=hover_texts,
                hoverinfo='text',
                hovertext=hover_texts,
                name='residue',
                showlegend=False
            ),
            row=1, col=i+1
        )
    
        fig.add_trace(
            go.Scatter3d(
                x=df['x_1'],
                y=df['y_1'],
                z=df['z_1'],
                mode='lines',
                line=dict(
                    color='gray',
                    width=2,
                    dash='solid'
                ),
                opacity=0.5,
                name='bone',
                showlegend=False
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        title='',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        margin=dict(r=20, l=10, b=10, t=50),
        legend=dict(
            title='resname',
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        hovermode='closest',
        template='plotly_white'
    )
    
    fig.show()


def matplotRNAStructure(df_labels, target_ids):
    sequence = sequences.filter(pl.col('target_id')==target_ids[0]).select('sequence').to_series()[0]
    print('sequence:', sequence)
    
    color_map = {
        'A': 'red',
        'C': 'blue',
        'G': 'green',
        'U': 'orange',
        'X': 'black',
        '-': 'gray'
    }
    column_num = len(target_ids)
    row_num = 1
    if column_num > 5:
        row_num += (column_num-1) // 5
        column_num = 5

    fig, axes = plt.subplots(row_num, column_num, figsize=(5 * column_num, 5 * row_num),
                            subplot_kw={'projection': '3d'})

    if len(target_ids) == 1:
        axes = [[axes]]
    elif len(target_ids) < 6:
        axes = [axes]
        
    for i in range(len(target_ids)):
        ax = axes[i//5][i%5]
        target_id = target_ids[i]
        df = df_labels.filter(pl.col('target_id')==target_id)
        colors = [color_map[base] for base in df['resname']]
        ax.scatter(df['x_1'], df['y_1'], df['z_1'], c=colors, s=100, alpha=0.7, marker='o')
        ax.plot(df['x_1'], df['y_1'], df['z_1'], 'k-', alpha=0.3)

        for i, txt in enumerate(df['resname']):
            if df['x_1'][i] is not None:
                ax.text(df['x_1'][i], df['y_1'][i], df['z_1'][i], 
                        f'{txt}{df["resid"][i]}', size=6, color='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(target_id)
        ax.view_init(elev=20, azim=30)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                label=base, markerfacecolor=color, markersize=10) 
                    for base, color in color_map.items()]
    fig.legend(handles=legend_elements, title='Nucleotides', loc='lower left')
    # sequence = sequences.filter(pl.col('target_id')==target_ids[0]).select('sequence').to_series()[0]
    # plt.suptitle(sequence, fontsize=6)
    plt.tight_layout()
    plt.show()


cnt_sequence = (
    train_sequences
    .group_by('sequence')
    .agg(
        pl.col("target_id").count().alias('duplicates')
    )
    .sort(['duplicates', 'sequence'])
)

cnt_duplicates = (cnt_sequence
    .group_by('duplicates')
    .agg(
        pl.col("sequence").count().alias('count')
    )
    .sort(['duplicates'])
)
plt.figure(figsize=(8, 5))
plt.bar(cnt_duplicates["duplicates"], cnt_duplicates["count"], color="skyblue")
plt.title("Number of data points with duplicate sequences")
plt.ylabel('num of sequences')
plt.xlabel('duplicates')
plt.show()

same_sequence = cnt_sequence.filter(pl.col('duplicates')>1)
print('Number of duplicate sequences:', same_sequence.height)


seq = 'ACGAGUGUCGUACCAAG'
target_ids = train_sequences.filter(pl.col('sequence')==seq).select(pl.col('target_id')).to_series()
plotlyRNAStructure(labels, target_ids)


for seq in same_sequence['sequence']:
    target_ids = train_sequences.filter(pl.col('sequence')==seq).select(pl.col('target_id')).to_series()
    matplotRNAStructure(labels, target_ids)


cnt_sequence = (
    valid_sequences
    .group_by('sequence')
    .agg(
        pl.col("target_id").count().alias('duplicates')
    )
    .sort(['duplicates', 'sequence'])
)

cnt_duplicates = (cnt_sequence
    .group_by('duplicates')
    .agg(
        pl.col("sequence").count().alias('count')
    )
    .sort(['duplicates'])
)
plt.figure(figsize=(8, 5))
plt.bar(cnt_duplicates["duplicates"], cnt_duplicates["count"], color="skyblue")
plt.title("Number of data points with duplicate sequences")
plt.ylabel('num of sequences')
plt.xlabel('duplicates')
plt.show()

same_sequence = cnt_sequence.filter(pl.col('duplicates')>1)
print('Number of duplicate sequences:', same_sequence.height)


seq = 'GCGUACAGGGAACACGCAACCCCGAAGGAUCGGGGAAGGGACGUCGCCAGGGAGGCGAUUCCAUCAGGAUGAUGACGAGGGACUGAAGAGUGGGCGGGGUAAUACCCCGCCCCUUUUU'
target_ids = valid_sequences.filter(pl.col('sequence')==seq).select(pl.col('target_id')).to_series()
plotlyRNAStructure(labels, target_ids)


for seq in same_sequence['sequence']:
    target_ids = valid_sequences.filter(pl.col('sequence')==seq).select(pl.col('target_id')).to_series()
    matplotRNAStructure(labels, target_ids)


same_sequence = list(set(train_sequences.select(pl.col('sequence')).to_series()) & set(valid_sequences.select(pl.col('sequence')).to_series()))
same_sequence.sort()


for seq in same_sequence:
    target_ids = sequences.filter(pl.col('sequence')==seq).select(pl.col('target_id')).to_series()
    matplotRNAStructure(labels, target_ids)

