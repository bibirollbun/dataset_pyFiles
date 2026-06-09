import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)


import pandas as pd
train_labels_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
print(train_labels.head(10))


import plotly.graph_objects as go
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

train_labels['protein_id'] = train_labels['ID'].str.split('_').str[:2].str.join('_')

unique_protein_ids = train_labels['protein_id'].unique()

output = widgets.Output()

protein_dropdown = widgets.Dropdown(
    options=unique_protein_ids,
    description='Protein ID:',
    disabled=False,
)

def show_plot(protein_id):
    with output:
        clear_output(wait=True)
        df = train_labels[train_labels['protein_id'] == protein_id]
        colors = {'A': 'red', 'C': 'green', 'G': 'blue', 'U': 'purple'}
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=df['x_1'], y=df['y_1'], z=df['z_1'],
                mode='lines',
                line=dict(color='gray', width=4),
                name='Backbone'
            )
        )
        for base in ['A', 'C', 'G', 'U']:
            subset = df[df['resname'] == base]
            if not subset.empty:
                fig.add_trace(
                    go.Scatter3d(
                        x=subset['x_1'], y=subset['y_1'], z=subset['z_1'],
                        mode='markers',
                        marker=dict(
                            color=colors[base],
                            size=6,
                            opacity=0.8,
                            symbol='circle'
                        ),
                        name=f'{base}'
                    )
                )
                
        fig.update_layout(
            title=f'3D RNA Structure: {protein_id}',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
            ),
            legend_title='Nucleotides',
            width=800,
            height=600
        )
        
        fig.show()

def on_protein_change(change):
    show_plot(change.new)

protein_dropdown.observe(on_protein_change, names='value')

show_plot(protein_dropdown.value)

display(widgets.VBox([protein_dropdown, output]))

