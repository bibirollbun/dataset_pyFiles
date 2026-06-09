import pandas as pd
import plotly.graph_objects as go

csv_path = "/kaggle/input/stanford-rna-3d-folding/train_labels.csv" 
df = pd.read_csv(csv_path)

df['target_id'] = df['ID'].apply(lambda x: x.split('_')[0])
target_ids = df['target_id'].unique()

def plot_rna_interactive(target_id, structure_idx=1):
    subset = df[df['target_id'] == target_id]

    x_col = f'x_{structure_idx}'
    y_col = f'y_{structure_idx}'
    z_col = f'z_{structure_idx}'

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=subset[x_col], y=subset[y_col], z=subset[z_col],
        mode='lines+markers+text',
        marker=dict(size=4, color='cyan'),
        line=dict(color='lightblue', width=2),
        text=subset['resname'],
        textposition='top center',
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"RNA 3D Structure: {target_id} (Structure {structure_idx})",
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            bgcolor='rgb(10,10,10)'
        ),
        paper_bgcolor='rgb(20,20,20)',
        font=dict(color='white'),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

plot_rna_interactive(target_ids[2])

