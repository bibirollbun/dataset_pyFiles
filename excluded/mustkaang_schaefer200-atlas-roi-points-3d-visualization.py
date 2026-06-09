import pandas as pd
import plotly.graph_objects as go

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Load your data
url = 'https://drive.google.com/uc?export=download&id=1EP7KzgEhsiqOg7bvfDYSYOg1CZA7pVnt'
df = pd.read_csv(url)

# Unique labels to color
unique_labels = df['Network Name'].unique()

# Base gray background (all points, low opacity)
gray_trace = go.Scatter3d(
    x=df['R'], y=df['A'], z=df['S'],
    mode='markers',
    marker=dict(size=2, color='black', opacity=0.3),
    name='All ROIs (background)',
    showlegend=False  # Hide from legend
)

# Add one colored trace per label
colored_traces = []
for label in unique_labels:
    sub = df[df['Network Name'] == label]
    trace = go.Scatter3d(
        x=sub['R'], y=sub['A'], z=sub['S'],
        mode='markers',
        marker=dict(size=3),
        name=label,
        visible='legendonly'  # All traces visible by default, but user can toggle
    )
    colored_traces.append(trace)

# Combine all traces
fig = go.Figure(data=[gray_trace] + colored_traces)

# Layout
fig.update_layout(
    title="Schaefer Atlas ROIs — Click Legend to Toggle Highlights",
    scene=dict(
        xaxis_title='R (Right-Left)',
        yaxis_title='A (Anterior-Posterior)',
        zaxis_title='S (Superior-Inferior)'
    ),
    legend=dict(itemsizing='constant')
)

fig.show(renderer='iframe')

