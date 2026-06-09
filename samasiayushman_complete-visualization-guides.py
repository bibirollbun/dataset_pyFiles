import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

# Load data (replace with your actual file path)
df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')


print("Dataset shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)



categorical_cols = ['sequence_type', 'subject', 'orientation', 'behavior', 'phase', 'gesture']

plt.figure(figsize=(18, 12))
gs = GridSpec(3, 2, figure=plt.gcf())

for i, col in enumerate(categorical_cols):
    ax = plt.subplot(gs[i//2, i%2])
    sns.countplot(y=col, data=df, ax=ax, order=df[col].value_counts().index)
    ax.set_title(f'Distribution of {col}')
    plt.tight_layout()

plt.suptitle('Categorical Variables Distribution', y=1.02)
plt.show()



plt.figure(figsize=(14, 6))
sns.lineplot(x='sequence_counter', y='acc_x', hue='gesture', data=df)
plt.title('Acceleration X over Time by Gesture')
plt.xlabel('Sequence Counter')
plt.ylabel('Acceleration X')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



plt.figure(figsize=(14, 6))
sns.boxplot(x='gesture', y='acc_x', data=df)
plt.title('Acceleration X by Gesture Type')
plt.xticks(rotation=45)
plt.show()



behavior_acc = df.groupby('behavior')['acc_x'].agg(['mean', 'std', 'count'])
behavior_acc = behavior_acc.sort_values('mean', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=behavior_acc.index, y='mean', data=behavior_acc, yerr=behavior_acc['std'])
plt.title('Average Acceleration by Behavior with Standard Deviation')
plt.xticks(rotation=45)
plt.ylabel('Mean Acceleration X')
plt.show()


from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

# First define tof_cols - all columns starting with 'tof_'
tof_cols = [col for col in df.columns if col.startswith('tof_')]



from pandas.plotting import parallel_coordinates

# Select relevant features for gesture classification
gesture_features = ['acc_x'] + tof_cols[:5]  # Using first 5 TOF sensors for visualization

# Sample data for visualization (to avoid overcrowding)
sample_df = df.sample(min(500, len(df)), random_state=42)
sample_df = sample_df[gesture_features + ['gesture']].dropna()

plt.figure(figsize=(16, 8))
parallel_coordinates(sample_df, 'gesture', alpha=0.5)
plt.title('Parallel Coordinates Plot for Gesture Classification')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



from mpl_toolkits.mplot3d import Axes3D

tof_3d_cols = tof_cols[:3]  # Take first 3 TOF sensors for 3D plot
plot_df = df[tof_3d_cols + ['gesture']].replace(-1, np.nan).dropna()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Encode gestures to numbers for coloring
le = LabelEncoder()
gesture_encoded = le.fit_transform(plot_df['gesture'])

scatter = ax.scatter(
    plot_df[tof_3d_cols[0]], 
    plot_df[tof_3d_cols[1]], 
    plot_df[tof_3d_cols[2]], 
    c=gesture_encoded, 
    cmap='viridis',
    alpha=0.6
)

ax.set_xlabel(tof_3d_cols[0])
ax.set_ylabel(tof_3d_cols[1])
ax.set_zlabel(tof_3d_cols[2])
plt.title('3D Visualization of TOF Sensor Readings by Gesture')

# Create a legend
legend_labels = le.classes_
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=plt.cm.viridis(i/len(legend_labels)), 
                      markersize=10) for i in range(len(legend_labels))]
ax.legend(handles, legend_labels, title='Gestures', bbox_to_anchor=(1.2, 1))
plt.show()


from matplotlib.animation import FuncAnimation

# Select a sequence to animate
sequence_id = 'SEQ_000007'
seq_df = df[df['sequence_id'] == sequence_id]

fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(seq_df['sequence_counter'].min(), seq_df['sequence_counter'].max())
ax.set_ylim(0, seq_df[tof_cols].max().max() + 10)
ax.set_xlabel('Sequence Counter')
ax.set_ylabel('TOF Value')
ax.set_title(f'TOF Sensor Data Animation for {sequence_id}')

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = seq_df['sequence_counter'].iloc[:i+1]
    y = seq_df['tof_5_v56'].iloc[:i+1]  # Using one TOF sensor for animation
    line.set_data(x, y)
    return line,

# For actual use, you would save this animation. Just TRY IT !!!!!

# ani = FuncAnimation(fig, animate, frames=len(seq_df), init_func=init, blit=True)
# ani.save('tof_animation.mp4', writer='ffmpeg', fps=10)

# Instead, we'll just show the final frame
animate(len(seq_df)-1)
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

!pip install -q squarify
import squarify

!pip install -q joypy
import joypy


!pip install -q missingno
import missingno as msno


!pip install -q ipywidgets
import ipywidgets as widgets
from IPython.display import display


!pip install -q networkx
import networkx as nx

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify
import joypy
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import missingno as msno
import ipywidgets as widgets
from IPython.display import display
import networkx as nx


fig = px.sunburst(
    df, 
    path=['orientation', 'behavior', 'gesture'], 
    values='acc_x',
    color='acc_x',
    color_continuous_scale='RdYlBu_r',
    title='Behavioral Hierarchy Sunburst'
)
fig.show()


fig = px.scatter(
    df, 
    x='sequence_counter', 
    y='acc_x', 
    animation_frame='sequence_counter',
    color='gesture',
    size=np.abs(df['acc_x']),  # Take absolute values
    hover_name='behavior',
    range_x=[df['sequence_counter'].min(), df['sequence_counter'].max()],
    range_y=[df['acc_x'].min(), df['acc_x'].max()],
    title='Animated Gesture Progression'
)
fig.show()


tof_cols = [col for col in df.columns if col.startswith('tof_')][:6]
gesture_avg = df.groupby('gesture')[tof_cols].mean().reset_index()

fig = go.Figure()
for gesture in gesture_avg['gesture'].unique():
    fig.add_trace(go.Scatterpolar(
        r=gesture_avg[gesture_avg['gesture']==gesture][tof_cols].values[0],
        theta=tof_cols,
        fill='toself',
        name=gesture
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title='TOF Sensor Radar by Gesture')
fig.show()


behavior_counts = df.groupby(['sequence_counter', 'behavior']).size().unstack().fillna(0)

plt.figure(figsize=(16, 8))
plt.stackplot(
    behavior_counts.index,
    behavior_counts.values.T,
    labels=behavior_counts.columns,
    alpha=0.7
)
plt.legend(loc='upper left')
plt.title('Behavior Streamgraph Over Time')
plt.xlabel('Sequence Counter')
plt.ylabel('Frequency')
plt.show()


# Create transition matrix
transitions = df.groupby(['sequence_id', 'behavior'])['behavior'].shift(-1).dropna()
edges = pd.crosstab(df['behavior'], transitions)

# Create network graph
G = nx.DiGraph()
for src in edges.index:
    for tgt in edges.columns:
        if edges.loc[src, tgt] > 0:
            G.add_edge(src, tgt, weight=edges.loc[src, tgt])

pos = nx.spring_layout(G)
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=15,
        color=[],
        line_width=2))

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Behavior Transition Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40)))
fig.show()


fig = px.scatter_ternary(
    df,
    a=df['tof_5_v56'].rank(pct=True),
    b=df['tof_5_v57'].rank(pct=True),
    c=df['tof_5_v58'].rank(pct=True),
    color='gesture',
    size=np.abs(df['acc_x']),  # Take absolute values
    hover_name='behavior',
    title="TOF Sensor Ternary Space"
)
fig.show()

