from pathlib import Path
competition_path = Path('/kaggle/input/MABe-mouse-behavior-detection/')

import pandas as pd
train_df = pd.read_csv(competition_path / 'train.csv', index_col='video_id')

train_df['arena_width_pix_approx'] = train_df['arena_width_cm'] * train_df['pix_per_cm_approx']
train_df['arena_height_pix_approx'] = train_df['arena_height_cm'] * train_df['pix_per_cm_approx']
train_df['width_frac'] = train_df['arena_width_pix_approx'] / train_df['video_width_pix']
train_df['height_frac'] = train_df['arena_height_pix_approx']/ train_df['video_height_pix']
train_df['area_frac'] = train_df['height_frac'] * train_df['height_frac']

voi = train_df.loc[[21954203, 1127527475, 2030735934, 9603]]
voi['label'] = voi['lab_id'] + '<br>' + voi.index.astype(str)


import plotly.express as px

frac_range = (0.32, 3.125)
fig = px.scatter(train_df, x='width_frac', y='height_frac', opacity=0.5, log_x=True, log_y=True, range_x=frac_range, range_y=frac_range)
fig.update_traces(marker=dict(size=6))
fig.add_hline(y=1, line_color='gray', opacity=0.3)
fig.add_vline(x=1, line_color='gray', opacity=0.3)
fig.add_scatter(
    mode='markers+text', x=voi['width_frac'], y=voi['height_frac'], text=voi['label'],
    marker=dict(size=12, symbol='circle-open'), textposition='bottom right', showlegend=False
)
fig.update_layout(
    width=600, height=600,
    title=dict(text='Approx. arena pixel dimensions<br>as fraction of frame', x=0.5, xanchor='center'),
    xaxis=dict(title='Approx. arena pixel width as fraction of frame'),
    yaxis=dict(title='Approx. arena pixel height as fraction of frame', scaleanchor='x', scaleratio=1)
)
fig.show()


import matplotlib.pyplot as plt
import numpy as np

start = np.log10(train_df['area_frac'].min())
stop = np.log10(train_df['area_frac'].max())
counts, bins = np.histogram(train_df['area_frac'], bins=np.logspace(start, stop, 20))

fig, ax = plt.subplots(figsize=(8,3))

# 3. Plot the histogram, passing the data and pre-defined bins
ax.hist(train_df['area_frac'], bins=bins)

xticks = [.2, .5, 1, 2]
yticks = [1, 10, 100, 1000]
ax.set(
    xscale='log', yscale='log', xlim=[.1, 5], ylim=[.8,1e4], xticks=xticks, xticklabels=xticks, yticks=yticks, yticklabels=yticks,
    xlabel='Arena area fraction of frame', ylabel='# of videos', title='Distribution of arena area fraction of frame')
ax.grid(True)


for video_id, row in voi.iterrows():
    tracking_df = pd.read_parquet(competition_path / 'train_tracking' / row['lab_id'] / f'{video_id}.parquet', columns=['x','y'])
    tracking_df['title'] = 'Tracked coordinates'
    
    symmetric_extent = np.array((-0.5, 0.5))
    frame_x = np.array((0.0, row['video_width_pix']))
    frame_y = np.array((0.0, row['video_height_pix']))
    arena_x = frame_x.mean() + row['arena_width_pix_approx'] * symmetric_extent
    arena_y = frame_y.mean() + row['arena_height_pix_approx'] * symmetric_extent
    all_x = np.concatenate((frame_x,arena_x))
    all_y = np.concatenate((frame_y,arena_y))
    
    padding = 20
    range_x = (all_x.min()-padding, all_x.max()+padding)
    range_y = (all_y.min()-padding, all_y.max()+padding)
    
    fig = px.scatter(tracking_df, x='x', y='y', color='title', title=f"{row['lab_id']}/{video_id}", opacity=0.5, range_x=range_x, range_y=range_y)
    fig.add_shape(
        type='rect', showlegend=True, name='Frame',
        x0=frame_x[0], x1=frame_x[1],
        y0=frame_y[0], y1=frame_y[1],
        line=dict(color='green', width=2)
    )
    fig.add_shape(
        type='rect', showlegend=True, name='Arena ("approx", centered)',
        x0=arena_x[0], x1=arena_x[1],
        y0=arena_y[0], y1=arena_y[1],
        line=dict(color='red', width=2)
    )
    fig.update_traces(marker=dict(size=1))
    axes_width = np.diff(range_x)[0]
    axes_height = np.diff(range_y)[0]
    scale_factor = min(1200.0/axes_width, 900.0/axes_height)
    fig.update_layout(
        width=min(axes_width * scale_factor,950), height=axes_height * scale_factor,
        xaxis_title='x (pix)', yaxis_title='y (pix)', yaxis_scaleanchor='x', yaxis_scaleratio=1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None)
    )
    fig.show()


MABe22_df = train_df[train_df['lab_id'].str.contains('MABe22')]
display(MABe22_df.reset_index()[['video_width_pix','video_height_pix']].drop_duplicates())
min_coords = np.array([np.inf, np.inf])
max_coords = np.array([-np.inf, -np.inf])
for video_id, row in MABe22_df.iterrows():
    tracking_df = pd.read_parquet(competition_path / 'train_tracking' / row['lab_id'] / f'{video_id}.parquet', columns=['x','y'])
    min_coords = np.minimum(min_coords,tracking_df.min())
    max_coords = np.maximum(max_coords,tracking_df.max())
print(min_coords)
print(max_coords)

