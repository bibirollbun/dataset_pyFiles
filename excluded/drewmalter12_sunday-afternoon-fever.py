
#import packages
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#read in data and parse dates, adjust other data types 
#df_plays = pd.read_csv('plays.csv', parse_dates = ['GameDate'])
df_plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
df_games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
df_players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
df_player_play = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
df_week1 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv', parse_dates = ['time'])
df_week2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_2.csv', parse_dates = ['time'])
df_week3 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_3.csv', parse_dates = ['time'])
df_week4 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_4.csv', parse_dates = ['time'])
df_week5 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_5.csv', parse_dates = ['time'])
df_week6 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_6.csv', parse_dates = ['time'])
df_week7 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_7.csv', parse_dates = ['time'])
df_week8 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_8.csv', parse_dates = ['time'])
df_week9 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/tracking_week_9.csv', parse_dates = ['time'])
#df_week1.head()

#Data Prep

#Concat all weekly df's into one
df_weeks = pd.concat([df_week1, df_week2, df_week3, df_week4, df_week5, df_week6, df_week7, df_week8, df_week9])
#df_weeks.shape




df_coverages_clusters = pd.read_csv('/kaggle/input/big-data-bowl-coverage-clusters/df_coverages_clusters.csv')
df_coverages_clusters.rename(columns=lambda x: x.replace('Unnamed: 0', ''), inplace=True)

# Use Pandas styling to display the DataFrame with formatting and hide the index
df_coverages_clusters_styled = df_coverages_clusters.style.format({
    'Coverage': lambda x: x,  # Leave 'Coverage' column as is
}).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#013369'),
                                 ('font-weight', 'bold'), ('color', 'white')]},  # Make column label text white
    {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]},  # Adjust font size here
    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},  # Lighter grey for even rows
    {'selector': 'tr:nth-child(odd)', 'props': [('background-color', 'white')]},   # Default grey for odd rows
], overwrite=False).set_properties(**{'border': 'none', 'padding': '5px'})  # Remove the borders

# Hide the index
df_coverages_clusters_styled = df_coverages_clusters_styled.hide(axis='index')

# Add a title as a caption
df_coverages_clusters_styled = df_coverages_clusters_styled.set_caption(
    '% of Post Snap Coverage Given Pre Snap Cluster'
).set_table_styles([
    {'selector': 'caption', 'props': [('caption-side', 'top'), ('text-align', 'center'), 
                                      ('font-size', '14px'), ('font-weight', 'bold'), ('color', 'black')]}
], overwrite=False)

# Display the styled DataFrame
from IPython.display import display
display(df_coverages_clusters_styled)



# mean_summary_transposed['Metric (Yards)'] = mean_summary_transposed['Metric (Yards)'].map(metric_names)
mean_summary_transposed = pd.read_csv('/kaggle/input/metric-cluster-averages/mean_summary_transposed.csv')

# Rename cluster columns
cluster_column_map = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'}
mean_summary_transposed.rename(columns=cluster_column_map, inplace=True)

# Apply formatting and hide the index
styled_df = mean_summary_transposed.style.format(precision=2).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#013369'),
                                 ('font-weight', 'bold'), ('color', 'white')]},  # Header styles
    {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '14px')]},  # Cell styles
    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},  # Lighter grey for even rows
    {'selector': 'tr:nth-child(odd)', 'props': [('background-color', 'white')]},   # Default white for odd rows
]).set_properties(**{'border': 'none', 'padding': '5px'})  # Cell padding and border adjustments

# Hide the index
styled_df = styled_df.hide(axis="index")

# Add a title as a caption
styled_df = styled_df.set_caption(
    'Average Value for Each Pre Snap Cluster'
).set_table_styles([
    {'selector': 'caption', 'props': [('caption-side', 'top'), ('text-align', 'center'), 
                                      ('font-size', '14px'), ('font-weight', 'bold'), ('color', 'black')]}
], overwrite=False)

# Display the styled DataFrame
from IPython.display import display
display(styled_df)


df_weeks_game1 = df_weeks.loc[df_weeks['gameId'] == 2022100911].reset_index()
df_weeks_game1 = df_weeks_game1.loc[df_weeks_game1['playId'] == 1566]

df_weeks_game2 = df_weeks.loc[df_weeks['gameId'] == 2022091804].reset_index()
df_weeks_game2 = df_weeks_game2.loc[df_weeks_game2['playId'] == 3409]

df_weeks_game3 = df_weeks.loc[df_weeks['gameId'] == 2022100904].reset_index()
df_weeks_game3 = df_weeks_game3.loc[df_weeks_game3['playId'] == 3001]

df_weeks_game4 = df_weeks.loc[df_weeks['gameId'] == 2022092900].reset_index()
df_weeks_game4 = df_weeks_game4.loc[df_weeks_game4['playId'] == 2598]

df_weeks_game5 = df_weeks.loc[df_weeks['gameId'] == 2022102307].reset_index()
df_weeks_game5 = df_weeks_game5.loc[df_weeks_game5['playId'] == 2733]

#df_weeks_game2.tail()


#Example of Cluster 1
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
import plotly.io as pio
pio.renderers.default = 'iframe'
#pio.renderers.default = 'notebook'

# Assuming df_weeks_game3 is your DataFrame
df_weeks_game1 = df_weeks_game1.loc[df_weeks_game1['time'].notnull()]
df_weeks_game1['time_str'] = df_weeks_game1['time'].astype(str)  # Convert time to string for Plotly

# Create scatter plot with animation
fig = px.scatter(
    df_weeks_game1,
    x='x',
    y='y',
    animation_frame='time_str',
    color='club',
    size=[1] * len(df_weeks_game1),  # Set dot sizes
    size_max=12,
    text='jerseyNumber',  # Add jersey numbers as text labels
    range_x=[0, 120],
    range_y=[0, 53.3],
    title='Eagles at Cardinals: Cluster 1 to Cover 3',
    color_discrete_map={'football': '#814d0f', 'PHI': '#58aab2', 'ARI': '#f43535'}  # Set colors for dots
)
# Center the title and customize the font
fig.update_layout(
    title={
        'x': 0.5,  # Center title horizontally
        'xanchor': 'center',  # Ensure it aligns correctly
        'yanchor': 'top',  # Top alignment for title
        'font': {
            'family': 'Arial, sans-serif',  # Change font family
            'size': 16,  # Set font size
            'color': 'black'  # Set font color
        }
    }
)
# Customize layout to resemble a football field
fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="#3f9b0b",  # Set field color
    xaxis=dict(
        showticklabels=False,  # Remove x-axis labels
        gridcolor="white",  # Vertical grid lines for yard markers
        tickvals=list(range(0, 121, 10)),  # Tick marks every 10 yards
        ticktext=[f"{i}" for i in range(0, 121, 10)],
    ),
    yaxis=dict(
        showgrid=False,  # Disable horizontal grid lines
        showticklabels=False  # No ticks on the y-axis
    ),
    height=500,  # Adjust height
    width=575,  # Adjust width
    legend_title_text="Team",  # Set legend title
)

# Add darker green end zones
fig.add_shape(
    type="rect",
    x0=0,
    x1=10,
    y0=0,
    y1=53.3,
    fillcolor="#f43535",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)
fig.add_shape(
    type="rect",
    x0=110,
    x1=120,
    y0=0,
    y1=53.3,
    fillcolor="#f43535",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)

# Add vertical yard markers
for x in range(10, 111, 10):
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=53.3,
        line=dict(color="white", width=2),
        layer="below"
    )

# Add a yellow line at x = 52
fig.add_shape(
    type="line",
    x0=32,
    x1=32,
    y0=0,
    y1=53.3,
    line=dict(color="yellow", width=3),  # Yellow color
    layer="below"
)


# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=22.8,
    y1=22.8,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=30.5,
    y1=30.5,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)
# Add yard marker annotations
yard_markers = {
    20: "10", 30: "20", 40: "30", 50: "40",
    60: "50", 70: "40", 80: "30", 90: "20", 100: "10"
}
for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=11,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center"
    )


for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=42.2,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center",
        textangle=180
    )
# Add the NFL logo image to the center of the field
fig.update_layout(
    images=[
        dict(
            source="https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg", 
            x=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            y=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            xref="paper",  # Reference x-axis in normalized coordinates
            yref="paper",  # Reference y-axis in normalized coordinates
            sizex=0.2,  # Width of the image (relative to the plot)
            sizey=0.2,  # Height of the image (relative to the plot)
            xanchor="center",  # Align image to its center on x
            yanchor="middle",  # Align image to its center on y
            layer="above"  # Place the image below the data
        )
    ]
)

# Customize Play/Pause buttons
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                'label': '▶',  # Play icon
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': '⏸',  # Pause icon
                'method': 'animate'
            }
        ],
        'direction': 'left',
        #'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Rename the animation label above the slider
fig.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': 'Time: ',  # Change "time_str:" to "time:"
            'font': {'size': 14, 'color': 'black'}
        }
    }]
)
# Ensure dots are on top by adjusting the z-order of the scatter plot trace
fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"), z=101)
# Show the plot
fig.show()


#Example of Cluster 2

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming df_weeks_game3 is your DataFrame
df_weeks_game2 = df_weeks_game2.loc[df_weeks_game2['time'].notnull()]
df_weeks_game2['time_str'] = df_weeks_game2['time'].astype(str)  # Convert time to string for Plotly

# Create scatter plot with animation
fig = px.scatter(
    df_weeks_game2,
    x='x',
    y='y',
    animation_frame='time_str',
    color='club',
    size=[1] * len(df_weeks_game2),  # Set dot sizes
    size_max=12,
    text='jerseyNumber',  # Add jersey numbers as text labels
    range_x=[0, 120],
    range_y=[0, 53.3],
    title='Bucs at Saints: Cluster 2 to Cover 1',
    color_discrete_map={'football': '#814d0f', 'TB': '#df4848', 'NO': '#D3BC8D'}  # Set colors for dots
)
# Center the title and customize the font
fig.update_layout(
    title={
        'x': 0.5,  # Center title horizontally
        'xanchor': 'center',  # Ensure it aligns correctly
        'yanchor': 'top',  # Top alignment for title
        'font': {
            'family': 'Arial, sans-serif',  # Change font family
            'size': 16,  # Set font size
            'color': 'black'  # Set font color
        }
    }
)
# Customize layout to resemble a football field
fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="#3f9b0b",  # Set field color
    xaxis=dict(
        showticklabels=False,  # Remove x-axis labels
        gridcolor="white",  # Vertical grid lines for yard markers
        tickvals=list(range(0, 121, 10)),  # Tick marks every 10 yards
        ticktext=[f"{i}" for i in range(0, 121, 10)],
    ),
    yaxis=dict(
        showgrid=False,  # Disable horizontal grid lines
        showticklabels=False  # No ticks on the y-axis
    ),
    height=500,  # Adjust height
    width=575,  # Adjust width
    legend_title_text="Team",  # Set legend title
)

# Add darker green end zones
fig.add_shape(
    type="rect",
    x0=0,
    x1=10,
    y0=0,
    y1=53.3,
    fillcolor="#D3BC8D",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)
fig.add_shape(
    type="rect",
    x0=110,
    x1=120,
    y0=0,
    y1=53.3,
    fillcolor="#D3BC8D",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)

# Add vertical yard markers
for x in range(10, 111, 10):
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=53.3,
        line=dict(color="white", width=2),
        layer="below"
    )

# Add a yellow line at x = 52
fig.add_shape(
    type="line",
    x0=33,
    x1=33,
    y0=0,
    y1=53.3,
    line=dict(color="yellow", width=3),  # Yellow color
    layer="below"
)


# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=22.8,
    y1=22.8,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=30.5,
    y1=30.5,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)
# Add yard marker annotations
yard_markers = {
    20: "10", 30: "20", 40: "30", 50: "40",
    60: "50", 70: "40", 80: "30", 90: "20", 100: "10"
}
for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=11,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center"
    )


for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=42.2,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center",
        textangle=180
    )
# Add the NFL logo image to the center of the field
fig.update_layout(
    images=[
        dict(
            source="https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg", 
            x=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            y=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            xref="paper",  # Reference x-axis in normalized coordinates
            yref="paper",  # Reference y-axis in normalized coordinates
            sizex=0.2,  # Width of the image (relative to the plot)
            sizey=0.2,  # Height of the image (relative to the plot)
            xanchor="center",  # Align image to its center on x
            yanchor="middle",  # Align image to its center on y
            layer="above"  # Place the image below the data
        )
    ]
)

# Customize Play/Pause buttons
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                'label': '▶',  # Play icon
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': '⏸',  # Pause icon
                'method': 'animate'
            }
        ],
        'direction': 'left',
        #'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Rename the animation label above the slider
fig.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': 'Time: ',  # Change "time_str:" to "time:"
            'font': {'size': 14, 'color': 'black'}
        }
    }]
)
# Ensure dots are on top by adjusting the z-order of the scatter plot trace
fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"), z=101)
# Show the plot
fig.show()


#Example of Cluster 3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming df_weeks_game3 is your DataFrame
df_weeks_game3 = df_weeks_game3.loc[df_weeks_game3['time'].notnull()]
df_weeks_game3['time_str'] = df_weeks_game3['time'].astype(str)  # Convert time to string for Plotly

# Create scatter plot with animation
fig = px.scatter(
    df_weeks_game3,
    x='x',
    y='y',
    animation_frame='time_str',
    color='club',
    size=[1] * len(df_weeks_game3),  # Set dot sizes
    size_max=12,
    text='jerseyNumber',  # Add jersey numbers as text labels
    range_x=[0, 120],
    range_y=[0, 53.3],
    title='Bears at Vikings: Cluster 3 to Cover 2',
    color_discrete_map={'football': '#814d0f', 'CHI': '#C83803', 'MIN': '#c326fa'}  # Set colors for dots
)

# Center the title and customize the font
fig.update_layout(
    title={
        'x': 0.5,  # Center title horizontally
        'xanchor': 'center',  # Ensure it aligns correctly
        'yanchor': 'top',  # Top alignment for title
        'font': {
            'family': 'Arial, sans-serif',  # Change font family
            'size': 16,  # Set font size
            'color': 'black'  # Set font color
        }
    }
)

# Customize layout to resemble a football field
fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="#3f9b0b",  # Set field color
    xaxis=dict(
        showticklabels=False,  # Remove x-axis labels
        gridcolor="white",  # Vertical grid lines for yard markers
        tickvals=list(range(0, 121, 10)),  # Tick marks every 10 yards
        ticktext=[f"{i}" for i in range(0, 121, 10)],
    ),
    yaxis=dict(
        showgrid=False,  # Disable horizontal grid lines
        showticklabels=False  # No ticks on the y-axis
    ),
    height=500,  # Adjust height
    width=575,  # Adjust width
    legend_title_text="Team",  # Set legend title
)

# Add darker green end zones
fig.add_shape(
    type="rect",
    x0=0,
    x1=10,
    y0=0,
    y1=53.3,
    fillcolor="#9f19ce",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)
fig.add_shape(
    type="rect",
    x0=110,
    x1=120,
    y0=0,
    y1=53.3,
    fillcolor="#9f19ce",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)

# Add vertical yard markers
for x in range(10, 111, 10):
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=53.3,
        line=dict(color="white", width=2),
        layer="below"
    )

# Add a yellow line at x = 52
fig.add_shape(
    type="line",
    x0=78,
    x1=78,
    y0=0,
    y1=53.3,
    line=dict(color="yellow", width=3),  # Yellow color
    layer="below"
)


# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=22.8,
    y1=22.8,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=30.5,
    y1=30.5,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)
# Add yard marker annotations
yard_markers = {
    20: "10", 30: "20", 40: "30", 50: "40",
    60: "50", 70: "40", 80: "30", 90: "20", 100: "10"
}
for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=11,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center"
    )


for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=42.2,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center",
        textangle=180
    )
# Add the NFL logo image to the center of the field
fig.update_layout(
    images=[
        dict(
            source="https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg",  # Replace with the URL of the NFL logo
            x=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            y=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            xref="paper",  # Reference x-axis in normalized coordinates
            yref="paper",  # Reference y-axis in normalized coordinates
            sizex=0.2,  # Width of the image (relative to the plot)
            sizey=0.2,  # Height of the image (relative to the plot)
            xanchor="center",  # Align image to its center on x
            yanchor="middle",  # Align image to its center on y
            layer="below"  # Place the image below the data
        )
    ]
)

# Customize Play/Pause buttons
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                'label': '▶',  # Play icon
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': '⏸',  # Pause icon
                'method': 'animate'
            }
        ],
        'direction': 'left',
        #'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Rename the animation label above the slider
fig.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': 'Time: ',  # Change "time_str:" to "time:"
            'font': {'size': 14, 'color': 'black'}
        }
    }]
)

# Ensure dots are on top by adjusting the z-order of the scatter plot trace
fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"), z=101)
# Show the plot
fig.show()


#Example of Cluster 4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming df_weeks_game5 is your DataFrame
df_weeks_game4 = df_weeks_game4.loc[df_weeks_game4['time'].notnull()]
df_weeks_game4['time_str'] = df_weeks_game4['time'].astype(str)  # Convert time to string for Plotly

# Create scatter plot with animation
fig = px.scatter(
    df_weeks_game4,
    x='x',
    y='y',
    animation_frame='time_str',
    color='club',
    size=[1] * len(df_weeks_game4),  # Set dot sizes
    size_max=12,
    text='jerseyNumber',  # Add jersey numbers as text labels
    range_x=[0, 120],
    range_y=[0, 53.3],
    title='Dolphins at Bengals: Cluster 4 to Quarters',
    color_discrete_map={'football': '#814d0f', 'CIN': '#FB4F14', 'MIA': '#09f6fa'}  # Set colors for dots
)
# Center the title and customize the font
fig.update_layout(
    title={
        'x': 0.5,  # Center title horizontally
        'xanchor': 'center',  # Ensure it aligns correctly
        'yanchor': 'top',  # Top alignment for title
        'font': {
            'family': 'Arial, sans-serif',  # Change font family
            'size': 16,  # Set font size
            'color': 'black'  # Set font color
        }
    }
)
# Customize layout to resemble a football field
fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="#3f9b0b",  # Set field color
    xaxis=dict(
        showticklabels=False,  # Remove x-axis labels
        gridcolor="white",  # Vertical grid lines for yard markers
        tickvals=list(range(0, 121, 10)),  # Tick marks every 10 yards
        ticktext=[f"{i}" for i in range(0, 121, 10)],
    ),
    yaxis=dict(
        showgrid=False,  # Disable horizontal grid lines
        showticklabels=False  # No ticks on the y-axis
    ),
    height=500,  # Adjust height
    width=575,  # Adjust width
    legend_title_text="Team",  # Set legend title
)

# Add darker green end zones
fig.add_shape(
    type="rect",
    x0=0,
    x1=10,
    y0=0,
    y1=53.3,
    fillcolor="#FB4F14",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)
fig.add_shape(
    type="rect",
    x0=110,
    x1=120,
    y0=0,
    y1=53.3,
    fillcolor="#FB4F14",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)

# Add vertical yard markers
for x in range(10, 111, 10):
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=53.3,
        line=dict(color="white", width=2),
        layer="below"
    )

# Add a yellow line at x = 52
fig.add_shape(
    type="line",
    x0=69,
    x1=69,
    y0=0,
    y1=53.3,
    line=dict(color="yellow", width=3),  # Yellow color
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=22.8,
    y1=22.8,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=30.5,
    y1=30.5,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)
# Add yard marker annotations
yard_markers = {
    20: "10", 30: "20", 40: "30", 50: "40",
    60: "50", 70: "40", 80: "30", 90: "20", 100: "10"
}
for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=11,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center"
    )


for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=42.2,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center",
        textangle=180
    )
# Add the NFL logo image to the center of the field
fig.update_layout(
    images=[
        dict(
            source="https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg",  # Replace with the URL of the NFL logo
            x=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            y=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            xref="paper",  # Reference x-axis in normalized coordinates
            yref="paper",  # Reference y-axis in normalized coordinates
            sizex=0.2,  # Width of the image (relative to the plot)
            sizey=0.2,  # Height of the image (relative to the plot)
            xanchor="center",  # Align image to its center on x
            yanchor="middle",  # Align image to its center on y
            layer="below"  # Place the image below the data
        )
    ]
)

# Customize Play/Pause buttons
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                'label': '▶',  # Play icon
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': '⏸',  # Pause icon
                'method': 'animate'
            }
        ],
        'direction': 'left',
        #'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Rename the animation label above the slider
fig.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': 'Time: ',  # Change "time_str:" to "time:"
            'font': {'size': 14, 'color': 'black'}
        }
    }]
)
# Ensure dots are on top by adjusting the z-order of the scatter plot trace
fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"), z=101)
# Show the plot
fig.show()


#Example of Motion 

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming df_weeks_game5 is your DataFrame
df_weeks_game5 = df_weeks_game5.loc[df_weeks_game5['time'].notnull()]
df_weeks_game5['time_str'] = df_weeks_game5['time'].astype(str)  # Convert time to string for Plotly

# Create scatter plot with animation
fig = px.scatter(
    df_weeks_game5,
    x='x',
    y='y',
    animation_frame='time_str',
    color='club',
    size=[1] * len(df_weeks_game5),  # Set dot sizes
    size_max=12,
    text='jerseyNumber',  # Add jersey numbers as text labels
    range_x=[0, 120],
    range_y=[0, 53.3],
    title='Jets at Broncos: Cluster 4 to Cover 1',
    color_discrete_map={'football': '#814d0f', 'NYJ': '#6ca893', 'DEN': '#FB4F14'}  # Set colors for dots
)
# Center the title and customize the font
fig.update_layout(
    title={
        'x': 0.5,  # Center title horizontally
        'xanchor': 'center',  # Ensure it aligns correctly
        'yanchor': 'top',  # Top alignment for title
        'font': {
            'family': 'Arial, sans-serif',  # Change font family
            'size': 16,  # Set font size
            'color': 'black'  # Set font color
        }
    }
)
# Customize layout to resemble a football field
fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    plot_bgcolor="#3f9b0b",  # Set field color
    xaxis=dict(
        showticklabels=False,  # Remove x-axis labels
        gridcolor="white",  # Vertical grid lines for yard markers
        tickvals=list(range(0, 121, 10)),  # Tick marks every 10 yards
        ticktext=[f"{i}" for i in range(0, 121, 10)],
    ),
    yaxis=dict(
        showgrid=False,  # Disable horizontal grid lines
        showticklabels=False  # No ticks on the y-axis
    ),
    height=500,  # Adjust height
    width=575,  # Adjust width
    legend_title_text="Team",  # Set legend title
)

# Add darker green end zones
fig.add_shape(
    type="rect",
    x0=0,
    x1=10,
    y0=0,
    y1=53.3,
    fillcolor="#FB4F14",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)
fig.add_shape(
    type="rect",
    x0=110,
    x1=120,
    y0=0,
    y1=53.3,
    fillcolor="#FB4F14",  # endzone
    opacity=1,
    layer="below",
    line_width=0,
)

# Add vertical yard markers
for x in range(10, 111, 10):
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=53.3,
        line=dict(color="white", width=2),
        layer="below"
    )

# Add a yellow line at x = 52
fig.add_shape(
    type="line",
    x0=52,
    x1=52,
    y0=0,
    y1=53.3,
    line=dict(color="yellow", width=3),  # Yellow color
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=22.8,
    y1=22.8,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)

# Add dotted white horizontal lines
fig.add_shape(
    type="line",
    x0=0,
    x1=120,
    y0=30.5,
    y1=30.5,
    line=dict(color="white", width=2, dash="dot"),  # Dotted white line
    layer="below"
)
# Add yard marker annotations
yard_markers = {
    20: "10", 30: "20", 40: "30", 50: "40",
    60: "50", 70: "40", 80: "30", 90: "20", 100: "10"
}
for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=11,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center"
    )


for x, text in yard_markers.items():
    fig.add_annotation(
        x=x,
        y=42.2,  # Fixed y coordinate for text placement
        text=text,
        showarrow=False,
        font=dict(color="white", size=15),  # White font
        align="center",
        textangle=180
    )
# Add the NFL logo image to the center of the field
fig.update_layout(
    images=[
        dict(
            source="https://upload.wikimedia.org/wikipedia/en/a/a2/National_Football_League_logo.svg",  # Replace with the URL of the NFL logo
            x=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            y=0.5,  # Center of the field (normalized coordinate: 0 to 1)
            xref="paper",  # Reference x-axis in normalized coordinates
            yref="paper",  # Reference y-axis in normalized coordinates
            sizex=0.2,  # Width of the image (relative to the plot)
            sizey=0.2,  # Height of the image (relative to the plot)
            xanchor="center",  # Align image to its center on x
            yanchor="middle",  # Align image to its center on y
            layer="below"  # Place the image below the data
        )
    ]
)

# Customize Play/Pause buttons
fig.update_layout(
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                'label': '▶',  # Play icon
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                'label': '⏸',  # Pause icon
                'method': 'animate'
            }
        ],
        'direction': 'left',
        #'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Rename the animation label above the slider
fig.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': 'Time: ',  # Change "time_str:" to "time:"
            'font': {'size': 14, 'color': 'black'}
        }
    }]
)
# Ensure dots are on top by adjusting the z-order of the scatter plot trace
fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"), z=101)
# Show the plot
fig.show()



df_team_disco1 = pd.read_csv('/kaggle/input/summarized-big-data-bowl-25-tables/df_team_disco1.csv')

# # Merge the logo and team into one column
# df_team_disco1['Team'] = ''
# df_team_disco1['Team'] = df_team_disco1.apply(
#     lambda row: f'<img src="{row["logo_url"]}" style="height:30px; padding-right: 10px;">{row["Team"]}',
#     axis=1
# )

# # Drop the original 'logo_url' and 'Team' columns
# df_team_disco1 = df_team_disco1.drop(columns=['logo_url'])
df_team_disco1 = pd.read_csv('/kaggle/input/summarized-big-data-bowl-25-tables/df_team_disco1.csv')

# Reorder columns so the new merged column is first
df_team_disco1 = df_team_disco1[['Team', 'DISCO %', 'DISCO Cluster 1 %', 'DISCO Cluster 2 %', 'DISCO Cluster 3 %', 'DISCO Cluster 4 %']]
df_team_disco1.rename(columns=lambda x: x.replace(' %', ''), inplace=True)

# Use Pandas styling to display the merged column and format the DataFrame
df_team_disco_styled = df_team_disco1.style.format({
    'Logo and Team': lambda x: x  # Keep the merged column as is
}).hide(axis='index')  # Hide the default index

# Set table styles to ensure correct alignment and appearance
df_team_disco_styled.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f4f4f4'), ('font-weight', 'bold')]},
    {'selector': 'td', 'props': [('text-align', 'center')]},
])#.set_properties(**{'border': '1px solid black', 'padding': '5px'})

# Use Pandas styling to display the logos and format the DataFrame
def render_logo(url):
    return f'<img src="{url}" style="height:30px;">'


# Set table styles to ensure correct alignment, appearance, and alternating row colors
df_team_disco_styled.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#013369'), 
                                 ('font-weight', 'bold'), ('color', 'white')]},  # Make column label text white
    {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '16px')]},  # Adjust font size here
    {'selector': 'tr:nth-child(aeven)', 'props': [('background-color', '#f9f9f9')]},  # Lighter grey for even rows
    {'selector': 'tr:nth-child(odd)', 'props': [('background-color', 'white')]},   # Default grey for odd rows
], overwrite=False).set_properties(**{'border': 'none', 'padding': '7px'})  # Remove the borders

# Add a title as a caption
df_team_disco_styled = df_team_disco_styled.set_caption(
    'Frequency of DISCO Plays by Team'
).set_table_styles([
    {'selector': 'caption', 'props': [('caption-side', 'top'), ('text-align', 'center'), 
                                      ('font-size', '14px'), ('font-weight', 'bold'), ('color', 'black')]}
], overwrite=False)


# Display the styled DataFrame with the merged column
from IPython.display import display
display(df_team_disco_styled)



df_performance = pd.read_csv('/kaggle/input/summarized-big-data-bowl-25-tables/df_performance.csv')

# Set colors for the bars
colors = ['#D50A0A', '#013369']  # NFL logo red and blue

# Create the bar plot and assign it to ax
plt.figure(figsize=(10, 6))
ax = sn.barplot(
    data=df_performance,
    x='coverage_cluster_cat',
    y='expectedPointsAdded',
    hue='disco',
    palette=colors
)

# Customize the plot
plt.title('Change in Expected Points per Play', fontsize=18, color='black', pad=15)
plt.xlabel('Cluster', fontsize=16, color='black')
plt.ylabel('Change in Expected Points Per Play', fontsize=16, color='black')

# Ensure x-axis shows integers
plt.xticks(fontsize=18, color='black')

# Set y-axis range
plt.ylim(-0.1, 0.4)
plt.yticks(fontsize=18, color='black')

# Adjust legend
plt.legend(title=None, fontsize=12, loc='best')

# Remove gridlines
plt.grid(False)

# Ensure layout fits
plt.tight_layout()
# Enable horizontal gridlines for y-axis ticks
ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.2, color='grey')
ax.grid(axis='x', visible=False)  # Disable vertical gridlines

# Set background colors to very light grey
ax.set_facecolor('#edebeb')  # Light grey for the plot area
plt.gcf().set_facecolor('#edebeb')  # Light grey for the figure background


# Show the plot
plt.show()

