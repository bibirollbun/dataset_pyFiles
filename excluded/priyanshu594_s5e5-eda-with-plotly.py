#Import the Libraries
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from warnings import filterwarnings
filterwarnings('ignore')
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df.sample(5)


num_records = len(df)
num_records


num_columns = len(df.columns)
num_columns


def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values
    summ['Duplicate'] = df.duplicated().sum()
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['avg'] = desc['mean'].values
    summ['std dev'] = desc['std'].values
    summ['top value'] = desc['top'].values
    summ['Freq'] = desc['freq'].values

    return summ

summary(df).style.background_gradient()


df.info()


df.nunique()


import plotly.express as px

cat_columns = df[['Sex']]

def univariateAnalysis_category(cols):
    print("Distribution of", cols)
    print("_" * 60)
    colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',  
        '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',  
        '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB', 
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'  
    ]
    value_counts = cat_columns[cols].value_counts()

    # Create bar plot
    fig = px.bar(
        value_counts,
        x=value_counts.index,
        y=value_counts.values,
        title=f'Distribution of {cols}',
        labels={'x': 'Categories', 'y': 'Count'},
        color_discrete_sequence=[colors]
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12), 
        title_font=dict(size=30),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )
    fig.show()

    # Calculate percentage
    percentage = (value_counts / value_counts.sum()) * 100
    
    # Create pie chart
    fig = px.pie(
        values=percentage,
        names=value_counts.index,
        labels={'names': 'Categories', 'values': 'Percentage'},
        hole=0.5,
        color_discrete_sequence=colors
    )
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f'{cols}',
        font=dict(size=18, color='white'),
        showarrow=False
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        title_font=dict(size=30),
        legend=dict(x=0.9, y=0.5),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )
    fig.show()
    print("       ")

for x in cat_columns:
    univariateAnalysis_category(x)



df.nunique()


df.info()


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from IPython.display import display, HTML

def groupby(data, x):
    result = data.groupby(x).size().rename('count').reset_index()
    return result

def create_combined_plot(data, feature, color, width=600, height=360):
    # Prepare grouped data for scatter
    grouped_data = groupby(data, feature)

    # Create subplot with 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Scatter_plot', 'Violin_plot'))

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=grouped_data[feature],
            y=grouped_data['count'],
            mode='markers',
            marker=dict(color=color, size=10),
            name='Scatter'
        ),
        row=1, col=1
    )

    # Violin plot
    fig.add_trace(
        go.Violin(
            y=data[feature],
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            name='Violin',
            orientation='v'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        width=width,
        height=height,
        showlegend=False
    )

    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False)

    fig.show()

# Assuming 'df' is already defined and loaded
color = [
    '#FFD700', '#FFA500', '#00FA9A', '#FFB6C1', '#FF1493',
    'red', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
]

numeric_cols = df.select_dtypes(include='number').columns

for i, feature in enumerate(numeric_cols):
    if feature == 'id':
        continue
    display(HTML(f"<h1 style='text-align:center; font-size:40px; font-weight:bold;'>{feature} Distribution</h1>"))
    create_combined_plot(df, feature, color[i % len(color)])
    print("\n\n\n")



df.nunique()


import pandas as pd
import plotly.express as px
from IPython.display import display, HTML

def groupby(data, x):
    result = data.groupby(x).size().rename('count').reset_index()
    return result

def plot_scatter(df, x_col, y_col, color=None):
    print("\n\n")
    display(HTML(f"<h2 style='text-align:left; font-size:27px; font-weight:bold;'>{x_col} vs {y_col}</h2>"))
    colors = ['red', 'yellow']

    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=df[color].astype(str) if color is not None else None,
        color_discrete_sequence=colors  
    )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=11),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white', showline=False),  
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='white', showline=False), 
        legend_title_text=color,
        legend_font=dict(color='white', size=12),
        width=470,  
        height=320,
        
        
    )
    
   
    fig.show()

plot_scatter(df, 'Height', 'Weight',  'Sex')
plot_scatter(df, 'Age', 'Heart_Rate',  'Sex')
plot_scatter(df, 'Age', 'Calories',  'Sex')
plot_scatter(df, 'Duration', 'Calories',  'Sex')
plot_scatter(df, 'Heart_Rate', 'Calories', 'Sex')
plot_scatter(df, 'Body_Temp', 'Calories', 'Sex')





import plotly.graph_objects as go

numeric_df = df.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr().round(2)  # round to 2 decimals for readability

fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    text=correlation_matrix.values,
    texttemplate="%{text}",
    colorscale='Tealrose',
    zmin=-1, zmax=1,
    colorbar=dict(title="Correlation")
))

fig.update_layout(
    title='Correlation Heatmap of numeric features',
    xaxis_showgrid=False,
    yaxis_showgrid=False
)

fig.show()




