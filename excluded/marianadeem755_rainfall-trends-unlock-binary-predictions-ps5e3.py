import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
# Import Plotly.go
import plotly.graph_objects as go
# import Subplots
from plotly.subplots import make_subplots
import plotly.io as pio
# Set the default renderer for both Plotly Express and Graph Objects
pio.renderers.default = 'iframe_connected'
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import random
from IPython.core.display import display, HTML
# ignore warnings
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import random
from IPython.display import display, HTML

# Function to style tables
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#FF4500")]}]  # Changed header color
    ).set_properties(**{"text-align": "center", "font-size": "14px"}).hide(axis="index")
    return styled_df.to_html()

# Function to create styled heading with emojis
def styled_heading(text, background_color='#FF4500', text_color='white'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        color: {text_color};
        padding: 18px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.15);
        border: 2px solid {background_color};
    ">
        {text}
    </div>
    """

# Function to print dataset analysis
def print_dataset_analysis(dataset, dataset_name, n_top=5, heading_color='#FF4500', text_color='white'):
    heading = styled_heading(f"ğŸ“Š {dataset_name} Overview", heading_color, text_color)
    display(HTML(heading))

    def subheader(text):
        return f"<h2 style='font-size: 18px; color: #FF4500; margin-top: 15px; text-decoration: underline;'>{text}</h2>"

    display(HTML(subheader("ğŸ“� Shape of the Dataset")))
    display(HTML(f"<p style='font-size: 16px;'>{dataset.shape[0]} rows and {dataset.shape[1]} columns</p>"))

    display(HTML(subheader("ğŸ”� First 5 Rows")))
    display(HTML(style_table(dataset.head(n_top))))

    display(HTML(subheader("ğŸ“Š Summary Statistics")))
    display(HTML(style_table(dataset.describe())))

    display(HTML(subheader("ğŸ”§ Null Values")))
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML("<p style='font-size: 16px;'>âœ… No null values found.</p>"))
    else:
        display(HTML(style_table(null_counts[null_counts > 0].to_frame(name='Null Values'))))

    display(HTML(subheader("â™»ï¸� Duplicate Rows")))
    duplicate_count = dataset.duplicated().sum()
    display(HTML(f"<p style='font-size: 16px;'>{duplicate_count} duplicate rows found.</p>"))

    display(HTML(subheader("ğŸ—‚ï¸� Data Types")))
    dtypes_table = pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns]
    })
    display(HTML(style_table(dtypes_table)))

    display(HTML(subheader("ğŸ“‹ Column Names")))
    display(HTML(f"<p style='font-size: 16px;'>{', '.join(dataset.columns)}</p>"))

    display(HTML(subheader("ğŸ”¢ Unique Values")))
    unique_values_table = pd.DataFrame({
        'Column Name': dataset.columns,
        'Unique Values': [', '.join(map(str, dataset[col].unique()[:7])) + (', ...' if len(dataset[col].unique()) > 7 else '') for col in dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))

# Load datasets
df_original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# Analyze datasets with updated colors
print_dataset_analysis(df_original, "Original Data", heading_color='#C2185B')  # Magenta
print_dataset_analysis(df_train, "Training Data", heading_color='#795548')  # Brown
print_dataset_analysis(df_test, "Test Data", heading_color='#AEEA00')      # Lime
print_dataset_analysis(sample_sub, "Sample Submission", heading_color='#00BCD4') # Cyan



# Checking uniqueness
print("\nUnique ID Counts:")
print(f"Training Data unique id values are: {df_train['id'].nunique()}")
print("========================================================================")

# Checking the count of unique values
unique_ids = df_train['id'].nunique()
total_rows = len(df_train)

if unique_ids == total_rows:
    print("All IDs are unique.")
else:
    print(f"There are {total_rows - unique_ids} duplicate IDs.")
print("========================================================================")

# Checking ID range and sequence pattern
print("\nID Range:")
print(f"Training Data: Min ID: {df_train['id'].min()}, Max ID: {df_train['id'].max()}")
print("========================================================================")

# Checking whether IDs are sequential
train_id_diff = df_train['id'].diff().dropna().unique()

if len(train_id_diff) == 1 and train_id_diff[0] == 1:
    print("\nYes, IDs are Sequential")
else:
    print("\nNo, IDs are Non-Sequential")



# Check unique values and range of 'day', 'pressure', and 'maxtemp'
print("Unique values count:")
print(df_train[['day', 'pressure', 'maxtemp']].nunique())

print("\nMinimum and Maximum values:")
print(df_train[['day', 'pressure', 'maxtemp']].agg(['min', 'max', 'mean']))



# Calculate the rate of change
df_train['pressure_change'] = df_train['pressure'].diff()
df_train['maxtemp_change'] = df_train['maxtemp'].diff()

# Summary of the changes
print("Pressure Change:\n", df_train['pressure_change'].describe())
print("==============================================================")
print("\nMaxtemp Change:\n", df_train['maxtemp_change'].describe())



# Step 1: Group by 'day' and compute the maximum values for 'pressure' and 'maxtemp'
daily_stats = df_train.groupby('day')[['pressure', 'maxtemp']].max().reset_index()

# Step 2: Sort separately
top_10_pressure = daily_stats.sort_values(by='pressure', ascending=False).head(10)
top_10_temp = daily_stats.sort_values(by='maxtemp', ascending=False).head(10)

# Step 3: Display results
print("Top Days with Maximum Pressure:\n", top_10_pressure)
print("======================================================================")
print("\nTop Days with Maximum Temperature:\n", top_10_temp)



# Step 1: Group by 'day' and compute max values
daily_stats = df_train.groupby('day')[['pressure', 'maxtemp']].max().reset_index()

# Step 2: Create subplots with shared x-axis and individual y-axis labels
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=(
        "<b>Daily Maximum Pressure</b>", 
        "<b>Daily Maximum Temperature</b>"
    ),
    vertical_spacing=0.15  # Adjust spacing between subplots
)

# Step 3: Add line plots with customized styling
fig.add_trace(go.Scatter(
    x=daily_stats['day'], 
    y=daily_stats['pressure'], 
    mode='lines+markers', 
    name='Max Pressure',
    line=dict(color='royalblue', width=3), 
    marker=dict(size=7, color='navy'),
    hovertemplate='<b>Day:</b> %{x}<br><b>Max Pressure:</b> %{y}'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=daily_stats['day'], 
    y=daily_stats['maxtemp'], 
    mode='lines+markers', 
    name='Max Temperature',
    line=dict(color='crimson', width=3),
    marker=dict(size=7, color='darkred'),
    hovertemplate='<b>Day:</b> %{x}<br><b>Max Temperature:</b> %{y}'
), row=2, col=1)

# Step 4: Update layout with x-axis and y-axis labels
fig.update_layout(
    title_text="<b>Daily Maximum Pressure and Temperature Trends</b>",
    height=700, width=1200,
    showlegend=True,
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", family="Arial", size=14),
    
    # Adjust x-axis settings
    xaxis=dict(title="Day", showgrid=True, gridcolor="lightgrey"),
    xaxis2=dict(title="Day", showgrid=True, gridcolor="lightgrey"),
    
    # Adjust y-axis settings for both plots
    yaxis=dict(title="Max Pressure", showgrid=True, gridcolor="lightgrey"),
    yaxis2=dict(title="Max Temperature", showgrid=True, gridcolor="lightgrey"),
    
    # Improve subplot spacing
    margin=dict(l=60, r=40, t=80, b=60),
    
    # Customize title font
    title_font=dict(size=18, family="Arial Black"),
)

# Step 5: Show figure
fig.show()



# Check how pressure changes from one day to the next
df_train_sorted = df_train.sort_values(by='day')
df_train_sorted['pressure_diff'] = df_train_sorted['pressure'].diff()

print("Largest daily pressure changes:")
print(df_train_sorted[['day', 'pressure', 'pressure_diff']].nlargest(5, 'pressure_diff'))
print("=====================================================================================")
print("\nSmallest daily pressure changes:")
print(df_train_sorted[['day', 'pressure', 'pressure_diff']].nsmallest(5, 'pressure_diff'))



# Step 1: Sort by 'day' and calculate pressure difference
df_train_sorted = df_train.sort_values(by='day')
df_train_sorted['pressure_diff'] = df_train_sorted['pressure'].diff()

# Step 2: Get largest & smallest pressure changes
largest_changes = df_train_sorted.nlargest(10, 'pressure_diff')
smallest_changes = df_train_sorted.nsmallest(10, 'pressure_diff')

# Step 3: Create scatter plot
fig = go.Figure()

# Step 4: Plot overall pressure changes (lighter opacity)
fig.add_trace(go.Scatter(
    x=df_train_sorted['day'],
    y=df_train_sorted['pressure_diff'],
    mode='markers',
    marker=dict(color='royalblue', size=8, opacity=0.6, line=dict(color='black', width=1.2)),
    name="Daily Pressure Change",
    hoverinfo="text",
    hovertext=[f"Day: {d}<br>Pressure Change: {p:.2f}" for d, p in zip(df_train_sorted['day'], df_train_sorted['pressure_diff'])]
))

# Step 5: Highlight Largest Increases (bigger bubble)
fig.add_trace(go.Scatter(
    x=largest_changes['day'],
    y=largest_changes['pressure_diff'],
    mode='markers',
    marker=dict(color='green', size=16, opacity=1, line=dict(color='black', width=2)),
    name="Largest Increases",
    hoverinfo="text",
    hovertext=[f"Day: {d}<br>Pressure Increase: {p:.2f}" for d, p in zip(largest_changes['day'], largest_changes['pressure_diff'])]
))

# Step 6: Highlight Largest Decreases (bigger bubble)
fig.add_trace(go.Scatter(
    x=smallest_changes['day'],
    y=smallest_changes['pressure_diff'],
    mode='markers',
    marker=dict(color='red', size=16, opacity=1, line=dict(color='black', width=2)),
    name="Largest Decreases",
    hoverinfo="text",
    hovertext=[f"Day: {d}<br>Pressure Decrease: {p:.2f}" for d, p in zip(smallest_changes['day'], smallest_changes['pressure_diff'])]
))

# Step 7: Layout Styling
fig.update_layout(
    title="<b>Daily Pressure Changes (Bubble Scatterplot)</b>",
    height=700, width=1200,  # Larger figure size
    showlegend=True,
    plot_bgcolor='white',  # White background
    paper_bgcolor='white',
    font=dict(color='black'),
    xaxis=dict(title="<b>Day</b>", showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title="<b>Pressure Change</b>", showgrid=True, gridcolor='lightgray')
)

# Step 8: Show plot
fig.show()



# Step 5: Relationship between 'day' & 'pressure'
pressure_variation = df_train.groupby('day')['pressure'].agg(['min', 'max', 'mean'])

# Sorting in descending order and resetting the index
pressure_variation = pressure_variation.sort_values(by='mean', ascending=False).reset_index()

# Display results
print("Pressure variation per day:\n", pressure_variation.head(10))



# Step 1: Group by 'day' and compute pressure statistics
pressure_variation = df_train.groupby('day')['pressure'].agg(['min', 'max', 'mean']).reset_index()

# Step 2: Sort by mean pressure in descending order
pressure_variation = pressure_variation.sort_values(by='mean', ascending=False)

# Step 3: Create subplots for scatterplots only
fig = sp.make_subplots(
    rows=3, cols=1, shared_xaxes=True, 
    subplot_titles=(
        "<b>Minimum Daily Pressure</b>", 
        "<b>Maximum Daily Pressure</b>", 
        "<b>Mean Daily Pressure</b>"
    ),
    vertical_spacing=0.12  # Adjust spacing for better visibility
)

# Step 4: Add Min Pressure Scatter Plot
fig.add_trace(go.Scatter(
    x=pressure_variation['day'], 
    y=pressure_variation['min'], 
    mode='markers', 
    marker=dict(color='deepskyblue', size=10, opacity=0.85, line=dict(color='black', width=1.2)),
    name="Min Pressure",
    hoverinfo="text",
    hovertext=[f"Day: {d}<br>Min Pressure: {p}" for d, p in zip(pressure_variation['day'], pressure_variation['min'])]
), row=1, col=1)

# Step 5: Add Max Pressure Scatter Plot
fig.add_trace(go.Scatter(
    x=pressure_variation['day'], 
    y=pressure_variation['max'], 
    mode='markers', 
    marker=dict(color='tomato', size=10, opacity=0.85, line=dict(color='black', width=1.2)),
    name="Max Pressure",
    hoverinfo="text",
    hovertext=[f"Day: {d}<br>Max Pressure: {p}" for d, p in zip(pressure_variation['day'], pressure_variation['max'])]
), row=2, col=1)

# Step 6: Add Mean Pressure Scatter Plot
fig.add_trace(go.Scatter(
    x=pressure_variation['day'], 
    y=pressure_variation['mean'], 
    mode='markers', 
    marker=dict(color='seagreen', size=10, opacity=0.85, line=dict(color='black', width=1.2)),
    name="Mean Pressure",
    hoverinfo="text",
    hovertext=[f"Day: {d}<br>Mean Pressure: {p:.2f}" for d, p in zip(pressure_variation['day'], pressure_variation['mean'])]
), row=3, col=1)

# Step 7: Update layout with labels for each subplot
fig.update_layout(
    title="<b>Daily Pressure Variations (Min, Max, Mean)</b>",
    height=1200, width=1300,  # Increased figure size
    showlegend=True,
    plot_bgcolor='white',  # White background
    paper_bgcolor='white',
    font=dict(color='black'),
    
    # Global axis settings
    xaxis=dict(title="<b>Day</b>", showgrid=True, gridcolor='lightgray'),
    xaxis2=dict(title="<b>Day</b>", showgrid=True, gridcolor='lightgray'),
    xaxis3=dict(title="<b>Day</b>", showgrid=True, gridcolor='lightgray'),
    
    yaxis=dict(title="<b>Min Pressure</b>", showgrid=True, gridcolor='lightgray'),
    yaxis2=dict(title="<b>Max Pressure</b>", showgrid=True, gridcolor='lightgray'),
    yaxis3=dict(title="<b>Mean Pressure</b>", showgrid=True, gridcolor='lightgray'),
    
    margin=dict(l=80, r=50, t=80, b=80),  # Adjust margins for better spacing
    title_font=dict(size=20, family="Arial Black"),
)

# Step 8: Show the interactive plot
fig.show()



# Step 5: Relationship between 'day' & 'maxtemp'
maxtemp_variation = df_train.groupby('day')['maxtemp'].agg(['min', 'max', 'mean'])

# Sorting in descending order by mean maxtemp and resetting the index
maxtemp_variation = maxtemp_variation.sort_values(by='mean', ascending=False).reset_index()

# Display results
print("Temperature variation per day:\n", maxtemp_variation.head(10))



# Step 1: Aggregate min, max, and mean temperature per day
maxtemp_variation = df_train.groupby('day')['maxtemp'].agg(['min', 'max', 'mean']).reset_index()

# Step 2: Create subplots (3 rows, 1 column)
fig = make_subplots(rows=3, cols=1, subplot_titles=("Min Temperature", "Max Temperature", "Mean Temperature"))

# Step 3: Add Min Temperature Line
fig.add_trace(go.Scatter(
    x=maxtemp_variation['day'],
    y=maxtemp_variation['min'],
    mode='lines',
    name='Min Temp',
    line=dict(color='blue', width=2),
    hovertemplate="Day: %{x}<br>Min Temp: %{y}Â°C"
), row=1, col=1)

# Step 4: Add Max Temperature Line
fig.add_trace(go.Scatter(
    x=maxtemp_variation['day'],
    y=maxtemp_variation['max'],
    mode='lines',
    name='Max Temp',
    line=dict(color='red', width=2),
    hovertemplate="Day: %{x}<br>Max Temp: %{y}Â°C"
), row=2, col=1)

# Step 5: Add Mean Temperature Line
fig.add_trace(go.Scatter(
    x=maxtemp_variation['day'],
    y=maxtemp_variation['mean'],
    mode='lines+markers',
    name='Mean Temp',
    line=dict(color='orange', width=3, dash='dot'),
    marker=dict(color='black', size=6, symbol='circle'),
    hovertemplate="Day: %{x}<br>Mean Temp: %{y}Â°C"
), row=3, col=1)

# Step 6: Update Layout for White Background
fig.update_layout(
    title="<b>Daily Temperature Variations</b>",
    height=1000, width=1000,  # Adjusting height for better spacing
    showlegend=False,
    template="plotly_white",
    font=dict(color='black', size=12),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Step 7: Customize X & Y Axes
for i in range(1, 4):
    fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", row=i, col=1)
    fig.update_yaxes(title="Temperature", showgrid=True, gridcolor="lightgray", row=i, col=1)

# Step 8: Show Final Chart
fig.show()



# Step 1: Group by 'day' and calculate min, max, and mean for both 'pressure' and 'maxtemp'
pressure_temp_variation = df_train.groupby('day')[['pressure', 'maxtemp']].agg(['min', 'max', 'mean'])

# Step 2: Sorting
# For 'pressure': sort 'max' and 'mean' in descending order, and 'min' in ascending order
pressure_sorted = pressure_temp_variation['pressure'].sort_values(
    by=['max', 'mean', 'min'], 
    ascending=[False, False, True]
).reset_index()

# For 'maxtemp': sort 'max' and 'mean' in descending order, and 'min' in ascending order
maxtemp_sorted = pressure_temp_variation['maxtemp'].sort_values(
    by=['max', 'mean', 'min'], 
    ascending=[False, False, True]
).reset_index()

# Step 3: Add 'day' column separately for both 'pressure' and 'maxtemp'
pressure_sorted['day'] = pressure_sorted['day']
maxtemp_sorted['day'] = maxtemp_sorted['day']

# Step 4: Display results
print("Pressure variation per day:\n", pressure_sorted.head(10))
print("=======================================================================")
print("\nTemperature variation per day:\n", maxtemp_sorted.head(10))


# Step 1: Aggregate min, max, and mean for pressure & temperature per day
pressure_temp_variation = df_train.groupby('day')[['pressure', 'maxtemp']].agg(['min', 'max', 'mean']).reset_index()

# Step 2: Create 3 row-wise subplots
fig = make_subplots(
    rows=3, cols=1, 
    subplot_titles=("Pressure Variation", "Temperature Variation", "Pressure vs Temperature"),
    vertical_spacing=0.1  # Space between subplots
)

# Step 3: Add Pressure Variation (Min, Max, Mean)
fig.add_trace(go.Scatter(
    x=pressure_temp_variation['day'], 
    y=pressure_temp_variation[('pressure', 'min')], 
    mode='lines+markers', name='Min Pressure', 
    line=dict(color='blue', width=2), 
    marker=dict(symbol='circle', size=6),
    hovertemplate="Day: %{x}<br>Min Pressure: %{y}"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=pressure_temp_variation['day'], 
    y=pressure_temp_variation[('pressure', 'max')], 
    mode='lines+markers', name='Max Pressure', 
    line=dict(color='red', width=2),
    marker=dict(symbol='diamond', size=6),
    hovertemplate="Day: %{x}<br>Max Pressure: %{y}"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=pressure_temp_variation['day'], 
    y=pressure_temp_variation[('pressure', 'mean')], 
    mode='lines', name='Mean Pressure', 
    line=dict(color='green', width=3, dash='dot'),
    hovertemplate="Day: %{x}<br>Mean Pressure: %{y}"
), row=1, col=1)

# Step 4: Add Temperature Variation (Min, Max, Mean)
fig.add_trace(go.Scatter(
    x=pressure_temp_variation['day'], 
    y=pressure_temp_variation[('maxtemp', 'min')], 
    mode='lines+markers', name='Min Temperature', 
    line=dict(color='cyan', width=2),
    marker=dict(symbol='triangle-up', size=6),
    hovertemplate="Day: %{x}<br>Min Temp: %{y}"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=pressure_temp_variation['day'], 
    y=pressure_temp_variation[('maxtemp', 'max')], 
    mode='lines+markers', name='Max Temperature', 
    line=dict(color='orange', width=2),
    marker=dict(symbol='triangle-down', size=6),
    hovertemplate="Day: %{x}<br>Max Temp: %{y}"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=pressure_temp_variation['day'], 
    y=pressure_temp_variation[('maxtemp', 'mean')], 
    mode='lines', name='Mean Temperature', 
    line=dict(color='purple', width=3, dash='dot'),
    hovertemplate="Day: %{x}<br>Mean Temp: %{y}"
), row=2, col=1)

# Step 5: Scatter Plot for Pressure vs Temperature
fig.add_trace(go.Scatter(
    x=pressure_temp_variation[('pressure', 'mean')], 
    y=pressure_temp_variation[('maxtemp', 'mean')], 
    mode='markers', name='Pressure vs Temperature',
    marker=dict(color='magenta', size=8, opacity=0.7, symbol='hexagon'),
    hovertemplate="Mean Pressure: %{x}<br>Mean Temp: %{y}"
), row=3, col=1)

# Step 6: Update Layout (White Background, Labels)
fig.update_layout(
    title="<b>Daily Pressure & Temperature Variations</b>",
    height=1000, width=1200,  # Adjust size for better visibility
    showlegend=True,
    template="plotly_white",
    font=dict(color='black', size=12),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Step 7: Customize X & Y Axes
for i in range(1, 4):
    fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", row=i, col=1)
    
fig.update_yaxes(title="Pressure", showgrid=True, gridcolor="lightgray", row=1, col=1)
fig.update_yaxes(title="Temperature", showgrid=True, gridcolor="lightgray", row=2, col=1)
fig.update_xaxes(title="Mean Pressure)", row=3, col=1)
fig.update_yaxes(title="Mean Temperature", row=3, col=1)

# Step 8: Show Final Chart
fig.show()



# When 'maxtemp' is above 30, analyze 'pressure' along with 'maxtemp'
high_temp_data = df_train[df_train['maxtemp'] > 30].groupby('day').agg({'pressure': ['max', 'mean'], 'maxtemp': 'max'})

# Sorting in descending order by 'max' and 'mean' pressure
high_temp_data = high_temp_data.sort_values(by=[('pressure', 'max'), ('pressure', 'mean')], ascending=[False, False]).reset_index()

# Display results
print("Highest max and mean pressure when max temp > 30:\n", high_temp_data.head(10))


# Step 1: Filter and aggregate data for max temp > 30
high_temp_data = df_train[df_train['maxtemp'] > 30].groupby('day').agg({
    'pressure': ['max', 'mean'], 
    'maxtemp': 'max'
}).reset_index()

# Step 2: Create 3 row-wise scatterplot subplots
fig = make_subplots(
    rows=3, cols=1, 
    subplot_titles=("Max Pressure vs. Day", "Mean Pressure vs. Day", "Pressure vs. Max Temperature"),
    vertical_spacing=0.1  # Space between subplots
)

# Step 3: Scatter Plot for Max Pressure vs. Day
fig.add_trace(go.Scatter(
    x=high_temp_data['day'], 
    y=high_temp_data[('pressure', 'max')], 
    mode='markers', 
    name='Max Pressure', 
    marker=dict(color='red', size=8),
    hovertemplate="Day: %{x}<br>Max Pressure: %{y}"
), row=1, col=1)

# Step 4: Scatter Plot for Mean Pressure vs. Day
fig.add_trace(go.Scatter(
    x=high_temp_data['day'], 
    y=high_temp_data[('pressure', 'mean')], 
    mode='markers', 
    name='Mean Pressure', 
    marker=dict(color='blue', size=8),
    hovertemplate="Day: %{x}<br>Mean Pressure: %{y}"
), row=2, col=1)

# Step 5: Scatter Plot for Pressure vs Max Temperature
fig.add_trace(go.Scatter(
    x=high_temp_data[('pressure', 'mean')], 
    y=high_temp_data[('maxtemp', 'max')], 
    mode='markers', 
    name='Pressure vs Temp', 
    marker=dict(color='purple', size=10, opacity=0.7),
    hovertemplate="Mean Pressure: %{x}<br>Max Temp: %{y}"
), row=3, col=1)

# Step 6: Update Layout (White Background, Labels)
fig.update_layout(
    title="<b>Scatterplots for Pressure When Max Temperature > 30Â°</b>",
    height=1000, width=1200,  
    showlegend=True,
    template="plotly_white",
    font=dict(color='black', size=12),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Step 7: Customize X & Y Axes
fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", row=1, col=1)
fig.update_yaxes(title="Max Pressure", showgrid=True, gridcolor="lightgray", row=1, col=1)

fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", row=2, col=1)
fig.update_yaxes(title="Mean Pressure", showgrid=True, gridcolor="lightgray", row=2, col=1)

fig.update_xaxes(title="Mean Pressure", showgrid=True, gridcolor="lightgray", row=3, col=1)
fig.update_yaxes(title="Max Temperature", showgrid=True, gridcolor="lightgray", row=3, col=1)

# Step 8: Show Final Chart
fig.show()



# When 'pressure' is highest, check corresponding 'maxtemp'
high_pressure_data = df_train[df_train['pressure'] == df_train['pressure'].max()].groupby('day').agg({
    'pressure': ['max', 'mean'], 
    'maxtemp': 'max'
})

# Sorting in descending order by max and mean pressure
high_pressure_data = high_pressure_data.sort_values(by=[('pressure', 'max'), ('pressure', 'mean')], ascending=[False, False]).reset_index()

# Display results
print("Max temperature when pressure is highest:\n", high_pressure_data)



# Identify extreme pressure drops within a day
daily_extreme_pressure_drops = df_train.groupby('day').apply(lambda x: x['pressure'].diff().min()).reset_index(name="Extreme_Pressure_Drop")

# Step 15: Identify maximum temperature spike within a day
daily_max_temp_spike = df_train.groupby('day').apply(lambda x: x['maxtemp'].diff().max()).reset_index(name="Max_Temperature_Spike")

# Display results
print("Extreme pressure drops per day:\n", daily_extreme_pressure_drops.head(10))
print("====================================================================")
print("Max temperature spikes per day:\n", daily_max_temp_spike.head(10))



# Step 1: Identify extreme pressure drops per day
daily_extreme_pressure_drops = df_train.groupby('day').apply(lambda x: x['pressure'].diff().min()).reset_index(name="Extreme_Pressure_Drop")

# Step 2: Identify max temperature spikes per day
daily_max_temp_spike = df_train.groupby('day').apply(lambda x: x['maxtemp'].diff().max()).reset_index(name="Max_Temperature_Spike")

# Step 3: Create row-wise line subplots
fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=("Extreme Pressure Drop per Day", "Max Temperature Spike per Day"),
    vertical_spacing=0.15  # Space between subplots
)

# Step 4: Line Plot for Extreme Pressure Drop per Day
fig.add_trace(go.Scatter(
    x=daily_extreme_pressure_drops['day'], 
    y=daily_extreme_pressure_drops['Extreme_Pressure_Drop'], 
    mode='lines+markers', 
    name='Extreme Pressure Drop', 
    line=dict(color='red', width=1, dash='solid'),
    marker=dict(size=5, symbol='circle'),
    hovertemplate="Day: %{x}<br>Extreme Drop: %{y}"
), row=1, col=1)

# Step 5: Line Plot for Max Temperature Spike per Day
fig.add_trace(go.Scatter(
    x=daily_max_temp_spike['day'], 
    y=daily_max_temp_spike['Max_Temperature_Spike'], 
    mode='lines+markers', 
    name='Max Temp Spike', 
    line=dict(color='blue', width=1, dash='solid'),
    marker=dict(size=5, symbol='square'),
    hovertemplate="Day: %{x}<br>Max Temp Spike: %{y}"
), row=2, col=1)

# Step 6: Update Layout (White Background, Labels)
fig.update_layout(
    title="<b>Extreme Pressure Drops & Max Temperature Spikes per Day</b>",
    height=800, width=1600,  
    showlegend=True,
    template="plotly_white",
    font=dict(color='black', size=12),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Step 7: Customize X & Y Axes
fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", row=1, col=1)
fig.update_yaxes(title="Extreme Pressure Drop", showgrid=True, gridcolor="lightgray", row=1, col=1)

fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", row=2, col=1)
fig.update_yaxes(title="Max Temperature Spike", showgrid=True, gridcolor="lightgray", row=2, col=1)

# Step 8: Show Final Chart
fig.show()



# Display first 10 rows of selected columns
print("Displaying 'pressure', 'maxtemp', 'temparature', 'mintemp' Analysis:")
display(df_train[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp']].head(10))


# Days with Highest and Lowest Pressure
highest_pressure_day = df_train[df_train['pressure'] == df_train['pressure'].max()]
lowest_pressure_day = df_train[df_train['pressure'] == df_train['pressure'].min()]

print("\nDay with Highest Pressure:\n", highest_pressure_day[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp']])
print("=======================================================================================")
print("\nDay with Lowest Pressure:\n", lowest_pressure_day[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp']])
print("=======================================================================================")
# Days with Highest and Lowest Temperatures
hottest_day = df_train[df_train['maxtemp'] == df_train['maxtemp'].max()]
coldest_day = df_train[df_train['mintemp'] == df_train['mintemp'].min()]

print("\nHottest Day:\n", hottest_day[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp']])
print("=======================================================================================")
print("\nColdest Day:\n", coldest_day[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp']])



# Identify days with lowest temperatures
lowest_temp_days = df_train.groupby('day')['mintemp'].min().reset_index(name="Lowest_Temperature")
lowest_temp_days = lowest_temp_days.sort_values(by="Lowest_Temperature", ascending=True)

# Display results
print("Days with the Lowest Recorded Temperatures:\n", lowest_temp_days.head(10))
print("="*80)



# Step 1: Identify lowest recorded temperatures per day
lowest_temp_days = df_train.groupby('day')['mintemp'].min().reset_index(name="Lowest_Temperature")
lowest_temp_days = lowest_temp_days.sort_values(by="day", ascending=True)  # Ensure correct order

# Step 2: Find extreme values
coldest_day = lowest_temp_days.loc[lowest_temp_days['Lowest_Temperature'].idxmin()]
warmest_day = lowest_temp_days.loc[lowest_temp_days['Lowest_Temperature'].idxmax()]

# Step 3: Create Line Plot with Enhancements
fig = go.Figure()

# Line with Spline Interpolation (Smooth Curves)
fig.add_trace(go.Scatter(
    x=lowest_temp_days['day'], 
    y=lowest_temp_days['Lowest_Temperature'], 
    mode='lines+markers', 
    name='Lowest Temperature',
    line=dict(color='royalblue', width=3, shape='spline'),  # Smooth curve
    marker=dict(
        size=5,
        color='red',  # Fixed color instead of color scale
        symbol='circle'
    ),
    hovertemplate="Day: %{x}<br>Lowest Temp: %{y}"
))

# Step 4: Highlight Extreme Values
fig.add_trace(go.Scatter(
    x=[coldest_day['day']], 
    y=[coldest_day['Lowest_Temperature']], 
    mode='markers+text',
    name="Coldest Day",
    marker=dict(color='cyan', size=10, symbol='diamond'),
    text=["â�„ Coldest Day"],
    textposition="top center"
))

fig.add_trace(go.Scatter(
    x=[warmest_day['day']], 
    y=[warmest_day['Lowest_Temperature']], 
    mode='markers+text',
    name="Warmest Day",
    marker=dict(color='darkred', size=10, symbol='triangle-up'),
    text=["ğŸ”¥ Warmest Day"],
    textposition="top center"
))

# Step 5: Customize Layout
fig.update_layout(
    title="<b>Daily Lowest Recorded Temperatures</b> ğŸŒ¡",
    height=700, width=1500,  
    showlegend=True,
    template="plotly_white",
    font=dict(color='black', size=13),
    plot_bgcolor="rgba(240, 248, 255, 0.5)",  # Light transparent background
    paper_bgcolor="white",
    hovermode="x unified"  # Unified hover display
)

# Step 6: Customize X & Y Axes
fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", tickangle=45)
fig.update_yaxes(title="Lowest Temperature", showgrid=True, gridcolor="lightgray")

# Step 7: Show Final Chart
fig.show()



# Verify daily temperature consistency
temperature_consistency = df_train[df_train['maxtemp'] < df_train['mintemp']]

# Display results in two separate rows
print("Days with inconsistent temperature readings:\n", temperature_consistency.iloc[0], "\n")
print("="*80)



# Compute day-to-day change in pressure and temperature
df_train['pressure_change'] = df_train['pressure'].diff()
df_train['maxtemp_change'] = df_train['maxtemp'].diff()
df_train['mintemp_change'] = df_train['mintemp'].diff()

# Finding days with the largest changes
largest_pressure_shifts = df_train[['day', 'pressure_change']].abs().nlargest(10, 'pressure_change')
largest_temp_shifts = df_train[['day', 'maxtemp_change']].abs().nlargest(10, 'maxtemp_change')

# Display results
print("Days with Largest Pressure Changes:\n", largest_pressure_shifts)
print("="*80)
print("Days with Largest Temperature Changes:\n", largest_temp_shifts)
print("="*80)



# Compute daily temperature range
df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']

# Identify largest differences
largest_temp_ranges = df_train[['day', 'temp_range']].nlargest(10, 'temp_range')

# Display results
print("Days with Largest Temperature Ranges:\n", largest_temp_ranges)
print("="*80)



# Step 1: Compute daily temperature range
df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']

# Step 2: Compute daily average temperature range
daily_temp_range = df_train.groupby('day')['temp_range'].mean().reset_index()

# Step 3: Identify extreme values
max_temp_range_day = daily_temp_range.loc[daily_temp_range['temp_range'].idxmax()]
min_temp_range_day = daily_temp_range.loc[daily_temp_range['temp_range'].idxmin()]

# Step 4: Create Line Plot
fig = go.Figure()

# Smooth Line Plot with Markers
fig.add_trace(go.Scatter(
    x=daily_temp_range['day'], 
    y=daily_temp_range['temp_range'], 
    mode='lines+markers', 
    name='Daily Temperature Range',
    line=dict(color='orange', width=3, shape='spline'),  
    marker=dict(size=6, symbol='circle', color='darkblue'),
    hovertemplate="Day: %{x}<br>Temp Range: %{y}"
))

# Step 5: Highlight Maximum & Minimum Temp Range Days
fig.add_trace(go.Scatter(
    x=[max_temp_range_day['day']], 
    y=[max_temp_range_day['temp_range']], 
    mode='markers+text',
    name="Max Temp Range Day",
    marker=dict(color='red', size=10, symbol='star'),
    text=["ğŸ”¥ Max Temp Range"],
    textposition="top center"
))

fig.add_trace(go.Scatter(
    x=[min_temp_range_day['day']], 
    y=[min_temp_range_day['temp_range']], 
    mode='markers+text',
    name="Min Temp Range Day",
    marker=dict(color='blue', size=10, symbol='diamond'),
    text=["â�„ Min Temp Range"],
    textposition="top center"
))

# Step 6: Customize Layout
fig.update_layout(
    title="<b>Daily Temperature Range (Max - Min)</b> ğŸŒ¡",
    height=700, width=1500,  
    showlegend=True,
    template="plotly_white",
    font=dict(color='black', size=13),
    plot_bgcolor="rgba(250, 250, 210, 0.3)",  # Light warm background
    paper_bgcolor="white",
    hovermode="x unified"  # Unified hover effect
)

# Step 7: Customize X & Y Axes
fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", tickangle=45)
fig.update_yaxes(title="Temperature Range", showgrid=True, gridcolor="lightgray")

# Step 8: Show Final Chart
fig.show()



# Compute day-wise temperature and pressure variation
daily_variation = df_train.groupby('day').agg({
    'maxtemp': 'max',
    'mintemp': 'min',
    'pressure': ['max', 'min']
}).reset_index()

# Calculate differences
daily_variation['Temp_Variation'] = daily_variation[('maxtemp', 'max')] - daily_variation[('mintemp', 'min')]
daily_variation['Pressure_Variation'] = daily_variation[('pressure', 'max')] - daily_variation[('pressure', 'min')]

# Sorting by the highest variations
daily_variation = daily_variation.sort_values(by=['Temp_Variation', 'Pressure_Variation'], ascending=[False, False])

# Display results
display("Days with the Most Fluctuating Weather (Temperature & Pressure)", daily_variation.head(10))
print("="*80)



# Compute day-wise temperature and pressure variation
daily_variation = df_train.groupby('day').agg({
    'maxtemp': 'max',
    'mintemp': 'min',
    'pressure': ['max', 'min']
}).reset_index()

# Calculate differences
daily_variation['Temp_Variation'] = daily_variation[('maxtemp', 'max')] - daily_variation[('mintemp', 'min')]
daily_variation['Pressure_Variation'] = daily_variation[('pressure', 'max')] - daily_variation[('pressure', 'min')]

# Sorting by the highest variations
daily_variation = daily_variation.sort_values(by=['Temp_Variation', 'Pressure_Variation'], ascending=[False, False])

# Step 2: Create Subplots for Scatter Plots
fig = sp.make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Temperature Variation per Day", "Pressure Variation per Day"),
    horizontal_spacing=0.15
)

# Scatterplot for Temperature Variation
fig.add_trace(go.Scatter(
    x=daily_variation['day'], 
    y=daily_variation['Temp_Variation'], 
    mode='markers',
    name='Temperature Variation',
    marker=dict(size=10, color='red', symbol='circle', opacity=0.8, line=dict(width=1, color='black')),
    hovertemplate="Day: %{x}<br>Temp Variation: %{y}"
), row=1, col=1)

# Scatterplot for Pressure Variation
fig.add_trace(go.Scatter(
    x=daily_variation['day'], 
    y=daily_variation['Pressure_Variation'], 
    mode='markers',
    name='Pressure Variation',
    marker=dict(size=10, color='blue', opacity=0.8, line=dict(width=1, color='black')),
    hovertemplate="Day: %{x}<br>Pressure Variation: %{y}"
), row=1, col=2)

# Customize Layout
fig.update_layout(
    title="<b>Daily Temperature & Pressure Variations</b>",
    height=600, width=1400,  
    showlegend=False,  
    template="plotly_white",
    font=dict(color='black', size=13),
    plot_bgcolor="rgba(240, 248, 255, 0.5)",  # Light pastel blue background
    paper_bgcolor="white"
)

# Customize X & Y Axes
fig.update_xaxes(title="Day", showgrid=True, gridcolor="lightgray", tickangle=45)
fig.update_yaxes(title="Temperature Variation", showgrid=True, gridcolor="lightgray", row=1, col=1)
fig.update_yaxes(title="Pressure Variation", showgrid=True, gridcolor="lightgray", row=1, col=2)

# Show Final Chart
fig.show()



# Step 1: Compute day-wise temperature and pressure variation
daily_variation = df_train.groupby('day').agg({
    'maxtemp': 'max',
    'mintemp': 'min',
    'pressure': ['max', 'min']
}).reset_index()

# Calculate differences
daily_variation['Temp_Variation'] = daily_variation[('maxtemp', 'max')] - daily_variation[('mintemp', 'min')]
daily_variation['Pressure_Variation'] = daily_variation[('pressure', 'max')] - daily_variation[('pressure', 'min')]

# Sorting by the highest variations
daily_variation = daily_variation.sort_values(by=['Temp_Variation', 'Pressure_Variation'], ascending=[False, False])

# Step 2: Create Row-wise Subplots
fig = sp.make_subplots(
    rows=2, cols=1,  # Create 2 rows, 1 column layout for the subplots
    subplot_titles=("Temperature Variation per Day", "Pressure Variation per Day"),
    vertical_spacing=0.15
)

# Step 3: Scatterplot for Temperature Variation
fig.add_trace(go.Scatter(
    x=daily_variation['day'], 
    y=daily_variation['Temp_Variation'], 
    mode='markers',
    name='Temperature Variation',
    marker=dict(
        size=12, 
        color=daily_variation['Temp_Variation'],  # Color by variation value
        colorscale='RdYlBu',  # Gradient color scale
        symbol='circle', 
        opacity=0.9, 
        line=dict(width=1, color='black')
    ),
    hovertemplate="Day: %{x}<br>Temp Variation: %{y}",
), row=1, col=1)

# Step 4: Scatterplot for Pressure Variation
fig.add_trace(go.Scatter(
    x=daily_variation['day'], 
    y=daily_variation['Pressure_Variation'], 
    mode='markers',
    name='Pressure Variation',
    marker=dict(
        size=12, 
        color=daily_variation['Pressure_Variation'],  # Color by variation value
        colorscale='Viridis',  # Gradient color scale
        opacity=0.9, 
        line=dict(width=1, color='black')
    ),
    hovertemplate="Day: %{x}<br>Pressure Variation: %{y}",
), row=2, col=1)

# Step 5: Customize Layout with White Background
fig.update_layout(
    title="<b>Daily Temperature & Pressure Variations</b>",
    height=800,  # Increased height to give more space for each plot
    width=1000,  
    showlegend=False,  
    template="plotly",  # Default template for white background
    font=dict(color='black', size=14),  # Black font color for better contrast
    plot_bgcolor="white",  # White background for the plot area
    paper_bgcolor="white",  # White paper background for the overall chart
)

# Step 6: Customize X & Y Axes for both subplots with the same grid
fig.update_xaxes(
    title="Day", 
    showgrid=True, 
    gridcolor="gray", 
    tickangle=45, 
    row=1, col=1
)
fig.update_yaxes(
    title="Temperature Variation", 
    showgrid=True, 
    gridcolor="gray", 
    row=1, col=1
)

# Same grid settings for Pressure Variation subplot (row 2)
fig.update_xaxes(
    title="Day", 
    showgrid=True, 
    gridcolor="gray", 
    tickangle=45, 
    row=2, col=1
)
fig.update_yaxes(
    title="Pressure Variation", 
    showgrid=True, 
    gridcolor="gray", 
    row=2, col=1
)

# Step 7: Show Final Chart
fig.show()



# Compute days with the least variation
steady_weather_days = daily_variation.sort_values(by=['Temp_Variation', 'Pressure_Variation'], ascending=[True, True])

# Display results
display("Days with the Steadiest Weather (Minimal Variation in Temperature & Pressure):", steady_weather_days.head(10))
print("="*80)



# Compute days with the least variation
steady_weather_days = daily_variation.sort_values(by=['Temp_Variation', 'Pressure_Variation'], ascending=[True, True])

# Display results
display("Days with the Steadiest Weather (Minimal Variation in Temperature & Pressure):", steady_weather_days.head(10))
print("="*80)



# Compute days with the least variation
steady_weather_days = daily_variation.sort_values(by=['Temp_Variation', 'Pressure_Variation'], ascending=[True, True])

# Create Row-wise Subplots for Steady Weather Days
fig = sp.make_subplots(
    rows=2, cols=1,  # Create 2 rows, 1 column layout for the subplots
    subplot_titles=("Temperature Variation per Day (Steady Weather)", "Pressure Variation per Day (Steady Weather)"),
    vertical_spacing=0.15
)

# Scatterplot for Temperature Variation (Steady Weather)
fig.add_trace(go.Scatter(
    x=steady_weather_days['day'], 
    y=steady_weather_days['Temp_Variation'], 
    mode='markers',
    name='Temperature Variation',
    marker=dict(
        size=10, 
        color=steady_weather_days['Temp_Variation'],  # Color by variation value
        colorscale='Jet',  # More prominent color scale
        symbol='circle', 
        opacity=0.9, 
        line=dict(width=2, color='black')
    ),
    hovertemplate="Day: %{x}<br>Temp Variation: %{y}",
), row=1, col=1)

# Scatterplot for Pressure Variation (Steady Weather)
fig.add_trace(go.Scatter(
    x=steady_weather_days['day'], 
    y=steady_weather_days['Pressure_Variation'], 
    mode='markers',
    name='Pressure Variation',
    marker=dict(
        size=10, 
        color=steady_weather_days['Pressure_Variation'],  # Color by variation value
        colorscale='Inferno',  # More prominent and different color scale for second subplot
        opacity=0.9, 
        line=dict(width=2, color='black')
    ),
    hovertemplate="Day: %{x}<br>Pressure Variation: %{y}",
), row=2, col=1)

# Customize Layout with White Background and Unique Styling
fig.update_layout(
    title="<b>Steady Weather Days with Minimal Temperature & Pressure Variation</b>",
    height=800,  # Increased height to give more space for each plot
    width=1000,  
    showlegend=False,  
    template="plotly",  # Default template for white background
    font=dict(color='black', size=14),  # Black font color for better contrast
    plot_bgcolor="white",  # White background for the plot area
    paper_bgcolor="white",  # White paper background for the overall chart
)

# Customize X & Y Axes for both subplots with consistent grid
fig.update_xaxes(
    title="Day", 
    showgrid=True, 
    gridcolor="lightgray", 
    tickangle=45, 
    row=1, col=1
)
fig.update_yaxes(
    title="Temperature Variation", 
    showgrid=True, 
    gridcolor="lightgray", 
    row=1, col=1
)

# Same grid settings for Pressure Variation subplot (row 2)
fig.update_xaxes(
    title="Day", 
    showgrid=True, 
    gridcolor="lightgray", 
    tickangle=45, 
    row=2, col=1
)
fig.update_yaxes(
    title="Pressure Variation", 
    showgrid=True, 
    gridcolor="lightgray", 
    row=2, col=1
)

# Show Final Chart
fig.show()



# Compute overall average temperature
avg_temperature = df_train['temparature'].mean()

print(f"Overall Average Temperature: {avg_temperature:.2f}")
# Compute day-wise average temperature
daily_avg_temp = df_train.groupby('day')['temparature'].mean().reset_index(name='Avg_Temperature')

# Sorting by highest and lowest temperatures
hottest_days = daily_avg_temp.sort_values(by='Avg_Temperature', ascending=False)
coldest_days = daily_avg_temp.sort_values(by='Avg_Temperature', ascending=True)

# Display results
print("Hottest Days on Average:\n", hottest_days.head(10))
print("=" * 80)

print("Coldest Days on Average:\n", coldest_days.head(10))
print("=" * 80)



# Compute overall average temperature
avg_temperature = df_train['temparature'].mean()

# Compute day-wise average temperature
daily_avg_temp = df_train.groupby('day')['temparature'].mean().reset_index(name='Avg_Temperature')

# Sorting by highest and lowest temperatures
hottest_days = daily_avg_temp.sort_values(by='Avg_Temperature', ascending=False)
coldest_days = daily_avg_temp.sort_values(by='Avg_Temperature', ascending=True)
# Create a Plotly figure
fig = go.Figure()

# Line plot for average temperature
fig.add_trace(go.Scatter(
    x=daily_avg_temp['day'], 
    y=daily_avg_temp['Avg_Temperature'], 
    mode='lines',
    name='Daily Average Temperature',
    line=dict(color='blue', width=2, dash='solid'),
    hovertemplate="Day: %{x}<br>Average Temp: %{y:.2f}",
))

# Highlight the hottest days with markers
fig.add_trace(go.Scatter(
    x=hottest_days['day'].head(10), 
    y=hottest_days['Avg_Temperature'].head(10),
    mode='markers',
    name='Hottest Days',
    marker=dict(color='red', size=8, symbol='star', line=dict(width=2, color='black')),
    hovertemplate="Hottest Day: %{x}<br>Temperature: %{y:.2f}",
))

# Highlight the coldest days with markers
fig.add_trace(go.Scatter(
    x=coldest_days['day'].head(10),
    y=coldest_days['Avg_Temperature'].head(10),
    mode='markers',
    name='Coldest Days',
    marker=dict(color='blue', size=8, symbol='circle', line=dict(width=2, color='black')),
    hovertemplate="Coldest Day: %{x}<br>Temperature: %{y:.2f}",
))

# Add a horizontal line for overall average temperature
fig.add_trace(go.Scatter(
    x=[daily_avg_temp['day'].min(), daily_avg_temp['day'].max()],
    y=[avg_temperature, avg_temperature],
    mode='lines',
    name='Overall Average Temperature',
    line=dict(color='green', width=3, dash='dash'),
    hovertemplate=f"Overall Avg Temp: {avg_temperature:.2f}",
))

# Customize Layout with Title, Axes, and Background
fig.update_layout(
    title="<b>Comparative Plot of Hottest, Coldest, and Overall Average Temperatures</b>",
    height=600,  
    width=1300,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",  
    xaxis=dict(title="Day", showgrid=True, gridcolor="lightgray"),
    yaxis=dict(title="Temperature (Â°C)", showgrid=True, gridcolor="lightgray"),
)

# Show the final plot
fig.show()



# Highest and Lowest Dewpoint Days
highest_dewpoint_days = df_train.groupby('day')['dewpoint'].max().reset_index().sort_values(by='dewpoint', ascending=False)
lowest_dewpoint_days = df_train.groupby('day')['dewpoint'].min().reset_index().sort_values(by='dewpoint', ascending=True)

# Display results
print("Days with Highest Dewpoint:\n", highest_dewpoint_days.head(10))
print("\n====================================================================\n")
print("Days with Lowest Dewpoint:\n", lowest_dewpoint_days.head(10))



# Compute highest and lowest dewpoint days
highest_dewpoint_days = df_train.groupby('day')['dewpoint'].max().reset_index().sort_values(by='dewpoint', ascending=False)
lowest_dewpoint_days = df_train.groupby('day')['dewpoint'].min().reset_index().sort_values(by='dewpoint', ascending=True)

# Create a Plotly figure
fig = go.Figure()

# Line plot for dewpoints
fig.add_trace(go.Scatter(
    x=df_train.groupby('day')['dewpoint'].mean().index, 
    y=df_train.groupby('day')['dewpoint'].mean().values, 
    mode='lines',
    name='Daily Average Dewpoint',
    line=dict(color='blue', width=2, dash='solid'),
    hovertemplate="Day: %{x}<br>Average Dewpoint: %{y:.2f}",
))

# Highlight the days with the highest dewpoint using markers
fig.add_trace(go.Scatter(
    x=highest_dewpoint_days['day'], 
    y=highest_dewpoint_days['dewpoint'],
    mode='markers',
    name='Highest Dewpoint Days',
    marker=dict(color='red', size=7, line=dict(width=2, color='black')),
    hovertemplate="Highest Dewpoint Day: %{x}<br>Dewpoint: %{y:.2f}",
))

# Highlight the days with the lowest dewpoint using markers
fig.add_trace(go.Scatter(
    x=lowest_dewpoint_days['day'],
    y=lowest_dewpoint_days['dewpoint'],
    mode='markers',
    name='Lowest Dewpoint Days',
    marker=dict(color='green', size=7, line=dict(width=2, color='black')),
    hovertemplate="Lowest Dewpoint Day: %{x}<br>Dewpoint: %{y:.2f}",
))

# Customize Layout with Title, Axes, and Background
fig.update_layout(
    title="<b>Comparative Plot of Highest and Lowest Dewpoint Days</b>",
    height=700,  
    width=1300,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",  
    xaxis=dict(title="Day", showgrid=True, gridcolor="lightgray"),
    yaxis=dict(title="Dewpoint", showgrid=True, gridcolor="lightgray"),
)

# Show the final plot
fig.show()



# Analysis of Pressure & Dewpoint when Max Temp > 30Â°C
hot_days_analysis = df_train[df_train['maxtemp'] > 30].groupby('day').agg({
    'pressure': ['max', 'mean'],
    'dewpoint': ['max', 'mean'],
    'maxtemp': 'max'
}).reset_index()

# Sorting by max pressure
hot_days_analysis = hot_days_analysis.sort_values(by=[('pressure', 'max')], ascending=False)

# Display results
print("Pressure & Dewpoint on Days with Max Temp > 30Â°C:\n")
display(hot_days_analysis.head(10))


# Temperature Analysis on Highest Pressure Days
high_pressure_temp = df_train[df_train['pressure'] == df_train['pressure'].max()].groupby('day').agg({
    'pressure': ['max', 'mean'],
    'maxtemp': 'max',
    'dewpoint': 'max'
}).reset_index()

# Sorting by max pressure
high_pressure_temp = high_pressure_temp.sort_values(by=[('pressure', 'max')], ascending=False)

# Display results
print("Temperature & Dewpoint on Days with Highest Pressure:\n")
display( high_pressure_temp)



# Identify extreme weather conditions
extreme_weather_days = df_train.groupby('day').agg({
    'maxtemp': 'max',
    'mintemp': 'min',
    'pressure': 'mean',
    'dewpoint': 'mean'
}).reset_index()

# Categorizing days based on extreme conditions
extreme_weather_days['Weather_Condition'] = 'Normal'

extreme_weather_days.loc[
    (extreme_weather_days['dewpoint'] > 20) & (extreme_weather_days['maxtemp'] > 30),
    'Weather_Condition'
] = 'Very Humid & Hot'

extreme_weather_days.loc[
    (extreme_weather_days['dewpoint'] < 5) & (extreme_weather_days['maxtemp'] > 30),
    'Weather_Condition'
] = 'Dry Heat'

extreme_weather_days.loc[
    (extreme_weather_days['pressure'] > 1020) & (extreme_weather_days['mintemp'] < 5),
    'Weather_Condition'
] = 'Stable Cold Weather'

extreme_weather_days.loc[
    (extreme_weather_days['pressure'] < 1000) & ((extreme_weather_days['maxtemp'] > 35) | (extreme_weather_days['mintemp'] < 0)),
    'Weather_Condition'
] = 'Stormy/Unstable'

# Display days with extreme conditions
print("Extreme Weather Conditions per Day:\n")
display(extreme_weather_days.sort_values(by='pressure', ascending=False).head(10))



# Step 1: Identify extreme weather conditions (already given)
extreme_weather_days = df_train.groupby('day').agg({
    'maxtemp': 'max',
    'mintemp': 'min',
    'pressure': 'mean',
    'dewpoint': 'mean'
}).reset_index()

# Categorizing days based on extreme conditions
extreme_weather_days['Weather_Condition'] = 'Normal'

extreme_weather_days.loc[
    (extreme_weather_days['dewpoint'] > 20) & (extreme_weather_days['maxtemp'] > 30),
    'Weather_Condition'
] = 'Very Humid & Hot'

extreme_weather_days.loc[
    (extreme_weather_days['dewpoint'] < 5) & (extreme_weather_days['maxtemp'] > 30),
    'Weather_Condition'
] = 'Dry Heat'

extreme_weather_days.loc[
    (extreme_weather_days['pressure'] > 1020) & (extreme_weather_days['mintemp'] < 5),
    'Weather_Condition'
] = 'Stable Cold Weather'

extreme_weather_days.loc[
    (extreme_weather_days['pressure'] < 1000) & ((extreme_weather_days['maxtemp'] > 35) | (extreme_weather_days['mintemp'] < 0)),
    'Weather_Condition'
] = 'Stormy/Unstable'

# Step 2: Create subplots (3 plots)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Weather Condition Distribution", 
                    "Max and Min Temperature over Days", 
                    "Temperature vs Dewpoint"),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "scatter"}, None]]
)

# Step 3: Bar plot for Weather Condition Distribution
weather_condition_count = extreme_weather_days['Weather_Condition'].value_counts().reset_index()
weather_condition_count.columns = ['Weather_Condition', 'Count']
fig.add_trace(go.Bar(
    x=weather_condition_count['Weather_Condition'],
    y=weather_condition_count['Count'],
    name="Weather Conditions",
    marker=dict(color='rgba(100, 150, 255, 0.7)'),
    text=weather_condition_count['Count'],
    textposition='auto',
    hovertemplate="Weather Condition: %{x}<br>Count: %{y}"  # Hover data for bar chart
), row=1, col=1)

# Step 4: Line plot for Max and Min Temperature
fig.add_trace(go.Scatter(
    x=extreme_weather_days['day'],
    y=extreme_weather_days['maxtemp'],
    mode='lines',
    name='Max Temperature',
    line=dict(color='red', width=2),
    hovertemplate="Day: %{x}<br>Max Temp: %{y}Â°C"  # Hover data for Max Temperature
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=extreme_weather_days['day'],
    y=extreme_weather_days['mintemp'],
    mode='lines',
    name='Min Temperature',
    line=dict(color='blue', width=2),
    hovertemplate="Day: %{x}<br>Min Temp: %{y}Â°C"  # Hover data for Min Temperature
), row=1, col=2)

# Step 5: Temperature vs Dewpoint Scatter Plot with Color Coding
fig.add_trace(go.Scatter(
    x=extreme_weather_days['maxtemp'],
    y=extreme_weather_days['dewpoint'],
    mode='markers',
    name='Temperature vs Dewpoint',
    marker=dict(
        color=extreme_weather_days['Weather_Condition'].map({
            'Very Humid & Hot': 'rgba(255, 0, 0, 0.7)',
            'Dry Heat': 'rgba(255, 165, 0, 0.7)',
            'Stable Cold Weather': 'rgba(0, 0, 255, 0.7)',
            'Stormy/Unstable': 'rgba(128, 0, 128, 0.7)',
            'Normal': 'rgba(0, 255, 0, 0.7)'
        }),
        size=10
    ),
    hovertemplate="Day: %{x}<br>Dewpoint: %{y}<br>Weather: %{text}",
    text=extreme_weather_days['Weather_Condition']
), row=2, col=1)

# Step 6: Customize Layout with Titles, Labels, and Background
fig.update_layout(
    title="<b>Extreme Weather Conditions Visualizations</b>",
    height=1000,  
    width=1500,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",
    title_x=0.5
)

# Step 7: Add X and Y Axis Labels
fig.update_xaxes(title_text="Weather Condition", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)

fig.update_xaxes(title_text="Day", row=1, col=2)
fig.update_yaxes(title_text="Temperature", row=1, col=2)

fig.update_xaxes(title_text="Max Temperature", row=2, col=1)
fig.update_yaxes(title_text="Dewpoint", row=2, col=1)

# Step 8: Show the final plot
fig.show()



# Identify Days with Largest Temperature Swings
df_train['temp_difference'] = df_train['maxtemp'] - df_train['mintemp']

temp_swing_days = df_train.groupby('day').agg({
    'temp_difference': 'max',
    'pressure': 'mean',
    'dewpoint': 'mean'
}).reset_index()

temp_swing_days = temp_swing_days.sort_values(by='temp_difference', ascending=False)

print("Top Days with Largest Temperature Swings:\n", temp_swing_days.head(10))



# Checking Pressure Impact on Temperature Extremes
pressure_temp_impact = df_train.groupby('pressure').agg({
    'maxtemp': 'max',
    'mintemp': 'min',
    'dewpoint': 'mean'
}).reset_index()

# Sorting by multiple columns with different orders
pressure_temp_impact = pressure_temp_impact.sort_values(
    by=['maxtemp', 'pressure', 'dewpoint', 'mintemp'],
    ascending=[False, False, False, True]
)

print("Pressure Influence on Temperature Extremes:\n", pressure_temp_impact.head(10))




# Group by day & analyze combined factors
combined_weather_analysis = df_train.groupby('day').agg({
    'pressure': 'mean',
    'maxtemp': 'max',
    'temparature': 'mean',
    'mintemp': 'min',
    'dewpoint': 'mean'
}).reset_index()

# Sort by highest temparature to see how pressure and dewpoint behave
combined_weather_analysis = combined_weather_analysis.sort_values(by='maxtemp', ascending=False)

# Display results
print("Weather Analysis by Max Temp:\n", combined_weather_analysis.head(10))



# Group by day & analyze combined factors
combined_weather_analysis = df_train.groupby('day').agg({
    'pressure': 'mean',
    'maxtemp': 'max',
    'temparature': 'mean',  # Fixed typo from 'temparature' to 'temperature'
    'mintemp': 'min',
    'dewpoint': 'mean'
}).reset_index()

# Sorting by highest max temperature
combined_weather_analysis = combined_weather_analysis.sort_values(
    by=['maxtemp', 'pressure', 'dewpoint', 'mintemp'],
    ascending=[False, False, False, True]
)

# Create subplots (4 plots)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Max Temperature vs Pressure", 
                    "Max Temperature vs Dewpoint", 
                    "Min Temperature vs Dewpoint",
                    "Mean Temperature vs Pressure"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Max Temperature vs Pressure
fig.add_trace(go.Scatter(
    x=combined_weather_analysis['maxtemp'],
    y=combined_weather_analysis['pressure'],
    mode='markers',
    name='Max Temp vs Pressure',
    marker=dict(color='rgba(255, 0, 0, 0.6)', size=10),
    hovertemplate="Max Temp: %{x}<br>Pressure: %{y}"
), row=1, col=1)

# Max Temperature vs Dewpoint
fig.add_trace(go.Scatter(
    x=combined_weather_analysis['maxtemp'],
    y=combined_weather_analysis['dewpoint'],
    mode='markers',
    name='Max Temp vs Dewpoint',
    marker=dict(color='rgba(0, 0, 255, 0.6)', size=10),
    hovertemplate="Max Temp: %{x}<br>Dewpoint: %{y}"
), row=1, col=2)

# Min Temperature vs Dewpoint
fig.add_trace(go.Scatter(
    x=combined_weather_analysis['mintemp'],
    y=combined_weather_analysis['dewpoint'],
    mode='markers',
    name='Min Temp vs Dewpoint',
    marker=dict(color='rgba(0, 255, 0, 0.6)', size=10),
    hovertemplate="Min Temp: %{x}<br>Dewpoint: %{y}"
), row=2, col=1)

# Mean Temperature vs Pressure
fig.add_trace(go.Scatter(
    x=combined_weather_analysis['temparature'],
    y=combined_weather_analysis['pressure'],
    mode='markers',
    name='Mean Temp vs Pressure',
    marker=dict(color='rgba(255, 165, 0, 0.6)', size=10),
    hovertemplate="Mean Temp: %{x}<br>Pressure: %{y}"
), row=2, col=2)

# Customize Layout with Title, Axes, and Background
fig.update_layout(
    title="<b>Weather Factors Correlation Analysis</b>",
    height=800,  
    width=1200,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",
    title_x=0.5
)

# Update axis labels
fig.update_xaxes(title_text="Max Temperature", row=1, col=1)
fig.update_yaxes(title_text="Pressure", row=1, col=1)

fig.update_xaxes(title_text="Max Temperature", row=1, col=2)
fig.update_yaxes(title_text="Dewpoint", row=1, col=2)

fig.update_xaxes(title_text="Min Temperature", row=2, col=1)
fig.update_yaxes(title_text="Dewpoint", row=2, col=1)

fig.update_xaxes(title_text="Mean Temperature", row=2, col=2)
fig.update_yaxes(title_text="Pressure", row=2, col=2)

# Step 11: Show the final plot
fig.show()



# Analysis of humidity across different days (only min, max, and mean)
humidity_daywise = df_train.groupby('day')['humidity'].agg(['min', 'mean', 'max']).reset_index()

# Get the top 10 days with the highest humidity
highest_humidity = humidity_daywise.sort_values(by='max', ascending=False).head(10)

# Get the top 10 days with the lowest humidity
lowest_humidity = humidity_daywise.sort_values(by='min', ascending=True).head(10)

# Display the result
print("Top Days with Highest Humidity:")
display(highest_humidity)
print("=================================================================================")
print("Top Days with Lowest Humidity:")
display(lowest_humidity)



# Group by day and get humidity analysis (min, mean, and max)
humidity_daywise = df_train.groupby('day')['humidity'].agg(['min', 'mean', 'max']).reset_index()

# Create subplots (2 rows, 1 column) with increased gap between the plots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Humidity Min, Mean, and Max across Days", 
                    "Humidity Max vs Mean across Days"),
    specs=[[{"type": "scatter"}],
           [{"type": "scatter"}]],
    vertical_spacing=0.1  # Increase the gap between the two subplots
)

# Humidity Min, Mean, and Max across Days
fig.add_trace(go.Scatter(
    x=humidity_daywise['day'],
    y=humidity_daywise['min'],
    mode='lines+markers',
    name='Min Humidity',
    marker=dict(color='rgba(0, 255, 0, 0.8)', size=4),
    line=dict(width=1, color='rgba(0, 255, 0, 0.8)'),
    hovertemplate="Day: %{x}<br>Min Humidity: %{y} "
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=humidity_daywise['day'],
    y=humidity_daywise['mean'],
    mode='lines+markers',
    name='Mean Humidity',
    marker=dict(color='rgba(255, 165, 0, 0.8)', size=4),
    line=dict(width=1, color='rgba(255, 165, 0, 0.8)'),
    hovertemplate="Day: %{x}<br>Mean Humidity: %{y} "
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=humidity_daywise['day'],
    y=humidity_daywise['max'],
    mode='lines+markers',
    name='Max Humidity',
    marker=dict(color='rgba(255, 0, 0, 0.8)', size=4),
    line=dict(width=1, color='rgba(255, 0, 0, 0.8)'),
    hovertemplate="Day: %{x}<br>Max Humidity: %{y} "
), row=1, col=1)

# Humidity Max vs Mean across Days
fig.add_trace(go.Scatter(
    x=humidity_daywise['max'],
    y=humidity_daywise['mean'],
    mode='markers',
    name='Max vs Mean Humidity',
    marker=dict(color='rgba(138, 43, 226, 0.8)', size=8),
    hovertemplate="Max Humidity: %{x} <br>Mean Humidity: %{y} "
), row=2, col=1)

# Customize Layout with Title, Axes, and Background
fig.update_layout(
    title="<b>Humidity Analysis Across Different Days</b>",
    height=800,  
    width=1400,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",
    title_x=0.5,
)

# Update x-axis and y-axis labels for both subplots
fig.update_xaxes(title_text="Days", row=1, col=1)
fig.update_yaxes(title_text="Humidity", row=1, col=1)

fig.update_xaxes(title_text="Max Humidity", row=2, col=1)
fig.update_yaxes(title_text="Mean Humidity", row=2, col=1)

# Step 11: Show the final plot
fig.show()



# Relationship between pressure and humidity by day
pressure_vs_humidity = df_train[['pressure', 'humidity']].groupby(df_train['day']).mean().reset_index()

# Get the 10 days with the highest mean pressure
highest_pressure_days = pressure_vs_humidity.sort_values(by='pressure', ascending=False).head(10)

# Get the 10 days with the highest mean humidity
highest_humidity_days = pressure_vs_humidity.sort_values(by='humidity', ascending=False).head(10)

# Comparative analysis of the 10 highest pressure days and highest humidity days
print("Top Days with Highest Mean Pressure:")
display(highest_pressure_days)
print("=================================================================================")
print("Top Days with Highest Mean Humidity:")
display(highest_humidity_days)



# Grouping the data and calculating mean for humidity, max for maxtemp, and min for mintemp
humidity_temp_extremes = df_train.groupby('day').agg({
    'humidity': 'mean',  # Mean value for humidity
    'maxtemp': 'max',    # Maximum value for maxtemp
    'mintemp': 'min'     # Minimum value for mintemp
}).reset_index()

# Sorting in ascending order by 'maxtemp' (maximum temperature)
humidity_temp_extremes_asc = humidity_temp_extremes.sort_values(by='maxtemp', ascending=True)

# Sorting in descending order by 'mintemp' (minimum temperature)
humidity_temp_extremes_desc = humidity_temp_extremes.sort_values(by='mintemp', ascending=False)

# Display the result
print("Humidity Distribution with Max and Min Temperature by Day (Sorted by Max Temp):")
display(humidity_temp_extremes_asc.head(10))
print("=================================================================================")
print("Humidity Distribution with Max and Min Temperature by Day (Sorted by Min Temp):")
display(humidity_temp_extremes_desc.head(10))



# Grouping the data and calculating mean for humidity, max for maxtemp, and min for mintemp
humidity_temp_extremes = df_train.groupby('day').agg({
    'humidity': 'mean',  # Mean value for humidity
    'maxtemp': 'max',    # Maximum value for maxtemp
    'mintemp': 'min'     # Minimum value for mintemp
}).reset_index()

# Sorting in ascending order by 'maxtemp' (maximum temperature)
humidity_temp_extremes_asc = humidity_temp_extremes.sort_values(by='maxtemp', ascending=True)

# Sorting in descending order by 'mintemp' (minimum temperature)
humidity_temp_extremes_desc = humidity_temp_extremes.sort_values(by='mintemp', ascending=False)

# Create a subplot (2x2 grid) for different comparisons
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Humidity vs Max Temp", 
                    "Humidity vs Min Temp", 
                    "Temperature Trends vs Humidity"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter", "colspan": 2}, None]]  # Merge second row to span across both columns
)

# Humidity vs Max Temp (Sorted by Max Temp) as Scatter Plot
fig.add_trace(go.Scatter(
    x=humidity_temp_extremes_asc['day'],
    y=humidity_temp_extremes_asc['humidity'],
    mode='markers',
    name='Humidity vs Max Temp',
    marker=dict(color='rgba(255, 99, 71, 0.8)', size=8),
    hovertemplate="Day: %{x}<br>Humidity: %{y}<br>Max Temp: %{x}"
), row=1, col=1)

# Humidity vs Min Temp (Sorted by Min Temp) as Scatter Plot
fig.add_trace(go.Scatter(
    x=humidity_temp_extremes_desc['day'],
    y=humidity_temp_extremes_desc['humidity'],
    mode='markers',
    name='Humidity vs Min Temp',
    marker=dict(color='rgba(0, 191, 255, 0.8)', size=8),
    hovertemplate="Day: %{x}<br>Humidity: %{y}<br>Min Temp: %{x}"
), row=1, col=2)

# Temperature Trends vs Humidity (Spanning across both columns in the second row)
fig.add_trace(go.Scatter(
    x=humidity_temp_extremes['day'],
    y=humidity_temp_extremes['maxtemp'],
    mode='lines+markers',
    name='Max Temp',
    marker=dict(color='rgba(255, 165, 0, 0.8)', size=4),
    line=dict(width=1, color='rgba(255, 165, 0, 0.8)'),
    hovertemplate="Day: %{x}<br>Max Temp: %{y}"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=humidity_temp_extremes['day'],
    y=humidity_temp_extremes['mintemp'],
    mode='lines+markers',
    name='Min Temp',
    marker=dict(color='rgba(34, 139, 34, 0.8)', size=4),
    line=dict(width=1, color='rgba(34, 139, 34, 0.8)'),
    hovertemplate="Day: %{x}<br>Min Temp: %{y}"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=humidity_temp_extremes['day'],
    y=humidity_temp_extremes['humidity'],
    mode='lines+markers',
    name='Humidity',
    marker=dict(color='rgba(255, 0, 0, 0.8)', size=4),
    line=dict(width=1, color='rgba(255, 0, 0, 0.8)'),
    hovertemplate="Day: %{x}<br>Humidity: %{y}"
), row=2, col=1)

# Customize Layout with Title, Axes, and Background
fig.update_layout(
    title="<b>Comparative Analysis of Humidity and Temperatures</b>",
    height=900,  
    width=1200,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",
    title_x=0.5
)

# Set x-axis and y-axis labels for all subplots
fig.update_xaxes(title_text="Day", row=1, col=1)
fig.update_yaxes(title_text="Humidity", row=1, col=1)

fig.update_xaxes(title_text="Day", row=1, col=2)
fig.update_yaxes(title_text="Humidity", row=1, col=2)

fig.update_xaxes(title_text="Day", row=2, col=1)
fig.update_yaxes(title_text="Temperature & Humidity", row=2, col=1)

# Show the final plot
fig.show()



# Grouping the data and calculating the maximum values for pressure, maxtemp, and humidity
max_values = df_train.groupby('day')[['pressure', 'maxtemp', 'humidity']].max().reset_index()

# Sorting the result in descending order
max_values_sorted = max_values.sort_values(by=['pressure', 'maxtemp', 'humidity'], ascending=False)

# Display the result: Top 10 rows after sorting
print("Maximum Pressure, Temperature, and Humidity by Day:")
display(max_values_sorted.head(10))



# Grouping the data and calculating the minimum values for pressure, mintemp, and humidity
min_values = df_train.groupby('day')[['pressure', 'mintemp', 'humidity']].min().reset_index()

# Sorting the result in ascending order
min_values_sorted = min_values.sort_values(by=['pressure', 'mintemp', 'humidity'], ascending=True)

# Display the result: Top 10 rows after sorting
print("Minimum Pressure, Temperature, and Humidity by Day:")
display(min_values_sorted.head(10))



# Grouping the data and calculating mean, max, and min for pressure, dewpoint, and humidity
pressure_dewpoint_humidity_trends = df_train.groupby('day')[['pressure', 'dewpoint', 'humidity']].agg(['mean', 'max', 'min']).reset_index()

# Sorting in descending order for mean and max values, and ascending order for min values
pressure_dewpoint_humidity_sorted = pressure_dewpoint_humidity_trends.sort_values(
    by=[('pressure', 'mean'), ('dewpoint', 'mean'), ('humidity', 'mean'),
        ('pressure', 'max'), ('dewpoint', 'max'), ('humidity', 'max'),
        ('pressure', 'min'), ('dewpoint', 'min'), ('humidity', 'min')],
    ascending=[False, False, False, False, False, False, True, True, True]
)

# Display the result: Top 10 rows after sorting
print("Pressure, Dewpoint, and Humidity Trends by Day:")
display(pressure_dewpoint_humidity_sorted.head(10))



# Grouping the data and calculating mean, max, and min for pressure, dewpoint, and humidity
pressure_dewpoint_humidity_trends = df_train.groupby('day')[['pressure', 'dewpoint', 'humidity']].agg(['mean', 'max', 'min']).reset_index()

# Sorting in descending order for mean and max values, and ascending order for min values
pressure_dewpoint_humidity_sorted = pressure_dewpoint_humidity_trends.sort_values(
    by=[('pressure', 'mean'), ('dewpoint', 'mean'), ('humidity', 'mean'),
        ('pressure', 'max'), ('dewpoint', 'max'), ('humidity', 'max'),
        ('pressure', 'min'), ('dewpoint', 'min'), ('humidity', 'min')],
    ascending=[False, False, False, False, False, False, True, True, True]
)

# Create a subplot (2x2 grid) for different comparisons (3 subplots now)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Pressure vs Dewpoint", 
                    "Pressure vs Humidity", 
                    "Dewpoint vs Humidity"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, None]]  # Remove the 4th subplot
)

# Pressure vs Dewpoint (Scatter plot)
fig.add_trace(go.Scatter(
    x=pressure_dewpoint_humidity_sorted[('pressure', 'mean')],
    y=pressure_dewpoint_humidity_sorted[('dewpoint', 'mean')],
    mode='markers',
    name='Pressure vs Dewpoint',
    marker=dict(color='rgba(255, 165, 0, 0.8)', size=8, line=dict(color='black', width=1)),
    hovertemplate="Pressure: %{x}<br>Dewpoint: %{y}"
), row=1, col=1)

# Pressure vs Humidity (Scatter plot)
fig.add_trace(go.Scatter(
    x=pressure_dewpoint_humidity_sorted[('pressure', 'mean')],
    y=pressure_dewpoint_humidity_sorted[('humidity', 'mean')],
    mode='markers',
    name='Pressure vs Humidity',
    marker=dict(color='rgba(255, 0, 0, 0.8)', size=8, line=dict(color='black', width=1)),
    hovertemplate="Pressure: %{x}<br>Humidity: %{y}"
), row=1, col=2)

# Dewpoint vs Humidity (Scatter plot)
fig.add_trace(go.Scatter(
    x=pressure_dewpoint_humidity_sorted[('dewpoint', 'mean')],
    y=pressure_dewpoint_humidity_sorted[('humidity', 'mean')],
    mode='markers',
    name='Dewpoint vs Humidity',
    marker=dict(color='rgba(0, 255, 255, 0.8)', size=8, line=dict(color='black', width=1)),
    hovertemplate="Dewpoint: %{x}<br>Humidity: %{y}"
), row=2, col=1)

# Customize Layout with Title, Axes, and Background
fig.update_layout(
    title="<b>Analysis of Pressure, Dewpoint, and Humidity</b>",
    height=800,  
    width=1200,  
    showlegend=True,  
    template="plotly",  
    font=dict(color='black', size=14),  
    plot_bgcolor="white",  
    paper_bgcolor="white",
    title_x=0.5
)

# Set x-axis and y-axis labels for all subplots
fig.update_xaxes(title_text="Pressure (Mean)", row=1, col=1)
fig.update_yaxes(title_text="Dewpoint (Mean)", row=1, col=1)

fig.update_xaxes(title_text="Pressure (Mean)", row=1, col=2)
fig.update_yaxes(title_text="Humidity (Mean)", row=1, col=2)

fig.update_xaxes(title_text="Dewpoint (Mean)", row=2, col=1)
fig.update_yaxes(title_text="Humidity (Mean)", row=2, col=1)

# Show the final plot
fig.show()


# Compute min, mean, and max values for selected columns
summary_stats = df_train[['pressure', 'temparature', 
                          'dewpoint', 'humidity', 'cloud', 'sunshine']].agg(['min', 'mean', 'max'])
display(summary_stats)



# Set display options to show full transposed DataFrame
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

# Group data by 'day' and compute min, mean, max for each column
daily_stats = df_train.groupby('day')[['pressure',  'temparature', 
                                       'dewpoint', 'humidity', 'cloud', 'sunshine']].agg(['min', 'mean', 'max']).reset_index()

# Apply sorting conditions for all columns
for col in ['pressure', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine']:
    daily_stats = daily_stats.sort_values((col, 'max'), ascending=False)   # Sort by max (descending)
    daily_stats = daily_stats.sort_values((col, 'mean'), ascending=False)  # Sort by mean (descending)
    daily_stats = daily_stats.sort_values((col, 'min'), ascending=True)    # Sort by min (ascending)

# Print statement before displaying results
print("Day-wise pressure, maxtemp, temparature, mintemp, dewpoint, humidity, cloud & sunshine Analysis:")

# Display the first 10 rows without an index
display(daily_stats.head(10).style.hide(axis="index"))



# Set display options to show full transposed DataFrame
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

# Group data by 'day' and compute min, mean, max for each column
daily_stats = df_train.groupby('day')[['pressure', 'temparature', 
                                       'dewpoint', 'humidity', 'cloud', 'sunshine']].agg(['min', 'mean', 'max']).reset_index()

# Apply sorting conditions for all columns
for col in ['pressure', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine']:
    daily_stats = daily_stats.sort_values((col, 'max'), ascending=False)   # Sort by max (descending)
    daily_stats = daily_stats.sort_values((col, 'mean'), ascending=False)  # Sort by mean (descending)
    daily_stats = daily_stats.sort_values((col, 'min'), ascending=True)    # Sort by min (ascending)
# Set up the figure and axes
fig, axes = plt.subplots(3, 2, figsize=(16, 18))  # 3 rows, 2 columns

# Titles for subplots
plot_titles = ['Pressure', 'Temperature', 'Dew Point', 
               'Humidity', 'Cloud Cover', 'Sunshine']

# Loop through each column and plot
for i, col in enumerate(['pressure', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine']):
    row, col_idx = divmod(i, 2)  # Calculate row and column index for subplot
    
    # Line plot for trends over days
    sns.lineplot(data=daily_stats, x='day', y=(col, 'mean'), ax=axes[row, col_idx], label='Mean', color='b')
    sns.lineplot(data=daily_stats, x='day', y=(col, 'min'), ax=axes[row, col_idx], label='Min', color='g', linestyle='dashed')
    sns.lineplot(data=daily_stats, x='day', y=(col, 'max'), ax=axes[row, col_idx], label='Max', color='r', linestyle='dotted')
    
    # Formatting
    axes[row, col_idx].set_title(plot_titles[i], fontsize=14, fontweight='bold')
    axes[row, col_idx].set_xlabel('Day')
    axes[row, col_idx].set_ylabel(col.capitalize())
    axes[row, col_idx].legend()
    axes[row, col_idx].grid(True, linestyle='--', alpha=0.6)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Find days with extreme max temperature
max_temp_day = df_train[df_train['maxtemp'] == df_train['maxtemp'].max()]['day'].values[0]

# Find days with extreme min temperature
min_temp_day = df_train[df_train['mintemp'] == df_train['mintemp'].min()]['day'].values[0]

# Find days with extreme humidity
max_humidity_day = df_train[df_train['humidity'] == df_train['humidity'].max()]['day'].values[0]

# Find days with extreme cloud cover
max_cloud_day = df_train[df_train['cloud'] == df_train['cloud'].max()]['day'].values[0]

# Print results
print(f"Day with highest max temperature: {max_temp_day}")
print(f"Day with lowest min temperature: {min_temp_day}")
print(f"Day with highest humidity: {max_humidity_day}")
print(f"Day with highest cloud cover: {max_cloud_day}")



# Find days with maximum and minimum sunshine
sunshine_analysis = df_train.groupby('day').agg({
    'sunshine': 'sum',
    'maxtemp': 'mean',
    'mintemp': 'mean',
    'pressure': 'mean',
    'humidity': 'mean',
    'cloud': 'mean'
}).reset_index()

# Sort by highest and lowest sunshine
max_sunshine_days = sunshine_analysis.sort_values(by='sunshine', ascending=False)
min_sunshine_days = sunshine_analysis.sort_values(by='sunshine', ascending=True)

# Display results
print("Top Sunniest Days:\n")
display(max_sunshine_days.head(10))
print("\n====================================================================\n")
print("Top Cloudiest Days (Least Sunshine):\n")
display(min_sunshine_days.head(10))



# Create subplots layout
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Daily Sunshine Duration", 
        "Sunshine vs. Humidity",
        "Sunshine vs. Cloud Cover",
        "Temperature Trends (Max & Min)", 
        "Daily Pressure Trends"
    ],
    horizontal_spacing=0.15,  # Adjust spacing for better visibility
    vertical_spacing=0.12
)

# Line plot for sunshine across days
fig.add_trace(go.Scatter(
    x=sunshine_analysis['day'], 
    y=sunshine_analysis['sunshine'], 
    mode='lines', 
    name='Sunshine', 
    line=dict(color='gold'),
    hovertemplate="Day: %{x}<br>Sunshine: %{y}"
), row=1, col=1)

# Scatter plot of sunshine vs humidity
fig.add_trace(go.Scatter(
    x=sunshine_analysis['sunshine'], 
    y=sunshine_analysis['humidity'], 
    mode='markers', 
    name='Humidity', 
    marker=dict(color='blue', opacity=0.6),
    hovertemplate="Sunshine: %{x} hours<br>Humidity: %{y}"
), row=1, col=2)

# Scatter plot of sunshine vs cloud cover
fig.add_trace(go.Scatter(
    x=sunshine_analysis['sunshine'], 
    y=sunshine_analysis['cloud'], 
    mode='markers', 
    name='Cloud Cover', 
    marker=dict(color='gray', opacity=0.6),
    hovertemplate="Sunshine: %{x} hours<br>Cloud Cover: %{y}"
), row=2, col=1)

# Line plot for max & min temperature
fig.add_trace(go.Scatter(
    x=sunshine_analysis['day'], 
    y=sunshine_analysis['maxtemp'], 
    mode='lines', 
    name='Max Temp', 
    line=dict(color='red'),
    hovertemplate="Day: %{x}<br>Max Temp: %{y}"
), row=2, col=2)

fig.add_trace(go.Scatter(
    x=sunshine_analysis['day'], 
    y=sunshine_analysis['mintemp'], 
    mode='lines', 
    name='Min Temp', 
    line=dict(color='blue'),
    hovertemplate="Day: %{x}<br>Min Temp: %{y}"
), row=2, col=2)

# Line plot for pressure trends
fig.add_trace(go.Scatter(
    x=sunshine_analysis['day'], 
    y=sunshine_analysis['pressure'], 
    mode='lines', 
    name='Pressure', 
    line=dict(color='green'),
    hovertemplate="Day: %{x}<br>Pressure: %{y}"
), row=3, col=1)

# Update axes labels
fig.update_xaxes(title_text="Day", row=1, col=1)
fig.update_yaxes(title_text="Sunshine", row=1, col=1)

fig.update_xaxes(title_text="Sunshine (hours)", row=1, col=2)
fig.update_yaxes(title_text="Humidity", row=1, col=2)

fig.update_xaxes(title_text="Sunshine (hours)", row=2, col=1)
fig.update_yaxes(title_text="Cloud Cover", row=2, col=1)

fig.update_xaxes(title_text="Day", row=2, col=2)
fig.update_yaxes(title_text="Temperature", row=2, col=2)

fig.update_xaxes(title_text="Day", row=3, col=1)
fig.update_yaxes(title_text="Pressure", row=3, col=1)

# Update layout
fig.update_layout(
    height=1000, width=1600, 
    title_text="Sunshine and Weather Variable Analysis",
    showlegend=True
)

fig.show()



# Compute daily temperature range
df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']

# Aggregate temperature range with pressure & humidity
temp_fluctuation_analysis = df_train.groupby('day').agg({
    'temp_range': 'max',
    'pressure': 'mean',
    'humidity': 'mean',
    'cloud': 'mean'
}).reset_index()

# Sort by highest temperature fluctuation
temp_fluctuation_analysis = temp_fluctuation_analysis.sort_values(by='temp_range', ascending=False)

# Display results
print("Days with Largest Temperature Differences:\n")
display(temp_fluctuation_analysis.head(10))



# Find high dewpoint days and their cloud, humidity, and temperature impact
dewpoint_analysis = df_train.groupby('day').agg({
    'dewpoint': 'max',
    'humidity': 'mean',
    'cloud': 'mean',
    'temparature': 'mean',
    'sunshine': 'sum'
}).reset_index()

# Sort by highest dewpoint
high_dewpoint_days = dewpoint_analysis.sort_values(by='dewpoint', ascending=False)

# Display results
print("Top Most Humid Days Based on Dewpoint:\n")
display(high_dewpoint_days.head(10))



import plotly.express as px

# Create the scatter bubble chart
fig = px.scatter(
    temp_fluctuation_analysis,
    x="day",
    y="temp_range",
    size="humidity",  # Bubble size based on humidity
    color="cloud",  # Color intensity based on cloud cover
    hover_data=["pressure", "humidity", "cloud"],  # Show extra details on hover
    color_continuous_scale="blues",
    title="Daily Temperature Fluctuations vs. Weather Variables"
)

# Update layout for better visualization
fig.update_layout(
    xaxis_title="Day",
    yaxis_title="Temperature Range",
    coloraxis_colorbar=dict(title="Cloud Cover"),
    template="plotly_white",  # Clean and elegant theme
    height=700,
    width=1200
)

fig.show()



# Compute average cloud cover per day
cloud_analysis = df_train.groupby('day').agg({
    'cloud': 'mean',
    'sunshine': 'sum',
    'maxtemp': 'mean',
    'mintemp': 'mean',
    'pressure': 'mean'
}).reset_index()

# Sort by highest cloud cover
high_cloud_days = cloud_analysis.sort_values(by='cloud', ascending=False)

# Display results
print("Days with Highest Cloud Cover:\n")
display(high_cloud_days.head(10))



# Define a prominent color scale
colors = ['#FF4500', '#FF8C00', '#FFD700', '#32CD32', '#1E90FF', '#4B0082']

# Create scatter bubbles with smaller sizes
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=high_cloud_days["day"],
    y=high_cloud_days["cloud"],
    mode="markers",
    marker=dict(
        size=high_cloud_days["sunshine"] * 5.0,  # Decrease bubble size
        sizemode="area",  # Ensure size is proportionate
        color=high_cloud_days["pressure"],  # Color based on pressure
        colorscale=colors,  # Apply bold custom color palette
        showscale=True,
        colorbar=dict(title="Pressure (hPa)")
    ),
    hovertemplate="<b>Day:</b> %{x}<br>" +
                  "<b>Cloud Cover:</b> %{y}%<br>" +
                  "<b>Sunshine:</b> %{marker.size}<br>" +
                  "<b>Max Temp:</b> %{customdata[0]}Â°C<br>" +
                  "<b>Min Temp:</b> %{customdata[1]}Â°C<br>" +
                  "<b>Pressure:</b> %{marker.color} hPa",
    name="Cloud vs. Sunshine",
    customdata=high_cloud_days[["maxtemp", "mintemp"]]
))

# Update layout for aesthetics
fig.update_layout(
    title="â˜�ï¸� Cloud Cover vs. Sunshine & Temperature",
    xaxis_title="Day",
    yaxis_title="Average Cloud Cover (%)",
    template="plotly_white",  # Clean and stylish theme
    height=750,
    width=1300,
    coloraxis_colorbar=dict(title="Pressure Levels")
)

fig.show()



# Compute daily humidity trends
humidity_analysis = df_train.groupby('day').agg({
    'humidity': 'mean',
    'maxtemp': 'mean',
    'mintemp': 'mean',
    'sunshine': 'sum',
    'cloud': 'mean'
}).reset_index()

# Sort by highest humidity
high_humidity_days = humidity_analysis.sort_values(by='humidity', ascending=False)

# Sort by lowest humidity
low_humidity_days = humidity_analysis.sort_values(by='humidity', ascending=True)

# Display results
print("Top Most Humid Days:\n")
display(high_humidity_days.head(10))
print("\n====================================================================\n")
print("Top Least Humid Days:\n")
display(low_humidity_days.head(10))


# Compute daily pressure trends
pressure_analysis = df_train.groupby('day').agg({
    'pressure': 'mean',
    'maxtemp': 'mean',
    'mintemp': 'mean',
    'humidity': 'mean',
    'cloud': 'mean',
    'sunshine': 'sum'
}).reset_index()

# Sort by highest pressure
high_pressure_days = pressure_analysis.sort_values(by='pressure', ascending=False)

# Display results
print("Top Highest Pressure Days:\n")
display(high_pressure_days.head(10))



# Compute daily average values
sun_cloud_humidity_analysis = df_train.groupby('day').agg({
    'sunshine': 'sum',
    'cloud': 'mean',
    'humidity': 'mean',
    'temparature': 'mean'
}).reset_index()

# Sort by highest sunshine
high_sunshine_days = sun_cloud_humidity_analysis.sort_values(by='sunshine', ascending=False)

# Display results
print("Days with Highest Sunshine & Their Cloud Cover & Humidity:\n")
display(high_sunshine_days.head(10))



# Create weather patterns based on categorical bins
df_train['temp_category'] = pd.cut(df_train['temparature'], bins=5, labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
df_train['humidity_category'] = pd.cut(df_train['humidity'], bins=4, labels=['Low', 'Moderate', 'High', 'Very High'])
df_train['sunshine_category'] = pd.cut(df_train['sunshine'], bins=3, labels=['Low', 'Medium', 'High'])

# Count the most common weather patterns
weather_patterns = df_train.groupby(['temp_category', 'humidity_category', 'sunshine_category']).size().reset_index(name='count')

# Sort by most frequent pattern
most_common_patterns = weather_patterns.sort_values(by='count', ascending=False)

# Display results
print("Most Common Weather Patterns:\n")
display(most_common_patterns.head(10))


# Define figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(22, 10), constrained_layout=True, facecolor="white")

# Set vibrant, unique color palettes
colors_temp = sns.color_palette("coolwarm", as_cmap=True)
colors_humidity = sns.color_palette("crest", as_cmap=True)
colors_sunshine = sns.color_palette("rocket", as_cmap=True)

# Custom function to add value labels
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=14, fontweight='bold', color='black', 
                    xytext=(0, 7), textcoords='offset points')

# Iterate through subplots and configure each
for ax, column, palette, title, title_color in zip(
    axes,
    ["temp_category", "humidity_category", "sunshine_category"],
    ["coolwarm", "crest", "rocket"],
    ["Temperature Categories", "Humidity Categories", "Sunshine Categories"],
    ["blue", "green", "red"]
):
    # Plot category count
    sns.barplot(
        x=df_train[column].value_counts().index, 
        y=df_train[column].value_counts().values, 
        ax=ax, 
        palette=palette,
        edgecolor="black",
        linewidth=2
    )

    # Title styling
    ax.set_title(title, fontsize=18, fontweight='bold', color=title_color, pad=15)

    # Set axis labels
    ax.set_xlabel(column.replace("_", " ").title(), fontsize=16, fontweight='bold', color='black', labelpad=10)
    ax.set_ylabel("Count", fontsize=16, fontweight='bold', color='black', labelpad=10)

    # Improve ticks: larger, bolder, and colored
    ax.tick_params(axis='x', labelsize=14, rotation=25, colors='black', width=2)
    ax.tick_params(axis='y', labelsize=14, colors='black', width=2)

    # Enhance grid
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray', linewidth=1.2)

    # Add labels to bars
    add_labels(ax)

# Set a main title
plt.suptitle('Weather Category Distributions', fontsize=22, fontweight='bold', color='black', y=1.05)

# Show plots
plt.show()



# Compute average cloud cover at different pressure levels
pressure_cloud_relation = df_train.groupby('pressure').agg({'cloud': 'mean'}).reset_index()

# Find pressure levels with highest cloud cover
highest_cloud_pressure = pressure_cloud_relation.sort_values(by='cloud', ascending=False).head(10)

# Display results
print("Pressure Levels with Highest Cloud Cover:\n", highest_cloud_pressure)



# Compute average cloud cover at different pressure levels
pressure_cloud_relation = df_train.groupby('pressure').agg({'cloud': 'mean'}).reset_index()

# Compute rolling average for smoother trends
window_size = 5  # Rolling average window size (e.g., 5 pressure levels)
pressure_cloud_relation['rolling_cloud'] = pressure_cloud_relation['cloud'].rolling(window=window_size, center=True).mean()

# Identify key pressure levels with highest cloud cover
top_pressure_levels = pressure_cloud_relation.nlargest(3, 'cloud')

# Create figure
fig = go.Figure()

# Original cloud cover trend (line)
fig.add_trace(go.Scatter(
    x=pressure_cloud_relation['pressure'], 
    y=pressure_cloud_relation['cloud'], 
    mode='lines+markers',
    marker=dict(size=8, color='blue', symbol='circle'),
    line=dict(width=2, dash='dot', color='blue'),
    name='Original Cloud Cover',
    hovertemplate='Pressure Level: %{x}<br>Cloud Cover: %{y:.2f}'
))

# Rolling average trendline (Smoothed)
fig.add_trace(go.Scatter(
    x=pressure_cloud_relation['pressure'], 
    y=pressure_cloud_relation['rolling_cloud'], 
    mode='lines',
    line=dict(width=4, color='red', dash='solid'),
    name=f'Smoothed Trend (Rolling Avg - {window_size} Levels)',
    hovertemplate='Pressure Level: %{x}<br>Rolling Avg Cloud: %{y:.2f}'
))

# Annotations for key pressure levels with highest cloud cover
for i, row in top_pressure_levels.iterrows():
    fig.add_annotation(
        x=row['pressure'], 
        y=row['cloud'], 
        text=f"High Cloud Cover: {row['cloud']:.2f}", 
        showarrow=True, 
        arrowhead=2, 
        arrowcolor="red",
        font=dict(size=10, color="red"),
        align="center",
        textangle=45  # Rotating the annotation text by 45 degrees for clarity
    )

# Customize layout with better insights
fig.update_layout(
    title='Cloud Cover Trend Across Pressure Levels',
    title_font_size=18,
    title_font_color='darkblue',
    xaxis=dict(
        title='Pressure Level', 
        title_font_size=14, 
        tickmode='linear',
        tickformat='.2f',  # Ensure better precision on the x-axis labels
        tickangle=45,  # Rotate tick labels for better legibility
        tickvals=pressure_cloud_relation['pressure'],
        ticktext=pressure_cloud_relation['pressure']
    ),
    yaxis=dict(
        title='Average Cloud Cover', 
        title_font_size=14,
        tickformat='.2f',
        showgrid=True,  # Adding gridlines for better readability
    ),
    template='plotly_white',
    legend=dict(title="Legend", font=dict(size=14), x=0.8, y=1),
    width=1800,  # Increased width for a larger figure
    height=700,  # Increased height for a larger figure
    margin=dict(l=40, r=40, t=80, b=60),  # Adjusted margins for better spacing
)

# Add a background color to the plot area to enhance visual appeal
fig.update_layout(
    plot_bgcolor='lightgray'
)

# Show plot
fig.show()



# Compute the top 10 days with highest rainfall
high_rain_days = df_train.sort_values(by='rainfall', ascending=False).head(10)

# Select relevant columns
print("Top Days with Highest Rainfall:\n")
display(high_rain_days[['day', 'rainfall', 'temparature', 'humidity', 'pressure', 'windspeed', 'cloud']])



df_train['windspeed_category'] = pd.cut(df_train['windspeed'], bins=4, labels=['Calm', 'Breezy', 'Windy', 'Stormy'])

# Group and count most common patterns
weather_patterns = df_train.groupby(['windspeed_category']).size().reset_index(name='count')

# Sort by most frequent pattern
most_common_weather = weather_patterns.sort_values(by='count', ascending=False).head(10)

# Display results
print("Most Common Weather Patterns:\n", most_common_weather)



# Custom-defined vibrant colors for windspeed categories
windspeed_colors = ['#FF5733', '#33A8FF', '#FFD700', '#9C27B0']  # Red, Blue, Gold, Purple

# Count occurrences of each windspeed category
windspeed_counts = df_train['windspeed_category'].value_counts()

# Define figure and subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)  # Increased figure size

# Barplot for windspeed categories
sns.barplot(
    x=windspeed_counts.index, 
    y=windspeed_counts.values, 
    ax=axes[0], 
    palette=windspeed_colors
)

# Add value labels on top of bars
for i, value in enumerate(windspeed_counts.values):
    axes[0].text(i, value + 1, f"{value}", ha='center', fontsize=12, fontweight='bold', color='black', 
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

axes[0].set_title('Windspeed Categories - Barplot', fontsize=14, fontweight='bold', color='darkred')
axes[0].set_xlabel('Windspeed Category', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Pie chart for windspeed categories
explode_values = (0.05, 0.07, 0.1, 0.05)  # Subtle explosion for better visibility

axes[1].pie(
    windspeed_counts, 
    labels=windspeed_counts.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=windspeed_colors, 
    explode=explode_values,  # Subtle explosion applied
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, 
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)

axes[1].set_title('Windspeed Categories - Pie Chart', fontsize=14, fontweight='bold', color='darkblue')

# Set a main title for the whole figure
plt.suptitle('Windspeed Category Distribution', fontsize=16, fontweight='bold', color='darkgreen')

# Show the plots
plt.show()



# Compute average rainfall for different wind directions
wind_rain_relation = df_train.groupby('winddirection').agg({'rainfall': 'mean'}).reset_index()

# Find wind directions with highest rainfall
highest_rain_wind = wind_rain_relation.sort_values(by='rainfall', ascending=False).head(10)

# Display results
print("Wind Directions Leading to Most Rainfall:\n", highest_rain_wind)



# Compute average rainfall for different wind directions
wind_rain_relation = df_train.groupby('winddirection').agg({'rainfall': 'mean'}).reset_index()

# Compute rolling average for smoother trends
window_size = 3  # Rolling average window size (e.g., 3 wind directions)
wind_rain_relation['rolling_rain'] = wind_rain_relation['rainfall'].rolling(window=window_size, center=True).mean()

# Identify key wind directions with highest rainfall
top_wind_directions = wind_rain_relation.nlargest(3, 'rainfall')

# Create figure
fig = go.Figure()

# Original rainfall trend (line)
fig.add_trace(go.Scatter(
    x=wind_rain_relation['winddirection'], 
    y=wind_rain_relation['rainfall'], 
    mode='lines+markers',
    marker=dict(size=8, color='blue', symbol='circle'),
    line=dict(width=2, dash='dot', color='blue'),
    name='Original Rainfall',
    hovertemplate='Wind Direction: %{x}<br>Rainfall: %{y:.2f} mm'
))

# Rolling average trendline
fig.add_trace(go.Scatter(
    x=wind_rain_relation['winddirection'], 
    y=wind_rain_relation['rolling_rain'], 
    mode='lines',
    line=dict(width=4, color='red', dash='solid'),
    name=f'Rolling Avg - {window_size} Wind Directions',  # Updated to reflect window size as wind directions
    hovertemplate='Wind Direction: %{x}<br>Rolling Avg: %{y:.2f} mm'
))

# Annotations for top wind directions with highest rainfall
for i, row in top_wind_directions.iterrows():
    fig.add_annotation(
        x=row['winddirection'], 
        y=row['rainfall'], 
        text=f"High Rainfall: {row['rainfall']:.2f} mm", 
        showarrow=True, 
        arrowhead=2, 
        arrowcolor="red",
        font=dict(size=10, color="red"),
        align="center",
        textangle=45  # Rotating the annotation text by 45 degrees
    )

# Customize layout with better insights
fig.update_layout(
    title='Rainfall Trend Across Wind Directions',
    title_font_size=18,
    title_font_color='darkblue',
    xaxis=dict(
        title='Wind Direction', 
        title_font_size=14, 
        tickangle=45,  # Rotate wind direction labels for better legibility
        tickmode='array',
        tickvals=wind_rain_relation['winddirection'],
        ticktext=wind_rain_relation['winddirection']
    ),
    yaxis=dict(
        title='Average Rainfall (mm)', 
        title_font_size=14,
        tickformat='.2f'
    ),
    template='plotly_white',
    legend=dict(title="Legend", font=dict(size=14), x=0.8, y=1),
    width=1800,  # Increased width for a larger figure
    height=700,  # Increased height for a larger figure
    margin=dict(l=40, r=40, t=80, b=60),  # Adjusted margins for better spacing
)

# Add a background color to the plot area to enhance visual appeal
fig.update_layout(
    plot_bgcolor='lightgray'
)

# Show plot
fig.show()



# Find the top 5 days with the highest sunshine hours
sunniest_days = df_train.nlargest(10, 'sunshine')
# Display results
print("Days with Longest Sunshine Hours:\n")
display(sunniest_days[['day', 'sunshine', 'temparature', 'humidity', 'cloud']])



# Categorize wind speeds into meaningful labels
def classify_wind(speed):
    if speed < 5:
        return "Calm"
    elif speed < 15:
        return "Breezy"
    elif speed < 25:
        return "Windy"
    else:
        return "Stormy"

df_train['wind_category'] = df_train['windspeed'].apply(classify_wind)

# Compute average cloud cover for different wind categories
cloud_wind_relation = df_train.groupby('wind_category')['cloud'].median().reset_index()

# Display results
print("Cloud Cover at Different Wind Speeds:\n", cloud_wind_relation)



# First, classify the wind speed into meaningful categories
def classify_wind(speed):
    if speed < 5:
        return "Calm"
    elif speed < 15:
        return "Breezy"
    elif speed < 25:
        return "Windy"
    else:
        return "Stormy"

df_train['wind_category'] = df_train['windspeed'].apply(classify_wind)

# Compute median cloud cover for different wind categories
cloud_wind_relation = df_train.groupby('wind_category')['cloud'].median().reset_index()

# Set up the subplots with a 1x2 grid for better organization (only first row)
fig = sp.make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Cloud Cover by Wind Category (Bar)", "Cloud Cover by Wind Category (Box)"),
    column_widths=[0.5, 0.5],  # Adjust column widths to keep them balanced
    horizontal_spacing=0.1
)

# 1. Bar Plot: Cloud Cover by Wind Category
fig.add_trace(go.Bar(
    x=cloud_wind_relation['wind_category'], 
    y=cloud_wind_relation['cloud'],
    name="Cloud Cover by Wind Category",
    marker=dict(color='royalblue', line=dict(color='darkblue', width=2)),
    text=cloud_wind_relation['cloud'],
    textposition='auto',
    hovertemplate='<b>Wind Category:</b> %{x}<br>' +
                  '<b>Median Cloud Cover:</b> %{y:.2f}<br>' +
                  'Cloud Cover Value: %{text:.2f}<extra></extra>',  # Show the text with cloud cover value
), row=1, col=1)

# 2. Box Plot: Cloud Cover Distribution by Wind Category
fig.add_trace(go.Box(
    x=df_train['wind_category'], 
    y=df_train['cloud'], 
    name="Cloud Cover Distribution",
    marker=dict(color='lightgreen', line=dict(color='darkgreen', width=2)),
    boxmean='sd',  # Show mean with standard deviation
    hovertemplate='<b>Wind Category:</b> %{x}<br>' +
                  '<b>Cloud Cover Value:</b> %{y:.2f}<br>' +
                  'Cloud Distribution: %{y}<extra></extra>',  # Show the cloud cover value in the hover
), row=1, col=2)

# Update layout for better visuals
fig.update_layout(
    title="Cloud Cover Analysis by Wind Categories",
    title_font_size=20,
    title_font_color='darkblue',
    title_x=0.5,  # Center the title
    title_xanchor='center',
    template="plotly_white",  # White background theme for clarity
    showlegend=False,  # No need for legend in these subplots
    width=1100,  # Adjust width for better viewing
    height=600,  # Adjust height for better viewing
    margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins for spacing
    plot_bgcolor='white',  # White background for plot area
    xaxis=dict(
        title="Wind Category",
        title_font_size=14,
        title_font_color='black',
        tickangle=45,
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray'  # Lighter gridlines for better clarity
    ),
    yaxis=dict(
        title="Cloud Cover Value",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray'  # Lighter gridlines for better clarity
    )
)

# Show the plot
fig.show()



dry_windy_days = df_train[(df_train['windspeed'] > 40) & (df_train['rainfall'] == 0)]

print("Windy but Dry Days:\n")
display(dry_windy_days[['day', 'windspeed', 'rainfall', 'temparature']])


# Filter the dataframe to get the dry and windy days
dry_windy_days = df_train[(df_train['windspeed'] > 40) & (df_train['rainfall'] == 0)]

# Set up subplots (1 row, 2 columns)
fig = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=("Wind Speed vs Temperature", "Distribution of Wind Speed and Temperature"),
    column_widths=[0.5, 0.5],  # Adjust column widths for balance
    horizontal_spacing=0.1  # Spacing between plots
)

# Scatter Plot: Wind Speed vs Temperature
fig.add_trace(go.Scatter(
    x=dry_windy_days['windspeed'], 
    y=dry_windy_days['temparature'], 
    mode='markers', 
    name='Wind Speed vs Temperature',
    marker=dict(size=10, color='orange', line=dict(color='darkorange', width=2)),
    hovertemplate='<b>Wind Speed:</b> %{x}<br>' +
                  '<b>Temperature:</b> %{y:.2f}<extra></extra>',
), row=1, col=1)

# Box Plot: Distribution of Wind Speed and Temperature
fig.add_trace(go.Box(
    y=dry_windy_days['windspeed'], 
    name='Wind Speed',
    boxmean='sd',  # Show standard deviation on the box plot
    marker=dict(color='lightcoral'),
    hovertemplate='<b>Wind Speed:</b> %{y}<extra></extra>',
), row=1, col=2)

fig.add_trace(go.Box(
    y=dry_windy_days['temparature'], 
    name='Temperature',
    boxmean='sd',  # Show standard deviation on the box plot
    marker=dict(color='lightseagreen'),
    hovertemplate='<b>Temperature:</b> %{y}<extra></extra>',
), row=1, col=2)

# Update layout for better clarity and presentation
fig.update_layout(
    title="Windy but Dry Days Visualization",
    title_font_size=20,
    title_font_color='darkblue',
    title_x=0.5,  # Center the title
    title_xanchor='center',
    template="plotly_white",  # White background for clarity
    showlegend=False,  # No legend needed for these plots
    width=1200,  # Adjust the width for a larger view
    height=600,  # Adjust the height for better spacing
    margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins for more space
    plot_bgcolor='white',  # White plot background for clean aesthetics
    xaxis=dict(
        title="Wind Speed",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
    yaxis=dict(
        title="Temperature",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
    xaxis2=dict(
        title="Wind Speed",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
    yaxis2=dict(
        title="Temperature",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
)

# Show the plot
fig.show()



# Group by wind direction and compute average temperature and rainfall
wind_weather = df_train.groupby('winddirection').agg({'temparature': 'mean', 'rainfall': 'mean'}).reset_index()

print("Wind Directions and Their On an average Weather Impact:\n")
display(wind_weather)



# Set up subplots (1 row, 2 columns)
fig = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=("Temperature vs Wind Direction", "Rainfall vs Wind Direction"),
    column_widths=[0.5, 0.5],  # Adjust column widths for balance
    horizontal_spacing=0.1  # Spacing between plots
)

# Scatter Plot: Temperature vs Wind Direction
fig.add_trace(go.Scatter(
    x=wind_weather['winddirection'], 
    y=wind_weather['temparature'], 
    mode='markers+lines', 
    name='Temperature',
    marker=dict(size=10, color='red', line=dict(color='darkred', width=2)),
    line=dict(width=2, color='red', dash='solid'),
    hovertemplate='<b>Wind Direction:</b> %{x}<br>' +
                  '<b>Average Temperature:</b> %{y:.2f}<extra></extra>',
), row=1, col=1)

# Line Plot: Rainfall vs Wind Direction
fig.add_trace(go.Scatter(
    x=wind_weather['winddirection'], 
    y=wind_weather['rainfall'], 
    mode='markers+lines', 
    name='Rainfall',
    marker=dict(size=10, color='blue', line=dict(color='darkblue', width=2)),
    line=dict(width=2, color='blue', dash='solid'),
    hovertemplate='<b>Wind Direction:</b> %{x}<br>' +
                  '<b>Average Rainfall:</b> %{y:.2f}<extra></extra>',
), row=1, col=2)

# Update layout for better clarity and presentation
fig.update_layout(
    title="Weather Impact by Wind Direction",
    title_font_size=20,
    title_font_color='darkblue',
    title_x=0.5,  # Center the title
    title_xanchor='center',
    template="plotly_white",  # White background for clarity
    showlegend=False,  # No legend needed for these plots
    width=1200,  # Adjust the width for a larger view
    height=600,  # Adjust the height for better spacing
    margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins for more space
    plot_bgcolor='white',  # White plot background for clean aesthetics
    xaxis=dict(
        title="Wind Direction",
        title_font_size=14,
        title_font_color='black',
        tickangle=45,
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
    yaxis=dict(
        title="Average Temperature",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
    xaxis2=dict(
        title="Wind Direction",
        title_font_size=14,
        title_font_color='black',
        tickangle=45,
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
    yaxis2=dict(
        title="Average Rainfall",
        title_font_size=14,
        title_font_color='black',
        tickfont=dict(size=12, color='black'),
        showgrid=True,
        gridcolor='lightgray',  # Subtle gridlines
    ),
)

# Show the plot
fig.show()



# Load datasets
df_original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
# Clean and preprocess original data
df_original.columns = df_original.columns.str.strip()
df_original['rainfall'] = df_original['rainfall'].str.lower().map({'yes': 1, 'no': 0})

# Combine datasets
df_combined = pd.concat([df_original, df_train], axis=0, ignore_index=True)

# Drop 'id' column
df_combined.drop(columns=['id'], inplace=True, errors='ignore')
test_ids = df_test['id']
df_test.drop(columns=['id'], inplace=True, errors='ignore')


# Convert 'day' to datetime
df_combined['day'] = pd.to_datetime(df_combined['day'], errors='coerce')
df_test['day'] = pd.to_datetime(df_test['day'], errors='coerce')

# Extract temporal features
df_combined['month'] = df_combined['day'].dt.month
df_combined['day_of_week'] = df_combined['day'].dt.dayofweek
df_combined['is_weekend'] = df_combined['day_of_week'].isin([5, 6]).astype(int)

df_test['month'] = df_test['day'].dt.month
df_test['day_of_week'] = df_test['day'].dt.dayofweek
df_test['is_weekend'] = df_test['day_of_week'].isin([5, 6]).astype(int)

# Temperature features
df_combined['temp_range'] = df_combined['maxtemp'] - df_combined['mintemp']
df_combined['avg_temp'] = (df_combined['maxtemp'] + df_combined['mintemp']) / 2
df_combined['temp_deviation'] = df_combined['temparature'] - df_combined['avg_temp']

df_test['temp_range'] = df_test['maxtemp'] - df_test['mintemp']
df_test['avg_temp'] = (df_test['maxtemp'] + df_test['mintemp']) / 2
df_test['temp_deviation'] = df_test['temparature'] - df_test['avg_temp']

# Dew point depression
df_combined['dew_point_depression'] = df_combined['temparature'] - df_combined['dewpoint']
df_test['dew_point_depression'] = df_test['temparature'] - df_test['dewpoint']

# Wind direction transformation
df_combined['wind_dir_sin'] = np.sin(np.deg2rad(df_combined['winddirection']))
df_combined['wind_dir_cos'] = np.cos(np.deg2rad(df_combined['winddirection']))

df_test['wind_dir_sin'] = np.sin(np.deg2rad(df_test['winddirection']))
df_test['wind_dir_cos'] = np.cos(np.deg2rad(df_test['winddirection']))

# Wind chill factor
df_combined['wind_chill'] = 13.12 + 0.6215 * df_combined['temparature'] - 11.37 * (df_combined['windspeed']**0.16) + 0.3965 * df_combined['temparature'] * (df_combined['windspeed']**0.16)
df_test['wind_chill'] = 13.12 + 0.6215 * df_test['temparature'] - 11.37 * (df_test['windspeed']**0.16) + 0.3965 * df_test['temparature'] * (df_test['windspeed']**0.16)

# Interaction features
df_combined['humidity_temp'] = df_combined['humidity'] * df_combined['temparature']
df_combined['cloud_sunshine'] = df_combined['cloud'] * df_combined['sunshine']

df_test['humidity_temp'] = df_test['humidity'] * df_test['temparature']
df_test['cloud_sunshine'] = df_test['cloud'] * df_test['sunshine']

# Rolling features
df_combined['rolling_temp_mean'] = df_combined['avg_temp'].rolling(window=7).mean()
df_combined['rolling_wind_mean'] = df_combined['windspeed'].rolling(window=7).mean()
df_combined['rolling_humidity_mean'] = df_combined['humidity'].rolling(window=7).mean()

df_test['rolling_temp_mean'] = df_test['avg_temp'].rolling(window=7).mean()
df_test['rolling_wind_mean'] = df_test['windspeed'].rolling(window=7).mean()
df_test['rolling_humidity_mean'] = df_test['humidity'].rolling(window=7).mean()

# Lag features
df_combined['temp_lag_1'] = df_combined['avg_temp'].shift(1)
df_combined['humidity_lag_1'] = df_combined['humidity'].shift(1)
df_combined['windspeed_lag_1'] = df_combined['windspeed'].shift(1)

df_test['temp_lag_1'] = df_test['avg_temp'].shift(1)
df_test['humidity_lag_1'] = df_test['humidity'].shift(1)
df_test['windspeed_lag_1'] = df_test['windspeed'].shift(1)

# Drop original 'day' column
df_combined.drop(columns=['day'], inplace=True)
df_test.drop(columns=['day'], inplace=True)

# Fill missing values
df_combined = df_combined.fillna(df_combined.mean())
df_test = df_test.fillna(df_test.mean())


# Separate features and target
X = df_combined.drop(columns=['rainfall'])
y = df_combined['rainfall']



# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(df_test)


# Define Purged Cross-Validation
def purged_cross_validation(X, y, n_splits=5, purge_length=1):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X):
        # Apply purging
        val_idx = val_idx[val_idx >= train_idx[-purge_length]]
        yield train_idx, val_idx
# Define base models for stacking
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42)),
    ('catboost', CatBoostClassifier(verbose=0, random_state=42))
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Define hyperparameter distributions
param_distributions = {
    'rf__n_estimators': randint(100, 500),
    'xgb__n_estimators': randint(100, 500),
    'xgb__learning_rate': uniform(0.01, 0.3),
    'lgbm__n_estimators': randint(100, 500),
    'lgbm__learning_rate': uniform(0.01, 0.3),
    'catboost__n_estimators': randint(100, 500),
    'catboost__learning_rate': uniform(0.01, 0.3)
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(stacking_model, param_distributions, n_iter=10, cv=purged_cross_validation(X_scaled, y), scoring='roc_auc', random_state=42)
random_search.fit(X_scaled, y)

# Best model
best_model = random_search.best_estimator_


# Train on full dataset
best_model.fit(X_scaled, y)

# Predict on test set
test_preds = best_model.predict_proba(test_scaled)[:, 1]


# Cross-validation loop with best model
auc_scores = []
for train_idx, val_idx in purged_cross_validation(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train best model
    best_model.fit(X_train, y_train)
    
    # Validation predictions
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_val, val_proba)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    print(f"AUC: {roc_auc:.4f}")

# Average AUC score
print(f"Average AUC: {np.mean(auc_scores):.4f}")


# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_preds
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created!")
submission.head(10)

