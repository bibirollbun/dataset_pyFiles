import numpy as np
import xgboost as xgb

# 4-by-2 Data matrix
X = np.array([[1, -1], [-1, 1], [0, 1], [1, 0]])
dtrain = xgb.DMatrix(X)

# Associate ranged labels with the data matrix.
# This example shows each kind of censored labels.
#                         uncensored    right     left  interval
y_lower_bound = np.array([      2.0,     3.0,     0.0,     4.0])
y_upper_bound = np.array([      2.0, +np.inf,     4.0,     5.0])
dtrain.set_float_info('label_lower_bound', y_lower_bound)
dtrain.set_float_info('label_upper_bound', y_upper_bound)



params = {'objective': 'survival:aft',
          'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal',
          'aft_loss_distribution_scale': 1.20,
          'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 2}
bst = xgb.train(params, dtrain, num_boost_round=5,
                evals=[(dtrain, 'train')])



# XGBoosting.com
# Train an XGBoost Model for Survival Analysis using AFT Model and scikit-learn API
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Generate a synthetic dataset with features and survival times
n_samples = 1000
n_features = 10
X = np.random.rand(n_samples, n_features)
# Generate survival times from a Weibull distribution
scale = np.exp(-X[:, 0])
shape = 1.5
y = np.random.weibull(shape, n_samples) * scale

# Create lower and upper bounds, here they are the same as y because there is no censoring
y_lower = y_upper = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, y_lower_train, y_lower_test, y_upper_train, y_upper_test = train_test_split(X, y, y_lower, y_upper, test_size=0.2, random_state=42)

# Convert data into DMatrix, specifying the label, label_lower_bound, and label_upper_bound
dtrain = xgb.DMatrix(X_train, label=y_train, label_lower_bound=y_lower_train, label_upper_bound=y_upper_train)
dtest = xgb.DMatrix(X_test, label=y_test, label_lower_bound=y_lower_test, label_upper_bound=y_upper_test)

# Initialize an XGBRegressor with the "survival:aft" objective
params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',
    'aft_loss_distribution_scale': 1.0,
    'learning_rate': 0.1
}

# Fit the model on the training data
bst = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the test set
y_pred = bst.predict(dtest)

# Output the predicted survival times for demonstration purposes
print("Predicted survival times:", y_pred[:5])



import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Generate a synthetic dataset with survival times and censoring indicators
n_samples = 1000
n_features = 10
X = np.random.rand(n_samples, n_features)
true_coef = np.random.rand(n_features)
survival_time = np.exp(np.dot(X, true_coef))
censoring = np.random.binomial(1, 0.9, n_samples)


# Initialize the XGBSurvivalAnalysis model
model = XGBRegressor(objective='survival:cox',
                     eval_metric='cox-nloglik',
                     tree_method='hist')

# Fit the model to the training data
model.fit(X, survival_time, sample_weight=censoring, verbose=False)

# Make predictions
predictions = model.predict(X)
print("Sample predictions:", predictions[:5])



import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic survival data
np.random.seed(42)
X, _ = make_classification(n_samples=1000, n_features=10, random_state=42)
time = np.random.exponential(scale=1, size=1000)  # Survival times
event = np.random.binomial(1, p=0.5, size=1000)  # Censoring indicator

X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(X, time, event, test_size=0.2, random_state=42)

# Prepare DMatrix for Cox Proportional Hazards
dtrain_cox = xgb.DMatrix(X_train, label=time_train, weight=event_train)
dtest_cox = xgb.DMatrix(X_test)

# XGBoost Cox Proportional Hazards Model
params_cox = {'objective': 'survival:cox'}
model_cox = xgb.train(params_cox, dtrain_cox, num_boost_round=50)
predictions_cox = model_cox.predict(dtest_cox)

# Prepare DMatrix for AFT model
lower_bound = np.zeros_like(time_train)  # Lower bounds of survival times, set to zero for non-censoring
upper_bound = np.inf * np.ones_like(time_train)  # Assume infinity where data is censored
upper_bound[event_train == 1] = time_train[event_train == 1]  # Actual survival times for uncensored data

dtrain_aft = xgb.DMatrix(X_train, label=time_train)
dtrain_aft.set_float_info('label_lower_bound', lower_bound)
dtrain_aft.set_float_info('label_upper_bound', upper_bound)
dtest_aft = xgb.DMatrix(X_test)

# XGBoost AFT Model
params_aft = {'objective': 'survival:aft', 'aft_loss_distribution': 'normal'}
model_aft = xgb.train(params_aft, dtrain_aft, num_boost_round=50)
predictions_aft = model_aft.predict(dtest_aft)

# Output survival estimates for comparison
print(f"Cox Proportional Hazards Predictions: {predictions_cox[:5]}")
print(f"Accelerated Failure Time Predictions: {predictions_aft[:5]}")



import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

from IPython.display import display
from math import ceil
import warnings
warnings.filterwarnings('ignore')


sample_submission_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
data_dict_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


train_df.head()


train_df.shape


data_dict_df.head()


data_dict_df['type'].value_counts()


# Filter categorical variables and their descriptions from the data dictionary
categorical_vars = data_dict_df[data_dict_df['type'] == 'Categorical']
var_names = categorical_vars['variable'].tolist()
var_descriptions = categorical_vars['description'].tolist()


# Set 2 plots per row
n_cols = 2
n_rows = ceil(len(var_names) / n_cols)

# Dynamically calculate vertical spacing, slightly reduced
max_spacing = 1 / (n_rows - 1) if n_rows > 1 else 0.05
vertical_spacing = min(0.03, max_spacing)  # Reduced spacing for compact layout

# Create a subplot grid
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=var_descriptions,  # Use descriptions as titles
    vertical_spacing=vertical_spacing,  # Adjusted vertical spacing
    horizontal_spacing=0.1,  # Adjust column spacing
)

# Loop through each categorical variable and create bar plots
for idx, var in enumerate(var_names):
    # Count occurrences including "Missing" values
    category_counts = train_df[var].fillna('Missing').value_counts().reset_index()
    category_counts.columns = [var, 'count']
    
    # Calculate percentages
    total_count = category_counts['count'].sum()
    category_counts['percentage'] = (category_counts['count'] / total_count * 100).round(2)
    
    # Define bar colors
    bar_colors = [
        '#A9A9A9' if val == 'Missing' else '#2B4F81'
        for val in category_counts[var]
    ]

    # Add bar chart to subplot
    fig.add_trace(
        go.Bar(
            x=category_counts[var],
            y=category_counts['count'],
            marker=dict(color=bar_colors),
            hovertemplate=(
                '<b>Category:</b> %{x}<br>' +
                '<b>Count:</b> %{y}<br>' +
                '<b>Percentage:</b> %{customdata:.2f}%<extra></extra>'
            ),
            customdata=category_counts['percentage'],  # Pass percentage data for hover
            showlegend=False,
        ),
        row=(idx // n_cols) + 1,
        col=(idx % n_cols) + 1
    )

# Update layout for better visualization
fig.update_layout(
    height=350 * n_rows,  # Slightly reduced height for compact layout
    width=1100,           # Adjust width for better readability
    title_text='Distribution of Categorical Variables',
    title_font_size=20,
    font=dict(size=12),
    plot_bgcolor='white',  # Remove grid background
    margin=dict(t=100, b=50, l=50, r=50),  # Increase top margin for space between title and first row
)

# Customize axes for better readability
fig.update_xaxes(
    tickangle=30,  # Adjust angle to prevent overlap
    automargin=True,  # Enable auto-margin for axes
)
fig.update_yaxes(showgrid=False, automargin=True)  # Remove grid lines and enable auto-margin

# Display the updated chart grid
fig.show()



# Set 2 plots per row
n_cols = 2
n_rows = ceil(len(var_names) / n_cols)

# Dynamically calculate vertical spacing, slightly reduced
max_spacing = 1 / (n_rows - 1) if n_rows > 1 else 0.05
vertical_spacing = min(0.03, max_spacing)  # Reduced spacing for compact layout

# Create a subplot grid
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=var_descriptions,  # Use descriptions as titles
    vertical_spacing=vertical_spacing,  # Adjusted vertical spacing
    horizontal_spacing=0.1,  # Adjust column spacing
)

# Loop through each categorical variable and create stacked bar plots
for idx, var in enumerate(var_names):
    # Group data by variable and target (`efs`) and calculate counts
    grouped = train_df.groupby([var, 'efs']).size().unstack(fill_value=0).reset_index()
    grouped.columns = [var, 'efs_0', 'efs_1']
    
    # Calculate percentages for hover information
    grouped['total'] = grouped['efs_0'] + grouped['efs_1']
    grouped['efs_0_percentage'] = (grouped['efs_0'] / grouped['total'] * 100).round(2)
    grouped['efs_1_percentage'] = (grouped['efs_1'] / grouped['total'] * 100).round(2)
    
    # Add bar chart for `efs=0`
    fig.add_trace(
        go.Bar(
            x=grouped[var],
            y=grouped['efs_0'],
            name='efs=0',
            marker=dict(color='#2B4F81'),
            hovertemplate=(
                '<b>Category:</b> %{x}<br>' +
                '<b>Count (efs=0):</b> %{y}<br>' +
                '<b>Percentage (efs=0):</b> %{customdata:.2f}%<extra></extra>'
            ),
            customdata=grouped['efs_0_percentage'],  # Pass percentage data for hover
            showlegend=(idx == 0),  # Show legend only for the first subplot
        ),
        row=(idx // n_cols) + 1,
        col=(idx % n_cols) + 1
    )
    
    # Add bar chart for `efs=1`
    fig.add_trace(
        go.Bar(
            x=grouped[var],
            y=grouped['efs_1'],
            name='efs=1',
            marker=dict(color='#A9A9A9'),
            hovertemplate=(
                '<b>Category:</b> %{x}<br>' +
                '<b>Count (efs=1):</b> %{y}<br>' +
                '<b>Percentage (efs=1):</b> %{customdata:.2f}%<extra></extra>'
            ),
            customdata=grouped['efs_1_percentage'],  # Pass percentage data for hover
            showlegend=(idx == 0),  # Show legend only for the first subplot
        ),
        row=(idx // n_cols) + 1,
        col=(idx % n_cols) + 1
    )

# Update layout for better visualization
fig.update_layout(
    height=350 * n_rows,  # Slightly reduced height for compact layout
    width=1100,           # Adjust width for better readability
    title_text='Distribution of Categorical Variables by EFS',
    title_font_size=20,
    font=dict(size=12),
    plot_bgcolor='white',  # Remove grid background
    margin=dict(t=100, b=50, l=50, r=50),  # Increase top margin for space between title and first row
    barmode='stack'  # Stack bars for each category
)

# Customize axes for better readability
fig.update_xaxes(
    tickangle=30,  # Adjust angle to prevent overlap
    automargin=True,  # Enable auto-margin for axes
)
fig.update_yaxes(showgrid=False, automargin=True)  # Remove grid lines and enable auto-margin

# Display the updated chart grid
fig.show()



# Handle `inf` values by replacing them with `NaN`
vis_train_df = train_df.replace([np.inf, -np.inf], np.nan)

# Filter numerical variables from the data dictionary
numerical_vars = data_dict_df[data_dict_df['type'] == 'Numerical']
var_names = numerical_vars['variable'].tolist()
var_descriptions = numerical_vars['variable'].tolist()

# Define number of rows and columns for the subplot grid
n_cols = 4  # Number of columns per row
n_rows = math.ceil(len(var_names) / n_cols)  # Calculate rows dynamically

# Create a large figure to hold all subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5), constrained_layout=True)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot density plots for each numerical variable
for idx, var in enumerate(var_names):
    ax = axes[idx]
    sns.kdeplot(data=vis_train_df[vis_train_df['efs'] == 0], x=var, fill=True, alpha=0.5, ax=ax, label='efs=0', color='blue')
    sns.kdeplot(data=vis_train_df[vis_train_df['efs'] == 1], x=var, fill=True, alpha=0.5, ax=ax, label='efs=1', color='orange')
    
    # Add title and labels
    ax.set_title(var_descriptions[idx], fontsize=12)
    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(loc='upper right')

# Hide any unused subplots
for idx in range(len(var_names), len(axes)):
    axes[idx].axis('off')

# Add a title for the entire figure
fig.suptitle('Distribution of Numerical Variables by EFS', fontsize=16)

# Show the plot
plt.show()





