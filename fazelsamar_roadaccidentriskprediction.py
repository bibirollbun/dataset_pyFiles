# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/working'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# !pip install kaggle
# from kaggle_secrets import UserSecretsClient
# import os

# # Access Kaggle secrets
# user_secrets = UserSecretsClient()
# kaggle_key = user_secrets.get_secret("kaggle_key")
# kaggle_username = user_secrets.get_secret("kaggle_username")

# # Make Kaggle API credentials available to the CLI
# os.environ['KAGGLE_USERNAME'] = kaggle_username
# os.environ['KAGGLE_KEY'] = kaggle_key

# !kaggle kernels output fazelsamar/roadaccidentriskprediction -p /kaggle/working


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# !pip install sweetviz
# !pip install vegafusion vegafusion-python-embed
# !pip install vl-convert-python
# !pip freeze


# !pip install ray==2.10.0
!pip install autogluon.tabular
!pip install "ray>=2.10.0,<2.45.0"
# !pip install -U ipywidgets


# --- Import Libraries ---

import pandas as pd
import numpy as np
import os
# import sweetviz as sv
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.graph_objs as go
# import plotly.io as pio
# import altair as alt
# from IPython.display import IFrame, display
# import missingno as msno
# import matplotlib.pyplot as plt
# import seaborn as sns

import warnings
from autogluon.tabular import TabularDataset, TabularPredictor



# --- Configure Environment ---

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# Set a default template for plotly for a clean, professional look
# px.defaults.template = "plotly_dark"
# pio.renderers.default = "kaggle"



# --- Load Data ---

train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
original_df = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')

# Extract test_ids for later use
test_ids = test_df['id']

# # Let's standardize column names for the original dataset to match the competition data
original_df.rename(columns={
    'Number of Lanes': 'num_lanes',
    'Road Curvature': 'curvature',
    'Speed Limit': 'speed_limit',
    'Lighting': 'lighting',
    'Weather': 'weather',
    'Road Signs Present': 'road_signs_present',
    'Public Road': 'public_road',
    'Time of Day': 'time_of_day',
    'Holiday': 'holiday',
    'School Season': 'school_season',
    'Number of Reported Accidents': 'num_reported_accidents',
    'Accident Risk Score': 'accident_risk',
    'Road Type': 'road_type'
}, inplace=True)


print(f"Original data shape: {original_df.shape}")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")



# --- Automated Interactive EDA with Sweetviz ---
# report = sv.analyze(train_df, target_feat='accident_risk')
# report.show_html('Sweetviz_EDA_Report.html')
# print("Sweetviz report saved as Sweetviz_EDA_Report.html")
# # Example with layout and scale adjustments
# report.show_notebook(layout='vertical', scale=0.9)



train_df.head()


train_df.info()


duplicated_rows_train = train_df.duplicated()
train_df.drop_duplicates(inplace=True)
sum(duplicated_rows_train)


duplicated_rows_original = original_df.duplicated()
original_df.drop_duplicates(inplace=True)
sum(duplicated_rows_original)


train_df.isna().sum()


#Check statistical information of numerical values

numerical_features = train_df.select_dtypes(include=[np.number])
train_df.describe(include=[np.number]).transpose()


#Check statistical information of categorical values

categorial_features = train_df.select_dtypes(include=object)
train_df.describe(include=object)


# Get the number of unique values for each column
unique_counts = train_df.nunique()
print(unique_counts)


# Set a threshold for the maximum number of unique values to display frequencies
threshold = 8

# Dictionary to hold value frequencies
value_frequencies = {}

# Iterate over columns to compute value frequencies
for col in train_df.columns:
    if unique_counts[col] <= threshold:
        value_counts = train_df[col].value_counts()
        value_frequencies[col] = value_counts

# Print the value frequencies for columns with fewer unique values
for col, frequencies in value_frequencies.items():
    print(f"Column '{col}':")
    print(f"Number of unique values: {unique_counts[col]}")
    print("Value frequencies:")
    print(frequencies)
    print()


# Enable the VegaFusion data transformer to handle large datasets
# alt.data_transformers.enable('vegafusion')

# def plot_distributions_revised(data: pd.DataFrame, n_cols: int = 3):
#     """
#     Creates a grid of distribution plots with high performance and custom control.
#     - Uses a loop instead of melt() for speed.
#     - Explicitly defines columns for histograms vs. bar charts.
#     - Customizes bins for better histogram readability.
#     """
    
#     # --- 1. Define plot types and theme ---
#     histogram_cols = ['curvature', 'accident_risk']
#     bar_chart_cols = [
#         'road_type', 'num_lanes', 'speed_limit', 'lighting', 'weather', 
#         'road_signs_present', 'public_road', 'time_of_day', 'holiday', 
#         'school_season', 'num_reported_accidents'
#     ]
    
#     # --- Dark Theme Configuration for Altair ---
#     dark_theme = {
#         "config": {
#             "background": "#282a36",
#             "title": {"color": "#f8f8f2"},
#             "style": {
#                 "guide-label": {"fill": "#f8f8f2"},
#                 "guide-title": {"fill": "#f8f8f2"}
#             },
#             "axis": {
#                 "domainColor": "#f8f8f2",
#                 "gridColor": "#44475a",
#                 "tickColor": "#f8f8f2"
#             }
#         }
#     }
#     alt.themes.register("my_dark_theme", lambda: dark_theme)
#     alt.themes.enable("my_dark_theme")

#     # --- 2. Create all charts ---
#     all_charts = []
    
#     # Histograms for continuous features
#     for col in histogram_cols:
#         chart = alt.Chart(data).mark_bar(color='#66c2a5').encode(
#             x=alt.X(f'{col}:Q', bin=alt.Bin(maxbins=40), title=col),
#             y=alt.Y('count()', title='Count'),
#             tooltip=[alt.Tooltip('count()', title='Count')]
#         ).properties(
#             width=280,
#             height=220,
#             title=f'Distribution of {col}'
#         )
#         all_charts.append(chart)
        
#     # Bar charts for categorical/discrete features
#     for col in bar_chart_cols:
#         chart = alt.Chart(data).mark_bar(color='#66c2a5').encode(
#             x=alt.X(f'{col}:N', title=col, sort='-y'), 
#             y=alt.Y('count()', title='Count'),
#             tooltip=[alt.Tooltip(f'{col}:N', title=col), alt.Tooltip('count()', title='Count')]
#         ).properties(
#             width=280,
#             height=220,
#             title=f'Distribution of {col}'
#         )
#         all_charts.append(chart)

#     # --- 3. Combine and display ---
#     if all_charts:
#         combined_chart = alt.concat(
#             *all_charts,
#             columns=n_cols
#         ).resolve_scale(
#             x='independent'
#         )
#         display(combined_chart)
#     else:
#         print("No columns were specified for plotting.")


# plot_distributions_revised(train_df)


# --- Target Variable Distribution ---

# from IPython.display import IFrame, display

# print("Analyzing Target Variable: accident_risk")
# fig_target = px.histogram(
#     train_df,
#     x='accident_risk',
#     marginal='box',
#     nbins=50,
#     title='Distribution of Accident Risk Scores'
# )
# fig_target.update_layout(bargap=0.1, title_x=0.5)

# # 1. Save the figure to a standalone HTML file
# html_filename = 'accident_risk_distribution.html'
# fig_target.write_html(html_filename)

# # 2. Display the saved HTML file within an IFrame
# IFrame(src=html_filename, width='100%', height=600)


numerical_features = train_df.select_dtypes(include=[np.number]).columns

# Calculate skewness for each numerical column
skew_newfeatures = train_df[numerical_features].skew().sort_values(ascending=False)

# Set skewness threshold
skew_limit = 0.75

# Identify numerical columns with unique values 0 and 1
binary_cols = [col for col in numerical_features if train_df[col].nunique() == 2]

# Filter out binary columns and apply skewness threshold
skew_cols = (
    skew_newfeatures
    .drop(index=binary_cols)  # Exclude binary columns
    .to_frame(name='Skew')    # Convert to DataFrame and rename the column to 'Skew'
    # .query('abs(Skew) > @skew_limit')  # Filter for skewness beyond the limit
)

print(skew_cols)


# Set the plot style to a dark theme
# plt.style.use('dark_background')

# # Define the number of columns per row
# n_cols = 4
# filtered_cols = ['num_reported_accidents', 'speed_limit', 'num_lanes', 'curvature']
# n_rows = (len(filtered_cols) + n_cols - 1) // n_cols

# # Create a figure and axes
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
# axes = axes.flatten()

# # Loop and create boxplots
# for i, col in enumerate(filtered_cols):
#     sns.boxplot(y=train_df[col].dropna(), ax=axes[i], color='#66c2a5') 
#     axes[i].set_title(f'Boxplot of {col}', color='white')
#     axes[i].set_xlabel(col, color='white')
#     axes[i].set_ylabel('Value', color='white')
#     axes[i].tick_params(colors='white') # Set tick colors to white

# # Remove any extra empty subplots
# for j in range(len(filtered_cols), len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()

# # Optional: Reset to default style after plotting
# plt.style.use('default')


# ---  2D Density Heatmap for Numerical Interactions ---
# print("Analyzing Numerical Interactions")
# fig_heatmap = go.Figure(go.Histogram2d(
#         x=train_df['speed_limit'],
#         y=train_df['curvature'],
#         z=train_df['accident_risk'],
#         histfunc='avg',
#         colorscale='Inferno'
# ))
# fig_heatmap.update_layout(
#     title='Average Accident Risk by Speed Limit and Road Curvature',
#     xaxis_title='Speed Limit',
#     yaxis_title='Road Curvature',
#     title_x=0.5
# )
# # fig_heatmap.show()

# # 1. Save the heatmap to a standalone HTML file
# html_filename = 'risk_heatmap.html'
# fig_heatmap.write_html(html_filename)

# # 2. Display the saved HTML file within an IFrame
# IFrame(src=html_filename, width='100%', height=600)


##  Box plots: accident_risk by num_reported_accidents
# print("\n---  Box Plots by num_reported_accidents ---")
# if {'num_reported_accidents', 'accident_risk'}.issubset(train_df.columns):
#     fig = px.box(train_df, x='num_reported_accidents', y='accident_risk', title='accident_risk by num_reported_accidents')
#     fig.update_layout(title_x=0.5)
    
#     # --- Workaround ---
#     html_filename = 'box_num_reported_accidents.html'
#     fig.write_html(html_filename)
#     display(IFrame(src=html_filename, width='100%', height=500))


# --- Hierarchical Categorical Analysis with Sunburst Chart ---
# print("Analyzing Categorical Hierarchies")
# fig_sunburst = px.sunburst(
#     train_df.dropna(),
#     path=['road_type', 'lighting', 'weather'],
#     values='num_reported_accidents',
#     title='Breakdown of Accidents by Road Conditions Hierarchy'
# )
# fig_sunburst.update_layout(title_x=0.5)
# # fig_sunburst.show()

# # 1. Save the sunburst chart to a standalone HTML file
# html_filename = 'road_conditions_sunburst.html'
# fig_sunburst.write_html(html_filename)

# # 2. Display the saved HTML file within an IFrame
# # Sunburst charts are often large, so a taller height is used
# IFrame(src=html_filename, width='100%', height=800)


# --- Feature Engineering Pipeline ---

print("Starting feature engineering...")
def feature_engineer(df, te_reference_df):
    """Applies TE, interactions, and label encoding."""
    # --- Cross-Dataset Target Encoding (TE) ---
    categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
    global_mean = te_reference_df['accident_risk'].mean()
    global_std  = te_reference_df['accident_risk'].std()

    for col in categorical_features:
        stats = te_reference_df.groupby(col)['accident_risk'].agg(['mean','std'])
        df[f'{col}_te_mean'] = df[col].map(stats['mean']).fillna(global_mean)
        df[f'{col}_te_std']  = df[col].map(stats['std']).fillna(global_std)

    # # --- Interaction Features ---
    # df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
    # df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    
    return df

# Apply feature engineering to train and test data, using original_df as the TE source
train_fe = feature_engineer(train_df.copy(), original_df)
test_fe  = feature_engineer(test_df.copy(), original_df)



# --- Prepare Final Data for AutoGluon ---

y = train_fe['accident_risk']
X = train_fe.drop(['accident_risk','id'], axis=1, errors='ignore')
X_test = test_fe.drop('id', axis=1, errors='ignore')

# AutoGluon expects target in the same dataframe
train_ag = X.copy()
train_ag['accident_risk'] = y


print('Shape of Train data is : ' , train_ag.shape)
print('Shape of Test data is : ' , X_test.shape)


from autogluon.tabular import TabularDataset, TabularPredictor

train_ag = TabularDataset(train_ag)
X_test = TabularDataset(X_test)
target = 'accident_risk'

# hyperparameters = {
#     "GBM": {},        # LightGBM
#     "XGB": {},        # XGBoost
#     "CAT": {},        # CatBoost
#     # Use FASTAI (or NN_TORCH) for neural nets which can use GPU.
#     # FASTAI is commonly available in autogluon; using ag_args_fit to request GPU.
#     "NN_TORCH": {"ag_args_fit": {"num_gpus": 1}},
#     "FASTAI": {"ag_args_fit": {"num_gpus": 1}},
#     "XT": {},         # ExtraTrees
#     "RF": {},         # RandomForest
# }

predictor_main = TabularPredictor(label=target, eval_metric ='rmse', 
                            problem_type="regression").fit(train_ag, 
                                                           # presets='best_quality',
                                                           presets = 'extreme',
                                                           # auto_stack = True,
                                                           # hyperparameters=hyperparameters,
                                                           time_limit=3600*10.5,
                                                           verbosity=3,
                                                           # excluded_model_types=['KNN'],
                                                           ag_args_fit={'num_gpus': 2}
                                                      )



# extra_hyperparameters_full = {
#     # tree / boosting / ensemble
#     "GBM": {},
#     "XGB": {},
#     "CAT": {},

#     # forests / ensembles
#     "RF": {},
#     "XT": {},

#     # nearest / linear (use LR for linear)
#     "KNN": {},
#     "LR": {},

#     # modern dense / MLP families
#     "REALMLP": {},

#     # neural nets â€” explicitly request GPUs and reasonable epochs
#     "NN_TORCH": {"ag_args_fit": {"num_gpus": 2, "epochs": 30}},
#     "FASTAI": {"ag_args_fit": {"num_gpus": 2, "epochs": 30}},

#     # advanced / extreme families from your version's list
#     "TABPFNV2": {},    # note: uppercase key as shown in your validator output
#     "TABM": {},
#     "TABICL": {},
#     "TABPFNMIX": {},
#     "MITRA": {},

#     # other possible model types shown in your valid list (optional; keep or remove)
#     "FT_TRANSFORMER": {},
#     "AG_TEXT_NN": {},
#     "AG_IMAGE_NN": {},
#     "AG_AUTOMM": {},
#     "FASTTEXT": {},

#     # Keep ensemble/meta keys out of hyperparameters; AutoGluon handles ensembles automatically
# }



# from autogluon.tabular import TabularDataset, TabularPredictor

# # Use this if you want to KEEP the existing predictor and add extra model families
# # extra_hyperparameters = {
# #     "GBM": {}, "XGB": {}, "CAT": {},
# #     "RF": {}, "XT": {}, "LINEAR": {}, "KNN": {},
# #     "NN_TORCH": {"ag_args_fit": {"num_gpus": 2, "epochs": 30}},
# #     "FASTAI": {"ag_args_fit": {"num_gpus": 2, "epochs": 30}},
# #     # include other families your AG version supports; remove unsupported keys if errors occur
# #     "TabPFNv2": {}, "RealMLP": {}, "TabM": {}, "TabNet": {}, "TabTransformer": {}
# # }

# predictor_path = "/kaggle/working/AutogluonModels/ag-20251029_084948"
# predictor = TabularPredictor.load(predictor_path, verbosity=3)

# # quick check of existing models
# # print("Existing models:", predictor.get_model_names())

# # add models â€” note: NO `presets` here
# predictor.fit_extra(
#     hyperparameters=extra_hyperparameters,
#     time_limit=int(60*60*10.5),   # seconds
#     ag_args_fit={'num_gpus': 2},  # global GPU request for models that honor it
#     num_bag_sets=1,               # number of bagging model sets to create (increase for stronger ensembling)
#     num_stack_levels=1,           # enable stacking (levels >0)
#     save_bag_folds=True,          # keep OOF predictions (useful for stacking/refit)
#     verbosity=3
# )

# # inspect results
# predictor.fit_summary(show_plot=True)



# results = predictor_main.fit_summary()
# print(results)


predictor_main.leaderboard()



y_pred = predictor_main.predict(X_test)



# Feature importance
# importances = predictor_main.feature_importance(train_ag)
# print("Feature importances:")
# print(importances.head(14))


# Plot feature importances

# plt.figure(figsize=(12, 10))
# sns.barplot(
#     x=importances['importance'],
#     y=importances.index,
#     palette='viridis'
# )
# plt.title('Feature Importances')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()



# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y_pred
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)
# submission.to_csv('submissionV1.csv', index=False)

# Display the first few rows of the predictions
print(submission.head(10))

