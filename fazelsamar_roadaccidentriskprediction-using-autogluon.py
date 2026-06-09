# --- Import Libraries ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sweetviz")

import pandas as pd
import numpy as np
import os
import sweetviz as sv
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.io as pio
import altair as alt
from IPython.display import IFrame, display
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from autogluon.tabular import TabularDataset, TabularPredictor



# --- Configure Environment ---

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# Set a default template for plotly for a clean, professional look
px.defaults.template = "plotly_dark"
pio.renderers.default = "kaggle"



# --- Load Data ---

train_df = pd.read_csv('train.csv', index_col='id')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')
original_df = pd.read_csv('synthetic_road_accidents_100k.csv')

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

    # --- Interaction Features ---
    df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1e-6)
    
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


# !pip uninstall -y lightgbm
# !apt update && apt install -y build-essential cmake libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-opencl-dev
# !pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON



from autogluon.tabular import TabularDataset, TabularPredictor

train_ag = TabularDataset(train_ag)
X_test = TabularDataset(X_test)
target = 'accident_risk'

predictor_main = TabularPredictor(label=target, eval_metric ='rmse', 
                            problem_type="regression").fit(train_ag, 
                                                           presets='best_quality',
                                                            # presets='medium_quality_faster_train',   # lighter than 'best_quality'
                                                           # presets = 'extreme',
                                                           # auto_stack = True,
                                                           time_limit=3600*3,
                                                           verbosity=3,
                                                           excluded_model_types=['KNN','NN_TORCH'],  # avoid torch NN which may OOM on 4GB
                                                           ag_args_fit={'num_gpus': 1}
                                                      )



# results = predictor_main.fit_summary()
# print(results)


predictor_main.leaderboard()



y_pred = predictor_main.predict(X_test)



# Feature importance
importances = predictor_main.feature_importance(train_ag)
print("Feature importances:")
print(importances.head(14))


# Plot feature importances

plt.figure(figsize=(12, 10))
sns.barplot(
    x=importances['importance'],
    y=importances.index,
    palette='viridis'
)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y_pred
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)
submission.to_csv('submissionV1.csv', index=False)

# Display the first few rows of the predictions
print(submission.head(10))

