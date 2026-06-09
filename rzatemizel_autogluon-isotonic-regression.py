!pip install autogluon.tabular


#basics
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import time

#preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, quantile_transform


#statistics
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import randint

#feature engineering
from sklearn.feature_selection import mutual_info_regression


#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import set_config


#algorithms
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR


#model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, cross_val_predict, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, make_scorer, mean_squared_log_error
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


#stacking
from sklearn.ensemble import StackingRegressor

random_state = 42



# Read the data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col=[0])
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col=[0])
original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
original.index.names = ['id']




original.head()


#train = pd.concat([train, original], ignore_index=True)

# reserved for pipeline
pipe_data = train.copy()
pipe_test = test.copy()
pipe_original = original.copy()

# use for preliminary analysis
train_df = train.copy()
test_df = test.copy()
original_df = original.copy()
train_df.head()


#descriptive statistics
train_df.describe().T


numerical_features = [feature for feature in train.columns if not feature  == "BeatsPerMinute"]

target = "BeatsPerMinute"



#sample data for eda
sampled_df = train_df.sample(frac = 0.01)


fig, ax = plt.subplots(3, 3, figsize=(40, 40))
for var, subplot in zip(numerical_features, ax.flatten()):
    sns.scatterplot(x=var, y='BeatsPerMinute',  data=sampled_df, ax=subplot, hue = 'BeatsPerMinute' )
    


# Display correlations between numerical features and target on heatmap.

sns.set(font_scale=1.1)
correlation_train = sampled_df.corr()
mask = np.triu(correlation_train.corr())
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_train,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            square=True,
            mask=mask,
            linewidths=1,
            cbar=False);


# Mutual Information score
y_sampled = sampled_df.BeatsPerMinute
mutual_df = sampled_df[numerical_features ]

mutual_info = mutual_info_regression(mutual_df, y_sampled, random_state=random_state)

mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns = ["Numerical_Feature_MI"] )
mutual_info.style.background_gradient("cool")


import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))

# Create the histogram plot
sns.histplot(x='BeatsPerMinute', data=train_df, bins=20, kde=True, color='skyblue', edgecolor='black')

# Add title and labels
plt.title('BeatsPerMinute Distribution')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')

# Show the plot
plt.show()


from autogluon.tabular import TabularDataset, TabularPredictor


train_data = TabularDataset(pipe_data)
label = 'BeatsPerMinute'


predictor = TabularPredictor(
    label=label,
    eval_metric='root_mean_squared_error'
)

predictor.fit(
    train_data,
    presets='extreme', # Keep this to get the powerful ensembles
    num_bag_folds=10,       # <<-- Key Change: Explicitly request 10 folds
    time_limit=32400 
)


# 3) See the model zoo & performance
predictor.leaderboard()


oof_preds_autogluon = predictor.predict_oof()


preds_test =  predictor.predict(pipe_test)


oof_preds_autogluon.to_csv('oof_predictions_autogluon.csv', index=True, header=True)
print("OOF predictions saved to oof_predictions_autogluon.csv")

# 2. Save the test predictions
# These are the predictions on the unseen test set.
# If you need to create a submission file, you'd format this file accordingly.
preds_test.to_csv('test_predictions_autogluon.csv', index=True, header=True)
print("Test predictions saved to test_predictions_autogluon.csv")



#from cir_model import CenteredIsotonicRegression
from sklearn.isotonic import IsotonicRegression

oof_preds_from_ensemble = oof_preds_autogluon

# --- Step 2: Calculate and Print the "Before" Metric ---
# This is your baseline ensemble performance on the CV.
rmse_before = np.sqrt(mean_squared_error(pipe_data.BeatsPerMinute, oof_preds_from_ensemble))
print(f"\nEnsemble OOF RMSE (Before Calibration): {rmse_before:.6f}")

# --- Step 3: Train the Isotonic Regressor ---
# It learns how to correct the systematic errors found in the OOF predictions.
print("Training the Isotonic Regression calibrator...")
ir_calibrator = IsotonicRegression(out_of_bounds="clip")
ir_calibrator.fit(oof_preds_from_ensemble, pipe_data.BeatsPerMinute)

# --- Step 4: Calibrate the OOF predictions to get the "After" predictions ---
calibrated_oof_preds = ir_calibrator.predict(oof_preds_from_ensemble)

# --- Step 5: Calculate and Print the "After" Metric ---
# This shows the performance after calibration.
rmse_after = np.sqrt(mean_squared_error(pipe_data.BeatsPerMinute, calibrated_oof_preds))
print(f"Ensemble OOF RMSE (After Isotonic Calibration): {rmse_after:.6f}")

# --- Step 6: Report the Improvement ---
improvement = rmse_before - rmse_after
print(f"\nImprovement in CV score: {improvement:.6f}")
if improvement > 0:
    print("Isotonic Regression improved the CV score! Applying to test predictions.")
else:
    print("Isotonic Regression did not improve the CV score. Proceeding with caution.")

print("-" * 50)

# --- Step 7: Calibrate the final test predictions for submission ---
# 'preds_test' is from your original code: 
# preds_test = ensemble_pipe.predict(submission_predictions_df[selected_models])
print("Calibrating the final test predictions...")
calibrated_preds_test = ir_calibrator.predict(preds_test)

# --- Step 8: Create the new submission file ---
print("Creating the new submission file: submission_isotonic.csv")
output_isotonic = pd.DataFrame({'id': pipe_test.index,
                                'BeatsPerMinute': calibrated_preds_test})

output_isotonic.to_csv('submission.csv', index=False)
display(output_isotonic.head())

# --- Step 9: (Optional) Visualize the calibration mapping ---
# This plot helps you understand what correction the calibrator learned.
plt.figure(figsize=(10, 8))
# Use a subset for the scatter plot to avoid overplotting
sample_indices = np.random.choice(len(oof_preds_from_ensemble), 5000, replace=False)
plt.scatter(oof_preds_from_ensemble[sample_indices], pipe_data.BeatsPerMinute.iloc[sample_indices], 
            alpha=0.1, s=10, label='Original OOF Preds vs True (Sampled)')

# Plot the calibration line on sorted predictions for a smooth curve
sorted_indices = np.argsort(oof_preds_from_ensemble)
plt.plot(oof_preds_from_ensemble[sorted_indices], calibrated_oof_preds[sorted_indices], 
         color='red', linewidth=2, label='Isotonic Calibration Function')

plt.xlabel("Ensemble's Predicted BPM")
plt.ylabel("True BPM")
plt.title("Isotonic Regression Calibration Map")
plt.legend()
plt.grid(True)
plt.show()


