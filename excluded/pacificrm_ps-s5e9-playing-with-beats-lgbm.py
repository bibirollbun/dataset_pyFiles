import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna
import shap
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
train_ = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
train


train = train.drop(["id"],axis =1)
train_



train = pd.concat([train, train_], ignore_index=True)
train


epsilon = 1e-6 
train['Acoustic_to_Instrumental_Ratio'] = train['AcousticQuality'] / (train['InstrumentalScore'] + 0.001)
train['Energy_x_Rhythm'] = train['Energy'] * train['RhythmScore']
train['Loudness_per_Second'] = train['AudioLoudness'] / (train['TrackDurationMs'] / 1000)
train['Danceability_Proxy'] = train['Energy'] * train['RhythmScore'] * (train['AudioLoudness'] - train['AudioLoudness'].min())
train['Vocal_Prominence'] = train['VocalContent'] / (train['InstrumentalScore'] + 0.001)
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + epsilon)
train['MoodRhythm'] = train['MoodScore'] * train['RhythmScore']
train['PerformanceIntensity'] = train['LivePerformanceLikelihood'] * train['AudioLoudness']
train['MoodAcoustic'] = train['MoodScore'] * train['AcousticQuality']


test['Acoustic_to_Instrumental_Ratio'] = test['AcousticQuality'] / (test['InstrumentalScore'] + 0.001)
test['Energy_x_Rhythm'] = test['Energy'] * test['RhythmScore']
test['Loudness_per_Second'] = test['AudioLoudness'] / (test['TrackDurationMs'] / 1000)
test['Danceability_Proxy'] = test['Energy'] * test['RhythmScore'] * (test['AudioLoudness'] - test['AudioLoudness'].min())
test['Vocal_Prominence'] = test['VocalContent'] / (test['InstrumentalScore'] + 0.001)
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + epsilon)
test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']


train.info()


train.isna().sum()


train.describe()


sns.set(style="whitegrid")
colors = sns.color_palette("husl", len(train.columns))

plt.figure(figsize=(25, 20))
for i, (col, color) in enumerate(zip(train.columns, colors), 1):
    plt.subplot(len(train.columns) // 3 + 1, 3, i)
    sns.histplot(train[col], bins=15, kde=True, color=color)
    plt.title(f'Distribution of {col}', color=color)
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(train.columns):
    plt.subplot(len(train.columns) // 2 + 1, 2, i + 1)
    color = 'purple' if i % 2 == 0 else 'orange'
    sns.boxplot(x=train[col], color=color)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


corr_matrix = train.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


correlation_matrix = train.corr()
correlation_with_response = correlation_matrix['BeatsPerMinute'].sort_values(ascending=False)
print(correlation_with_response)


# --- Logarithmic Transformation ---
skewed_features = [ 'Danceability_Proxy','Vocal_Prominence','Energy_Acoustic_Ratio','RhythmScore',
                   'Loudness_per_Second']

# Apply the log1p transformation on train
for feature in skewed_features:
    train[feature] = np.log1p(train[feature])

# Apply the log1p transformation on test
for feature in skewed_features:
    test[feature] = np.log1p(test[feature])

print("Applied log transformation features.")


# --- Power Transformation ---
power_features = [ 'LivePerformanceLikelihood','MoodAcoustic','TrackDurationMs','InstrumentalScore',
                   'Energy_x_Rhythm','Energy']

power_transformer = PowerTransformer(method='yeo-johnson')

# # Fit the transformer on the training data and then transform it
train[power_features] = power_transformer.fit_transform(train[power_features])

# # Apply the transformation on test
test[power_features] = power_transformer.transform(test[power_features])

print("Applied power transformation features.")


train.columns


numerical_features= [ 'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 
                     'LivePerformanceLikelihood', 'MoodScore','TrackDurationMs', 'Energy',
                     'Acoustic_to_Instrumental_Ratio', 'Energy_x_Rhythm','Loudness_per_Second', 'Danceability_Proxy',
                     'Vocal_Prominence','Energy_Acoustic_Ratio', 'MoodRhythm', 'PerformanceIntensity','MoodAcoustic']
# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('robust_scale', RobustScaler(), numerical_features)
    ],
    remainder='passthrough'
)


X_raw =  train.drop(['BeatsPerMinute'], axis =1)
y = train['BeatsPerMinute']


X_test_raw = test.drop(['id'],axis =1)
X_test_raw


pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(X_raw)
X_test_processed = pipeline.transform(X_test_raw)


X = pd.DataFrame(X_processed, columns =numerical_features )
X_test = pd.DataFrame(X_test_processed, columns =numerical_features )
X


def objective(trial):
    """
    Objective function for Optuna to minimize.
    It performs k-fold cross-validation and returns the average RMSE.
    """
    # Optuna suggests hyperparameters for each trial
    param = {
        'objective': 'regression_l2',  
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 100.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    # Iterate through each fold
    # X and y should be your preprocessed feature and target dataframes
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)
    
    # Return the mean of the RMSE scores across all folds
    return np.mean(rmse_scores)



# # Create an Optuna study object and run the optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100) 

# # Print the best hyperparameters and their score
# print('Best trial:')
# print(f'  Value (Avg. RMSE): {study.best_value}')
# print('  Params:')
# for key, value in study.best_params.items():
#     print(f'    {key}: {value}')


best_params={'n_estimators': 1307,          
    'learning_rate': 0.06748646663694965,
    'num_leaves': 9,
    'max_depth': 8,
    'min_child_samples': 460,
    'subsample': 0.8278422593438073,
    'colsample_bytree': 0.9945980359117047,
    'reg_alpha': 0.00042155616855236246,
    'reg_lambda': 0.02788082464431462,
    'verbose': -1,
    'n_jobs': -1,
    
            
}


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Use the best hyperparameters found by Optuna
# best_params = study.best_params 


# Initialize KFold
kf = KFold(n_splits=15, shuffle=True, random_state=67)

# Array to store test predictions from each fold
test_predictions = np.zeros(len(X_test))

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize and train the model for this fold
    model = lgb.LGBMRegressor(**best_params, random_state=67)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Predict on the test data and add to the predictions array
    # We divide by n_splits here to directly compute the mean
    test_predictions += model.predict(X_test) / kf.get_n_splits()

# --- Create and save the submission file ---
submission_df = pd.DataFrame({
    'id': sample_submission['id'],
    'BeatsPerMinute': test_predictions
})

submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file created by averaging predictions from 15 folds.")
submission_df.head()


# Create a dataframe for feature importance from the final model
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value','Feature'])

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Feature Importance (from final model)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()



# It's recommended to set js_init for nicer plots in notebooks
shap.initjs()

# Create a SHAP TreeExplainer object, which is optimized for tree-based models like LightGBM
explainer = shap.TreeExplainer(model)

# For performance reasons, we'll calculate SHAP values on a sample of the data
# A sample of 10,000 is often sufficient for a good summary plot
X_sample = X.sample(n=10000, random_state=42)

# Calculate SHAP values for the sample
shap_values = explainer.shap_values(X_sample)

# --- Create the SHAP summary plot ---
print("Generating SHAP summary plot...")
plt.title('SHAP Summary Plot')
shap.summary_plot(shap_values, X_sample)

