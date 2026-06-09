# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Create a copy of train_df for correlation analysis
train_corr = train_df.copy()

# Encode the categorical variable 'Sex'
label_encoder = LabelEncoder()
train_corr['Sex_encoded'] = label_encoder.fit_transform(train_corr['Sex'])

# Calculate correlations with target using the encoded feature
print("Correlation with Calories:")
correlations = train_corr.drop('Sex', axis=1).corr()['Calories'].sort_values(ascending=False)
print(correlations)

# Count the number of each sex category
print("\nSex distribution:")
print(train_df['Sex'].value_counts())

# Feature distributions
print("\nFeature distributions:")
print(train_df.describe())

# Plot distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Calories'], kde=True)
plt.title('Distribution of Calories')
plt.savefig('calories_distribution.png')
plt.close()

# Visualize correlations using the encoded feature
plt.figure(figsize=(10, 8))
sns.heatmap(train_corr.drop('Sex', axis=1).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Boxplot of calories by sex
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='Calories', data=train_df)
plt.title('Calories by Sex')
plt.savefig('calories_by_sex.png')
plt.close()

# Pairplot of the main numerical features
features_to_plot = ['Age', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
plt.figure(figsize=(15, 10))
sns.pairplot(train_df[features_to_plot].sample(5000), diag_kind='kde')  # Sample to make plot faster
plt.savefig('pairplot.png')
plt.close()

# Examine relationships between important features and target
feature_cols = ['Duration', 'Heart_Rate', 'Weight', 'Body_Temp']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(feature_cols):
    sns.scatterplot(x=feature, y='Calories', data=train_df.sample(5000), hue='Sex', ax=axes[i])
    axes[i].set_title(f'{feature} vs Calories')
    
plt.tight_layout()
plt.savefig('feature_relationships.png')
plt.close()

# Check for outliers in numerical features
plt.figure(figsize=(15, 10))
train_df.drop(['id', 'Sex'], axis=1).boxplot(figsize=(15, 10))
plt.title('Boxplots of Numerical Features')
plt.savefig('outliers.png')
plt.close()

# Create additional scatter plots for relationships with duration (likely important)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Duration', y='Calories', data=train_df.sample(5000), hue='Sex')
plt.title('Duration vs Calories by Sex')
plt.savefig('duration_calories.png')
plt.close()

# Calculate average calories by duration, age groups, and sex
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[19, 30, 40, 50, 60, 80], labels=['20-30', '31-40', '41-50', '51-60', '61+'])
duration_groups = pd.cut(train_df['Duration'], bins=[0, 10, 20, 30], labels=['0-10', '11-20', '21-30'])

# Calculate average calories by duration and sex
avg_by_duration_sex = train_df.groupby(['Duration', 'Sex'])['Calories'].mean().reset_index()
pivot_duration_sex = avg_by_duration_sex.pivot(index='Duration', columns='Sex', values='Calories')

plt.figure(figsize=(10, 6))
pivot_duration_sex.plot()
plt.title('Average Calories by Duration and Sex')
plt.xlabel('Duration')
plt.ylabel('Average Calories')
plt.savefig('avg_calories_duration_sex.png')
plt.close()

# Check for possible interactions between features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Heart_Rate', y='Calories', data=train_df.sample(5000), hue='Body_Temp', palette='viridis')
plt.title('Heart Rate vs Calories, colored by Body Temperature')
plt.savefig('heart_rate_calories_temp.png')
plt.close()

# See if the relationship is different across age groups
plt.figure(figsize=(15, 10))
g = sns.FacetGrid(train_df.sample(10000), col='Age_Group', hue='Sex', col_wrap=3)
g.map(sns.scatterplot, 'Duration', 'Calories')
g.add_legend()
plt.savefig('calories_by_age_duration.png')
plt.close()

# Examine any differences in distributions between train and test
features_to_compare = [col for col in train_df.columns if col != 'Calories' and col != 'id' and col != 'Age_Group']

for feature in features_to_compare:
    plt.figure(figsize=(10, 6))
    if feature == 'Sex':
        # For categorical features, use countplot
        sns.countplot(x=feature, data=pd.concat([
            train_df[feature].to_frame().assign(dataset='train'),
            test_df[feature].to_frame().assign(dataset='test')
        ]))
    else:
        # For numerical features, use histplot
        sns.histplot(train_df[feature], label='train', alpha=0.6, kde=True)
        sns.histplot(test_df[feature], label='test', alpha=0.6, kde=True)
        plt.legend()
    
    plt.title(f'Distribution of {feature} in Train vs Test')
    plt.savefig(f'dist_compare_{feature}.png')
    plt.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import time

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Process data
def preprocess_data(df, is_train=True):
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    result['Sex'] = le.fit_transform(result['Sex'])
    
    # Feature engineering based on our analysis
    # 1. Interaction between Duration and Heart_Rate (both highly correlated with Calories)
    result['Duration_Heart'] = result['Duration'] * result['Heart_Rate']
    
    # 2. Interaction between Duration and Body_Temp
    result['Duration_Temp'] = result['Duration'] * result['Body_Temp']
    
    # 3. BMI calculation (Weight/Height^2)
    result['BMI'] = result['Weight'] / ((result['Height']/100) ** 2)
    
    # 4. Age groups as categorical
    result['Age_Group'] = pd.cut(result['Age'], bins=[19, 30, 40, 50, 60, 80], 
                                labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 5. Heart rate to max heart rate ratio (rough estimate of exercise intensity)
    # Max HR formula: 220 - Age
    result['HR_Intensity'] = result['Heart_Rate'] / (220 - result['Age'])
    
    # Define feature columns and target
    if is_train:
        X = result.drop(['id', 'Calories'], axis=1)
        y = result['Calories']
        return X, y
    else:
        return result.drop(['id'], axis=1)

# Preprocess the train and test data
X, y = preprocess_data(train_df, is_train=True)
X_test = preprocess_data(test_df, is_train=False)

# Define categorical features for CatBoost
cat_features = ['Sex', 'Age_Group']

# Define the CatBoost model
def rmsle(pred, actual):
    return np.sqrt(mean_squared_log_error(actual, pred))

# Define evaluation function for cross-validation
def rmsle_cv_score(model, X=X, y=y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmsle_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Make sure predictions are positive (for log calculation)
        y_pred = np.maximum(y_pred, 0.1)
        
        rmsle_val = rmsle(y_pred, y_val)
        rmsle_scores.append(rmsle_val)
    
    return np.mean(rmsle_scores)

# Basic CatBoost model parameters
params = {
    'loss_function': 'RMSE',  # We'll optimize for RMSE during training
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 100
}

# Train the model and evaluate with cross-validation
print("Training CatBoost model with cross-validation...")
start_time = time.time()

model = CatBoostRegressor(**params)
cv_score = rmsle_cv_score(model, X, y, n_folds=5)

print(f"5-fold CV RMSLE: {cv_score:.6f}")
print(f"Training took {time.time() - start_time:.2f} seconds")

# Train final model on all data
print("\nTraining final model on all data...")
final_model = CatBoostRegressor(**params)
final_model.fit(X, y, cat_features=cat_features, verbose=100)

# Feature importance
feature_importance = final_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Display features from top to bottom in descending order
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nTop 10 important features:")
print(importance_df.head(10))

# Make predictions on test set
print("\nMaking predictions on test set...")
test_predictions = final_model.predict(X_test)

# Ensure all predictions are positive
test_predictions = np.maximum(test_predictions, 0.1)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': test_predictions
})

submission.to_csv('catboost_submission.csv', index=False)
print("Submission file created.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, Pool
import time
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Process data with enhanced feature engineering
def preprocess_data(df, is_train=True):
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    result['Sex'] = le.fit_transform(result['Sex'])
    
    # Basic feature engineering based on our analysis
    # 1. Interaction between Duration and Heart_Rate (top feature from first model)
    result['Duration_Heart'] = result['Duration'] * result['Heart_Rate']
    
    # 2. Interaction between Duration and Body_Temp
    result['Duration_Temp'] = result['Duration'] * result['Body_Temp']
    
    # 3. BMI calculation (Weight/Height^2)
    result['BMI'] = result['Weight'] / ((result['Height']/100) ** 2)
    
    # 4. Age groups as categorical
    result['Age_Group'] = pd.cut(result['Age'], bins=[19, 30, 40, 50, 60, 80], 
                               labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 5. Heart rate to max heart rate ratio (rough estimate of exercise intensity)
    # Max HR formula: 220 - Age
    result['HR_Intensity'] = result['Heart_Rate'] / (220 - result['Age'])
    
    # 6. Squared terms for top features (to capture non-linear relationships)
    result['Duration_Squared'] = result['Duration'] ** 2
    result['Heart_Rate_Squared'] = result['Heart_Rate'] ** 2
    
    # 7. Interaction between all three top features
    result['Duration_Heart_Temp'] = result['Duration'] * result['Heart_Rate'] * result['Body_Temp']
    
    # 8. Weight to Height ratio
    result['Weight_Height_Ratio'] = result['Weight'] / result['Height']
    
    # 9. Age and Heart Rate interaction
    result['Age_Heart'] = result['Age'] * result['Heart_Rate']
    
    # 10. Sex and Duration interaction
    result['Sex_Duration'] = result['Sex'] * result['Duration']
    
    # Define feature columns and target
    if is_train:
        X = result.drop(['id', 'Calories'], axis=1)
        y = result['Calories']
        return X, y
    else:
        return result.drop(['id'], axis=1)

# Calculate RMSLE
def rmsle(pred, actual):
    return np.sqrt(mean_squared_log_error(actual, pred))

# Preprocess the train and test data
X, y = preprocess_data(train_df, is_train=True)
X_test = preprocess_data(test_df, is_train=False)

# Define categorical features for CatBoost
cat_features = ['Sex', 'Age_Group']

# Create bins for target stratification in cross-validation
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')

# Use Optuna for hyperparameter optimization
def objective(trial):
    # Define hyperparameters to tune
    params = {
        'iterations': 1000,  # Fixed number of iterations for consistency
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_seed': 42,
        'verbose': 0
    }
    
    # 5-fold cross-validation with stratification by binned target
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y_bins):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create CatBoost Pool objects with categorical features
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # Train model
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True, verbose=0)
        
        # Predict
        y_pred = model.predict(val_pool)
        # Ensure positive predictions for RMSLE
        y_pred = np.maximum(y_pred, 0.1)
        
        # Calculate RMSLE
        rmsle_val = rmsle(y_pred, y_val)
        scores.append(rmsle_val)
    
    return np.mean(scores)

# Run hyperparameter optimization
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)  # Adjust n_trials based on time constraints

# Get the best parameters
best_params = study.best_params
best_params['iterations'] = 2000  # Increase iterations for final model
best_params['verbose'] = 100
best_params['random_seed'] = 42

print(f"Best parameters: {best_params}")
print(f"Best RMSLE: {study.best_value:.6f}")

# Train final model with best parameters
print("\nTraining final model with best parameters...")
final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y, cat_features=cat_features, verbose=100)

# Feature importance
feature_importance = final_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 10))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_tuned.png')
plt.close()

print("\nTop 15 important features:")
print(importance_df.head(15))

# Make predictions on test set
print("\nMaking predictions on test set...")
test_predictions = final_model.predict(X_test)

# Ensure all predictions are positive
test_predictions = np.maximum(test_predictions, 0.1)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': test_predictions
})

submission.to_csv('tuned_catboost_submission.csv', index=False)
print("Submission file created.")

# Analyze predictions
plt.figure(figsize=(10, 6))
plt.hist(test_predictions, bins=50, alpha=0.7)
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Calories')
plt.ylabel('Frequency')
plt.savefig('prediction_distribution.png')
plt.close()

# Calculate learning curves for best model
print("\nTraining model with learning curves...")
eval_set = [(X, y)]
learning_model = CatBoostRegressor(**best_params)
learning_model.fit(X, y, cat_features=cat_features, eval_set=eval_set, verbose=False)

# Plot learning curve
evals_result = learning_model.get_evals_result()
plt.figure(figsize=(10, 6))
plt.plot(evals_result['learn']['RMSE'])
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.grid(True)
plt.savefig('learning_curve.png')
plt.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from catboost import CatBoostRegressor, Pool
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Process data with enhanced feature engineering
def preprocess_data(df, is_train=True):
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    result['Sex'] = le.fit_transform(result['Sex'])
    
    # Basic feature engineering from original model
    result['Duration_Heart'] = result['Duration'] * result['Heart_Rate']
    result['Duration_Temp'] = result['Duration'] * result['Body_Temp']
    result['BMI'] = result['Weight'] / ((result['Height']/100) ** 2)
    result['Age_Group'] = pd.cut(result['Age'], bins=[19, 30, 40, 50, 60, 80], 
                              labels=[0, 1, 2, 3, 4]).astype(int)
    result['HR_Intensity'] = result['Heart_Rate'] / (220 - result['Age'])
    result['Duration_Squared'] = result['Duration'] ** 2
    result['Heart_Rate_Squared'] = result['Heart_Rate'] ** 2
    result['Duration_Heart_Temp'] = result['Duration'] * result['Heart_Rate'] * result['Body_Temp']
    result['Weight_Height_Ratio'] = result['Weight'] / result['Height']
    result['Age_Heart'] = result['Age'] * result['Heart_Rate']
    result['Sex_Duration'] = result['Sex'] * result['Duration']
    
    # ENHANCED FEATURES BASED ON TOP PREDICTORS
    
    # 1. More variants of the top interactions
    result['Duration_Heart_Squared'] = result['Duration_Heart'] ** 2
    result['Duration_Heart_Cubed'] = result['Duration_Heart'] ** 3
    result['HR_Intensity_Squared'] = result['HR_Intensity'] ** 2
    
    # 2. Log transformations of important features
    result['Log_Duration'] = np.log1p(result['Duration'])
    result['Log_Heart_Rate'] = np.log1p(result['Heart_Rate'])
    result['Log_Duration_Heart'] = np.log1p(result['Duration_Heart'])
    
    # 3. Physiological composite features
    result['Efficiency_Factor'] = result['Duration_Heart'] / (result['BMI'] + 1)
    result['HR_Efficiency'] = result['Heart_Rate'] / (result['Age'] + 20)  # +20 to avoid division issues with young people
    
    # 4. Additional interaction terms
    result['BMI_Heart'] = result['BMI'] * result['Heart_Rate']
    result['Weight_Duration'] = result['Weight'] * result['Duration']
    result['Age_Duration'] = result['Age'] * result['Duration']
    result['Heart_Temp'] = result['Heart_Rate'] * result['Body_Temp']
    
    # 5. Specialized ratios and differences
    result['Duration_Per_kg'] = result['Duration'] / result['Weight']
    result['Heart_Per_Temp'] = result['Heart_Rate'] / result['Body_Temp']
    result['HR_Reserve'] = (220 - result['Age']) - result['Heart_Rate']  # Heart rate reserve
    
    # 6. Cubic features for the top predictors
    result['Duration_Cubed'] = result['Duration'] ** 3
    result['HR_Intensity_Cubed'] = result['HR_Intensity'] ** 3
    
    # Define feature columns and target
    if is_train:
        X = result.drop(['id', 'Calories'], axis=1)
        y = result['Calories']
        return X, y
    else:
        return result.drop(['id'], axis=1)

# Calculate RMSLE
def rmsle(pred, actual):
    return np.sqrt(mean_squared_log_error(actual, pred))

# Preprocess the train and test data
X, y = preprocess_data(train_df, is_train=True)
X_test = preprocess_data(test_df, is_train=False)

# Define categorical features for CatBoost
cat_features = ['Sex', 'Age_Group']

# Create bins for target stratification in cross-validation
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')

# Enhanced Optuna objective with expanded parameter space
def objective(trial):
    # Define hyperparameters to tune with expanded search space
    params = {
        'iterations': 1000,  # Fixed for optimization trials
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
        'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': 42,
        'verbose': 0
    }
    
    # 5-fold cross-validation with stratification by binned target
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y_bins):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create CatBoost Pool objects with categorical features
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # Train model with early stopping
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, 
                 use_best_model=True, verbose=0)
        
        # Predict and ensure positive values for RMSLE
        y_pred = model.predict(val_pool)
        y_pred = np.maximum(y_pred, 0.1)
        
        # Calculate RMSLE
        rmsle_val = rmsle(y_pred, y_val)
        scores.append(rmsle_val)
    
    return np.mean(scores)

# Run hyperparameter optimization with more trials
print("Starting enhanced hyperparameter optimization...")
study = optuna.create_study(direction='minimize', study_name="calorie_prediction")
study.optimize(objective, n_trials=15)  # Increased from 10 to 15 trials

# Get the best parameters
best_params = study.best_params
best_params['iterations'] = 3000  # Increased from 2000 to 3000 for final model
best_params['verbose'] = 100
best_params['random_seed'] = 42

print(f"Best parameters: {best_params}")
print(f"Best RMSLE: {study.best_value:.6f}")

# Ensemble approach: Train multiple models with slightly different seeds
print("\nTraining ensemble of models with best parameters...")
num_models = 5
models = []
predictions = []

for i in range(num_models):
    print(f"\nTraining model {i+1}/{num_models}...")
    model_params = best_params.copy()
    model_params['random_seed'] = 42 + i  # Different seed for each model
    
    model = CatBoostRegressor(**model_params)
    model.fit(X, y, cat_features=cat_features, verbose=100)
    models.append(model)
    
    # Make predictions
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0.1)  # Ensure positive predictions
    predictions.append(preds)

# Create ensemble prediction (average of all models)
ensemble_predictions = np.mean(predictions, axis=0)

# Feature importance from the first model (for analysis)
feature_importance = models[0].get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\nTop 20 important features:")
print(importance_df.head(20))

# Visualize feature importance
plt.figure(figsize=(12, 10))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_enhanced.png')
plt.close()

# Create submission file with ensemble predictions
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': ensemble_predictions
})

submission.to_csv('submission.csv', index=False)
print("Enhanced submission file created.")

# Analyze predictions
plt.figure(figsize=(10, 6))
plt.hist(ensemble_predictions, bins=50, alpha=0.7)
plt.title('Distribution of Ensemble Predictions')
plt.xlabel('Predicted Calories')
plt.ylabel('Frequency')
plt.savefig('ensemble_prediction_distribution.png')
plt.close()

# Optional: Save the individual models for future use
import pickle
for i, model in enumerate(models):
    with open(f'catboost_model_{i+1}.pkl', 'wb') as f:
        pickle.dump(model, f)
print("Models saved for future use.")

