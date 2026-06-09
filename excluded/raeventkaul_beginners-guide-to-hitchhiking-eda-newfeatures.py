# --- 0. Import Necessary Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import warnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Modelling libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib 

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Set a consistent style for all plots
sns.set_style('whitegrid')
import scipy.stats as stats

# Visibility of full data
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


print("--- 1. Loading and Inspecting Data ---")
df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id',axis=1)

print("Training Data Head:")
display(df_train.head())
print("\nTraining Data Info:")
df_train.info()
print("\nTraining Data Statistical Summary:")
display(df_train.describe())
print("-" * 50)



# In this section, we visualize the data to understand distributions and relationships.
print("\n--- 2. Exploratory Data Analysis (EDA) ---")

# Visualize the distribution of the target variable 'accident_risk'
plt.figure(figsize=(10, 6))
sns.histplot(df_train['accident_risk'], kde=True, bins=30, color='blue')
plt.title('Distribution of Accident Risk (Target Variable)')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()

# Logic: A Q-Q plot helps us see if the data distribution is similar to a normal distribution.
# If the points fall along the red line, the data is normally distributed.
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
stats.probplot(df_train['accident_risk'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Accident Risk')

plt.tight_layout()
plt.show()


print("--- Plotting Numerical Feature Distributions ---")

# Identify all numerical columns, excluding any ID columns
numerical_features = df_train.select_dtypes(include=np.number)

for col in numerical_features:
    plt.figure(figsize=(10, 6))
    # Plotting a histogram with a Kernel Density Estimate curve
    sns.histplot(df_train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

print("-" * 50)


categorical_features = df_train.select_dtypes(include=['object', 'bool']).columns

# Determine the grid size (2 columns, and as many rows as needed)
n_cols = 3
n_features = len(categorical_features)
n_rows = math.ceil(n_features / n_cols)

# Create the figure and a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 6))
# Flatten the axes array to make it easy to iterate over
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    # Calculate the frequency of each category
    value_counts = df_train[col].value_counts()
    
    # Select the current subplot
    ax = axes[i]
    
    # Create the pie chart on the current subplot
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
           startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title(f'Distribution of {col}', fontsize=14)
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

# If there's an odd number of plots, hide the last empty subplot
if n_features % 2 != 0:
    axes[-1].set_visible(False)

# Adjust layout to prevent titles from overlapping
plt.tight_layout()
plt.show()

print("-" * 50)


# Visualize categorical features against the target variable using boxplots
categorical_features = df_train.select_dtypes(include='object').columns
print("\nPlotting categorical features vs. accident risk...")
for col in categorical_features:
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=df_train[col], y=df_train['accident_risk'])
    plt.title(f'Distribution of Accident Risk across {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
print("-" * 50)


# This section handles data cleaning and creates new features from existing ones.
print("\n--- 3. Data Cleaning & Feature Engineering ---")

print("\nChecking for Missing Values:")
print(df_train.isnull().sum())

# The initial check showed no missing values, so no imputation is needed.
print("No missing values found in the dataset.")


# Feature Engineering: 

def engineer_all_features(df_train):
    """
    Applies a comprehensive set of feature engineering steps to the input DataFrame.
    This includes creating interactions, polynomial features, human-factor proxies, and ratios.
    """
    print("--- Starting Comprehensive Feature Engineering ---")

    # General Ideas
    # a) Rush Hour Feature 
    # Logic We hypothesize that morning and evening times are rush hours and might correlate with risk.
    df_train['is_rush_hour'] = df_train['time_of_day'].apply(lambda x: 1 if x in ['morning', 'evening'] else 0)
    
    # b) Adverse Conditions Score
    # Logic: Certain combinations of lighting and weather are more dangerous than others.
    def get_adverse_score(row):
        score = 0
        # Bad lighting conditions
        if row['lighting'] in ['night']:
            score += 2
        if row['lighting'] in ['dim']:
            score += 1
        # Bad weather conditions
        if row['weather'] in ['rainy', 'foggy']:
            score += 1
        return score
    
    df_train['adverse_conditions'] = df_train.apply(get_adverse_score, axis=1)
    
    
    # c) Speed x Curvature Interaction
    # Logic: High speed on a sharp curve is more dangerous than the same speed on a straight road.
    df_train['speed_x_curvature'] = df_train['speed_limit'] * df_train['curvature']
    
    
    # d) Lane Density Proxy
    # Logic: More lanes on a slower urban road imply different traffic dynamics than on a highway.
    # A higher value might suggest more potential for complex interactions and side-swipes.
    df_train['lane_density'] = df_train['num_lanes'] / df_train['speed_limit']
    
    # e) Logarithmic Transformation 
    # Logic: The impact of some variables might follow a "diminishing returns" pattern. For instance, 
    # The difference in risk between 0 and 1 prior accidents is likely much more significant than the difference between 10 and 11. 
    #A log transform compresses the range of large numbers and expands the range of small numbers, helping the model capture this effect
    df_train['log_num_accidents'] = np.log1p(df_train['num_reported_accidents'])
    
    # f) High Risk Flag: A binary flag for unambiguously dangerous road conditions.
    df_train['high_risk_combo'] = ((df_train['curvature'] > 0.5) & \
                                   (df_train['speed_limit'] >= 60)).astype(int)
    
    
    # --- 2. Polynomial Features ---
    # These can help models capture non-linear relationships.
    
    print("\n--- Creating Polynomial Features ---")
    
    # a) Speed Limit Squared
    # Logic: Kinetic energy increases with the square of velocity, which often correlates with accident severity and risk.
    df_train['speed_limit_sq'] = df_train['speed_limit']**2
    
    # b) Curvature Squared
    # Logic: The effect of a curve on vehicle dynamics might be non-linear.
    df_train['curvature_sq'] = df_train['curvature']**2
    
    # c) Curvature cubed
    # Logic: The effect of a curve on vehicle dynamics might be non-linear.
    df_train['curvature_cube'] = df_train['curvature']**3
    
    # d ) Centripetal Force Proxy
    # Logic: Models the physical force on a vehicle during a turn. Risk increases exponentially with speed.
    df_train['centripetal_force_proxy'] = (df_train['speed_limit']**2) * df_train['curvature']
    
    
    # --- 3. Human Factor & Environmental Proxies ---
    
    # a) "False Sense of Security" Index
    # Logic: Perfectly straight, well-lit, clear roads with high speed limits might encourage
    # drivers to be less attentive or to speed excessively.
    df_train['false_security_index'] = ((df_train['curvature'] < 0.1) & \
                                       (df_train['lighting'] == 'daylight') & \
                                       (df_train['weather'] == 'clear')).astype(int)
    
    # b) "Holiday Night" Effect
    # Logic: Driving at night during a holiday might increase the probability of encountering
    # recreational or impaired drivers.
    df_train['holiday_night_effect'] = ((df_train['time_of_day'] == 'night') & \
                                       (df_train['holiday'] == True)).astype(int)
    
    # c) School Season Rush Hour
    # Logic: Rush hour during the school season might have different traffic patterns (school buses, parents)
    # than rush hour when school is out.
    df_train['school_rush_hour'] = df_train['is_rush_hour'] * df_train['school_season']
    
    #d) Hidden Danger: Flags seemingly "safe" roads (straight, daylight) that have signs, hinting at a non-obvious risk.
    df_train['hidden_danger_warning'] = ((df_train['road_signs_present'] == True) & \
                                             (df_train['curvature'] < 0.1) & \
                                             (df_train['lighting'] == 'daylight')).astype(int)
    
    # e) Unusual Complexity: Flags rural roads with an unusually high number of lanes.
    df_train['unusual_rural_complexity'] = ((df_train['road_type'] == 'rural') & \
                                                (df_train['num_lanes'] > 2)).astype(int)
    
    # f) Night Driving Load: The cognitive load of night driving is amplified by complex roads (more lanes, sharper curves).
    df_train['night_driving_load'] = ((df_train['time_of_day'] == 'night').astype(int) * \
                                          (df_train['num_lanes'] + df_train['curvature'] * 5))
    # g) Afternoon Slump Risk: The post-lunch dip in alertness is most dangerous on roads that already demand high concentration.
    df_train['afternoon_slump_risk'] = ((df_train['time_of_day'] == 'afternoon').astype(int) * \
                                            df_train['speed_x_curvature'])
    
    
    # --- 4. Ratio Features ---
    
    # a) Accident History Per Lane
    # Logic: A road with 10 past accidents and 5 lanes is different from a road with 10 accidents and 1 lane.
    # We add 1 to the denominator to avoid division by zero, just in case.
    df_train['accidents_per_lane'] = df_train['num_reported_accidents'] / (df_train['num_lanes'] + 1)
    
    # b) Historical Risk vs. Speed Limit
    # Logic: A road with many past accidents despite a low speed limit is a major red flag.
    df_train['historical_risk_density'] = df_train['num_reported_accidents'] / df_train['speed_limit']
    
    # c) Curvature Per Lane: Higher curvature in lesser lanes
    df_train['curvature_per_lane'] = df_train['curvature'] / (df_train['num_lanes'] + 1)
    
    # --- Displaying the New Features ---
    print("\n--- Sample of the DataFrame with New Features ---")
    display(df_train.sample(10))
    return df_train
df_train=engineer_all_features(df_train)


# --- 3. Using Other Techniques---
# PCA can be used to reduce dimensionality and create new, uncorrelated components from numerical data.
print("\n--- Setting up Preprocessing Pipeline ---")

# Separate features from the target
X = df_train.drop(['accident_risk'], axis=1)
y = df_train['accident_risk']


# Prepare the data for modeling by encoding and scaling features.
print("\n--- 4. Preprocessing and Feature Selection ---")


# Separate features (X) from the target variable (y)
X = df_train.drop('accident_risk', axis=1)
y = df_train['accident_risk']

# Identify original numerical features 
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

print("\n--- a. Numerical features---")
print(f"\n {list(numerical_features)}")

print("\n--- b. Categorical features for OneHotEncoding ---")
print(f"{list(categorical_features)}")



# Create a preprocessing pipeline for numerical features that scales and then applies PCA.
# n_components=0.97 means PCA will retain the number of components needed to explain 97% of the variance.
# - StandardScaler: Scales numerical features to have a mean of 0 and standard deviation of 1.

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    #,('pca', PCA(n_components=0.97))
])

# Create the master preprocessor using ColumnTransformer
# - OneHotEncoder: Converts categorical features into a numerical format.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        # We still need to one-hot encode the original categorical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# This 'preprocessor_with_pca' object is now ready to be used as the first step in your main XGBoost model pipeline.
print("\nâœ… Preprocessing pipeline has been created successfully.")
print("This can now be passed to your main model pipeline for training.")



# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training and validation sets:")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print("-" * 50)



from sklearn import preprocessing

# Train the baseline model and analyze which features are most important.
print("\n--- 5. XGBoost Model Training & Feature Importance ---")

# Create the full pipeline by combining the preprocessor and the XGBoost regressor
# Assuming 'preprocessor_with_pca' and other variables are already defined
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
tree_method='hist', device='cuda',eval_metric='rmse'
  ))])

# Train the baseline model on the training data
xgb_pipeline.fit(X_train, y_train)

# --- Feature Importance Extraction ---
print("Extracting feature importances from the baseline model...")

# Get feature names after one-hot encoding
ohe_feature_names = xgb_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numerical_features, ohe_feature_names])

importances = xgb_pipeline.named_steps['regressor'].feature_importances_

# Now the lengths will match, and this DataFrame creation will work
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot the top 15 most important features
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Feature Importances from Baseline XGBoost Model')
plt.tight_layout()
plt.show()

y_pred = xgb_pipeline.predict(X_val)
print(f"mse score {mean_squared_error(y_val,y_pred)}")
print("-" * 50)


# import optuna
# from sklearn.model_selection import cross_val_score
# # --- 1. Define the Objective Function for Optuna ---
# # This function tells Optuna how to evaluate a set of hyperparameters.
# def objective(trial):
#     # Define the hyperparameters to search.
#     # Note the 'regressor__' prefix, which tells the pipeline to pass
#     # these parameters to the 'regressor' step (our XGBRegressor).
#     params = {
#         'regressor__tree_method': 'hist',
#         'regressor__device': 'cuda',
#         'regressor__n_estimators': trial.suggest_int('n_estimators', 400, 2000),
#         'regressor__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'regressor__max_depth': trial.suggest_int('max_depth', 4, 10),
#         'regressor__subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'regressor__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'regressor__reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
#         'regressor__reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
#     }

#     # Set the parameters for the current trial
#     xgb_pipeline.set_params(**params)

#     # Perform cross-validation
#     scores = cross_val_score(
#         xgb_pipeline, X_train, y_train,
#         cv=5,
#         scoring='neg_root_mean_squared_error',
#         n_jobs=-1
#     )

#     # Return the mean of the cross-validation scores.
#     # Optuna minimizes the return value, so we return the positive RMSE.
#     return -np.mean(scores)

# # --- 2. Create and Run the Optuna Study ---
# # The direction is 'minimize' because we want to minimize the RMSE.
# study = optuna.create_study(direction='minimize', study_name="XGB_Pipeline_Optimization")

# # Start the optimization process
# # n_trials=30 means Optuna will test 30 different combinations.
# study.optimize(objective, n_trials=30)


# # --- 3. Print the Best Results ---
# print("\n--- Optuna Tuning Complete ---")
# print("Best trial:")
# trial = study.best_trial

# print(f"  Value (RMSE): {trial.value:.5f}")

# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


best_params = {
    'n_estimators': 1070, 
    'learning_rate':  0.01398742944434015, 
    'max_depth': 7, 'subsample': 0.9086801010758693, 
    'colsample_bytree': 0.7555821753489727, 
    'reg_alpha': 0.00820231059861153, 
    'reg_lambda': 0.09923092739138861,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda'
}


# Evaluate the best model found during tuning on the unseen validation data.
print("\n--- 7. Final Model Evaluation ---")

# The best model is automatically refit on the entire training data
# best_model = random_search.best_estimator_

# We define the pipeline again, but this time we will populate it with the best parameters.
best_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(**best_params))
])
best_model.fit(X_train, y_train)


# Make predictions on the validation set
y_pred = best_model.predict(X_val)

# Calculate performance metrics
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print(f"Validation Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Validation R-squared (R2 Score): {r2:.4f}")

# Visualize the model's performance by plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.3, color='purple')
# Plot a line representing perfect predictions
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--', color='red', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Final Model: Actual vs. Predicted Accident Risk')
plt.legend()
plt.show()
print("-" * 50)


# --- 8. Save the Final Model ---
print("\n--- 8. Saving the Final Model ---")

# Use joblib to save the entire pipeline object to a file
model_filename = 'XGB_optuma.joblib'
joblib.dump(best_model, model_filename) 


# Create a submission file for a competition.
print("\n--- 8. Submission File Generation ---")
print("NOTE: A 'test.csv' file is required to generate predictions for submission.")
print("The following is a template for how to generate the submission file.")

df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = df_test['id']

df_test_engineered = engineer_all_features(df_test) # Apply same feature engineering
df_test_engineered = df_test_engineered.drop(['id'], axis=1) # Drop id
# X_test_transformed = xgb_pipeline.named_steps['preprocessor'].transform(df_test_engineered)

test_predictions = best_model.predict(df_test_engineered)
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("Generated final submission.csv")

