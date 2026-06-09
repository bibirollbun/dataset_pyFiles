import pandas as pd

# Set the option to display all columns
pd.set_option('display.max_columns', None)

# Read the CSV files into pandas DataFrames

d = pd.read_csv('/kaggle/input/nwds-k/train.csv', sep=',')
n = pd.read_csv('/kaggle/input/nwds-k/test.csv', sep=',' )
s = pd.read_csv('/kaggle/input/nwds-k/sample_solution.csv', sep=',' )  

# Display the DataFrame with all columns
d.head()


# Filter for strikes == 2
d = d[d['strikes'] == 2]

# Display basic info about filtered dataset
print("Shape of filtered dataset:", d.shape)
print("\nFirst few rows:")
display(d.head())

# Optional: Show count of rows with strikes == 2
print(f"\nTotal rows where strikes = 2: {len(d)}")


#drop column strikes from dataset d
d = d.drop(columns=['strikes'])
# drop column is_strike from dataset n
d = d.drop(columns=['is_strike'])
# drop column strikes from dataset n
n = n.drop(columns=['strikes'])

d.shape, n.shape, s.shape


# mean of column k in dataset d
d['k'].mean()


def generate_new_metrics(df):
    """
    Generate new physics-related pitching metrics that incorporate 'arm_angle'.
    The input dataframe must contain the following columns:
    'sz_top', 'sz_bot', 'pfx_x', 'pfx_z', 'arm_angle', 
    'release_speed', 'release_pos_x', 'release_extension', 
    'release_pos_z', 'release_spin_rate', 'spin_axis'
    
    Returns the dataframe with new feature columns added.
    """
    # Calculate the horizontal component of the release speed, influenced by arm angle.
    df['velocity_horizontal'] = df['release_speed'] * np.cos(np.radians(df['arm_angle']))
    
    # Calculate the vertical component of the release speed.
    df['velocity_vertical'] = df['release_speed'] * np.sin(np.radians(df['arm_angle']))
    
    # Create a metric that adjusts spin rate based on the arm angle.
    df['spin_effect'] = df['release_spin_rate'] * np.sin(np.radians(df['arm_angle']))
    
    # Create a metric that combines release extension and arm angle to capture effective pitch extension.
    df['extension_control'] = df['release_extension'] * np.cos(np.radians(df['arm_angle']))
    
    # Create a metric that adjusts the strike zone based on the vertical distance between sz_top and sz_bot.
    df['strike_zone_adjustment'] = (df['sz_top'] - df['sz_bot']) * np.sin(np.radians(df['arm_angle']))

    # Create a metric that captures the vertical height of batter's strike zone.
    df['sz_height'] = df['sz_top'] - df['sz_bot']
    
    return df


import numpy as np
d = generate_new_metrics(d)
n = generate_new_metrics(n)

d.shape, n.shape


# Fast xgboodt model with no tuning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
import xgboost as xgb

# Drop unnecessary columns
d.drop(columns=['index', 'pitch_name'], inplace=True)
n.drop(columns=['index', 'pitch_name'], inplace=True)

# Impute missing values with 0 for bat_speed and swing_length
d['bat_speed'].fillna(0, inplace=True)
d['swing_length'].fillna(0, inplace=True)
n['bat_speed'].fillna(0, inplace=True)
n['swing_length'].fillna(0, inplace=True)

# Convert logical columns to integers
for base in ['on_3b', 'on_2b', 'on_1b']:
    d[base] = d[base].astype(int)
    n[base] = n[base].astype(int)

# Encode categorical features
d['inning_topbot'] = d['inning_topbot'].map({'Top': 1, 'Bot': 0})
n['inning_topbot'] = n['inning_topbot'].map({'Top': 1, 'Bot': 0})

for col in ['stand', 'p_throws']:
    d[col] = d[col].map({'R': 1, 'L': 0})
    n[col] = n[col].map({'R': 1, 'L': 0})

# One-hot encode pitch_type
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
pitch_type_encoded_d = encoder.fit_transform(d[['pitch_type']])
pitch_type_encoded_n = encoder.transform(n[['pitch_type']])

pitch_type_cols = encoder.get_feature_names_out(['pitch_type'])
d = pd.concat([d.drop('pitch_type', axis=1), pd.DataFrame(pitch_type_encoded_d, columns=pitch_type_cols, index=d.index)], axis=1)
n = pd.concat([n.drop('pitch_type', axis=1), pd.DataFrame(pitch_type_encoded_n, columns=pitch_type_cols, index=n.index)], axis=1)

# Separate features and target
X = d.drop('k', axis=1)
y = d['k']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model with early stopping
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=10,
          verbose=False)

# Validation log loss
val_pred = model.predict_proba(X_val)[:, 1]
print(f"Validation Log Loss: {log_loss(y_val, val_pred):.4f}")

# Retrain on full data with best iteration
best_iter = model.best_iteration
params = model.get_params()
params['n_estimators'] = best_iter + 1  # Override n_estimators with best iteration
model_full = xgb.XGBClassifier(**params)
model_full.fit(X, y)

# Predict on test data
test_pred = model_full.predict_proba(n)[:, 1]

# Create submission file
s['k'] = test_pred
s.to_csv('submission.csv', index=False)


# Plot feature importance
import matplotlib.pyplot as plt

# Define the new features created by generate_new_metrics
generated_features = [
    'velocity_horizontal', 
    'velocity_vertical', 
    'spin_effect', 
    'extension_control', 
    'strike_zone_adjustment',
    'sz_height'
]

# Get feature importance from the full model
importance_dict = {X.columns[i]: model_full.feature_importances_[i] for i in range(len(X.columns))}
sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

# Create color list based on feature type
colors = ['#2ecc71' if feature in generated_features else '#3498db' 
          for feature in sorted_importance.keys()]

# Create plot
plt.figure(figsize=(15, 7))
bars = plt.bar(range(len(sorted_importance)), 
               list(sorted_importance.values()), 
               color=colors)

# Customize plot
plt.xticks(range(len(sorted_importance)), 
           list(sorted_importance.keys()), 
           rotation=45, 
           ha='right')
plt.title('XGBoost Feature Importance\n(Blue: Original Features, Green: Generated Features)', 
         pad=20)
plt.xlabel('Features')
plt.ylabel('Importance Score')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Original Features'),
    Patch(facecolor='#2ecc71', label='Generated Features')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

# Print top 10 most important features with type indication
print("\nTop 10 Most Important Features:")
for idx, (feature, importance) in enumerate(sorted_importance.items()):
    if idx < 10:
        feature_type = "Generated" if feature in generated_features else "Original"
        print(f"{feature} ({feature_type}): {importance:.4f}")


# Calculate log loss stepwise
import numpy as np
from tqdm import tqdm

# Get sorted features by importance
sorted_features = list(sorted_importance.keys())
feature_scores = []
feature_counts = []

# Initialize lists to store results
scores = []
n_features = []

# Iterate through features, adding one at a time
for i in tqdm(range(1, len(sorted_features) + 1)):
    # Select top i features
    selected_features = sorted_features[:i]
    
    # Prepare datasets with selected features
    X_selected = X[selected_features]
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with same parameters
    model_i = xgb.XGBClassifier(**params)
    model_i.fit(X_train, y_train)
    
    # Compute log loss
    val_pred = model_i.predict_proba(X_val)[:, 1]
    current_score = log_loss(y_val, val_pred)
    
    # Store results
    scores.append(current_score)
    n_features.append(i)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(n_features, scores, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Validation Log Loss')
plt.title('Log Loss vs Number of Features')
plt.grid(True)

# Find optimal number of features
optimal_n_features = n_features[np.argmin(scores)]
optimal_score = min(scores)

plt.axvline(x=optimal_n_features, color='r', linestyle='--', 
            label=f'Optimal: {optimal_n_features} features')
plt.axhline(y=optimal_score, color='r', linestyle='--')

plt.legend()
plt.tight_layout()
plt.show()

# Print optimal features
print(f"\nOptimal number of features: {optimal_n_features}")
print(f"Optimal log loss: {optimal_score:.4f}")
print("\nOptimal feature set:")
optimal_features = sorted_features[:optimal_n_features]
for i, feature in enumerate(optimal_features, 1):
    feature_type = "Generated" if feature in generated_features else "Original"
    print(f"{i}. {feature} ({feature_type})")

