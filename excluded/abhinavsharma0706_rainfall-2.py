# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Verify shapes
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)


# Display sample data
print("\nTrain Data Sample:")
display(train_data.head())

print("\nTest Data Sample:")
display(test_data.head())


# Display sample data
print("\nTrain Data Sample (Tail):")
display(train_data.tail())

print("\nTest Data Sample (Tail):")
display(test_data.tail())


# Display information about the DataFrames
print("\nTrain Data Info:")
train_data.info()

print("\nTest Data Info:")
test_data.info()


# Display NULL Counts
print("Train NULL Count:",train_data.isnull().sum().sum())
print("Test NULL Count:",test_data.isnull().sum().sum())


# Display sample data
print("\nTrain Data Columns: ", train_data.columns)
print("\nTest Data Columns: ", test_data.columns)


# Display information about the DataFrames
print("\nTrain Data Describe: ")
train_data.describe().T


# Display information about the DataFrames
print("\nTest Data Describe:")
test_data.describe().T


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = []

# fill the missing data as columns' mean
test_data['winddirection'].fillna(test_data["winddirection"].mean(), inplace=True)


# Analysis of all NUMERICAL features
# Define a custom color palette
custom_palette = ['#f1b963', '#c4c1e0']

# Define numerical features
variables = [col for col in train_data.columns if col in numerical_variables]

# Function to create and display plots for a single numerical variable
def create_variable_plots(variable):
    sns.set_theme(style='whitegrid')

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train_data.copy()
    test_temp = test_data.copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    combined_data = pd.concat([train_temp, test_temp])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    sns.boxplot(data=combined_data, x=variable, y="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot of {variable}")

    # Histogram
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of {variable} [Train, Test]")
    axes[1].legend()

    # Adjust spacing and show
    plt.tight_layout()
    plt.show()

# Perform univariate analysis for each numerical variable
for variable in variables:
    create_variable_plots(variable)


# Target variable
plt.figure(figsize=(6,4))
sns.countplot(x=train_data['rainfall'], palette='coolwarm')
plt.title('Rainfall Class Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Count')
plt.show()


# KDE plot for Feature-Target Relationship
plt.figure(figsize=(14, 10))
for i, col in enumerate(numerical_variables, 1):
    plt.subplot(3, 4, i)
    sns.kdeplot(train_data[col][train_data['rainfall'] == 1], color='red', label='Rainfall: 1')
    sns.kdeplot(train_data[col][train_data['rainfall'] == 0], color='blue', label='Rainfall: 0')
    plt.title(f'Distribution of {col} by Rainfall')
    plt.legend()
plt.tight_layout()
plt.show()


# Pairplot Analysis: Visualising relationships between variables
sns.pairplot(train_data[numerical_variables + ['rainfall']], hue='rainfall', palette='coolwarm', diag_kind='kde')
plt.show()


# Filter data based on rainfall
rain_data = train_data[train_data['rainfall'] > 0]
no_rain_data = train_data[train_data['rainfall'] == 0]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

# First wind rose plot (rain)
ax1 = axes[0]
ax1.set_theta_direction(-1)
ax1.set_theta_offset(np.pi / 2.0)
bars1 = ax1.bar(
    np.deg2rad(rain_data['winddirection']),
    rain_data['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color="b"  
)
ax1.set_title('Wind Speed and Direction with Rain')

# Second wind rose plot (no rain)
ax2 = axes[1]
ax2.set_theta_direction(-1)
ax2.set_theta_offset(np.pi / 2.0)
bars2 = ax2.bar(
    np.deg2rad(no_rain_data['winddirection']),
    no_rain_data['windspeed'],
    width=np.pi/8,
    bottom=0.0,
    color="r"  
)
ax2.set_title('Wind Speed and Direction without Rain')

plt.tight_layout()
plt.show()



# Correlation heatmap
def plot_correlation_heatmap(data, title, annot_size=12):
    plt.figure(figsize=(12, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, annot_kws={"size": annot_size},cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap - {title}', fontsize=16)
    plt.show()

plot_correlation_heatmap(train_data, "Train Data")


def preprocess_weather_data(data):
    # Feature Engineering
    #data["temp_range"] = data["maxtemp"] - data["mintemp"]
    #data["temp_from_dewpoint"] = data["temparature"] - data["dewpoint"] # **?
    data["dew_humidity"] = data["dewpoint"] * data["humidity"] # ***
    #data["wind_speeddirection"] = data["windspeed"] * data["winddirection"]
    data["cloud_windspeed"] = data["cloud"] * data["windspeed"] # ***
    data["cloud_to_humidity"] = data["cloud"] / data["humidity"]
    data["temp_to_sunshine"] = data["sunshine"] / data["temparature"] # ***
    
    data['humidity_cloud_interaction'] = data['humidity'] * data['cloud'] # ***
    #data['wind_temp_interaction'] = data['windspeed'] * data['temparature']
    #data['sunshine_wind_interaction'] = data['sunshine'] + data['windspeed'] # *
    #data['cloud_humidity_ratio'] = data['cloud'] / (data['humidity'] + 1)  # Avoid division by zero
    #data['pressure_temp_ratio'] = data['pressure'] / (data['temparature'] + 1)  # Avoid division by zero
    #data['cloud_wind_ratio'] = data['cloud'] / (data['windspeed'] + 1)  # Avoid division by zero
    #data['cloud_coverage_rate'] = data['cloud'] / 100  # Normalize to 0-1 range
    data['weather_severity'] = (data['cloud'] * data['humidity']) / (data['pressure'] * (data['sunshine'] + 1))
    data['cloud_sun_ratio'] = data['cloud'] / (data['sunshine'] + 1) # ***
    #data["cloud_sunshine_+"] = data["cloud"] + data["sunshine"]
    #data["cloud_sunshine_-"] = data["cloud"] - data["sunshine"]
    data["dew_humidity/sun"] = data["dewpoint"] * data["humidity"] / (data['sunshine'] + 1)
    data["dew_humidity_+"] = data["dewpoint"] * data["humidity"]
    
    #data['cloud_humidity_plus'] = data['cloud'] + data["humidity"] 
    #data['cloud_humidity_sunshine_plus'] = data['cloud'] + data["humidity"] + data['sunshine']
    #data['cloud_sunshine_*'] = data['cloud'] * data['sunshine']
    data['humidity_sunshine_*'] = data["humidity"] * data['sunshine']

    data["cloud_humidity/pressure"] = (data["cloud"] * data["humidity"]) / data["pressure"]
    
    # Drop redundant columns
    #data = data.drop(columns=["maxtemp", "mintemp"])
    data = data.drop(columns=["winddirection","temparature","maxtemp"])

    # Extract temporal features
    data['month'] = ((data['day'] - 1) // 30 + 1).clip(upper=12)
    data['season'] = data['month'].apply(lambda x: 1 if 3 <= x <= 5  # Spring
                                         else 2 if 6 <= x <= 8  # Summer
                                         else 3 if 9 <= x <= 11  # Autumn
                                         else 0)  # Winter
    data = data.drop(columns=["month"])

    return data

# Apply to train and test datasets
train_data = preprocess_weather_data(train_data)
test_data = preprocess_weather_data(test_data)


plot_correlation_heatmap(train_data, "Train Data", 7)



# Select features and target variable
X = train_data.drop(['rainfall', 'id'], axis=1)
y = train_data['rainfall']
X_test = test_data.drop(['id'], axis=1)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42,max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=100, hidden_layer_sizes=(50,50))
}

# Train models using StratifiedKFold CV
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
auc_scores = {}
roc_curves = {}

for name, model in models.items():
    oof_preds = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    auc_score = roc_auc_score(y, oof_preds)
    auc_scores[name] = auc_score
    fpr, tpr, _ = roc_curve(y, oof_preds)
    roc_curves[name] = (fpr, tpr, auc_score)
    print(f"{name}: AUC = {auc_score:.4f}")


# Plot ROC curves
plt.figure(figsize=(8, 6))
for model_name, (fpr, tpr, auc_score) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# Plot AUC scores
plt.figure(figsize=(8, 6))
sns.barplot(x=list(auc_scores.keys()), y=list(auc_scores.values()))
plt.xticks(rotation=45)
plt.ylabel("AUC Score")
plt.title("Model AUC Score Comparison")
plt.show()


# Find the best model
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]
print(f"Best Model: {best_model_name} with AUC = {auc_scores[best_model_name]:.4f}")


# Check if the model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

# Create a DataFrame to combine feature names and their importance values
feature_df = pd.DataFrame({
    'Feature': train_data.drop(['rainfall', 'id'], axis=1).columns,
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title(f"{importance_type} ({best_model_name})")
plt.show()


# Select the best model based on AUC
best_model_name = max(auc_scores, key=auc_scores.get)
best_model = models[best_model_name]

# Check if the model has feature_importances_ attribute
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
    importance_type = 'Feature Importance'
else:
    # For logistic regression, use coefficients as importance
    feature_importance = np.abs(best_model.coef_[0])
    importance_type = 'Coefficient Magnitudes'

# Create a DataFrame to combine feature names and their importance values
feature_df = pd.DataFrame({
    'Feature': train_data.drop(['rainfall', 'id'], axis=1).columns,
    'Importance': feature_importance
})

# Sort the features by importance in descending order
feature_df = feature_df.sort_values(by='Importance', ascending=False)


# List of top N features to try 
top_feature_counts = [5, 6, 7, 8, 9, 10, 12, 13, 14, 15]

# Variables to track the best AUC and corresponding top features
best_auc = 0
best_top_n = 0
best_oof_preds_top = None

# Loop over different top feature counts
for top_n in top_feature_counts:
    # Select the top N features
    top_features = feature_df.head(top_n)['Feature']
    
    # Prepare the data with the selected top N features
    X_top = X[:, [train_data.drop(['rainfall', 'id'], axis=1).columns.get_loc(col) for col in top_features]]
    X_test_top = X_test[:, [train_data.drop(['rainfall', 'id'], axis=1).columns.get_loc(col) for col in top_features]]

    # Retrain the best model using the top N features
    best_model.fit(X_top, y)

    # Make predictions and calculate AUC for the top N features
    oof_preds_top = np.zeros(len(y))
    for train_idx, val_idx in skf.split(X_top, y):
        X_train, X_val = X_top[train_idx], X_top[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_model.fit(X_train, y_train)
        oof_preds_top[val_idx] = best_model.predict_proba(X_val)[:, 1]

    # Calculate and print AUC score for top N features model
    auc_score_top = roc_auc_score(y, oof_preds_top)
    print(f"AUC for top {top_n} features model: {auc_score_top:.4f}")
    
    # Track the best AUC and corresponding features
    if auc_score_top > best_auc:
        best_auc = auc_score_top
        best_top_n = top_n
        best_oof_preds_top = oof_preds_top

# Now plot the feature importance for the set with the highest AUC
best_features = feature_df.head(best_top_n)



# Plotting the feature importance for the best model with the highest AUC
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=best_features)
plt.title(f"{importance_type} for Top {best_top_n} Features ({best_model_name})")
plt.show()

# You can also display the best AUC score
print(f"Best AUC: {best_auc:.4f} for top {best_top_n} features")


# Predictions for the test set with the top 10 features
test_preds = best_model.predict_proba(X_test_top)[:, 1]

# Submission
submission = pd.DataFrame({'id': test_data['id'], 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as 'submission.csv'.")


submission

