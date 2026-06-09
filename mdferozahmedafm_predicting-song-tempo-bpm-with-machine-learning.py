# Import essential libraries
import numpy as np      # For numerical operations
import pandas as pd     # For data manipulation and analysis
import os               # For interacting with the operating system

# Walk through the Kaggle input directory and print file paths
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# Import essential libraries
import numpy as np                  # For numerical operations
import pandas as pd                 # For data manipulation
import matplotlib.pyplot as plt     # For data visualization
import seaborn as sns               # For advanced statistical plots

# Set display options for better visualization
pd.set_option('display.max_columns', None)  # Show all columns in output
sns.set(style="whitegrid", palette="muted", font_scale=1.1)  # Set seaborn plot style

# Load dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  
print("Shape of dataset:", df.shape)  # Print number of rows and columns

# Display first 5 rows
df.head()



# Display dataset information
df.info()  # Shows column names, non-null counts, and data types

# Statistical summary of dataset
df.describe(include='all')  # Summary statistics for all columns (numeric + categorical)

# Check for missing values
missing = df.isnull().sum()  # Count missing values per column
missing = missing[missing > 0].sort_values(ascending=False)  # Filter columns with missing values

if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')  # Visualize missing values per feature
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("No missing values found.")

# Check for duplicate rows
print("Duplicate rows:", df.duplicated().sum())  # Count duplicate rows



# Print descriptive statistics of numerical features
print("\n--- Descriptive Statistics ---")
print(df.describe().T)  # Transposed table for easier readability



# Set target variable
target = "BeatsPerMinute"

# Plot distribution of target variable
plt.figure(figsize=(8,5))
sns.histplot(df[target], kde=True, bins=30)  # Histogram with KDE overlay
plt.title(f"Distribution of {target}")
plt.show()



# Select all numeric features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove("id")  # Remove 'id' column if present

# Plot histograms for all numeric features
df[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))  # Multiple histograms in a grid
plt.suptitle("Feature Distributions")  # Overall title for all plots
plt.show()



# Plot correlation heatmap for numeric features
plt.figure(figsize=(10,8))
sns.heatmap(df[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")  # Heatmap with annotations
plt.title("Correlation Heatmap")
plt.show()

# Display correlation of each feature with the target
print("\n--- Correlation with Target ---")
print(df.corr()[target].sort_values(ascending=False))  # Sorted correlations



# Plot scatter plots of numeric features vs target
for col in num_features:
    if col != target:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df[target])  # Scatter plot for each feature against target
        plt.title(f"{col} vs {target}")
        plt.show()



# Plot boxplots for numeric features to detect outliers
for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])  # Boxplot to visualize distribution and outliers
    plt.title(f"Outliers in {col}")
    plt.show()



#df['RhythmScore'] = np.where(df['RhythmScore'] < lower, lower,
#                             np.where(df['RhythmScore'] > upper, upper, df['RhythmScore']))
#df['AudioLoudness'] = np.where(df['AudioLoudness'] < lower, lower,
#                             np.where(df['AudioLoudness'] > upper, upper, df['AudioLoudness']))

features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality']

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Winsorization (capping)
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))


# Feature Engineering

# Convert track duration from milliseconds to minutes
df['TrackDurationMin'] = df['TrackDurationMs'] / 60000  

# Create ratio features
df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-5)  # Avoid division by zero
df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)  

# Interaction features
df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']



# Import necessary libraries for modeling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Prepare features and target
X = df.drop(columns=['id', target])  # Drop 'id' and target column
y = df[target]                       # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter search space
param_dist = {
    'max_depth': [3, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Initialize XGBoost regressor
xgb = XGBRegressor(random_state=42)

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the model to training data
random_search.fit(X_train, y_train)

# Display the best hyperparameters
print("Best parameters:", random_search.best_params_)



# Get the best model from randomized search
best_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))  # Root Mean Squared Error
print("RÂ² Score:", r2_score(y_test, y_pred))                        # R-squared metric



# Plot feature importance from the best XGBoost model
plt.figure(figsize=(10,6))
plt.barh(X.columns, best_model.feature_importances_)  # Horizontal bar plot of feature importances
plt.title("XGBoost Feature Importance")
plt.show()



# Load test dataset
import pandas as pd
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# Feature engineering on test set (same as training)
test_df['TrackDurationMin'] = test_df['TrackDurationMs'] / 60000
test_df['Energy_Acoustic_Ratio'] = test_df['Energy'] / (test_df['AcousticQuality'] + 1e-5)
test_df['Vocal_Instrument_Balance'] = test_df['VocalContent'] / (test_df['InstrumentalScore'] + 1e-5)
test_df['MoodRhythm'] = test_df['MoodScore'] * test_df['RhythmScore']
test_df['PerformanceIntensity'] = test_df['LivePerformanceLikelihood'] * test_df['AudioLoudness']
test_df['RhythmEnergy'] = test_df['RhythmScore'] * test_df['Energy']
test_df['MoodAcoustic'] = test_df['MoodScore'] * test_df['AcousticQuality']

# Select features used in the trained model
train_features = best_model.get_booster().feature_names 
X_test_final = test_df[train_features]  

# Predict on the test set
y_pred_test = best_model.predict(X_test_final)

# Create submission dataframe
output = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_BPM": y_pred_test
})

# Save predictions to CSV
output.to_csv("test_predictions.csv", index=False)

# Display confirmation and first rows
print("Predictions saved to test_predictions.csv")
print(output.head())


