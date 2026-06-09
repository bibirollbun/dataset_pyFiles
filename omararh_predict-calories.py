# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()



# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

playground_series_s5e5_path = kagglehub.competition_download('playground-series-s5e5')

print('Data source import complete.')



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

# Display basic information and a few rows from each dataframe
print("Training Data (train.csv) Info:")
df_train.info()
print("\nTraining Data Head:")
print(df_train.head())

print("\nTest Data (test.csv) Info:")
df_test.info()
print("\nTest Data Head:")
print(df_test.head())

print("\nSample Submission (sample_submission.csv) Info:")
df_sample_submission.info()
print("\nSample Submission Head:")
print(df_sample_submission.head())


# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform 'Sex' column in training data
df_train['Sex_Encoded'] = le.fit_transform(df_train['Sex'])

# Transform 'Sex' column in test data (using the encoder fitted on training data)
df_test['Sex_Encoded'] = le.transform(df_test['Sex'])

# Define features (X) and target (y) for training
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex_Encoded']
X_train = df_train[features]
y_train = df_train['Calories']

# Define features for the test set
X_test = df_test[features]


# RMSLE scorer function
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)



# Initialize and train the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, loss='squared_error')
y_train_log = np.log1p(y_train)
gbr.fit(X_train, y_train_log)


# Predict on the test set
y_pred_log_test = gbr.predict(X_test)

# Transform predictions back to original scale
y_pred_test = np.expm1(y_pred_log_test)

# Ensure no negative predictions if any (though log1p/expm1 should handle this)
y_pred_test[y_pred_test < 0] = 0  # Calories cannot be negative


# Create submission DataFrame
submission_df = pd.DataFrame({'id': df_test['id'], 'Calories': y_pred_test})

# Display first few rows of submission
print("Submission DataFrame Head:")
print(submission_df.head())

# Save submission file
submission_df.to_csv("submission.csv", index=False)

print("\nSubmission file 'submission.csv' created successfully.")


plt.figure(figsize=(10, 6))
sns.histplot(df_train['Calories'], kde=True)
plt.title('Distribution of Calories Burned')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.savefig("calories_distribution.png")
plt.close()
print("Plot 1 (calories_distribution.png) generated.")


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Duration', y='Calories', hue='Sex', data=df_train, alpha=0.6)
plt.title('Calories Burned vs. Workout Duration by Sex')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories Burned')
plt.savefig("duration_vs_calories_by_sex.png")
plt.close()
print("Plot 2 (duration_vs_calories_by_sex.png) generated.")


plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='Calories', data=df_train)
plt.title('Calories Burned by Sex')
plt.xlabel('Sex')
plt.ylabel('Calories Burned')
plt.savefig("calories_by_sex_boxplot.png")
plt.close()
print("Plot 3 (calories_by_sex_boxplot.png) generated.")


plt.figure(figsize=(10, 8))
numerical_features = df_train.select_dtypes(include=np.number).columns.tolist()
if 'id' in numerical_features:
    numerical_features.remove('id')
correlation_matrix = df_train[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig("correlation_heatmap.png")
plt.close()
print("Plot 4 (correlation_heatmap.png) generated.")


numerical_cols_for_dist = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_for_dist):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("numerical_features_distribution.png")
plt.close()
print("Plot 5 (numerical_features_distribution.png) generated.")


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Heart_Rate', y='Calories', data=df_train, alpha=0.6)
plt.title('Calories Burned vs. Heart Rate')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Calories Burned')
plt.savefig("heart_rate_vs_calories.png")
plt.close()
print("Plot 6 (heart_rate_vs_calories.png) generated.")


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight', y='Calories', data=df_train, alpha=0.6)
plt.title('Calories Burned vs. Weight')
plt.xlabel('Weight (kg)')
plt.ylabel('Calories Burned')
plt.savefig("weight_vs_calories.png")
plt.close()
print("Plot 7 (weight_vs_calories.png) generated.")


# The 'features' list used for training: ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex_Encoded']
importances = gbr.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importances from Gradient Boosting Regressor')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig("feature_importances.png")
plt.close()
print("Plot 8 (feature_importances.png) generated.")


pairplot_features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Calories']
plt.figure()
sns.pairplot(df_train[pairplot_features], diag_kind='kde', corner=True)
plt.suptitle('Pair Plot of Key Features and Calories', y=1.02)
plt.savefig("pairplot_key_features_calories.png")
plt.close()
print("Plot 9 (pairplot_key_features_calories.png) generated.")


y_train_pred_log = gbr.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)
residuals = y_train - y_train_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot (Actual vs. Predicted Calories on Training Data)')
plt.xlabel('Predicted Calories')
plt.ylabel('Residuals (Actual - Predicted)')
plt.savefig("residual_plot.png")
plt.close()
print("Plot 10 (residual_plot.png) generated.")

