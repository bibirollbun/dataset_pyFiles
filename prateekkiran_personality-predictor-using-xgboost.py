import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# Load Dataset
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df.info(), df.head()


test_df.info(), test_df.head()


# Check for missing values in the test dataset
missing_values_test = test_df.isnull().sum()

# Display columns with missing values
missing_values_test[missing_values_test > 0]


# Copy the original dataframe to preserve raw data
df_cleaned = df.copy()

# Drop the 'id' column as it's not informative
df_cleaned.drop(columns=['id'], inplace=True)

# Binary encode 'Stage_fear' and 'Drained_after_socializing' (Yes=1, No=0)
binary_map = {'Yes': 1, 'No': 0}
df_cleaned['Stage_fear'] = df_cleaned['Stage_fear'].map(binary_map)
df_cleaned['Drained_after_socializing'] = df_cleaned['Drained_after_socializing'].map(binary_map)

# Fill missing values
for col in df_cleaned.select_dtypes(include=['float64']).columns:
    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

for col in ['Stage_fear', 'Drained_after_socializing']:
    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

# Encode target variable 'Personality' as binary: Introvert=0, Extrovert=1
df_cleaned['Personality'] = df_cleaned['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Check cleaned data
df_cleaned.info(), df_cleaned.head()


# Copy the original dataframe to preserve raw data
test_df_cleaned = test_df.copy()


# Binary encode 'Stage_fear' and 'Drained_after_socializing' (Yes=1, No=0)
binary_map = {'Yes': 1, 'No': 0}
test_df_cleaned['Stage_fear'] = test_df_cleaned['Stage_fear'].map(binary_map)
test_df_cleaned['Drained_after_socializing'] = test_df_cleaned['Drained_after_socializing'].map(binary_map)

# Fill missing values
for col in df_cleaned.select_dtypes(include=['float64']).columns:
    test_df_cleaned[col].fillna(test_df_cleaned[col].median(), inplace=True)

for col in ['Stage_fear', 'Drained_after_socializing']:
    test_df_cleaned[col].fillna(test_df_cleaned[col].mode()[0], inplace=True)



# Check cleaned data
test_df_cleaned.info(), test_df_cleaned.head()


# Set plot aesthetics
sns.set(style='whitegrid')
plt.rcParams["figure.figsize"] = (10, 6)

# Plot distributions of numeric features by Personality
numeric_cols = df_cleaned.drop(columns='Personality').columns
personality_map = {0: 'Introvert', 1: 'Extrovert'}

# Plot each numeric feature
for col in numeric_cols:
    plt.figure()
    sns.kdeplot(data=df_cleaned, x=col, hue=df_cleaned['Personality'].map(personality_map), fill=True)
    plt.title(f'{col} Distribution by Personality')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()


# Compute correlation matrix
correlation_matrix = df_cleaned.corr()

# Plot heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()


# Prepare Training Data
X_train = df_cleaned.drop(columns='Personality')
y_train = df_cleaned['Personality']

# Best Model: XGBoost with Hyperparameter Tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 1, 5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)



#Training Metrics
y_train_preds = best_model.predict(X_train)
print("Accuracy:", accuracy_score(y_train, y_train_preds))
print("Classification Report:\n", classification_report(y_train, y_train_preds))


test_ids = test_df_cleaned['id']
X_test = test_df_cleaned.drop(columns='id')

# Final Predictions
final_preds = best_model.predict(X_test)
results = pd.DataFrame({
    'id': test_ids,
    'Personality': ['Introvert' if p == 0 else 'Extrovert' for p in final_preds]
})

# Save predictions to CSV
results.to_csv('/kaggle/working/submission.csv', index=False)




