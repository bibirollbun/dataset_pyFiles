!pip install --upgrade xgboost

!pip install scikit-learn imbalanced-learn


# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')





df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df.info()


df.head()


#Visualize missing values

# Calculating the percentage of missing values for each column
missing_data = df.isnull().sum()
missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100

# Prepare values
missing_percentage.sort_values(ascending=True, inplace=True)

# Plot the barh chart
fig, ax = plt.subplots(figsize=(15, 4))
ax.barh(missing_percentage.index, missing_percentage, color='#ff6200')

# Increase font size of y-axis labels
ax.set_yticks(range(len(missing_percentage)))
ax.set_yticklabels(missing_percentage.index, fontsize=14)

# Annotate the values and indexes
for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
    ax.text(value+0.5, i, f"{value:.2f}%", ha='left', va='center', fontweight='bold', fontsize=14, color='black')

# Set x-axis limit
ax.set_xlim([0, 12])

# Add title and xlabel
plt.title("Percentage of Missing Values", fontweight='bold', fontsize=18)
plt.xlabel('Percentages (%)', fontsize=12)
plt.show()



sns.countplot(data=df, x='Personality', width=.5, color='darkgreen')
plt.title('Personality Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Personality Type')
plt.ylabel('Count')



# Select numerical columns
numerical_cols = df.select_dtypes(include=['float']).columns

# Subplot layout
n_cols = 2
n_rows = math.ceil(len(numerical_cols) / n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = ax.flatten()

# Define a clean and readable palette
palette = sns.color_palette("Dark2", n_colors=2)  

# Plot numerical columns
for i, col in enumerate(numerical_cols):
    if i < len(axes):
        sns.histplot(
            data=df,
            x=col,
            hue='Personality',
            ax=axes[i],
            palette=palette,
            edgecolor='white'
        )
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)

# Hide unused plots
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



cols = ['Stage_fear', 'Drained_after_socializing']

# Set subplot configuration
n_cols = 2
n_rows = 1
fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 4))
axes = ax.flatten()

# Create countplots for each binary column
for i, col in enumerate(cols):
    sns.countplot(data=df, x=col, hue='Personality', ax=axes[i])
    axes[i].set_title(f'{col.replace("_", " ").title()} by Personality')
    axes[i].set_xlabel(col.replace("_", " ").title())
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()



# Pairwise correlation matrix
corr = df[numerical_cols].corr(method='pearson')

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap with the mask
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title("Lower Triangle Correlation Matrix")
plt.tight_layout()
plt.show()



# Step 1: Impute binary categorical columns first
for col in ['Drained_after_socializing', 'Stage_fear']:
    if df[col].isnull().sum():
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])

# Step 2: Encode binary columns
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})

# Step 3: Impute numeric
for col in df.select_dtypes(include=['float64', 'int64']):
    if df[col].isnull().sum():
        df[col] = df[col].fillna(df[col].median())

# Step 4: Impute remaining object columns
for col in df.select_dtypes(include='object'):
    if df[col].isnull().sum():
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])


# Confirming no missing values after imputation
missing_data = df.isnull().sum()
print("Missing values after imputation:")
print(missing_data[missing_data > 0])



# âœ… Encode target column
df['Personality'] = df['Personality'].str.strip().str.capitalize()
df['Personality'] = df['Personality'].replace({'Introvert': 0, 'Extrovert': 1})


# Feature Engineering ğŸ§ 
df['social_activity'] = df['Social_event_attendance'] * df['Friends_circle_size']
df['isolation_score'] = df['Time_spent_Alone'] / (df['Going_outside'] + 1)
df['energy_drain_index'] = df['Drained_after_socializing'] * df['Social_event_attendance']



print(df['Personality'].unique())  # Should only be [0, 1]



# âœ… Define X and y
X = df.drop(columns=['Personality', 'id'])
y = df['Personality']


# âœ… Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE only on training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

y_train_res = y_train_res.replace({'Introvert': 0, 'Extrovert': 1})
y_train_res = y_train_res.astype(int)



print("y_train unique values:", y_train.unique())
print("y_train has NaNs:", y_train.isnull().any())
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)




rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model with grid search
grid_search.fit(X_train_res, y_train_res)
best_rf = grid_search.best_estimator_



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
brf = BalancedRandomForestClassifier(random_state=42)
param_grid_brf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_brf = GridSearchCV(brf, param_grid_brf, cv=cv, scoring='accuracy', n_jobs=-1)
grid_brf.fit(X_train_res, y_train_res)


# %% Model 3: XGBoost
scale = y_train.value_counts()[0] / y_train.value_counts()[1]
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1],
    'scale_pos_weight': [1, scale]
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train_res, y_train_res)


# Ensure predictions and true labels are in the same format (strings)
rf_preds = best_rf.predict(X_test)
brf_preds = grid_brf.best_estimator_.predict(X_test)
xgb_preds = grid_xgb.best_estimator_.predict(X_test)


# Sanity check
print("rf_preds unique values:", pd.Series(rf_preds).unique())
assert not pd.Series(rf_preds).isnull().any(), "â�Œ Predictions contain NaN"




# Accuracy scores (safe comparison: both are int)
rf_score = accuracy_score(y_test, rf_preds)
brf_score = accuracy_score(y_test, brf_preds)
xgb_score = accuracy_score(y_test, xgb_preds)

scores = {'Random Forest': rf_score, 'Balanced RF': brf_score, 'XGBoost': xgb_score}
model_names = list(scores.keys())
model_scores = list(scores.values())

# %% Accuracy Barplot
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, model_scores, color=['skyblue', 'orange', 'green'])
plt.ylim(0, 1.25)
plt.title('Model Accuracy Comparison on Test Set', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score')
for bar, score in zip(bars, model_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{score:.4f}", ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% Select final model
best_model_name = max(scores, key=scores.get)
final_model = {'Random Forest': best_rf, 'Balanced RF': grid_brf.best_estimator_, 'XGBoost': grid_xgb.best_estimator_}[best_model_name]
print(f"âœ… Final model selected: {best_model_name}")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print("Train features:", list(X.columns))
print("Test features :", list(test_df.columns))


# âœ… Step 1: Missing Value Imputation
for col in ['Drained_after_socializing', 'Stage_fear']:
    if test_df[col].isnull().sum():
        mode = test_df[col].mode()
        if not mode.empty:
            test_df[col] = test_df[col].fillna(mode[0])

for col in test_df.select_dtypes(include=['float64', 'int64']):
    if test_df[col].isnull().sum():
        test_df[col] = test_df[col].fillna(test_df[col].median())

for col in test_df.select_dtypes(include='object'):
    if test_df[col].isnull().sum():
        mode = test_df[col].mode()
        if not mode.empty:
            test_df[col] = test_df[col].fillna(mode[0])

# âœ… Step 2: Encoding
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
test_df['Stage_fear'] = test_df['Stage_fear'].map({'Yes': 1, 'No': 0})

# âœ… Step 3: Feature Engineering
test_df['social_activity'] = test_df['Social_event_attendance'] * test_df['Friends_circle_size']
test_df['isolation_score'] = test_df['Time_spent_Alone'] / (test_df['Going_outside'] + 1)
test_df['energy_drain_index'] = test_df['Drained_after_socializing'] * test_df['Social_event_attendance']

# âœ… Step 4: Align features
features = X.columns.tolist()
X_final_test = test_df[features]

# âœ… Step 5: Predict
preds = final_model.predict(X_final_test)

# âœ… Step 6: Map predictions if numeric
if isinstance(preds[0], (int, np.integer)):
    submission_labels = pd.Series(preds).map({0: 'Introvert', 1: 'Extrovert'})
else:
    submission_labels = preds

# âœ… Step 7: Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': submission_labels
})

# âœ… Step 8: Preview and Save
print(submission.head())
print(submission['Personality'].value_counts())

submission.to_csv("submission.csv", index=False)
print("âœ… Final submission saved correctly!")



