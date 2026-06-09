# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from scipy import stats

import pickle

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Basic
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# Loading the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


# Train Dataset Preview
train_df.head()


# Test Dataset Preview
test_df.head()


# Encoding columns 'Stage_fear' and 'Drained_after_socializing'
encoding = {'Yes': 1, 'No': 0}
train_df['Stage_fear'] = train_df['Stage_fear'].map(encoding)
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map(encoding)
print(f"Encoded: {encoding}")


df_extrovert = train_df[train_df['Personality'] == "Extrovert"]
df_introvert = train_df[train_df['Personality'] == "Introvert"]
print(f"Created seperate dataframes for introvert and extrovert...")


# Train dataframe
train_df


# Test dataframe
test_df


# Shape of train and test
print(f"Shape of train dataset: {train_df.shape}")
print(f"Shape of test dataset:  {test_df.shape}")


# Count of unique values for each column
print(f"Number of unique values for each column:\n{train_df.nunique()}")


# Extrovert dataframe inspection
print(f"{df_extrovert.info()}\n")
df_extrovert.describe()


# Introvert dataframe inspection
print(f"{df_introvert.info()}\n")
df_introvert.describe()


# Reviewing the percentage of null values in each column for both the personalities separately
nulls = [
    (col, float(df_extrovert[col].isnull().sum() / df_extrovert.shape[0]), float(df_introvert[col].isnull().sum() / df_introvert.shape[0]))
    for col in df_extrovert.columns[1:-1]
    ]
print(f"Columns{'':<19}: Extrovert Null %{'':<4}: Introvert Null %")
for i in nulls:
    print(f"{i[0]:<26}: {(i[1]*100):.2f}%{'':15}: {(i[2]*100):.2f}%")


duplicate_count =train_df.duplicated().sum()
print("Number of duplicate records:", duplicate_count)


# Categorical and numerical columns
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
numerical_cols = ['Time_spent_Alone', 'Friends_circle_size', 'Post_frequency', 'Social_event_attendance', 'Going_outside']


# Checking median for numerical columns, and mode for categorical columns
mean_values_extrovert = df_extrovert[numerical_cols].mean()
median_values_extrovert = df_extrovert[numerical_cols].median()
mode_values_extrovert = df_extrovert[categorical_cols].mode().iloc[0]

mean_values_introvert = df_introvert[numerical_cols].mean()
median_values_introvert = df_introvert[numerical_cols].median()
mode_values_introvert = df_introvert[categorical_cols].mode().iloc[0]

mean_summary_df = pd.DataFrame({
    'Column Name': numerical_cols,
    'Introvert': mean_values_introvert.values,
    'Extrovert': mean_values_extrovert.values
})

median_summary_df = pd.DataFrame({
    'Column Name': numerical_cols,
    'Introvert': median_values_introvert.values,
    'Extrovert': median_values_extrovert.values
})

mode_summary_df = pd.DataFrame({
    'Column Name': categorical_cols,
    'Introvert': mode_values_introvert.values,
    'Extrovert': mode_values_extrovert.values
})

# Displaying the results
print("Summary Table: Median Values (Numerical):")
print(mean_summary_df)

print(f"\n{'='*50}\n")

print("Summary Table: Median Values (Numerical):")
print(median_summary_df)

print(f"\n{'='*50}\n")

print("Summary Table: Mode Values (Categorical):")
print(mode_summary_df)


# Imputation of numerical and categorical columns of train dataset
print("Imputing Numerical Columns with Group-Wise Median...")

for col in numerical_cols:
    train_df[col].fillna(
        train_df.groupby('Personality')[col].transform('median'),
        inplace=True
    )
    print(f"  - Filled NaN in '{col}' using group median.")

print("\nNumerical Imputation Complete.")

print("\nImputing Categorical Columns with Group-Wise Mode...")

for col in categorical_cols:
    train_df[col].fillna(
        train_df.groupby('Personality')[col].transform(lambda x: x.mode()[0]),
        inplace=True
    )
    print(f"  - Filled NaN in '{col}' using group mode.")

print("\nCategorical Imputation Complete.")


# Checking after removing null values
train_df.isna().sum()


# Histplot for each column
fig, axes = plt.subplots(7, 1, figsize=(20, 50))
personality_types = train_df['Personality'].unique()
palette = sns.color_palette('Set1', n_colors=len(personality_types))
color_map = dict(zip(personality_types, palette))
for i, column in enumerate(train_df.columns[1:-1]):
    # axes[i].set_facecolor('black')
    sns.histplot(data=train_df, x=column, hue='Personality', ax=axes[i], palette='Set1')
    # sns.histplot(data=train_df, x=column, hue='Personality', ax=axes[i], palette='Set2')

    for personality in personality_types:
        median_value = train_df[train_df['Personality'] == personality][column].median()

        axes[i].axvline(
            median_value,
            color=color_map[personality],
            linestyle='--',
            linewidth=2,
            label=f'Median ({personality})'
        )
    axes[i].legend(title='Personality', loc='upper right')
    axes[i].set_title(f'{column} by Personality', fontsize=14)
    axes[i].grid(True)


# Box plot for each numerical column
df_long = pd.melt(
    train_df,
    id_vars=['Personality'],  
    value_vars=numerical_cols,
    var_name='Feature',      
    value_name='Value'      
)

plt.figure(figsize=(18, 9))
sns.boxplot(
    x='Feature',
    y='Value',
    hue='Personality',
    data=df_long,
    palette='Set1'
)

# 3. Add title and adjust x-axis labels for readability
plt.title('Distribution of Features by Personality')
plt.xlabel('Feature')
plt.ylabel('Value')
plt.legend(title='Personality', loc='upper right')
# plt.tight_layout()
plt.show()


# Count plot for each categorical column
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
total = len(train_df)
for j, col in enumerate(categorical_cols):
    # plot_index = len(continuous_columns) + j
    ax = sns.countplot(
        x=col,
        hue='Personality',
        data=train_df,
        ax=axes[j],
        palette='Set1'
    )
    for p in ax.patches:
        count = p.get_height()
        percent = 100 * count / total
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(f'{percent:.1f}%', (x, y + total * 0.01), ha='center', fontsize=10)
    axes[j].set_title(f'Count Plot of {col} by Personality', fontsize=14)
    axes[j].set_xlabel(col, fontsize=12)
    axes[j].set_ylabel('Count', fontsize=12)
    axes[j].legend(title='Personality', loc='upper right')


# Countplot for target variable
plt.figure(figsize=(8, 6))

ax = sns.countplot(
        x='Personality',
        data=train_df,
        palette='Set1'
    )
total = len(train_df)
for p in ax.patches:
    count = p.get_height()
    percent = 100 * count / total
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(f'{percent:.1f}%', (x, y + total * 0.01), ha='center', fontsize=10)
plt.title('Count Plot of Personality')


# Encoding the target variable
personality_map = {'Extrovert': 0, 'Introvert': 1}
train_df['Personality'] = train_df['Personality'].map(personality_map)
print("Encoded")


# Correlation matrix including the target variable
sns.heatmap(train_df[train_df.columns[1:]].corr(), annot=True, cmap='viridis')
plt.title('Correlation Matrix (Including Target)')


# Pairplot for all the columns except id
# sns.pairplot(train_df[train_df.columns[1:]], hue='Personality', palette='Set1')
sns.pairplot(train_df[numerical_cols + ['Personality']], hue='Personality', palette='Set1')
plt.suptitle('ðŸ§  Pairplot of Features by Personality Type', y=1.02)
plt.show()


test_df


test_df.isna().sum()


encoding = {'Yes': 1, 'No': 0}
test_df['Stage_fear'] = test_df['Stage_fear'].map(encoding)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map(encoding)
print("Encoded")


# Imputation of numerical and categorical columns of train dataset
print("Imputing Numerical Columns with Group-Wise Median...")

for col in numerical_cols:
    train_df[col].fillna(
        train_df.groupby('Personality')[col].transform('median'),
        inplace=True
    )
    print(f"  - Filled NaN in '{col}' using group median.")

print("\nNumerical Imputation Complete.")

print("\nImputing Categorical Columns with Group-Wise Mode...")

for col in categorical_cols:
    train_df[col].fillna(
        train_df.groupby('Personality')[col].transform(lambda x: x.mode()[0]),
        inplace=True
    )
    print(f"  - Filled NaN in '{col}' using group mode.")

print("\nCategorical Imputation Complete.")


for col in numerical_cols:
    test_df[col] = test_df[col].fillna(train_df[col].median())


for col in categorical_cols:
    # Fill categorical with mode
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])


test_df.isna().sum()


test_df.dropna(inplace=True)


test_df.info()


test_df.describe()


test_df.info()


X = train_df.drop(columns=['id', 'Personality'])  # Drop target & ID from the beginning  
y = train_df['Personality']
X_test = test_df.drop(columns=['id'])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original Train rows: {len(train_df)}")
print(f"Now training on: {len(X_train)} rows")
print(f"Checking accuracy on: {len(X_val)} rows")


# Calculate class weights (inverse frequency)
extrovert_ratio = (y == 0).mean()
introvert_ratio = (y ==1).mean()

class_weights_dict = {
    0: 1 / extrovert_ratio,
    1: 1 / introvert_ratio
}
print("Class weights created using inverse frequency")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# model = LogisticRegression(class_weight=class_weights_dict)
# model.fit(X_train, y_train) # [Extrovert, Introvert]


# Define models
models = {
    "Logistic Regression": LogisticRegression(class_weight=class_weights_dict, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight=class_weights_dict, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=2.8) # Adjust weight based on your class ratio
}


# 2. Setup Plotting for Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
results = []

# 3. Train and Evaluate
for i, (name, model) in enumerate(models.items()):
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_val_scaled)
    y_probs = model.predict_proba(X_val_scaled)[:, 1]
    
    # Generate Metrics
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_probs)
    
    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1, "ROC-AUC": auc})
    
    # Print Detailed Report
    print(f"\n{'='*30}")
    print(f"REPORT FOR: {name}")
    print(f"{'='*30}")
    print(classification_report(y_val, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(f"Confusion Matrix: {name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 4. Display Comparison Table
comparison_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("\n--- Summary Table (Sorted by F1-Score) ---")
print(comparison_df)


comparison_df.set_index('Model')[['Accuracy', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()


chosen_model = models["Logistic Regression"]


# 1. Get coefficients (Logistic Regression uses .coef_)
importances = np.abs(chosen_model.coef_[0])

# 2. Get feature names 
feature_names = X_train.columns

# 3. Create DataFrame
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 4. Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette="viridis")
plt.title("ðŸŽ¯ Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.show()


with open('model.pkl', 'wb') as f:
    pickle.dump(chosen_model, f)

print("Model saved as model.pkl")

