import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")
%matplotlib inline


train = pd.read_csv(r"/kaggle/input/playground-series-s5e7/train.csv")
test  = pd.read_csv(r"/kaggle/input/playground-series-s5e7/test.csv")

train['dataset'] = 'train'
test['dataset'] = 'test'

df = pd.concat([train, test], axis=0).reset_index(drop=True)
print("Dataset shape:", df.shape)
df.head()


df.tail()


df.shape



df.info()



# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


df[numerical_cols].describe()



for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())



plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Personality', palette='pastel', edgecolor='black')
plt.title('Distribution of Personality Types', fontsize=14)
plt.xlabel('Personality Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\nðŸ“Š Personality Value Counts (Proportions):")
print(df['Personality'].value_counts(normalize=True).round(3))


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f'\nðŸ“Š Descriptive Stats for {col}:\n')
    print(df[col].describe(), '\n' + '-'*40)


plt.figure(figsize=(14, 6))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df, y=col, color='#FFA726')
    plt.title(f"Boxplot: {col}")
plt.tight_layout()
plt.show()


for col, palette in zip(['Stage_fear', 'Drained_after_socializing'], ['Set1', 'Set1']):
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=col,
        hue='Personality',
        palette=palette,
        edgecolor='black'
    )
    plt.title(f'Distribution of {col} by Personality', fontsize=14)
    plt.xlabel(f'{col} (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Personality', labels=['Introvert (0)', 'Extrovert (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 4))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(
        data=df,
        x='Personality',
        y=col,
        palette='Set2',
        linewidth=1.2,
        fliersize=4
    )
    plt.title(f'{col} by Personality', fontsize=14, fontweight='semibold', color='#2E4057')
    plt.xlabel('Personality', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(df.isnull().sum())



mapping_yes_no = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(mapping_yes_no)
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(mapping_yes_no)

mapping_personality = {'Extrovert': 1, 'Introvert': 0}
df['Personality'] = df['Personality'].map(mapping_personality)


train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns =['dataset'], errors='ignore')

train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['Personality'], errors='ignore')


X = train_df.drop('Personality', axis=1)
y = train_df['Personality']




xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    enable_categorical=False,
    random_state=42,
    n_estimators=1000,
    learning_rate=0.006358,
    max_depth=8,
    subsample=0.8854,
    colsample_bytree=0.6,
    reg_lambda=0.8295,
    reg_alpha=5.5149,
    gamma=0.0395,
    min_child_weight=2,
    use_label_encoder=False
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy_scores = cross_val_score(
    xgb_model, X, y, cv=cv, scoring='accuracy'
)
cv_f1_scores = cross_val_score(
    xgb_model, X, y, cv=cv, scoring='f1_weighted'
)

print("XGBoost Model Crossâ€‘Validation Results")
print("CV Accuracy Scores:", cv_accuracy_scores)
print("Mean CV Accuracy:", cv_accuracy_scores.mean())
print("Std CV Accuracy:", cv_accuracy_scores.std())
print("CV Weighted F1 Scores:", cv_f1_scores)
print("Mean CV Weighted F1 Score:", cv_f1_scores.mean())
print("Std CV Weighted F1 Score:", cv_f1_scores.std())

xgb_model.fit(X, y)

y_pred = xgb_model.predict(X)

test_accuracy = accuracy_score(y, y_pred)
test_f1 = f1_score(y, y_pred, average='weighted')
print("\nFull Data Accuracy:", test_accuracy)
print("Full Data Weighted F1 Score:", test_f1)
print("\ Classification Report:\n",
      classification_report(y, y_pred, digits=4))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=xgb_model.classes_)
disp.plot(cmap='Blues')
plt.title("XGBoost Confusion Matrix - Full Data")
plt.show()

importances = xgb_model.feature_importances_
feature_names = (X.columns 
                 if hasattr(X, 'columns') 
                 else [f"Feature {i}" for i in range(X.shape[1])])
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nXGBoost Feature Importances:\n", feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_importance_df, 
    palette='viridis'
)
plt.title('XGBoost Top Feature Importances')
plt.tight_layout()
plt.show()


test_features = test_df.drop(columns=['id'], errors='ignore')
predictions = xgb_model.predict(test_features)

mapping = {1: 'Extrovert', 0: 'Introvert'}

vec_map = np.vectorize(mapping.get)
predictions = vec_map(predictions.astype(int))

submission = pd.DataFrame({
    'id': test_df['id'],  
    'Fertilizer Name': predictions
})

submission.to_csv('submission.csv', index=False)


submission 

