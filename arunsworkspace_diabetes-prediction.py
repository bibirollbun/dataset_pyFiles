import pandas as pd

df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')

print("First 5 rows of the DataFrame:")
print(df.head())

print("\nDataFrame Information:")
df.info()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.dropna(inplace=True)
X = df.drop(columns=['id', 'diagnosed_diabetes'])
y = df['diagnosed_diabetes']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_split, y_train_split)

try:
    external_test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

    submission_ids = external_test_df['id']

    X_external_test = external_test_df.drop(columns=['id'], errors='ignore')

    X_external_test_preprocessed = preprocessor.transform(X_external_test)

    y_pred_proba_submission = model.predict_proba(X_external_test_preprocessed)[:, 1]

    submission_df = pd.DataFrame({
        'id': submission_ids,
        'diagnosed_diabetes': y_pred_proba_submission
    })

    print("First 5 rows of the submission file:")
    display(submission_df.head())

    submission_df.to_csv('submission.csv', index=False)
    print(f"\n'submission.csv' created successfully with {len(submission_df)} entries!")

except FileNotFoundError:
    print("Warning: '/content/test.csv' not found. ")
    print("The submission file could not be generated for an external test set. ")
    print("The current code still produces a submission.csv but it is based on the test split of the training data (140,000 rows). ")
    print("If you have a separate test.csv file with 300,000 rows, please upload it to '/content/test.csv' and rerun this cell to generate the correct submission.")

    y_pred_proba_local = model.predict_proba(X_test_split)[:, 1]
    test_ids_local = df.loc[y_test_split.index, 'id'] # Get original IDs for the local test split
    submission_df_local = pd.DataFrame({
        'id': test_ids_local,
        'diagnosed_diabetes': y_pred_proba_local
    })
    print("\nGenerating submission based on training data test split (140,000 rows) as fallback...")
    display(submission_df_local.head())
    submission_df_local.to_csv('submission.csv', index=False)
    print(f"'submission.csv' created successfully with {len(submission_df_local)} entries (from training split).")



import pandas as pd
import os

file_path = '/kaggle/input/playground-series-s5e12/test.csv'

if os.path.exists(file_path):
    print(f"'{file_path}' found. Loading file...")
    try:
        test_df_check = pd.read_csv(file_path)
        print("\nFirst 5 rows of test.csv:")
        display(test_df_check.head())
        print(f"\nShape of test.csv: {test_df_check.shape}")
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
else:
    print(f"Error: '{file_path}' not found. Please ensure the file is uploaded to the /content/ directory.")


print("Missing values before dropping:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df.dropna(inplace=True)

print("\nMissing values after dropping:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nShape of DataFrame after dropping missing values: {df.shape}")


X = df.drop(columns=['id', 'diagnosed_diabetes'])
y = df['diagnosed_diabetes']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"Categorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X)

print("Shape of preprocessed features (X_preprocessed):")
print(X_preprocessed.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print(f"\nShape of predicted probabilities (y_pred_proba): {y_pred_proba.shape}")
print(f"First 5 predicted probabilities: {y_pred_proba[:5]}")


print("Descriptive statistics for numerical columns:")
print(df.describe())


print("\nValue counts for categorical columns:")
categorical_cols_df = df.select_dtypes(include='object').columns
for col in categorical_cols_df:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())



import seaborn as sns
import matplotlib.pyplot as plt

numerical_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(20, 15))
sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()

print("Top 10 correlations with diagnosed_diabetes:")
print(numerical_df.corr()['diagnosed_diabetes'].sort_values(ascending=False).head(10))


print("\nCross-tabulations for categorical features with diagnosed_diabetes:")
categorical_cols_df = df.select_dtypes(include='object').columns
for col in categorical_cols_df:
    print(f"\nCross-tabulation for {col} vs diagnosed_diabetes:")
    print(pd.crosstab(df[col], df['diagnosed_diabetes'], normalize='index'))


import matplotlib.pyplot as plt
import seaborn as sns

key_numerical_features = ['age', 'bmi', 'systolic_bp', 'cholesterol_total', 'triglycerides']

for feature in key_numerical_features:
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df[df['diagnosed_diabetes'] == 0][feature], kde=True, color='skyblue', bins=30)
    plt.title(f'{feature.replace("_", " ").title()} Distribution (Non-Diabetic)')
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df[df['diagnosed_diabetes'] == 1][feature], kde=True, color='salmon', bins=30)
    plt.title(f'{feature.replace("_", " ").title()} Distribution (Diabetic)')
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel('Frequency')

    plt.suptitle(f'Distribution of {feature.replace("_", " ").title()} by Diabetes Status', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

categorical_cols_to_plot = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']

for col in categorical_cols_to_plot:

    crosstab_df = pd.crosstab(df[col], df['diagnosed_diabetes'], normalize='index')

    crosstab_df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Proportion of Diabetes Status by {col.replace("_", " ").title()}')
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Diagnosed Diabetes', labels=['Non-Diabetic', 'Diabetic'])
    plt.tight_layout()
    plt.show()


from sklearn.metrics import roc_curve, auc, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Logistic Regression')
roc_display.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_display.plot()
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
cmd_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cmd_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.show()


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully.")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully.")


from sklearn.ensemble import GradientBoostingClassifier

gbc_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
gbc_model.fit(X_train, y_train)

print("GradientBoostingClassifier model trained successfully.")


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.dropna(inplace=True) 

X = df.drop(columns=['id', 'diagnosed_diabetes'])
y = df['diagnosed_diabetes']

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"RandomForestClassifier Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"ROC AUC Score: {roc_auc_rf:.4f}")

plt.figure(figsize=(8, 6))
roc_display_rf = RocCurveDisplay(fpr=roc_curve(y_test, y_pred_proba_rf)[0], tpr=roc_curve(y_test, y_pred_proba_rf)[1], roc_auc=roc_auc_rf, estimator_name='RandomForestClassifier')
roc_display_rf.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve - RandomForestClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
cmd_display_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[0, 1])
cmd_display_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - RandomForestClassifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

y_pred_gbc = gbc_model.predict(X_test)
y_pred_proba_gbc = gbc_model.predict_proba(X_test)[:, 1]

accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
precision_gbc = precision_score(y_test, y_pred_gbc)
recall_gbc = recall_score(y_test, y_pred_gbc)
f1_gbc = f1_score(y_test, y_pred_gbc)
roc_auc_gbc = roc_auc_score(y_test, y_pred_proba_gbc)

print(f"GradientBoostingClassifier Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy_gbc:.4f}")
print(f"Precision: {precision_gbc:.4f}")
print(f"Recall: {recall_gbc:.4f}")
print(f"F1-Score: {f1_gbc:.4f}")
print(f"ROC AUC Score: {roc_auc_gbc:.4f}")

plt.figure(figsize=(8, 6))
roc_display_gbc = RocCurveDisplay(fpr=roc_curve(y_test, y_pred_proba_gbc)[0], tpr=roc_curve(y_test, y_pred_proba_gbc)[1], roc_auc=roc_auc_gbc, estimator_name='GradientBoostingClassifier')
roc_display_gbc.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve - GradientBoostingClassifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

cm_gbc = confusion_matrix(y_test, y_pred_gbc)
plt.figure(figsize=(8, 6))
cmd_display_gbc = ConfusionMatrixDisplay(confusion_matrix=cm_gbc, display_labels=[0, 1])
cmd_display_gbc.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - GradientBoostingClassifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.show()


import pandas as pd
import numpy as np

accuracy_lr = 0.6617
precision_lr = 0.6803
recall_lr = 0.8639
f1_lr = 0.7612
roc_auc_lr = 0.6943

accuracy_rf = 0.6639
precision_rf = 0.6854
recall_rf = 0.8529
f1_rf = 0.7600
roc_auc_rf = 0.6964

accuracy_gbc = 0.6749
precision_gbc = 0.6944
recall_gbc = 0.8558
f1_gbc = 0.7667
roc_auc_gbc = 0.7150

model_comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'RandomForestClassifier', 'GradientBoostingClassifier'],
    'Accuracy': [accuracy_lr, accuracy_rf, accuracy_gbc],
    'Precision': [precision_lr, precision_rf, precision_gbc],
    'Recall': [recall_lr, recall_rf, recall_gbc],
    'F1-Score': [f1_lr, f1_rf, f1_gbc],
    'ROC AUC': [roc_auc_lr, roc_auc_rf, roc_auc_gbc]
})

print("\nModel Performance Comparison:")
print(model_comparison_df.set_index('Model'))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

fpr_lr, tpr_lr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc_lr_calc = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf_calc = auc(fpr_rf, tpr_rf)

fpr_gbc, tpr_gbc, _ = roc_curve(y_test, gbc_model.predict_proba(X_test)[:, 1])
roc_auc_gbc_calc = auc(fpr_gbc, tpr_gbc)

plt.figure(figsize=(10, 8))

display_lr = RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr, roc_auc=roc_auc_lr_calc, estimator_name='Logistic Regression')
display_lr.plot(ax=plt.gca(), name=f'Logistic Regression (AUC = {roc_auc_lr_calc:.4f})', color='darkorange')

display_rf = RocCurveDisplay(fpr=fpr_rf, tpr=tpr_rf, roc_auc=roc_auc_rf_calc, estimator_name='RandomForestClassifier')
display_rf.plot(ax=plt.gca(), name=f'RandomForestClassifier (AUC = {roc_auc_rf_calc:.4f})', color='green')

display_gbc = RocCurveDisplay(fpr=fpr_gbc, tpr=tpr_gbc, roc_auc=roc_auc_gbc_calc, estimator_name='GradientBoostingClassifier')
display_gbc.plot(ax=plt.gca(), name=f'GradientBoostingClassifier (AUC = {roc_auc_gbc_calc:.4f})', color='red')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

