# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

# Data load
url = '/kaggle/input/train-data/train (1).csv'
df = pd.read_csv(url)


# Feature Engineering

'''Helpfull to know the categories in the particular column'''
print(df['education_level'].unique())
print(df['income_level'].unique())
print(df['gender'].unique())

# Replacement into numeric
df['gender'] = df['gender'].map({'Male':1, 'Female':0, 'Other':2})
df['education_level'] = df['education_level'].map({'No formal': 1,'Highschool': 2,'Graduate': 3,'Postgraduate': 4})
df['income_level'] =  df['income_level'].map({'Lower-Middle':1, 'Upper-Middle':2, 'Low':3, 'Middle':4, 'High':5})

# New columns
# df['Pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
# df['MAP'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
# df['HDL_Ratio'] =  df['cholesterol_total'] / df['hdl_cholesterol']
# df['bad_cholestrol'] =  df['cholesterol_total']  - df['hdl_cholesterol']
# df['Sedentary_Ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] + 1)
# df['HDL_BMI_Index'] = df['HDL_Ratio'] * df['bmi']
# df['Vascular_Age_Index'] = df['Pulse_pressure'] * df['age']
# df['MAP_Obesity_Load'] = df['MAP'] * df['bmi']
df['Pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
df['MAP'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
df['HDL_Ratio'] =  df['cholesterol_total'] / df['hdl_cholesterol']
df['bad_cholestrol'] =  df['cholesterol_total']  - df['hdl_cholesterol']
df['Sedentary_Ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] + 1)


# Data cleaning

print('Top Values')
print(df.head(10))
print("Bottom Values")
print(df.tail(10))
print("Information")
print(df.info())
print("Staticis")
print(df.describe())
print('Shape of the data')
print(df.shape)
print('Columns')
print(df.columns)
print("Duplicates value")
print(df.duplicated().sum())
''' There are no duplicates'''
print("Missing Values")
print(df.isnull().sum())

print("Unique Values in each column")
for col in df.columns:
  print(col, df[col].nunique())

# One hot encoding
df = pd.get_dummies(df, columns=['smoking_status', 'employment_status', 'ethnicity'], drop_first=True)



# Visulaizing outliners and removing

numeric_cols_diabetes = [
    'age',
    'bmi',
    'Pulse_pressure',
    'MAP',
    'HDL_Ratio',
    'bad_cholestrol',
    'Sedentary_Ratio',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'alcohol_consumption_per_week',
    'waist_to_hip_ratio'
]

for col in numeric_cols_diabetes:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# As there are outliners but to know which one to remove, check the min and max of the statics to see if it is normal range or not, if abnormal like negative pulse remove outliner

def remove_outliers_pulse(df):
    Q1 = df['Pulse_pressure'].quantile(0.25)
    Q3 = df['Pulse_pressure'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df['Pulse_pressure'] >= lower_bound) & (df['Pulse_pressure'] <= upper_bound)]
    return df_filtered

df = remove_outliers_pulse(df)
print("Rows after removing outliers:", df.shape[0])




# Relations

# 1

bins = [0, 30, 45, 60, 100]
labels = ['<30','30-45','45-60','60+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

age_diabetes = df.groupby('age_group', observed=True)['diagnosed_diabetes'].mean()
age_diabetes.plot(kind='bar')
plt.title('Diabetes Rate by Age Group')
plt.ylabel('Diabetes Rate')
plt.xlabel('Age Group')
plt.show()

# 2
gender_diabetes = df.groupby('gender')['diagnosed_diabetes'].mean()

gender_diabetes.plot(kind='bar')
plt.title('Diabetes Rate by Gender')
plt.ylabel('Diabetes Rate')
plt.xlabel('Gender (0 = Female, 1 = Male, 2 = Others)')
plt.show()

# Target distribution
# 3

sns.countplot(x='diagnosed_diabetes', data=df)
plt.title('Diabetes vs Non-Diabetes Count')
plt.xlabel('Diagnosed Diabetes (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 4

sns.barplot(x='diagnosed_diabetes', y='bmi', data=df)
plt.title('BMI by Diabetes Outcome')
plt.xlabel('Diagnosed Diabetes')
plt.ylabel('BMI')
plt.show()

# 5

numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title('Correlation Map')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.show()

# Which one relates the most by numbers

numeric_cols_diabetes = [
    'age',
    'bmi',
    'Pulse_pressure',
    'MAP',
    'HDL_Ratio',
    'bad_cholestrol',
    'Sedentary_Ratio',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'alcohol_consumption_per_week',
    'waist_to_hip_ratio'
]

num = df.groupby('diagnosed_diabetes')[numeric_cols_diabetes].mean().sort_values(by=numeric_cols_diabetes,ascending=False)
print(num)


# Data processing

X = df.select_dtypes(include=['number']).drop(['diagnosed_diabetes'], axis=1)
y = df['diagnosed_diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# stratify=y just makes sure both train and test have the same proportion of diabetes vs non-diabetes.
# target (diagnosed_diabetes) is imbalanced

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Note: Where we use fit transform we also use transform
 # Applies the exact same parameters learned during the fit_transform step to the test features


# Logistic regression

model = LogisticRegression(max_iter=1000, class_weight='balanced', penalty='elasticnet', C=100, solver='saga', l1_ratio=0.5)
model.fit(X_train_scaled, y_train)

# ---- Probabilities -------
y_test_probs  = model.predict_proba(X_test_scaled)[:, 1]

# --- Threshold tuning ---
threshold = 0.5  # set your threshold (default 0.5)
y_test_pred  = (y_test_probs >= threshold).astype(int)

# As the accuracy does not except probablities so we have to work indiviual for it
y_train_pred = model.predict(X_train_scaled)
y_test_pred  = model.predict(X_test_scaled)

train = accuracy_score(y_train, y_train_pred)
test = accuracy_score(y_test, y_test_pred)
print("Train Accuracy:", train)
print("Test Accuracy:", test)

# Note: In classifier we comapre accuracy score for overfitting and underfitting not MSE

print("\nClassification Report:\n")
print(classification_report(y_test,y_test_pred))

# plot

plt.figure(figsize=(6,4))
plt.bar(['Train accuracy', 'Test sccuracy'], [train, test])
plt.title('Train vs Test Accuracy (Logistic Regression)')
plt.ylabel('Accuracy')
plt.show()

# Plot
cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Diabetes", "Diabetes"]  # or 0 / 1
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic")
plt.show()

# For imbalanced data, the ROC-AUC score is a better measure of how well your model separates the two classes regardless of the threshold. 
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_probs))


# Xgboost

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.2,
    scale_pos_weight= (y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance (check the ratio of negative and postive and then the given result shows or tells which values we should apply the counter like here 2.5 x diabetes 'pay more attention')
    random_state=42,
    min_child_weight = 10, # This is a regularization control.
    gamma = 0.5,
    eval_metric='logloss' # Penalty
)

xgb_model.fit(X_train, y_train)

y_test_probs_xgb  = xgb_model.predict_proba(X_test)[:,1]

# --- Threshold tuning ---
threshold = 0.5  # set your threshold (default 0.5)
y_test_pred_xgb  = (y_test_probs_xgb >= threshold).astype(int)

# As the accuracy does not except probablities so we have to work indiviual for it

y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb  = xgb_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred_xgb)
test_acc  = accuracy_score(y_test, y_test_pred_xgb)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# plot

plt.figure(figsize=(6,4))
plt.bar(['Train accuracy', 'Test sccuracy'], [train_acc, test_acc])
plt.title('Train vs Test Accuracy (Logistic Regression)')
plt.ylabel('Accuracy')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title(f"XGBoost Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report")
print(classification_report(y_test, y_test_pred_xgb))

# For imbalanced data, the ROC-AUC score is a better measure of how well your model separates the two classes regardless of the threshold. 
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_probs_xgb))


# Ensembled model

ensemble_probs = 0.6 * y_test_probs_xgb + 0.4 * y_test_probs

best_t = 0.5
final_ensemble_preds = (ensemble_probs >= best_t).astype(int)

# Classification Report

print("Final Ensemble Performance:")
print(classification_report(y_test, final_ensemble_preds))

# Plot
cm = confusion_matrix(y_test, final_ensemble_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title(f"Ensemble Confusion Matrix (Threshold: 0.5)")
plt.show()

# Recall Curve
# ( _ ) this is for threshold but we are not using threshold here so that is why
precision, recall, _ = precision_recall_curve(y_test, ensemble_probs)
pr_auc = auc(recall, precision)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()



# 1. Feature Engineering & Mapping
X_test_final = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv') 
submission_ids = X_test_final['id']

# Replacements
X_test_final['gender'] = X_test_final['gender'].map({'Male':1, 'Female':0, 'Other':2})
X_test_final['education_level'] = X_test_final['education_level'].map({'No formal': 1,'Highschool': 2,'Graduate': 3,'Postgraduate': 4})
X_test_final['income_level'] =  X_test_final['income_level'].map({'Lower-Middle':1, 'Upper-Middle':2, 'Low':3, 'Middle':4, 'High':5})

# New Columns 
X_test_final['Pulse_pressure'] = X_test_final['systolic_bp'] - X_test_final['diastolic_bp']
X_test_final['MAP'] = (X_test_final['systolic_bp'] + 2 * X_test_final['diastolic_bp']) / 3
X_test_final['HDL_Ratio'] =  X_test_final['cholesterol_total'] / X_test_final['hdl_cholesterol']
X_test_final['bad_cholestrol'] =  X_test_final['cholesterol_total']  - X_test_final['hdl_cholesterol']
X_test_final['Sedentary_Ratio'] = X_test_final['screen_time_hours_per_day'] / (X_test_final['physical_activity_minutes_per_week'] + 1)
X_test_final['HDL_BMI_Index'] = X_test_final['HDL_Ratio'] * X_test_final['bmi']
X_test_final['Vascular_Age_Index'] = X_test_final['Pulse_pressure'] * X_test_final['age']
X_test_final['MAP_Obesity_Load'] = X_test_final['MAP'] * X_test_final['bmi']

# 2. Encoding & Reindexing
X_test_final = pd.get_dummies(X_test_final, columns=['smoking_status', 'employment_status', 'ethnicity'], drop_first=True)
# Ensure columns match training data exactly
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

# 3. Scaling (for models that need it, like Logistic Regression)
X_test_final_scaled = scaler.transform(X_test_final) 
# 4. Generate Ensemble PROBABILITIES (No thresholding!)
lr_probs_final = model.predict_proba(X_test_final_scaled)[:, 1]
xgb_probs_final = xgb_model.predict_proba(X_test_final)[:, 1]

# Weighted Average Ensemble
ensemble_probs_final = (0.6 * xgb_probs_final) + (0.4 * lr_probs_final)

# 5. Create and Save Submission
submission = pd.DataFrame({
    'id': submission_ids, 
    'diagnosed_diabetes': ensemble_probs_final # Submit the raw decimals!
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully with probabilities!")

