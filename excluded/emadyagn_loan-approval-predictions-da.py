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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE



!pip install --upgrade scikit-learn



!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1



df = pd.read_csv("/kaggle/input/loan-approval-predictions/train.csv")
df



df.info()


del df['id']


for col in df.columns:
    print(f"{col}({df[col].nunique()} unique out of {df[col].count()} values):")
    print(df[col].value_counts(normalize=True).head(10))
    print(f"The type of Column is >>>>>> {df[col].dtype}")
    print("-----------------------------")


df['person_age'].describe().round(1) 


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# First plot: all data including ages above 60
plt.subplot(1, 2, 1)
sns.histplot(df['person_age'], bins=30, kde=True)
plt.title("Before Removing Outliers")
plt.xlabel("person_age")
plt.ylabel("Count")

# Second plot: after filtering ages > 20 and <= 50
plt.subplot(1, 2, 2)
sns.histplot(df[(df['person_age'] > 21) & (df['person_age'] <= 50)]['person_age'], bins=30, kde=True)
plt.title("After Filtering Ages > 20 and <= 50")
plt.xlabel("person_age")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



print(df['person_age'].value_counts())


df = df[(df['person_age'] > 21) & (df['person_age'] <= 50)]
print(df['person_age'].describe())



counts = df['person_age'].value_counts().sort_index()

percentages = (counts / counts.sum()) * 100

age_stats = pd.DataFrame({
    'count': counts,
    'percentage': percentages
})

print(age_stats)



print(df['loan_status'].value_counts())



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))  

plt.subplot(1, 2, 1)  
sns.histplot(df['person_income'], bins=30, kde=True)
plt.title(" bevore Outliers ")
plt.xlabel("person_income")
plt.ylabel("Count")


df_clean = df[df['person_income'] <= 100000]
plt.subplot(1, 2, 2)
sns.histplot(df_clean['person_income'], bins=30, kde=True)
plt.title("after Outliers ")
plt.xlabel("person_income")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



df = df[df['person_income'] <= 100000]


print(df['person_income'].value_counts())


print(df['cb_person_default_on_file'].value_counts())      


df= df[df['person_emp_length'] <= 40]


print(df['person_emp_length'].value_counts())


import pandas as pd
import matplotlib.pyplot as plt

features = [
    'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
    'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
]

df_copy = df1.copy()

df_copy['person_age'] = pd.cut(df_copy['person_age'], bins=[18, 25, 35, 50], labels=['18-25', '26-35', '36-50'])

df_copy['person_income'] = pd.qcut(df_copy['person_income'], q=5, duplicates='drop')

df_copy['loan_amnt'] = pd.qcut(df_copy['loan_amnt'], q=4, duplicates='drop')
df_copy['loan_int_rate'] = pd.qcut(df_copy['loan_int_rate'], q=4, duplicates='drop')
df_copy['loan_percent_income'] = pd.qcut(df_copy['loan_percent_income'], q=4, duplicates='drop')
df_copy['cb_person_cred_hist_length'] = pd.qcut(df_copy['cb_person_cred_hist_length'], q=4, duplicates='drop')

for col in features:
    counts = df_copy.groupby([col, 'loan_status']).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    ax = percentages.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Pastel1')

    plt.title(f'Stacked Percentage Bar Chart: {col} vs Loan Status')
    plt.xlabel(col)
    plt.ylabel('Percentage (%)')
    plt.legend(title='Loan Status', labels=['Not Paid', 'Paid'])

    for i, vals in enumerate(zip(percentages[0], percentages[1])):
        not_paid, paid = vals
        ax.text(i, not_paid / 2, f'{not_paid:.1f}%', ha='center', va='center', color='black', fontsize=9)
        ax.text(i, not_paid + paid / 2, f'{paid:.1f}%', ha='center', va='center', color='black', fontsize=9)

    plt.tight_layout()
    plt.show()




loan_vs_default = pd.crosstab(df['cb_person_default_on_file'], df['loan_status'], normalize='index') * 100

print(loan_vs_default)



import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='cb_person_default_on_file', hue='loan_status', data=df)
plt.title('person_default')
plt.show()



correlation = df.corr(numeric_only=True)['loan_status'].sort_values(ascending=False)
 

top_corr = correlation.drop('loan_status').head(10)
 

plt.figure(figsize=(10,6))

sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')

plt.title("Top 10 Features Correlated with loan_status")

plt.xlabel("Correlation with loan_status")

plt.ylabel("Feature")

plt.tight_layout()

plt.show()


import pandas as pd
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

outlier_ratios = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5* IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    ratio = len(outliers) / len(df) * 100
    outlier_ratios.append((col, ratio))

outlier_df = pd.DataFrame(outlier_ratios, columns=['Column', 'Outlier_Percentage'])

plt.figure(figsize=(10, 6))
plt.bar(outlier_df['Column'], outlier_df['Outlier_Percentage'])
plt.title(' (Outliers) ')
plt.ylabel('(%)')
plt.xlabel('ا')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



Q1 = df['person_age'].quantile(0.25)
Q3 = df['person_age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 -2.5 * IQR
upper_bound = Q3 + 2.5 * IQR

condition = (df['person_age'] < lower_bound) | (df['person_age'] > upper_bound)
outliers = df.loc[condition]

print("The number of outliers in person_age:", len(outliers))
print(outliers['person_age'])



print(df['person_age'].value_counts())



print(len(df[df['person_age'] == 40]))



rows_40 = df[df['person_age'] == 40]
print(rows_40.duplicated().sum())   



print(df.duplicated().sum())



df.info()


print(df['person_home_ownership'].value_counts())


print(df['loan_intent'].value_counts())


print(df['loan_grade'].value_counts())


print(df['cb_person_default_on_file'].value_counts())


grade_A = df[df['loan_grade'] == 'A']

# Count of people in grade 'A' who did not get a loan
no_loan_count = grade_A[grade_A['loan_status'] == 0].shape[0]

# Total number of people in grade 'A'
total_A_count = grade_A.shape[0]

print(f"Number of people in grade 'A' who did not get a loan: {no_loan_count}")
print(f"Total number of people in grade 'A': {total_A_count}")

# Calculate the percentage of people in grade 'A' who did not get a loan
percentage_no_loan = (no_loan_count / total_A_count) * 100
print(f"Percentage of people in grade 'A' who did not get a loan: {percentage_no_loan:.2f}%")



approved_A = df[(df['loan_grade'] == 'A') & (df['loan_status'] == 1)]

ages_approved_A = approved_A['person_age']

age_counts = ages_approved_A.value_counts()

age_percentages = (age_counts / age_counts.sum()) * 100

print(age_percentages.sort_index())



approved_A = df[(df['loan_grade'] == 'A') & (df['loan_status'] == 1)]

# For numerical columns
for col in ['loan_amnt', 'person_income']:
    max_value = approved_A[col].max()
    min_value = approved_A[col].min()
    total_sum = approved_A[col].sum()
    max_percentage = (max_value / total_sum) * 100
    min_percentage = (min_value / total_sum) * 100
    print(f"In column {col}:")
    print(f"  Maximum value = {max_value}, represents {max_percentage:.2f}% of the total sum")
    print(f"  Minimum value = {min_value}, represents {min_percentage:.6f}% of the total sum\n")

# For categorical columns
for col in ['loan_intent', 'person_home_ownership']:
    value_counts = approved_A[col].value_counts()
    top_value = value_counts.idxmax()
    top_count = value_counts.max()
    bottom_value = value_counts.idxmin()
    bottom_count = value_counts.min()
    total_count = value_counts.sum()
    top_percentage = (top_count / total_count) * 100
    bottom_percentage = (bottom_count / total_count) * 100
    print(f"In column {col}:")
    print(f"  Most frequent value = '{top_value}', with {top_percentage:.2f}% of the total")
    print(f"  Least frequent value = '{bottom_value}', with {bottom_percentage:.2f}% of the total\n")



df.info()


df_encoded = df.copy()

label_encoder = LabelEncoder()
df_encoded['cb_person_default_on_file'] = label_encoder.fit_transform(df_encoded['cb_person_default_on_file'])

categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade']
df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)



df_encoded


print(df_encoded['person_age'].value_counts())



X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

X_test_scaled = X_test.copy()
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])



smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("Before balancing:", Counter(y_train))
print("After balancing:", Counter(y_train_res))



X_train_res_scaled




X_test_scaled


from sklearn.model_selection import train_test_split

X = df_scaled.drop('loan_status', axis=1)
y = df_scaled['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")



X_train


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

roc_data = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    roc_data[name] = (fpr, tpr, roc_auc)

plt.figure(figsize=(10, 7))

for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Random Forest vs XGBoost')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

print("Before SMOTE balancing:")
roc_data_before = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Classification Report (Before SMOTE):")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_data_before[name] = (fpr, tpr, roc_auc)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE balancing:")
print("Training set class distribution:", Counter(y_train_res))

roc_data_after = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Classification Report (After SMOTE):")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_data_after[name] = (fpr, tpr, roc_auc)

plt.figure(figsize=(12, 6))

for name, (fpr, tpr, roc_auc) in roc_data_before.items():
    plt.plot(fpr, tpr, lw=2, linestyle='--', label=f'{name} Before SMOTE (AUC = {roc_auc:.3f})')

for name, (fpr, tpr, roc_auc) in roc_data_after.items():
    plt.plot(fpr, tpr, lw=2, label=f'{name} After SMOTE (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Before and After SMOTE')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)  
con = confusion_matrix(y_test, y_pred)   

print(con)



from collections import Counter
from sklearn.metrics import classification_report, accuracy_score

model.fit(X_train_res, y_train_res)


y_train_pred = model.predict(X_train_res)
print("تقييم على بيانات التدريب (بعد SMOTE):")
print(f"الدعم في التدريب: {Counter(y_train_res)}")
print(classification_report(y_train_res, y_train_pred))


y_test_pred = model.predict(X_test)
print("تقييم على بيانات الاختبار (الأصلية):")
print(f"الدعم في الاختبار: {Counter(y_test)}")
print(classification_report(y_test, y_test_pred))


