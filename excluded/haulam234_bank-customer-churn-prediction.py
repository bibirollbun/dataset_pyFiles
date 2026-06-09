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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif


import warnings
warnings.filterwarnings('ignore')


df_raw = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/train.csv")
df_raw.head()


df_test = pd.read_csv("/kaggle/input/bank-customer-churn-prediction-challenge/test.csv")


df_raw.info()


df_raw.describe()


print("Duplicated check: " ,df_raw.duplicated().sum())
print("Null value check:\n",df_raw.isna().sum())


df = df_raw.copy()


fig, axes = plt.subplots(1, 2, figsize=(7, 5))
exited_counts = df['Exited'].value_counts(normalize=True).sort_index()
# Number of customers by Exited
sns.countplot(x='Exited', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Number of customers by Exited')
axes[0].set_xlabel('Exited')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['0', '1'])

for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].annotate(f'{height}', (p.get_x() + p.get_width() / 2, p.get_y() + height / 2),
                         ha='center', va='top', color='black', fontsize=12)

# Exited distribution
set2_colors = sns.color_palette('Set2')
axes[1].bar(['Exited'], [exited_counts.get(1.0, 0)], label='Yes', color=set2_colors[1], bottom=[exited_counts.get(0.0, 0)])
axes[1].bar(['Exited'], [exited_counts.get(0.0, 0)], label='No', color=set2_colors[0])
axes[1].set_title('Exited Distribution (%)')
axes[1].set_ylabel('Percentage')
axes[1].set_ylim(0, 1)
axes[1].legend(title='Exited')
axes[1].set_xticks([0])
axes[1].set_xticklabels(['Exited'])

for p in axes[1].patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2, p.get_y() + height / 2),
                         ha='center', va='center', color='black', fontsize=12)

plt.tight_layout()
plt.show()


cat_cols = ['Gender', 'Geography','NumOfProducts', 'HasCrCard', 'IsActiveMember']
n_features = len(cat_cols)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(x=col, hue='Exited', data=df, ax=axes[i], palette="tab10")
    axes[i].set_title(f'{col} by Exited')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


enc = OneHotEncoder(sparse_output=False)
geo_encoded = enc.fit_transform(df[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=enc.get_feature_names_out(['Geography']))
df = pd.concat([df, geo_encoded_df], axis=1)


df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})


number_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
n_col = 3
n_features = len(df[number_cols].columns)
n_row = math.ceil(n_features / n_col)
fig, axes = plt.subplots(n_row, n_col, figsize=(5*n_col, 4* n_row))
axes=axes.flatten()
for i, col in enumerate(number_cols):
    sns.histplot(df[col], ax=axes[i], kde=True)
    axes[i].set_title(col, fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].grid(True)
    axes[i].tick_params(axis='x', rotation=45, size=15)
    axes[i].tick_params(axis='y', rotation=0)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# Check outlier via boxplots
number_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
n_features = len(number_cols)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
axes = axes.flatten()

for i, col in enumerate(number_cols):
    sns.boxplot(y=col, data=df, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_ylabel(col)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


number_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
n_features = len(number_cols)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(number_cols):
    sns.boxplot(x='Exited', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} by Exited')
    axes[i].set_xlabel('Exited (0=No, 1=Yes)')
    axes[i].set_ylabel(col)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


class IQR():
    def __init__(self):
        self.lower = {}
        self.upper = {}
        self.outliers = 0
    def lower_upper_iqr(self, df, col, col_gr='Exited', factor=1.5):
        outlier_g = []
        for g in df[col_gr].unique():
            s = df.loc[df[col_gr] == g, col]
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            lower = s[s >= (Q1 - factor * IQR)].min()
            upper = s[s <= (Q3 + factor * IQR)].max()
            print(f"Group {g} Lower:{len(s[s<lower])} Upper:{len(s[s>upper])}")
            self.lower[g] = lower
            self.upper[g] = upper
            outlier_g.append(len(s[s<lower]) + len(s[s>upper]))
        self.outliers = sum(outlier_g)


# Create objects for required columns
iqr_age = IQR()
iqr_crescore = IQR()
iqr_estsalary = IQR()


df1 = df.copy()


# Check the number of outliers
iqr_age.lower_upper_iqr(df1, 'Age')
iqr_age.lower, iqr_age.upper, iqr_age.outliers


iqr_crescore.lower_upper_iqr(df1, 'CreditScore')
iqr_crescore.lower, iqr_crescore.upper, iqr_crescore.outliers


iqr_estsalary.lower_upper_iqr(df1, 'EstimatedSalary')
iqr_estsalary.lower, iqr_estsalary.upper, iqr_estsalary.outliers


# Create function to filter outliers for columns simultaneously
def compute_all_thresholds(df, cols, group_col, factor=1.5):
    thresholds = {}
    for col in cols:
        iqr = IQR()
        iqr.lower_upper_iqr(df, col, col_gr=group_col, factor=factor)
        thresholds[col] = (iqr.lower.copy(), iqr.upper.copy()) # save the values of upper, lower thresholds of each column
    return thresholds

def apply_all_thresholds(df, cols, group_col, thresholds):
    mask = pd.Series(True, index=df.index)
    for col in cols:
        lower_dict, upper_dict = thresholds[col]
        lower = df[group_col].map(lower_dict)
        upper = df[group_col].map(upper_dict)
        mask &= (df[col] >= lower) & (df[col] <= upper)
        #group_mask = df[group_col] == value
        #mask &= ~group_mask | ((df[col] >= lower) & (df[col] <= upper))
    return df[mask]


# apply all thresholds at once for 3 variables for group Non_Exited
cols = ['EstimatedSalary','CreditScore', 'Age']
thresholds = compute_all_thresholds(df1, cols, 'Exited')
df1 = apply_all_thresholds(df1, cols, 'Exited', thresholds)


print("df: ",df.shape)
print("df1: ",df1.shape)


number_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
n_features = len(number_cols)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(number_cols):
    sns.boxplot(x='Exited', y=col, data=df1, ax=axes[i])
    axes[i].set_title(f'{col} by Exited')
    axes[i].set_xlabel('Exited (0=No, 1=Yes)')
    axes[i].set_ylabel(col)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


df1.drop(columns=['CustomerId', 'id', 'Geography', 'Surname'], inplace=True)


# Check correlation
corr = df1.corr()
plt.figure(figsize=(10,8)) 
sns.heatmap(corr, annot=True, cmap='coolwarm')


def feature_engineering(df):
    df['Tenure_Products_Ratio'] = df.apply(lambda x: (x['Tenure']+1)/x['NumOfProducts'], axis=1)
    df['Age_Tenure_Ratio'] = df['Age'] / (df['Tenure'] + 1)
    df['LogBalance'] = np.log(df['Balance'] + 1)
    #df['HasBal_CrCard'] = df['Balance'] * df['HasCrCard']
    return df


df1 = feature_engineering(df1)


# List of numerical features to compare
number_cols = df1.drop(columns=['Exited']).select_dtypes(include=['int64', 'float64']).columns.tolist()
n_features = len(number_cols)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(number_cols):
    sns.kdeplot(data=df1, x=col, hue='Exited', fill=True, common_norm=False, 
                palette={0: 'blue', 1: 'red'}, alpha=0.4, ax=axes[i])
    axes[i].set_title(f'{col} by Exited')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()


# Create and fit the selector
df_select = df1.drop(columns=['Exited'])
selector = SelectKBest(f_classif)
selector.fit(df_select, df1['Exited'])

# Get scores for all features
feature_scores = selector.scores_
selected_indices = selector.get_support(indices=True)

# Get feature names
if hasattr(df_select, 'columns'):
    feature_names = df_select.columns
else:
    feature_names = [f'Feature_{i}' for i in range(df_select.shape[1])]

# Create ranking
feature_ranking = list(zip(feature_names, feature_scores))
feature_ranking.sort(key=lambda x: x[1], reverse=True)

# Print results
print("Feature Name                Score")
print("-" * 35)
for feature_name, score in feature_ranking:
    print(f"{feature_name:<20} {score:.4f}")

# plt.subplots(figsize = (5,5))
# sns.heatmap(corr,annot = True,linewidths = 0.4,linecolor = 'black');
# plt.title('Correlation w.r.t Outcome');


corr = df1.corr()
plt.figure(figsize=(15,12))
sns.heatmap(corr, annot=True, cmap='coolwarm')


X = df1[['Age', 'NumOfProducts', 'Geography_Germany', 'IsActiveMember', 
        'Tenure_Products_Ratio', 'Gender', 'Age_Tenure_Ratio', 'LogBalance', 
          'Geography_France']]
y = df1['Exited']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2)


# Train the model by using Random Forest
threshold = 0.5

# Create complete pipeline (prevents data leakage)
def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

# Define parameter grid
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 10, 20]
}

# Single GridSearchCV with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = create_pipeline()
grid_search_rf = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    refit=True
)

grid_search_rf.fit(X_train, y_train)



best_f1_score = grid_search_rf.best_score_
best_f1_score


best_model = grid_search_rf.best_estimator_
y_test_prob= best_model.predict_proba(X_test)[:,1]
y_test_pred = best_model.predict(X_test)

print(classification_report(y_test, y_test_pred))

fpr, tpr, _ = metrics.roc_curve(y_test, y_test_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Validation Set")
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix
y_test_pred = grid_search_rf.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()


## Find best k with XGBoost
model_xgb = XGBClassifier(use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    class_weight='balanced')
param_grid_xgb = {
    'clf__n_estimators': [100, 200, 300],  
    'clf__learning_rate': [0.01, 0.1, 0.3],  
    'clf__max_depth': [3, 6, 9],         
}
def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model_xgb)
    ])

# GridSearchCV with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = create_pipeline()

grid_search_xgb = GridSearchCV(
    pipeline,
    param_grid=param_grid_xgb,
    cv=cv,
    scoring='f1',
    refit=True
)

grid_search_xgb.fit(X_train, y_train)


best_f1_score = grid_search_xgb.best_score_
best_f1_score


threshold = 0.5

y_test_pred_xgb = grid_search_xgb.predict(X_test)
y_test_prob_xgb = grid_search_xgb.predict_proba(X_test)

print(classification_report(y_test, y_test_pred_xgb))

fpr, tpr, _ = metrics.roc_curve(y_test, y_test_prob_xgb[:,1])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Validation Set")
plt.legend()
plt.grid(True)
plt.show()
cm = confusion_matrix(y_test, y_test_pred_xgb)
ConfusionMatrixDisplay(cm).plot()
plt.show()


## Train model using Logistic Regression
param_grid = {
    'selector__k': range(5, X_train.shape[1]+1),
    'clf__C': [0.001, 0.01, 0.3], 
    'clf__penalty': ['l1', 'l2'],          
    'clf__solver': ['liblinear', 'saga'],
    'clf__max_iter': [100, 200, 300],
    'clf__class_weight': ['balanced']
}
def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif)),
        ('clf', LogisticRegression())
    ])

# GridSearchCV with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = create_pipeline()

grid_search_lr = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    refit=True
)

grid_search_lr.fit(X_train, y_train)


y_test_pred_lr = grid_search_lr.predict(X_test)
y_test_prob_lr = grid_search_lr.predict_proba(X_test)

print(classification_report(y_test, y_test_pred_lr))

fpr, tpr, _ = metrics.roc_curve(y_test, y_test_prob_lr[:,1])
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Validation Set")
plt.legend()
plt.grid(True)
plt.show()
cm = confusion_matrix(y_test, y_test_pred_lr)
ConfusionMatrixDisplay(cm).plot()
plt.show()


best_model = grid_search_rf.best_estimator_.named_steps['clf']
feature_importances = best_model.feature_importances_

selected_features = X_train.columns
# Combine into a DataFrame
feat_imp = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)

# Combine into a DataFrame
feat_imp = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_imp['Feature'], feat_imp['Importance'])
plt.gca().invert_yaxis()
plt.title(f'Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# print top features
print(feat_imp.head(10))


# Transform
df_test['Gender'] = df_test['Gender'].map({'Male': 1, 'Female': 0})

enc = OneHotEncoder(sparse_output=False)
geo_encoded = enc.fit_transform(df_test[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=enc.get_feature_names_out(['Geography']))
df_test = pd.concat([df_test, geo_encoded_df], axis=1)


# Feature engineering
df_test1 = feature_engineering(df_test)
df_test1.head()


# Select required features
features = X_train.columns.tolist()
X_test_sub = df_test1[features]


# Apply random forest model earlier
y_test_sub = grid_search_rf.predict_proba(X_test_sub)[:,1].round(1)
y_test_sub = pd.DataFrame(y_test_sub, columns=['Exited'])
y_test_sub.head()


df_test = pd.concat([df_test, y_test_sub], axis='columns')
df_test.head()


df_test_sub = df_test[['id', 'Exited']]


df_test_sub.to_csv('df_test_sub.csv', index=False)




