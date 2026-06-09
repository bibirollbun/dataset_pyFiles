
import kagglehub
kagglehub.login()
playground_series_s4e10_path = kagglehub.competition_download('playground-series-s4e10')
print('Data source import complete.')   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df_train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
df_submision = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')


def stacked_bar_plot(df, feature, target='loan_status'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='coolwarm')
    plt.title(f'Stacked Bar Plot of {feature} vs {target}')
    plt.ylabel('Proportion')
    plt.show()
stacked_bar_plot(df_train, 'loan_intent')

# stacked_bar_plot(df_train, 'person_home_ownership')

sns.countplot(data=df_train, x='person_home_ownership')
plt.title('Distribution of Loan Applicants by Home Ownership')
plt.show()

# sns.countplot(data=df_train, x='loan_intent')
# plt.title('Loan Intent Distribution')
# plt.xticks(rotation=45)
# plt.show()

# sns.countplot(data=df_train, x='loan_grade', hue='loan_status')
# plt.title('Loan Default Rate by Loan Grade')
# plt.show()

# sns.countplot(data=df_train, x='cb_person_default_on_file', hue='loan_status')
# plt.title('Loan Default by Prior Default Status')
# plt.show()

def plot_boxplots(df, columns):
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.boxplot(y=df[col], color='lightblue')
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.show()

# plot_boxplots(df_train, ['person_income', 'loan_amnt'])

# plt.figure(figsize=(10, 6))
# sns.kdeplot(df_train[df_train['loan_status'] == 1]['loan_amnt'], label='Default', fill=True)
# sns.kdeplot(df_train[df_train['loan_status'] == 0]['loan_amnt'], label='Non-Default', fill=True)
# plt.title('CDF of Loan Amount by Loan Status')
# plt.xlabel('Loan Amount')
# plt.ylabel('Density')
# plt.legend()
# plt.show()



# num_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']
# plt.figure(figsize=(15, 10))
# for i, feature in enumerate(num_features, 1):
#     plt.subplot(2, 2, i)
#     sns.histplot(df_train[feature], bins=30, kde=True)
#     plt.title(f'Distribution of {feature}')
# plt.tight_layout()
# plt.show()

# df_train.hist(figsize=(12, 10), color='skyblue', edgecolor='black')
# plt.gcf().set_facecolor('yellow')
# plt.tight_layout()
# plt.show()

# subset_features = ['loan_amnt', 'loan_int_rate', 'person_income', 'person_age', 'loan_status']
# sns.pairplot(df_train[subset_features], hue='loan_status')
# plt.title('Pair Plot of Selected Features')
# plt.show()

from sklearn.preprocessing import LabelEncoder
privat_path = kagglehub.dataset_download('ulyanar/privat')
df_submission = pd.read_csv('/kaggle/input/privat/test_data.csv')
def preprocess_data(df_train, df_test):
    label_enc = LabelEncoder()
    label_cols = ['person_home_ownership', 'loan_grade', 'cb_person_default_on_file']
    for col in label_cols:
        df_train[col] = label_enc.fit_transform(df_train[col])
        df_test[col] = label_enc.transform(df_test[col])
    df_train = pd.get_dummies(df_train, columns=['loan_intent'], drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['loan_intent'], drop_first=True)
    target_col = 'loan_status'
    train_columns = df_train.drop(columns=[target_col]).columns
    df_test = df_test.reindex(columns=train_columns, fill_value=0)
    return df_train, df_test

df_train_processed, df_test_processed = preprocess_data(df_train, df_test)

df_train = df_train_processed
df_test = df_test_processed

df_train.shape,df_test.shape

correlation_matrix = df_train.corr()
plt.figure(figsize=(15, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.2)
plt.gcf().set_facecolor('#00FFFF')
plt.title('Correlation Matrix')
plt.show()

df_train.columns

df_test.isnull().sum()

def feature_engineering(df):

    df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate']
    df['income_per_year_emp'] = df['person_income'] / (df['person_emp_length'])
    df['cred_hist_to_age_ratio'] = df['cb_person_cred_hist_length'] / df['person_age']
    df['int_to_loan_ratio'] = df['loan_int_rate'] / df['loan_amnt']
    df['loan_int_emp_interaction'] = df['loan_int_rate'] * df['person_emp_length']
    df['debt_to_credit_ratio'] = df['loan_amnt'] / df['cb_person_cred_hist_length']
    df['int_to_cred_hist'] = df['loan_int_rate'] / df['cb_person_cred_hist_length']
    df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'])
    df['loan_amt_per_emp_year'] = df['loan_amnt'] / (df['person_emp_length'])
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['int_to_loan_ratio'] = df['loan_int_rate'] / df['loan_amnt']
    df['loan_to_age_ratio'] = df['loan_amnt'] / df['person_age']
    df['normalized_interest_burden'] = df['loan_int_rate'] * df['loan_percent_income']
    df['emp_to_age_ratio'] = df['person_emp_length'] / df['person_age']
    
    return df

#median_income = df_train['person_income'].median()
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)

y = df_train['loan_status']
df_train = df_train.drop(['loan_status'],axis=1)
X = df_train


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_train_data = df_train
scaled_test_data = df_test

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

lgb_params = {
    'objective': 'binary',
    'n_estimators': 3000,
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.025,
    'num_leaves': 36,
    'max_depth': 15,
    'min_data_in_leaf': 25,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.9596685778433888,
    'bagging_freq': 3,
    'verbose': -1
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
lgbm_predictions = np.zeros(len(scaled_train_data))
lgbm_true_labels = np.zeros(len(scaled_train_data))
lgbm_test_predictions = np.zeros(len(scaled_test_data))

for fold, (train_idx, val_idx) in enumerate(skf.split(scaled_train_data, y)):
    X_train, X_val = scaled_train_data.iloc[train_idx], scaled_train_data.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgbm_model = LGBMClassifier(**lgb_params, early_stopping_rounds=10)
    lgbm_model.fit(X_train, y_train,
                   eval_set=[(X_val, y_val)],
                   eval_metric='auc'
                   )

    lgbm_fold_preds = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_fold_test_preds = lgbm_model.predict_proba(scaled_test_data)[:, 1]
    lgbm_predictions[val_idx] = lgbm_fold_preds
    lgbm_true_labels[val_idx] = y_val
    lgbm_test_predictions += lgbm_fold_test_preds / n_splits
overall_metric_lgbm = roc_auc_score(lgbm_true_labels, lgbm_predictions)
print("Overall AUC (LGBMClassifier with StratifiedKFold):", overall_metric_lgbm)

catboost_params = {
    'depth': 9,
    'learning_rate': 0.08,
    'bagging_temperature': 0.7979373495258176,
    'l2_leaf_reg': 7,
    'loss_function': 'Logloss',
    'iterations': 600,
    'grow_policy': 'Lossguide',
    'eval_metric': 'AUC',
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

catboost_predictions = np.zeros(len(scaled_train_data))
catboost_true_labels = np.zeros(len(scaled_train_data))
catboost_test_predictions = np.zeros(len(scaled_test_data))

for fold, (train_idx, val_idx) in enumerate(skf.split(scaled_train_data, y)):
    X_train, X_val = scaled_train_data.iloc[train_idx], scaled_train_data.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    catboost_model = CatBoostClassifier(**catboost_params, early_stopping_rounds=10, verbose=0)
    catboost_model.fit(X_train, y_train,
                       eval_set=(X_val, y_val)
                       )

    catboost_fold_preds = catboost_model.predict_proba(X_val)[:, 1]
    catboost_fold_test_preds = catboost_model.predict_proba(scaled_test_data)[:, 1]
    catboost_predictions[val_idx] = catboost_fold_preds
    catboost_true_labels[val_idx] = y_val
    catboost_test_predictions += catboost_fold_test_preds / n_splits
overall_metric_catboost = roc_auc_score(catboost_true_labels, catboost_predictions)
print("Overall AUC (CatBoostClassifier with StratifiedKFold):", overall_metric_catboost)



from lightgbm import plot_importance
plot_importance(lgbm_model,figsize=(10, 6))


lgbm_model

import lime
import lime.lime_tabular

df_train.shape

df_train.head()

df_test.shape

df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_train.fillna(df_train.mean(), inplace=True)
print("DataFrame after replacing inf with column means:")
df_train.head()

df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.fillna(df_test.mean(), inplace=True)
print("DataFrame after replacing inf with column means:")
df_test.head()

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(X_train.mean(), inplace=True)
X = X_train.values
explainer = lime.lime_tabular.LimeTabularExplainer(X,
                                                   feature_names=X_train.columns,
                                                   class_names=['0', '1'],
                                                   kernel_width=5)

redict_fn_lgb = lambda x: lgbm_model.predict_proba(x).astype(float)

df_submision.head()

from sklearn.metrics import roc_curve, auc
fpr_lgbm, tpr_lgbm, _ = roc_curve(lgbm_true_labels, lgbm_predictions)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

fpr_catboost, tpr_catboost, _ = roc_curve(catboost_true_labels, catboost_predictions)
roc_auc_catboost = auc(fpr_catboost, tpr_catboost)

df_submision['loan_status'] = lgbm_test_predictions*0.2 + catboost_test_predictions*0.8 


plt.figure(figsize=(10, 6))
plt.plot(fpr_lgbm, tpr_lgbm, color='blue', lw=2, label=f'LightGBM ROC Curve (AUC = {roc_auc_lgbm:.4f})')
plt.plot(fpr_catboost, tpr_catboost, color='green', lw=2, label=f'CatBoost ROC Curve (AUC = {roc_auc_catboost:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

df_submission.to_csv('submission.csv', index=False)

