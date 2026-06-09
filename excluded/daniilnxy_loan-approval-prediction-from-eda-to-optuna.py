import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import torch as t
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


test_df = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv').drop(columns=['id'], axis=1)
train_df  = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv').drop(columns=['id'], axis=1)
sub = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')


train_df.head()


train_df.isna().sum()


train_df.nunique()



train_df.describe()



train_df.dtypes



train_df.shape


test_df.head()


test_df.info()


test_df.isna().sum()


test_df.shape


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


sns.set_style('whitegrid')
sns.set_palette('pastel')

ax = sns.countplot(x='loan_status', data=train_df, order=[0, 1]) 
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title('Distribution of Loan Status', fontsize=16)
plt.xlabel('Loan Status (0 or 1)', fontsize=14)
plt.ylabel('Number of People', fontsize=14)

plt.show()


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_df['person_home_ownership']
df_label_encoded = label_encoder.fit_transform(train_df['person_home_ownership'])


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


sns.set_style('whitegrid')
sns.set_palette('pastel')

ax = sns.countplot(x='person_home_ownership', data=train_df) 
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title('Person home ownership', fontsize=16)
plt.xlabel('Ownership category', fontsize=14)
plt.ylabel('Number of People', fontsize=14)

plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


sns.set_style('whitegrid')
sns.set_palette('pastel')

ax = sns.countplot(x='loan_intent', data=train_df) 
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title('Loan intentions', fontsize=16)
plt.xticks(rotation=40)
plt.xlabel('Type of loan', fontsize=14)
plt.ylabel('Number of People', fontsize=14)
plt.show()


plt.figure(figsize=(18,6))
ax = sns.barplot(x='person_age', y='person_income', data=train_df, estimator=sum) 
plt.xticks(rotation=60)
plt.title("Salary based on age")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('ticks')
sns.set_palette('pastel')

num_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(num_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(train_df[feature], bins=30, kde=True, color=sns.color_palette('pastel')[i-1], edgecolor='black', linewidth=1.2)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))  
sns.boxplot(x='person_age', y='loan_amnt', data=train_df, whis=3)
plt.xlabel('Age in years') 
plt.ylabel('Loan Amount') 

plt.show()


df_encoded = pd.get_dummies(train_df, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])
correlation_matrix = df_encoded.corr()
loan_status_corr = correlation_matrix['loan_status'].sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(y=loan_status_corr.index, x=loan_status_corr.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Features correlation with loan_status")
plt.show()


from sklearn.preprocessing import LabelEncoder
import pandas as pd
ordinal_cols = ['loan_grade']
nominal_cols = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']


def preprocess_data(df_train, df_test, ordinal_cols, nominal_cols):

    label_enc = LabelEncoder()
    for col in ordinal_cols:
        df_train[col] = label_enc.fit_transform(df_train[col])
        df_test[col] = label_enc.transform(df_test[col])
    
    df_train = pd.get_dummies(df_train, columns=nominal_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=nominal_cols, drop_first=True)
    
    train_columns = df_train.drop(columns=['loan_status']).columns
    df_test = df_test.reindex(columns=train_columns, fill_value=0)
    
    return df_train, df_test

df_train_processed, df_test_processed = preprocess_data(train_df, test_df, ordinal_cols, nominal_cols)


def feature_engineering(df):
    median_emp_length = df['person_emp_length'].median()
    df['person_emp_length'] = df['person_emp_length'].replace(0, median_emp_length)

    df['financial_burden'] = df['loan_amnt']*df['loan_int_rate']
    df['income_per_year_emp'] = df['person_income'] / (df['person_emp_length'])
    df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'])
    return df 


df_train_processed = feature_engineering(df_train_processed)
df_test_processed = feature_engineering(df_test_processed)


correlation_matrix = df_train_processed.corr()

high_corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
high_corr_pairs = high_corr_pairs[(high_corr_pairs < 1) & (high_corr_pairs > 0.8)]

features_to_drop = []
for (feature1, feature2), corr in high_corr_pairs.items():
    corr1 = correlation_matrix.loc[feature1, 'loan_status']
    corr2 = correlation_matrix.loc[feature2, 'loan_status']
    if corr1 < corr2:
        features_to_drop.append(feature1)
    else:
        features_to_drop.append(feature2)




features_to_drop


df_train_processed  = df_train_processed.drop(columns=features_to_drop)
df_test_processed = df_test_processed.drop(columns=features_to_drop)


from sklearn.ensemble import RandomForestClassifier

target_feature = 'loan_status'
X_train = df_train_processed.drop(target_feature, axis=1)
y_train = df_train_processed[target_feature]

model = RandomForestClassifier(n_estimators = 100, 
                               random_state=42)
model.fit(X_train, y_train)

feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)




feature_importance



mask = feature_importance>0.01 
feature_importance_filtered = feature_importance[mask].index.to_list()
feature_importance_filtered


df_train_processed = df_train_processed[[target_feature]+ feature_importance_filtered]
df_test_processed = df_test_processed[feature_importance_filtered]


X = df_train_processed.drop(columns=['loan_status'])
y= df_train_processed['loan_status']



from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2,
                                                   random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


X_test_scaled = scaler.transform(df_test_processed)


import optuna

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


def xbgrclassifier_training(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42
    }
    
    model = Pipeline([
        ('scaler', StandardScaler()),  
        ('xgb', XGBClassifier(**params, eval_metric='logloss')) 
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    return scores.mean()



optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize')
study.optimize(xbgrclassifier_training, n_trials=10)

print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)


best_params = study.best_params 

final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(**best_params, random_state=42))
])
final_model.fit(X_train, y_train)


from sklearn.metrics import roc_auc_score

y_val_proba = final_model.predict_proba(X_val)[:, 1] 
auc_val = roc_auc_score(y_val, y_val_proba)
print("Validation AUC ROC:", auc_val)



y_test_pred = final_model.predict_proba(df_test_processed)[:,1]
y_test_pred


y_test_pred.shape


sub['loan_status'] = y_test_pred


sub.to_csv('optuna_proba.csv', index=False)


from catboost import CatBoostClassifier
def catboost_training(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42,
        'verbose': 0  
    }
    
    model = Pipeline([
        ('scaler', StandardScaler()),  
        ('catboost', CatBoostClassifier(**params)) 
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()



optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(catboost_training, n_trials=5)

best_params = study.best_params
best_params['random_state'] = 42


print("Best hyperparameters:", best_params)
print("Best AUC ROC on validation:", study.best_value)


final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('catboost', CatBoostClassifier(**best_params, verbose=0))
])

final_model.fit(X_train, y_train)


y_val_proba = final_model.named_steps['catboost'].predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_val_proba)
print("Validation AUC ROC:", auc_val)



y_test_proba = final_model.named_steps['catboost'].predict_proba(df_test_processed)[:, 1]


sub['loan_status'] = y_test_proba
sub.to_csv('catboost_optuna_proba.csv', index=False)




