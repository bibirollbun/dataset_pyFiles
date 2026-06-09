import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import warnings


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')

X = train.drop(columns='y', axis=1)
y = train['y']


numerical_features = []
categorical_features = []

for col in X.columns:
  if X[col].dtype != 'object':
    numerical_features.append(col)
  else:
    categorical_features.append(col)


print(numerical_features)
print(categorical_features)


print(y.value_counts())


def numerical_features_plot(df, feature, y):
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  plt.suptitle(f'Analysis of {feature}')

  sns.boxplot(data=df, x=feature, y=y, ax=axes[0], orient='h')
  sns.violinplot(df, x=feature, y=y, ax=axes[1], orient='h')
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    sns.histplot(df, x=feature, hue=(y), ax=axes[2], kde=True)

  plt.tight_layout(rect=[0, 0, 1, 0.98])
  plt.show()
  print('\n')

for feature in numerical_features:
  numerical_features_plot(X, feature, y)


for feature in categorical_features:
  value_count = X[feature].value_counts()
  value_count.sort_values()
  value_count = value_count.head(5)

  print(value_count, "\n")


def plot_heatmap(feature1, feature2):
  ct = pd.crosstab(index=X[feature1], columns=X[feature2], normalize='index') * 100

  plt.figure(figsize=(5, 5))
  sns.heatmap(ct, cmap='YlGnBu', linewidths=.5, annot=True)
  plt.title(f'{feature1} vs {feature2}')
  plt.xlabel('')
  plt.ylabel('')

  plt.show()


plot_heatmap('job', 'education')


plot_heatmap('loan', 'marital')


plot_heatmap('education', 'marital')


print('na: ', X.isna().sum().sum())
print('duplicates: ', X.duplicated().sum())


def preprocess(df):
  df = df.copy()

  # feature engineering
  df['long_duration'] = (df['duration'] > 300).astype(int)
  df['multiple_contacts'] = (df['campaign'] >= 2).astype(int)
  df['previous_fail'] = ((df['poutcome'] == 'failure') & (df['multiple_contacts'] == True)).astype(int)
  df['log_balance']  = np.log1p(df['balance'] - df['balance'].min() + 1)
  df['log_duration'] = np.log1p(df['duration'])

  # encoding

  # mapping bool values
  df['default'] = df['default'].map({'no' : 0, 'yes' : 1})
  df['housing'] = df['housing'].map({'no' : 0, 'yes' : 1})
  df['loan'] = df['loan'].map({'no' : 0, 'yes' : 1})

  # encoding remaining cat values
  from sklearn.preprocessing import OneHotEncoder
  enc = OneHotEncoder(handle_unknown='ignore')

  cols_to_encode = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

  encoded_columns = enc.fit_transform(df[cols_to_encode]).toarray()
  encoded_columns = pd.DataFrame(encoded_columns, columns=enc.get_feature_names_out(cols_to_encode))

  df = df.drop(cols_to_encode, axis=1)
  df = pd.concat([df, encoded_columns], axis=1)

  return df


X_preprocessed = preprocess(X)


X_preprocessed


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


positive = y_train.sum()
negative = y_train.shape[0] - positive

scale_pos_weight_value = negative / positive


def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 150, 250),
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.08),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'scale_pos_weight' : scale_pos_weight_value,
        'random_state': 42
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = xgb.XGBClassifier(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))

    return sum(auc_scores) / len(auc_scores)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=5)

# print("best params:", study.best_params)
# print("best rocauc:", study.best_value)


best_params = {
  'n_estimators': 205,
  'learning_rate': 0.0636418936875509,
  'max_depth': 8,
  'subsample': 0.5284304312609354,
  'colsample_bytree': 0.860984665584364,
  'scale_pos_weight' : scale_pos_weight_value,
  'objective': 'binary:logistic',
  'eval_metric': 'logloss',
}


model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
X_test = preprocess(test)

test_pred_prob = model.predict_proba(X_test.drop(columns='id', axis=1))[:, 1]

submission = pd.DataFrame({
    'id': test['id'],
    'y': test_pred_prob
})

submission.to_csv('submission.csv', index=False)
print("Submission saved!")

