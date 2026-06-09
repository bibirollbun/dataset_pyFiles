import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

%pip -q install git+https://github.com/iseedeep/deeprage.git@main
from deeprage.core import val_pie, val_bar, val_all_hist, compare_columns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv').drop('id', axis=1)
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_ids = df_test['id']
df_test  = df_test.drop('id', axis=1)

display(df_train.head())
display(df_test.head())

display(df_train.info())
display(df_test.info())


display(df_train.isnull().sum())
display(df_test.isnull().sum())


val_pie(df_train, 'Soil Type')


val_bar(df_train, 'Crop Type')


val_pie(df_train, 'Fertilizer Name')


val_all_hist(df_train, kde=True, freq=True)


numeric_cols = df_train.select_dtypes(include=['float64', 'int64'])
compare_columns(numeric_cols)


# 1) Encode target
le_target = LabelEncoder().fit(df_train['Fertilizer Name'].astype(str))
y = le_target.transform(df_train['Fertilizer Name'].astype(str))

# 2) Encode Soil & Crop as pandas categories
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder().fit(df_train[col].astype(str))
    df_train[col + '_enc'] = le.transform(df_train[col].astype(str))
    df_test [col + '_enc'] = le.transform(df_test [col].astype(str))
    df_train[col + '_enc'] = df_train[col + '_enc'].astype('category')
    df_test [col + '_enc'] = df_test [col + '_enc'].astype('category')

# 3) Feature engineering
for df in (df_train, df_test):
    df['N_to_P']    = df['Nitrogen']   / (df['Phosphorous'] + 1)
    df['N_to_K']    = df['Nitrogen']   / (df['Potassium']   + 1)
    df['P_to_K']    = df['Phosphorous'] / (df['Potassium']   + 1)
    df['NPK_total'] = df[['Nitrogen','Phosphorous','Potassium']].sum(axis=1)
    df['Temp_Hum']  = df['Temparature'] * df['Humidity']

# 4) Define features
features = [
    'Temparature','Humidity','Moisture',
    'Nitrogen','Potassium','Phosphorous',
    'Soil Type_enc','Crop Type_enc',
    'N_to_P','N_to_K','P_to_K','NPK_total','Temp_Hum'
]

X = df_train[features]
X_test = df_test [features]

# 5) Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 6) LightGBM params & model
params = {
    'objective'        : 'multiclass',
    'num_class'        : len(le_target.classes_),
    'learning_rate'    : 0.05,
    'num_leaves'       : 31,
    'min_data_in_leaf' : 20,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'lambda_l1'        : 0.1,
    'lambda_l2'        : 0.2,
    'metric'           : ['multi_logloss','multi_error'],
    'verbose'          : -1,
    'seed'             : 42
}

model = lgb.LGBMClassifier(**params, n_estimators=3000, n_jobs=-1)

# 7) Fit with callbacks for early stopping & logging
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ]
)

# 8) Local MAP@3 evaluation
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

probs_val   = model.predict_proba(X_val)
top3_idx    = np.argsort(probs_val, axis=1)[:, -3:][:, ::-1]
top3_labels = [[le_target.classes_[i] for i in row] for row in top3_idx]
true_labels = le_target.inverse_transform(y_val)

val_map3 = mapk(true_labels.tolist(), top3_labels, k=3)
val_acc  = accuracy_score(y_val, model.predict(X_val))

print(f'Validation MAP@3: {val_map3:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}')

# 9) Retrain on full data & make submission
best_iter = model.best_iteration_
model.set_params(n_estimators=best_iter)
model.fit(X, y)


probs_test = model.predict_proba(X_test)
top3_test_idx    = np.argsort(probs_test, axis=1)[:, -3:][:, ::-1]
submission_labels = [" ".join(le_target.classes_[i] for i in row) for row in top3_test_idx]

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': submission_labels
})
submission_df.to_csv('submission.csv', index=False)

print("submission.csv created!")

