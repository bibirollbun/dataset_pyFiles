!pip install scikeras



import pandas as pd
import numpy as np
import optuna
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb
import catboost as cat
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from scikeras.wrappers import KerasClassifier


df = pd.read_csv(r'/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e8/test.csv')


print('Train missing sum')
train_missing = df.isna().sum()
print(train_missing)


education = ['unknown','primary','secondary','tertiary']
month = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
default = ['no','yes']
housing = ['no','yes']
loan = ['no','yes']


ord = OrdinalEncoder(categories = [education, month, default, housing, loan])
ordinal_column_encode = ['education','month', 'default', 'housing', 'loan']

df[ordinal_column_encode] = ord.fit_transform(df[ordinal_column_encode]).astype(int)
df_test[ordinal_column_encode] = ord.fit_transform(df_test[ordinal_column_encode]).astype(int)


column_to_encode = df.select_dtypes('object')
for i in column_to_encode:
    label = LabelEncoder()
    df[i] = label.fit_transform(df[i])
    df_test[i] = label.transform(df_test[i])


df['bal_loan'] = df['balance'] * df['loan']
df['bal_housing'] = df['balance'] * df['housing']
df['dur_by_campaign'] = df['duration'] / (df['campaign'] + 1)
df['prev_poutcome'] = df['previous'] * df['poutcome']

df_test['bal_loan'] = df_test['balance'] * df_test['loan']
df_test['bal_housing'] = df_test['balance'] * df_test['housing']
df_test['dur_by_campaign'] = df_test['duration'] / (df_test['campaign'] + 1)
df_test['prev_poutcome'] = df_test['previous'] * df_test['poutcome']



X = df.drop(columns = ['id','y'])
Y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


X


# models = [
#     ('RF', RandomForestClassifier()),
#     ('XGB', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
#     ('CatBoost', cat.CatBoostClassifier(verbose=0)),
#     ('LGBM', lgb.LGBMClassifier())
# ]

# for i, (name, model) in enumerate(models):
#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     result = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
#     print(f'{name}: {result.mean()}')


X_train = X_train.to_numpy().astype(np.float32)
X_test = X_test.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)



def build_nn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

BATCH_SIZE = 1024
EPOCH = 300

nn_clf = KerasClassifier(
    model=build_nn,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    verbose=0
)



xgb_param = {
    'n_estimators': 1703,
    'learning_rate':  0.021675185548401536,
    'gamma': 0.5069007000161877,
    'max_depth': 14,
    'min_child_weight': 1.8684037245714267,
    'subsample': 0.888357145364407,
    'colsample_bytree': 0.44548516105944713,
    'reg_lambda': 3.379795153213554,
    'reg_alpha': 4.790750514668356
}
cat_param = {
    'iterations': 2519,
    'depth': 7,
    'learning_rate': 0.07776383402823714,
    'l2_leaf_reg': 1.7593596817227857,
    'bagging_temperature': 0.19375681623434676,
    'border_count': 240,
    'verbose': 100,
    'random_state': 42
}
lgbm_param = {
    'num_leaves': 112,
    'learning_rate': 0.06303277984422288,
    'max_depth': 11, 'n_estimators': 1073,
    'min_child_samples': 58,
    'subsample': 0.5763146517812537,
    'colsample_bytree': 0.4266264605394735,
    'reg_alpha': 4.743334668241185,
    'reg_lambda': 6.452586855834584
}


xgb_clf = xgb.XGBClassifier(**xgb_param)
cat_clf = cat.CatBoostClassifier(**cat_param)
lgbm_clf = lgb.LGBMClassifier(**lgbm_param)

meta = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)

stack_model = StackingClassifier(
    estimators=[("nn",nn_clf),("cat", cat_clf),("xgb", xgb_clf),("lgbm",lgbm_clf)],
    final_estimator=meta,
    stack_method="predict_proba",   
    cv=5,
    n_jobs=-1
)

stack_model.fit(X_train, y_train)

y_pred = stack_model.predict_proba(X_test)[:, 1]

# đánh giá
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred)
print("Stacking ROC-AUC:", score)


X_test_sub = df_test.drop(columns=["id"]).to_numpy().astype(np.float32)

y_pred = stack_model.predict_proba(X_test_sub)[:, 1]

submission_df = pd.DataFrame({
    "id": df_test["id"],
    "y": y_pred
})
submission_df.to_csv(r'submission.csv', index = False)


submission_df.head()

