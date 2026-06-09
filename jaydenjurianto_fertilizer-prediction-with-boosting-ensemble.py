import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import gc


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train['Soil_Crop'] = train['Soil Type'] + "_" + train['Crop Type']
test['Soil_Crop'] = test['Soil Type'] + "_" + test['Crop Type']


cat_cols = ['Soil Type', 'Crop Type', 'Soil_Crop']


for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(all_vals)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


for col in cat_cols:
    freq_enc = train[col].value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq_enc)
    test[f'{col}_freq'] = test[col].map(freq_enc)


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train['label'] = LabelEncoder().fit_transform(train['Fertilizer Name'])
target = train['label'].values


for col in cat_cols:
    train[f'{col}_te'] = np.nan
    for train_idx, val_idx in folds.split(train, target):
        mean_target = train.iloc[train_idx].groupby(col)['label'].mean()
        train.loc[train.index[val_idx], f'{col}_te'] = train.loc[train.index[val_idx], col].map(mean_target)
    mean_target_full = train.groupby(col)['label'].mean()
    test[f'{col}_te'] = test[col].map(mean_target_full)


features = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
    'Soil Type', 'Crop Type', 'Soil_Crop',
    'Soil Type_freq', 'Crop Type_freq', 'Soil_Crop_freq',
    'Soil Type_te', 'Crop Type_te', 'Soil_Crop_te'
]


X = train[features]
X_test = test[features]
y = target


cat_preds = np.zeros((len(X_test), len(np.unique(y))))
lgb_preds = np.zeros_like(cat_preds)
xgb_preds = np.zeros_like(cat_preds)

cat_features_idx = [features.index(c) for c in ['Soil Type', 'Crop Type', 'Soil_Crop']]


print("Training CatBoost...")
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"CatBoost fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_pool = cb.Pool(X_train, y_train, cat_features=cat_features_idx)
    val_pool = cb.Pool(X_val, y_val, cat_features=cat_features_idx)

    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    cat_preds += model.predict_proba(X_test) / folds.n_splits
    gc.collect()



from lightgbm import LGBMClassifier

print("Training LightGBM...")
lgb_preds = np.zeros((len(X_test), len(np.unique(y))))

lgb_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_estimators=1000,
    verbosity=-1
)

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"LightGBM fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        early_stopping_rounds=50,
        verbose=100
    )

    lgb_preds += model.predict_proba(X_test) / folds.n_splits
    gc.collect()


from xgboost import XGBClassifier

print("Training XGBoost...")
xgb_preds = np.zeros((len(X_test), len(np.unique(y))))

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_estimators=1000,
    use_label_encoder=False,
    verbosity=1
)

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"XGBoost fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mlogloss',
        early_stopping_rounds=50,
        verbose=100
    )

    xgb_preds += model.predict_proba(X_test) / folds.n_splits
    gc.collect()


print("Ensembling predictions...")
final_preds = 0.4 * cat_preds + 0.3 * lgb_preds + 0.3 * xgb_preds


label_freq = train['label'].value_counts(normalize=True).sort_index()
alpha = 0.5
logits = np.log(final_preds + 1e-12)
logits += alpha * np.log(1 / (label_freq.values + 1e-12))
probs = np.exp(logits)
probs /= probs.sum(axis=1, keepdims=True)


le_fert = LabelEncoder()
le_fert.fit(train['Fertilizer Name'])


top_3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top_3_ferts = [[le_fert.classes_[idx] for idx in row] for row in top_3_preds]


submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(preds) for preds in top_3_ferts]
})


submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


pd.read_csv("submission.csv")

