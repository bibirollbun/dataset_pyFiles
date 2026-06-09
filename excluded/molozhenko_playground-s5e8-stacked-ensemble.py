import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === Configuration ===
class Config:
    state = 42
    n_splits = 5
    target = 'y'
    train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

# === Data preprocessing ===
def transform_data(train, test, target, state=42):
    # Adding dataset to boost performance
    org = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')
    org[target] = (org[target] == 'yes').astype(int)
    train = pd.concat([train, org], ignore_index=True).drop_duplicates()

    # Separating categorical and numerical features
    cat_features = train.select_dtypes(include='object').columns.tolist()
    num_features = train.select_dtypes(include=np.number).columns.drop(target).tolist()

    # Handling missing values
    for df in [train, test]:
        df[cat_features] = df[cat_features].fillna('NaN')
        df[num_features] = df[num_features].fillna(df[num_features].median())

    # Encoding categorical features
    encoder = OrdinalEncoder(dtype=int)
    all_cat_data = pd.concat([train[cat_features], test[cat_features]])
    encoder.fit(all_cat_data)
    train[cat_features] = encoder.transform(train[cat_features])
    test[cat_features] = encoder.transform(test[cat_features])

    X = train.drop(target, axis=1)
    y = train[target]
    return X, y, test, cat_features, num_features

# === Model definitions ===
models = {
    'XGB': XGBClassifier(
        tree_method='hist', objective='binary:logistic', eval_metric='auc',
        n_estimators=10000, early_stopping_rounds=100, learning_rate=0.0078,
        max_depth=12, reg_lambda=1.43, reg_alpha=5.63, subsample=0.94,
        colsample_bytree=0.71, random_state=Config.state, device='cuda'
    ),
    'LGBM': LGBMClassifier(
        objective='binary', metric='AUC', n_estimators=10000, early_stopping_round=100,
        learning_rate=0.0173, max_depth=18, num_leaves=402, min_child_samples=97,
        subsample=0.55, colsample_bytree=0.55, reg_alpha=0.018, reg_lambda=5.73,
        random_state=Config.state, verbose=-1
    )
}

# === Training and prediction ===
X, y, X_test, cat_features, num_features = transform_data(Config.train, Config.test, Config.target)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = {}

skf = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.state)

for name, model in models.items():
    oof_split = np.zeros(len(X))
    test_split = np.zeros(len(X_test))
    print(f"\nTraining {name}...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if 'XGB' in name:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif 'LGBM' in name:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], categorical_feature=cat_features, callbacks=[lambda env: env.model.best_iteration])
        else:
            model.fit(X_train, y_train)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_split[val_idx] = val_pred
        test_split += model.predict_proba(X_test)[:, 1] / Config.n_splits

        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_pred):.5f}")

    score = roc_auc_score(y, oof_split)
    scores[name] = score
    print(f"{name} Mean AUC: {score:.5f}")

    oof_preds += oof_split / len(models)
    test_preds += test_split / len(models)

# === Ensemble with logistic regression meta-model ===
meta_model = LogisticRegression(C=0.1, random_state=Config.state, max_iter=1000)
meta_model.fit(oof_preds.reshape(-1, 1), y)
final_preds = meta_model.predict_proba(test_preds.reshape(-1, 1))[:, 1]

# === Creating submission ===
Config.submission[Config.target] = final_preds
Config.submission.to_csv("submission.csv", index=False)
print("Submission saved!")

