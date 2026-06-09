import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

X = train.drop(['Fertilizer Name', 'id'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

for col in num_features:
    median = X[col].median()
    X[col].fillna(median, inplace=True)
    X_test[col].fillna(median, inplace=True)

for col in cat_features:
    mode = X[col].mode()[0]
    X[col].fillna(mode, inplace=True)
    X_test[col].fillna(mode, inplace=True)

X['N_P_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + 1)
X['N_K_ratio'] = X['Nitrogen'] / (X['Potassium'] + 1)
X_test['N_P_ratio'] = X_test['Nitrogen'] / (X_test['Phosphorous'] + 1)
X_test['N_K_ratio'] = X_test['Nitrogen'] / (X_test['Potassium'] + 1)

for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)


def map3_score(y_true, y_pred_proba):
    score = 0.0
    for i in range(len(y_true)):
        true_label = y_true[i]
        top3_preds = np.argsort(y_pred_proba[i])[::-1][:3]
        if true_label in top3_preds:
            rank = np.where(top3_preds == true_label)[0][0] + 1
            score += 1 / rank
    return score / len(y_true)


params = {
    'objective': 'multi:softprob',
    'num_class': num_classes,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), num_classes))
test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    print(f"===== 训练第{fold + 1}/5折 =====")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    oof_preds[val_idx] = model.predict(dval)
    dtest = xgb.DMatrix(X_test)
    test_preds += model.predict(dtest) / 5

cv_score = map3_score(y_encoded, oof_preds)
print(f"5折交叉验证MAP@3分数：{cv_score:.4f}")

top3_preds = []
for proba in test_preds:
    top3_indices = np.argsort(proba)[::-1][:3]
    top3_types = label_encoder.inverse_transform(top3_indices)
    top3_preds.append(' '.join(top3_types))

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': top3_preds
})

submission.to_csv('submission.csv', index=False)
print("提交文件已生成：submission.csv")

