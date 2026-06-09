import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


def preprocess_data(train, test, target_col):
    combined = pd.concat([train, test], ignore_index=True)

    numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != target_col and col in combined.columns:
            combined[col].fillna(combined[col].median(), inplace=True)

    cat_cols = combined.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != target_col and col in combined.columns:
            combined[col] = combined[col].astype('category').cat.codes

    train_processed = combined[combined[target_col].notna()].copy()
    test_processed = combined[combined[target_col].isna()].drop(columns=[target_col]).copy()

    return train_processed, test_processed


def map_at5(y_true, y_pred_proba):
    map_sum = 0.0
    n_samples = y_pred_proba.shape[0]

    for i in range(n_samples):
        true_label = y_true[i]
        top5_pred = np.argsort(y_pred_proba[i])[::-1][:5]
        precision_sum = 0.0
        correct_count = 0

        for k in range(5):
            if top5_pred[k] == true_label:
                correct_count += 1
                precision = correct_count / (k + 1)
                precision_sum += precision
                break

        map_sum += precision_sum

    return map_sum / n_samples


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 目标列使用提交要求的 'Fertilizer Name'
target_col = 'Fertilizer Name'  

train_processed, test_processed = preprocess_data(train_df, test_df, target_col)

X = train_processed.drop(columns=['id', target_col])
y = train_processed[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective':'multi:softprob',
    'num_class': len(le.classes_),
    'eval_metric':'mlogloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=100
)

val_proba = model.predict(dval)
val_map = map_at5(y_val, val_proba)
print(f"Validation MAP@5: {val_map:.4f}")

dtest = xgb.DMatrix(test_processed.drop(columns=['id']))
test_proba = model.predict(dtest)

top5_indices = np.argsort(test_proba, axis=1)[:, ::-1][:, :5]
top5_predictions = []
for indices in top5_indices:
    top5_labels = le.inverse_transform(indices)
    top5_predictions.append(' '.join(top5_labels))

# 提交文件列名改为 'Fertilizer Name'，适配要求
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': top5_predictions  
})

submission.to_csv('submission.csv', index=False)
print("submission.csv generated")

