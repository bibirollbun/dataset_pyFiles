import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 2. Handle IDs
test_ids = test['id']
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# 3. Categorical feature handling
categorical_cols = train.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Fertilizer Name')
for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# 4. Label encode target
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# 5. Prepare full training data
X_full = train.drop(columns=['Fertilizer Name'])
y_full = train['Fertilizer Name']

# 6. Optuna-tuned parameters
best_params = {
    'learning_rate': 0.03772,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0.18861,
    'subsample': 0.90268,
    'colsample_bytree': 0.62924,
    'reg_alpha': 0.22981,
    'reg_lambda': 0.62083,
    'objective': 'multi:softprob',
    'num_class': len(le.classes_),
    'tree_method': 'hist',
    'device': 'cuda',
    'verbosity': 1,
    'seed': 42
}

# 7. temp_model for rank estimation with same hyperparams (n_estimators=500)
temp_model = xgb.XGBClassifier(
    **best_params,
    enable_categorical=True,
    n_estimators=500
)
temp_model.fit(X_full, y_full)
train_preds = temp_model.predict_proba(X_full)
train_ranks = np.argsort(-train_preds, axis=1)

# 8. Categorize samples by rank
sample_category = np.empty(len(X_full), dtype=object)
for i in range(len(X_full)):
    rank = np.where(train_ranks[i] == y_full.iloc[i])[0][0] + 1
    if rank == 1:
        sample_category[i] = "high"
    elif rank == 2:
        sample_category[i] = "mid"
    elif rank == 3:
        sample_category[i] = "low"
    else:
        sample_category[i] = "missed"

# 9. Apply Optuna-tuned weights
weights = np.ones(len(X_full))
for i, cat in enumerate(sample_category):
    if cat == "mid":
        weights[i] = 1.634991
    elif cat == "low":
        weights[i] = 1.032604
    elif cat == "missed":
        weights[i] = 1.198181

# 10. Prepare DMatrix
dtrain_full = xgb.DMatrix(X_full, label=y_full, weight=weights, enable_categorical=True)
dtest = xgb.DMatrix(test, enable_categorical=True)

# 11. Train final model
final_model = xgb.train(
    best_params,
    dtrain_full,
    num_boost_round=2000
)

# 12. Predict
test_preds = final_model.predict(dtest)
top3_preds = np.argsort(-test_preds, axis=1)[:, :3]
top3_labels = le.inverse_transform(top3_preds.flatten()).reshape(top3_preds.shape)

# 13. Format submission
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)

