import cudf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)


# Load data using CuDF
train = cudf.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
test = cudf.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col='id')


# Add constant column
train['const'] = 1
test['const'] = 1


# Define target and features
target = ['Fertilizer Name']
cat_columns = [i for i in train.columns if train[i].dtype == 'object'][:-1]
num_columns = [i for i in train.columns if i not in cat_columns and i != 'Fertilizer Name' and i != 'const']


# Label encode categorical columns
label_enc = LabelEncoder()
for i in cat_columns:
    train[i] = label_enc.fit_transform(train[i].to_pandas())
    test[i] = label_enc.transform(test[i].to_pandas())
    train[i] = train[i].astype('category')
    test[i] = test[i].astype('category')
train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'].to_pandas())


# Define MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Prepare features and target
X = train.drop(['Fertilizer Name'], axis=1)
y = train["Fertilizer Name"].to_pandas()  # Convert to Pandas for compatibility with StratifiedKFold
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


# Initialize arrays for out-of-fold predictions and test predictions
oof_xgb_1 = np.zeros(shape=(len(train), y.nunique()))
pred_prob_xgb_1 = np.zeros(shape=(len(test), y.nunique()))
oof_xgb_2 = np.zeros(shape=(len(train), y.nunique()))
pred_prob_xgb_2 = np.zeros(shape=(len(test), y.nunique()))
oof_xgb_3 = np.zeros(shape=(len(train), y.nunique()))
pred_prob_xgb_3 = np.zeros(shape=(len(test), y.nunique()))


# Define XGBoost models with adjusted colsample_bytree for constant column
xgb_model_1 = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467 * (X.shape[1] - 1) / X.shape[1],
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state=42,
    eval_metric='mlogloss',
    enable_categorical=True,
    device='cuda'
)

xgb_model_2 = XGBClassifier(
    max_depth=13,
    colsample_bytree=0.4309 * (X.shape[1] - 1) / X.shape[1],
    subsample=0.7281,
    n_estimators=1500,
    learning_rate=0.0179,
    gamma=0.0895,
    max_delta_step=4,
    reg_alpha=0.4187,
    reg_lambda=2.4665,
    early_stopping_rounds=100,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=13,
    enable_categorical=True,
    device='cuda'
)

xgb_model_3 = XGBClassifier(
    max_depth=9,
    colsample_bytree=0.4930 * (X.shape[1] - 1) / X.shape[1],
    subsample=0.9753,
    n_estimators=2353,
    learning_rate=0.0158,
    gamma=0.0035,
    max_delta_step=1,
    reg_alpha=0.5994,
    reg_lambda=4.5972,
    early_stopping_rounds=100,
    eval_metric='mlogloss',
    objective='multi:softprob',
    random_state=25,
    enable_categorical=True,
    device='cuda'
)


correlations = []

# Cross-validation loop
for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print('#' * 15, i+1, '#' * 15)
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    # XGB 1 
    xgb_model_1.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
    oof_xgb_1[valid_idx] = xgb_model_1.predict_proba(x_valid)
    pred_prob_xgb_1 += xgb_model_1.predict_proba(test)
    top_3_preds_xgb_1 = np.argsort(oof_xgb_1[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score_xgb_1 = mapk(actual, top_3_preds_xgb_1)
    print(f"âœ… FOLD {i+1}: MAP@3 XGB_1 Score: {map3_score_xgb_1:.5f}")

    # XGB 2 
    xgb_model_2.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
    oof_xgb_2[valid_idx] = xgb_model_2.predict_proba(x_valid)
    pred_prob_xgb_2 += xgb_model_2.predict_proba(test)
    top_3_preds_xgb_2 = np.argsort(oof_xgb_2[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score_xgb_2 = mapk(actual, top_3_preds_xgb_2)
    print(f"â˜‘ï¸� FOLD {i+1}: MAP@3 XGB_2 Score: {map3_score_xgb_2:.5f}")

    # XGB 3 
    xgb_model_3.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
    oof_xgb_3[valid_idx] = xgb_model_3.predict_proba(x_valid)
    pred_prob_xgb_3 += xgb_model_3.predict_proba(test)
    top_3_preds_xgb_3 = np.argsort(oof_xgb_3[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score_xgb_3 = mapk(actual, top_3_preds_xgb_3)
    print(f"âœ”ï¸� FOLD {i+1}: MAP@3 XGB_3 Score: {map3_score_xgb_3:.5f}")
    
    # Correlations
    corr_12 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_2[valid_idx].ravel())[0, 1]
    corr_13 = np.corrcoef(oof_xgb_1[valid_idx].ravel(), oof_xgb_3[valid_idx].ravel())[0, 1]
    corr_23 = np.corrcoef(oof_xgb_2[valid_idx].ravel(), oof_xgb_3[valid_idx].ravel())[0, 1]
    
    print(f"ğŸ”— FOLD {i+1}:")
    print(f'Corr XGB_1 vs XGB_2: {corr_12:.5f}')
    print(f'Corr XGB_1 vs XGB_3: {corr_13:.5f}')
    print(f'Corr XGB_2 vs XGB_3: {corr_23:.5f}')
    correlations.append((corr_12 + corr_13 + corr_23) / 3)

print(f"\nğŸ“Š Mean after 5 Folds: {np.mean(correlations):.5f}")


actual = [[label] for label in y]

# Final MAP@3 scores
top_3_preds_xgb_1 = np.argsort(oof_xgb_1, axis=1)[:, -3:][:, ::-1]
map3_score_xgb_1 = mapk(actual, top_3_preds_xgb_1)
print(f'âœ… Final XGB_1 MAP@3 Score: {map3_score_xgb_1:.5f}')

top_3_preds_xgb_2 = np.argsort(oof_xgb_2, axis=1)[:, -3:][:, ::-1]
map3_score_xgb_2 = mapk(actual, top_3_preds_xgb_2)
print(f'âœ… Final XGB_2 MAP@3 Score: {map3_score_xgb_2:.5f}')

top_3_preds_xgb_3 = np.argsort(oof_xgb_3, axis=1)[:, -3:][:, ::-1]
map3_score_xgb_3 = mapk(actual, top_3_preds_xgb_3)
print(f'âœ… Final XGB_3 MAP@3 Score: {map3_score_xgb_3:.5f}')

pred_prob_xgb_1 /= FOLDS
pred_prob_xgb_2 /= FOLDS
pred_prob_xgb_3 /= FOLDS


# Weight optimization
from scipy.optimize import minimize

def map3_multi_weight(weights, y_true, oof1, oof2, oof3):
    w1, w2 = weights
    w3 = 1.0 - w1 - w2
    if w3 < 0 or w3 > 1:
        return 1
    combined = w1 * oof1 + w2 * oof2 + w3 * oof3
    top_3 = np.argsort(combined, axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_true]
    return -mapk(actual, top_3)

best_result = None
for start1 in np.linspace(0.3, 0.7, 8):
    for start2 in np.linspace(0.2, 0.6, 4):
        if start1 + start2 >= 1.0:
            continue
        res = minimize(
            map3_multi_weight,
            [start1, start2],
            args=(y, oof_xgb_1, oof_xgb_2, oof_xgb_3),
            bounds=[(0, 1), (0, 1)],
            options={'disp': False}
        )
        w1, w2 = res.x
        w3 = 1.0 - w1 - w2
        score = -res.fun
        print(f"Start: w1={start1:.2f}, w2={start2:.2f} â†’ MAP@3: {score:.6f}, Weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")
        if best_result is None or res.fun < best_result.fun:
            best_result = res

w1, w2 = best_result.x
w3 = 1.0 - w1 - w2

print(f"ğŸ“Œ Best-found weights:")
print(f"XGB_1 = {w1:.4f}, XGB_2 = {w2:.4f}, XGB_3 = {w3:.4f}")
print(f"ğŸ“ˆ MAP@3 score after weight optimization {-best_result.fun:.5f}")  # Fixed error here

print(f'weight 1: {w1}, weight 2: {w2}, weight 3: {w3}')


#Final predictions
pred_prob = w1 * pred_prob_xgb_1 + w2 * pred_prob_xgb_2 + w3 * pred_prob_xgb_3

top_3_pred = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_label = label_enc.inverse_transform(top_3_pred.ravel()).reshape(top_3_pred.shape)

df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_label]
})

submission.to_csv('submission-04-const-cudf.csv', index=False)
print("âœ… Submission file saved as 'submission-const.csv'")




