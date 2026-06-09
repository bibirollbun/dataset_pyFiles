import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === Data Preprocessing Function ===
def preprocess_data(raw_df):
    assert len(raw_df.shape) == 2
    y = raw_df['label'].to_numpy()
    assert y.shape == (raw_df.shape[0],)

    cols = [
        'X363', 'X405', 'X321',
        'X175', 'X179', 'X137', 'X197', 'X22', 'X40', 'X181',
        'X28', 'X169', 'X198', 'X173',
        'X338', 'X288', 'X385', 'X344', 'X427', 'X587', 'X450',
        'X97', 'X52', 'X444',
        'X598', 'X379', 'X696', 'X297', 'X138',
        'X572', 'X343', 'X586', 'X466', 'X438', 'X452', 'X459',
        'X435', 'X386', 'X55', 'X341', 'X683', 'X428', 'X605',
        'X445', 'X272', 'X180', 'X593', 'X680',
        'X686', 'X692', 'X695',
        "X603", "X674", "X421", "X333",
        "X415", "X345", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume",
        "bid_qty", "ask_qty",
    ]

    df = raw_df[cols]
    assert df.isna().sum().sum() == 0

    df = pd.concat([
        df,
        df.shift(-10).add_suffix(f'_lead_10'),
        df.shift(-30).add_suffix(f'_lead_30'),
        df.shift(-60).add_suffix(f'_lead_60')
    ], axis=1)

    df = df.fillna(0.0)
    assert 'label' not in df.columns
    assert raw_df.shape[0] == df.shape[0]
    assert df.isna().sum().sum() == 0
    return df, y

# === Load and preprocess training data ===
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
X_train, y_train = preprocess_data(train_df)
train_df = None
print("Train shape:", X_train.shape, y_train.shape)

# === Ridge Regression with Scaling ===
pipeline = make_pipeline(StandardScaler(), Ridge())

# === Cross-validation evaluation ===
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print("Cross-validated R² scores:", cv_scores)
print("Average CV R² score:", np.mean(cv_scores))

# === Pearson Correlation via Cross-Validation Predictions ===
y_cv_pred = cross_val_predict(pipeline, X_train, y_train, cv=5)
pearson_corr = pearsonr(y_train, y_cv_pred)[0]
print("Cross-validated Pearson Correlation Coefficient:", pearson_corr)

# === Hyperparameter tuning for Ridge alpha ===
param_grid = {"ridge__alpha": [100]}
grid = GridSearchCV(pipeline, param_grid, scoring='r2', cv=3)
grid.fit(X_train, y_train)

print("Best alpha:", grid.best_params_["ridge__alpha"])
print("Best CV R² score from grid search:", grid.best_score_)

# === Train final model with best alpha ===
final_model = grid.best_estimator_
final_model.fit(X_train, y_train)

# === Evaluate on training set ===
y_train_pred = final_model.predict(X_train)
print("Final Training R² score:", r2_score(y_train, y_train_pred))
print("Final Training Pearson Correlation:", pearsonr(y_train, y_train_pred)[0])

# === Free memory ===
X_train = None
y_train = None

# === Load and reorder test data ===
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
t = pd.Series(pd.read_csv(
    '/kaggle/input/close-row1/closest_rows.csv'
)['0'].to_numpy())

assert t.shape == (test_df.shape[0],)
print('Reconstructed timestamps share:', len(t[t >= 0]) / len(t))

# === Adjust timestamps and reorder ===
t -= 10080
t[t < 0] = 538149
t = t.sort_values()
t[t <= len(t)] = np.arange(t[t <= len(t)].shape[0])
t = t.sort_index()
t = pd.Series(np.arange(538150), index=t.to_numpy()).sort_index()

test_df = test_df.iloc[t.to_numpy()]

# === Preprocess test data and make predictions ===
X_test, _ = preprocess_data(test_df)
test_df = None
print("Test shape:", X_test.shape)

y_pred = final_model.predict(X_test)
print("Prediction stats:")
print(pd.Series(y_pred).describe())

# === Save submission ===
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission = submission.iloc[t.to_numpy()]
submission['prediction'] = y_pred
submission = submission.sort_index()
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")





