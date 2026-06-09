import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np




# Load data
train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
print('done')


train.head(1)


def drop_inf_columns(df):
    """Drop columns from DataFrame if any value is inf or -inf in that column."""
    return df.loc[:, ~df.isin([float('inf'), float('-inf')]).any()]


train = drop_inf_columns(train)
train.shape


# Drop columns with all NaNs
train = train.dropna(axis=1, how='all')

# Drop rows with any NaNs
train = train.dropna(axis=0, how='any')

# Drop same columns in test as in train
test = test[train.drop(columns=["label"]).columns]


X = train.drop(columns = ['label'])
y = train['label']



X_fit, X_val, y_fit, y_val = train_test_split(X, y, shuffle=False)


import gc
del X,y,train
gc.collect()


X_test = drop_inf_columns(test)


# # X_all = train.drop(columns=["label"])
# # y = train["label"]
# # # Compute correlation of each column with the target (faster than .corr())
# # correlations = X_all.corrwith(y).abs().sort_values(ascending=False)

# # top_features = correlations.index

# # Final training and test sets
# X = traintrain.drop(columns=["label"])
# y = train["label"]
# X_test = test[train.columns]

# print('done')


# # Train-validation split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Split data
# X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import numpy as np

top_features = X_fit.shape[1]
models = []
alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

# Train per-feature RidgeCV
for i in range(top_features):
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
    )
    Xi = X_fit.iloc[:, [i]]     # âœ… Retain feature name
    model.fit(Xi, y_fit)
    models.append(model)

# Predict on validation set
preds = np.zeros((X_val.shape[0], top_features))
for i, model in enumerate(models):
    Xi_val = X_val.iloc[:, [i]]
    preds[:, i] = model.predict(Xi_val)

# Average predictions
y_pred = preds.mean(axis=1)

# Pearson correlation
corr, _ = pearsonr(y_pred, y_val)
print(f"ğŸ“ˆ Pearson Correlation on Validation Set: {corr:.4f}")



preds = np.zeros((test.shape[0], top_features))

for i, model in enumerate(models):
    i_test = X_test.iloc[:, [i]]  
    preds[:, i] = model.predict(i_test)

y_pred = preds.mean(axis=1)




submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
submission["prediction"] = y_pred
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission file saved as 'submission.csv'")
submission.head()


# from sklearn.metrics import r2_score

# feature_scores = []
# for i, model in enumerate(models):
#     Xi_val = X_val.iloc[:, [i]]
#     y_pred_i = model.predict(Xi_val)
#     score = r2_score(y_val, y_pred_i)
#     feature_scores.append((X_val.columns[i], score))

# # Sort by RÂ²
# top_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)



# print("Top 20 features by RÂ² score:")
# for name, score in top_features[:20]:
#     print(f"{name}: {score:.4f}")



# from scipy.stats import pearsonr

# feature_scores = []
# for i, model in enumerate(models):
#     Xi_val = X_val.iloc[:, [i]]
#     y_pred_i = model.predict(Xi_val)
#     score, _ = pearsonr(y_val, y_pred_i)
#     feature_scores.append((X_val.columns[i], score))

# # Sort by absolute Pearson
# top_features = sorted(feature_scores, key=lambda x: abs(x[1]), reverse=True)



# print("Top 20 features by corr score:")
# for name, score in top_features[:20]:
#     print(f"{name}: {score:.4f}")



# from sklearn.metrics import mean_squared_error

# feature_scores = []
# for i, model in enumerate(models):
#     Xi_val = X_val.iloc[:, [i]]
#     y_pred_i = model.predict(Xi_val)
#     mse = mean_squared_error(y_val, y_pred_i)
#     feature_scores.append((X_val.columns[i], mse))

# # Sort by ascending MSE (lower is better)
# top_features = sorted(feature_scores, key=lambda x: x[1])



# print("Top 20 features by corr score:")
# for name, score in top_features[:20]:
#     print(f"{name}: {score:.4f}")





