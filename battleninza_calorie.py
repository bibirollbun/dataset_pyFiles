import pandas as pd
import numpy as np
train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train_data.head()


from matplotlib import pyplot as plt
test_data.plot(kind='scatter', x='Height', y='Weight', s=32, alpha=.8)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.title("Height vs Weight")
plt.show()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['Sex'] = le.fit_transform(train_data['Sex'])
test_data['Sex'] = le.transform(test_data['Sex'])


features = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
X = train_data[features].copy()
y = np.log1p(train_data["Calories"])
X_test = test_data[features].copy()

# One-hot encoding
X_encoded = pd.get_dummies(X, columns=["Sex"])
X_test_encoded = pd.get_dummies(X_test, columns=["Sex"])
X_encoded, X_test_encoded = X_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)


from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

models = {
    'cat': CatBoostRegressor(verbose=0, random_seed=42),
    'xgb': XGBRegressor(verbosity=0, random_state=42),
    'lgbm': LGBMRegressor(random_state=42)
}

oof_preds = np.zeros((X.shape[0], len(models)))
test_preds = np.zeros((X_test.shape[0], len(models)))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (name, model) in enumerate(models.items()):
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if name == 'cat':
            model.fit(X_tr, y_tr, cat_features=["Sex"])
            oof_preds[val_idx, i] = model.predict(X_val)
            test_preds[:, i] += model.predict(X_test) / kf.n_splits
        else:
            model.fit(X_encoded.iloc[train_idx], y_tr)
            oof_preds[val_idx, i] = model.predict(X_encoded.iloc[val_idx])
            test_preds[:, i] += model.predict(X_test_encoded) / kf.n_splits

meta_model = Ridge()
meta_model.fit(oof_preds, y)
final_preds_log = meta_model.predict(test_preds)
final_preds = np.expm1(final_preds_log)


oof_meta_preds = meta_model.predict(oof_preds)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_meta_preds)))
print(f"Out-of-Fold RMSLE: {rmsle:.5f}")


submission = pd.DataFrame({'id': test_data['id'], 'Calories': np.maximum(final_preds, 0)})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved. Ready for Kaggle upload!")

