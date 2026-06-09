
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings
import lightgbm as lgb
import scipy

warnings.filterwarnings('ignore')
lgb_params = {'verbosity': -1}


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=1000)
X_full = tfidf.fit_transform(train['SMILES'])
X_test = tfidf.transform(test['SMILES'])

final_preds = np.zeros((len(test), len(targets)))

for i, col in enumerate(targets):
    
    valid_idx = train[col].notnull().values
    y = train.loc[valid_idx, col].values
    X = X_full[valid_idx]

   
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    lgbm = LGBMRegressor(n_estimators=150, learning_rate=0.03, random_state=42, **lgb_params)

    rf.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    # Validation Prediction
    val_preds = (rf.predict(X_val) + lgbm.predict(X_val)) / 2
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f"{col} RMSE: {rmse:.4f}")

    # Test Prediction
    test_preds = (rf.predict(X_test) + lgbm.predict(X_test)) / 2
    final_preds[:, i] = test_preds

# ✅ Save Submission
submission = pd.DataFrame({
    'id': test['id'],
    'Tg': final_preds[:, 0],
    'FFV': final_preds[:, 1],
    'Tc': final_preds[:, 2],
    'Density': final_preds[:, 3],
    'Rg': final_preds[:, 4],
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission saved.")



from sklearn.metrics import mean_absolute_error

weights = {'Tg': 0.3, 'FFV': 0.2, 'Tc': 0.2, 'Density': 0.1, 'Rg': 0.2}
wmae_total = 0

for i, col in enumerate(targets):
    valid_idx = train[col].notnull().values
    y = train.loc[valid_idx, col].values
    X = X_full[valid_idx]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    lgbm = LGBMRegressor(n_estimators=150, learning_rate=0.03, random_state=42, **lgb_params)

    rf.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    val_preds = (rf.predict(X_val) + lgbm.predict(X_val)) / 2
    mae = mean_absolute_error(y_val, val_preds)
    wmae_total += weights[col] * mae
    print(f"{col} MAE: {mae:.4f}")

print(f"\n✅ Final WMAE: {wmae_total:.4f}")


