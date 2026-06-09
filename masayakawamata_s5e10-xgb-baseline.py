import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)

train.head(3)


TARGET = 'accident_risk'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']
print(f'{len(BASE)} Base Features:{BASE}')


FEATURES = BASE
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = train[TARGET]


from sklearn.model_selection import KFold

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'---Fold {fold+1}/{N_SPLITS}---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

    X_test = test[FEATURES].copy()

    X_train[CATS] = X_train[CATS].astype('category')    
    X_val[CATS] = X_val[CATS].astype('category')    
    X_test[CATS] = X_test[CATS].astype('category')    
    
    model = XGBRegressor(
        n_estimators=100000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        device='cuda',
        early_stopping_rounds=200,
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=500, 
             )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] += val_preds

    test_preds += model.predict(X_test)

    print(f"Fold {fold+1} RMSE: {mean_squared_error(y_val, val_preds, squared=False)}")
    print(f"Fold {fold+1} R2: {r2_score(y_val, val_preds)}")

test_preds /= N_SPLITS

print(f"Overall OOF RMSE: {mean_squared_error(y, oof_preds, squared=False):.5f}")
print(f"Overall OOF R2: {r2_score(y, oof_preds):.5f}")


import seaborn as sns
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'feature': FEATURES, 
    'importance': feature_importances
})

importance_df = importance_df.sort_values('importance', ascending=False)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 20))
sns.barplot(x='importance', 
            y='feature', 
            data=importance_df.head(50)) 
plt.title('Feature Importance (Fold5 model)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


pd.DataFrame({'id': train.id, TARGET: oof_preds}).to_csv('oof_xgb_baseline.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds}).to_csv('test_xgb_baseline.csv', index=False)

