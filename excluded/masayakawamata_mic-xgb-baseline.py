import warnings
warnings.simplefilter('ignore')


import pandas as pd, numpy as np

train = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/train.csv')
test = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/test.csv')
print('Train Shape:', train.shape)
print('Test Shape:', test.shape)

train.head(3)


TARGET = 'charges'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['sex', 'smoker', 'region']
print(f'{len(BASE)} Base Features:{BASE}')


FEATURES = BASE
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = np.log1p(train[TARGET])


from sklearn.model_selection import KFold

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


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
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        device='cuda',
        early_stopping_rounds=200,
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000, 
             )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] += val_preds

    test_preds += model.predict(X_test)

    print(f"Fold {fold+1} RMSLE: {mean_squared_error(y_val, val_preds, squared=False)}")

test_preds /= N_SPLITS

print(f"Overall OOF RMSLE: {mean_squared_error(y, oof_preds, squared=False):.5f}")


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


pd.DataFrame({'id': train.id, TARGET: np.expm1(oof_preds)}).to_csv('oof_xgb_baseline.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: np.expm1(test_preds)}).to_csv('test_xgb_baseline.csv', index=False)




