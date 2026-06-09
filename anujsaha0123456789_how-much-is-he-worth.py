import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
print("Libraries imported!")


train = pd.read_csv('/kaggle/input/how-much-is-he-worth/train.csv')
test = pd.read_csv('/kaggle/input/how-much-is-he-worth/test.csv')
print(f"Train: {train.shape} | Test: {test.shape}")
train.head()


print(train.describe())
print("\nMissing values:\n", train.isnull().sum())
print("\nTarget stats:\n", train['net_worth'].describe())

### Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(train['net_worth'], bins=50, edgecolor='black')
axes[0].set_title('Net Worth Distribution')
axes[1].hist(np.log1p(train['net_worth']), bins=50, edgecolor='black', color='coral')
axes[1].set_title('Net Worth (Log Scale)')
plt.tight_layout()
plt.show()


### Separate features and target
y = train['net_worth']
X = train.drop(['net_worth'], axis=1)
test_ids = test['ID'] if 'ID' in test.columns else None
if 'ID' in test.columns: test = test.drop(['ID'], axis=1)
if 'name' in X.columns: X = X.drop(['name'], axis=1)
if 'name' in test.columns: test = test.drop(['name'], axis=1)

### Define column types
categorical_cols = ['profession', 'country', 'favorite_color', 'preferred_transportati']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

### Convert and fill numerical columns
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')
    median_val = X[col].median()
    X[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

### Encode categorical variables
for col in categorical_cols:
    if col in X.columns:
        X[col].fillna('Unknown', inplace=True)
        test[col].fillna('Unknown', inplace=True)
        le = LabelEncoder()
        combined = pd.concat([X[col], test[col]], axis=0)
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

print("Preprocessing complete!")


### Experience features
X['exp_squared'] = X['years_experience'] ** 2
test['exp_squared'] = test['years_experience'] ** 2
X['exp_valuation'] = X['years_experience'] * X['company_valuation']
test['exp_valuation'] = test['years_experience'] * test['company_valuation']

### Stock ownership features
X['stocks_valuation'] = X['owns_stocks'] * X['company_valuation']
test['stocks_valuation'] = test['owns_stocks'] * test['company_valuation']
X['stocks_exp'] = X['owns_stocks'] * X['years_experience']
test['stocks_exp'] = test['owns_stocks'] * test['years_experience']

### Lifestyle features
X['coffee_sleep_ratio'] = X['number_of_cups_coffee'] / (X['average_daily_sleep_hours'] + 1)
test['coffee_sleep_ratio'] = test['number_of_cups_coffee'] / (test['average_daily_sleep_hours'] + 1)
X['social_pets'] = X['hours_spent_on_social_media'] * X['number_of_pets']
test['social_pets'] = test['hours_spent_on_social_media'] * test['number_of_pets']

### High earner indicator
X['high_valuation'] = (X['company_valuation'] > X['company_valuation'].quantile(0.75)).astype(int)
test['high_valuation'] = (test['company_valuation'] > X['company_valuation'].quantile(0.75)).astype(int)

### Log transformation
X['log_valuation'] = np.log1p(X['company_valuation'])
test['log_valuation'] = np.log1p(test['company_valuation'])

### Clean data
X.replace([np.inf, -np.inf], 0, inplace=True)
test.replace([np.inf, -np.inf], 0, inplace=True)
X.fillna(0, inplace=True)
test.fillna(0, inplace=True)
print(f"Total features: {X.shape[1]}")


scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape} | Validation: {X_val.shape}")



models = {}
val_predictions = {}
rmse_scores = {}

### XGBoost
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05, 
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
val_predictions['XGBoost'] = xgb.predict(X_val)
rmse_scores['XGBoost'] = np.sqrt(mean_squared_error(y_val, val_predictions['XGBoost']))
models['XGBoost'] = xgb
print(f"RMSE: {rmse_scores['XGBoost']:,.2f}")

### LightGBM
print("Training LightGBM...")
lgb = LGBMRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
lgb.fit(X_train, y_train)
val_predictions['LightGBM'] = lgb.predict(X_val)
rmse_scores['LightGBM'] = np.sqrt(mean_squared_error(y_val, val_predictions['LightGBM']))
models['LightGBM'] = lgb
print(f"RMSE: {rmse_scores['LightGBM']:,.2f}")

### Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=3,
                           min_samples_leaf=1, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
val_predictions['RF'] = rf.predict(X_val)
rmse_scores['RF'] = np.sqrt(mean_squared_error(y_val, val_predictions['RF']))
models['RF'] = rf
print(f"RMSE: {rmse_scores['RF']:,.2f}")

### Extra Trees
print("Training Extra Trees...")
et = ExtraTreesRegressor(n_estimators=300, max_depth=25, min_samples_split=3,
                         random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
val_predictions['ET'] = et.predict(X_val)
rmse_scores['ET'] = np.sqrt(mean_squared_error(y_val, val_predictions['ET']))
models['ET'] = et
print(f"RMSE: {rmse_scores['ET']:,.2f}")


### Weighted ensemble
ensemble_val = (0.35 * val_predictions['XGBoost'] + 0.35 * val_predictions['LightGBM'] + 
                0.20 * val_predictions['RF'] + 0.10 * val_predictions['ET'])
rmse_scores['Ensemble'] = np.sqrt(mean_squared_error(y_val, ensemble_val))

print("\n=== MODEL PERFORMANCE (RMSE) ===")
for name, score in sorted(rmse_scores.items(), key=lambda x: x[1]):
    print(f"{name}: {score:,.2f}")


print("\nRetraining on full dataset...")
for name, model in models.items():
    model.fit(X_scaled, y)
print("Complete!")


### Generate test predictions
test_preds = {
    'XGBoost': xgb.predict(test_scaled),
    'LightGBM': lgb.predict(test_scaled),
    'RF': rf.predict(test_scaled),
    'ET': et.predict(test_scaled)
}

### Final ensemble
final_predictions = (0.35 * test_preds['XGBoost'] + 0.35 * test_preds['LightGBM'] + 
                    0.20 * test_preds['RF'] + 0.10 * test_preds['ET'])



submission = pd.DataFrame({
    'ID': test_ids if test_ids is not None else range(1, len(final_predictions) + 1),
    'net_worth': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\n✅ Submission created!")
print(submission.head(10))
print(f"\nTotal predictions: {len(submission)}")
print(submission['net_worth'].describe())


### Plot prediction distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(submission['net_worth'], bins=50, edgecolor='black', color='skyblue')
axes[0].set_title('Predicted Net Worth Distribution')
axes[1].hist(np.log1p(submission['net_worth']), bins=50, edgecolor='black', color='coral')
axes[1].set_title('Predicted Net Worth (Log Scale)')
plt.tight_layout()
plt.show()


