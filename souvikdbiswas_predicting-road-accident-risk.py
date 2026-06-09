import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train.head(10)



train.shape



train.columns



train.isnull().sum()


num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
for col in num_cols:
    sns.scatterplot(x=train[col], y=train['accident_risk'])
    plt.title(f"{col} vs Accident Risk")
    plt.show()



cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cat_cols:
    sns.boxplot(x=col, y='accident_risk', data=train)
    plt.show()



bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    sns.boxplot(x=col, y='accident_risk', data=train)
    plt.show()



# Convert boolean columns to integers
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

# Compute correlation for numeric columns and plot
corr = train.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap='mako', fmt='.2f', annot_kws={"size":8})
plt.title("Correlation Heatmap")
plt.show()



if 'id' in train.columns:
    train.drop(columns=['id'], inplace=True)
    print('id is dropped in train ....')
else:
    print('id NOT found in train')

if 'id' in test.columns:
    test.drop(columns=['id'], inplace=True)
    print('id is dropped in test ....')
else:
    print('id NOT found in test')



from sklearn.preprocessing import LabelEncoder

cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

train.head(10)



# Scale numeric columns: fit on train, apply to test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])
train.head(10)



from sklearn.model_selection import train_test_split

X = train.drop(columns=['accident_risk'])
y = train['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



X_train.head()


from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score



models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
}



results = []

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({'Model': name, 'MSE': mse, 'R2': r2})



comparison_df = pd.DataFrame(results)
print(comparison_df.sort_values(by='MSE'))



plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='R2', data=comparison_df)
plt.title('Model R2 Score Comparison')
plt.ylim(
    comparison_df['R2'].min() - 0.05,
    comparison_df['R2'].max() + 0.05
)
plt.show()




xgb_full = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)

if 'accident_risk' in train.columns:
    X_full = train.drop(columns=['accident_risk'])
    y_full = train['accident_risk']
else:
    raise ValueError('Target column "accident_risk" not found in train')



xgb_full.fit(X_full, y_full)



y_test_pred = xgb_full.predict(test)



try:
    test_orig = pd.read_csv('test.csv')
    if 'id' in test_orig.columns:
        submission = pd.DataFrame({'id': test_orig['id'], 'accident_risk': y_test_pred})
    else:
        submission = pd.DataFrame({'accident_risk': y_test_pred})
except Exception:
    submission = pd.DataFrame({'accident_risk': y_test_pred})



submission.to_csv('submission_xgb.csv', index=False)
print('Saved submission_xgb.csv')








