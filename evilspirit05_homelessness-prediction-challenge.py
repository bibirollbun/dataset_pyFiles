import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')
sample_sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Columns: {train.columns.tolist()}")
print("\nTarget stats:", train['HOMELESS_RATE'].describe())

numeric_cols = train.select_dtypes(include=np.number).columns
corr = train[numeric_cols].corrwith(train['HOMELESS_RATE']).sort_values(ascending=False)
print("\nTop 10 correlated features:\n", corr.head(10))

X = train.drop(columns=['HOMELESS_RATE', 'ID'])
y = train['HOMELESS_RATE']
X_test = test.drop(columns=['ID'])

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        upper = X[col].quantile(0.99)
        lower = X[col].quantile(0.01)
        X[col] = np.clip(X[col], lower, upper)
        X_test[col] = np.clip(X_test[col], lower, upper)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
gb.fit(X_scaled, y)

test_preds = gb.predict(X_test_scaled)
submission = pd.DataFrame({'ID': test['ID'], 'HOMELESS_RATE': test_preds})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved!")


test.head()


pd.set_option("display.max_columns",None)
df=pd.read_csv("/kaggle/input/california-homelessness-prediction-challenge/train.csv")
test=pd.read_csv("/kaggle/input/california-homelessness-prediction-challenge/test.csv")
ID=test.ID
test.drop(columns=["ID"],axis=1,inplace=True)
print(f"data shape: {df.shape}")
print(f"Check null values: {df.isnull().sum()}")
df.drop(columns=["ID"],axis=1,inplace=True)

X = df.drop(columns=['HOMELESS_RATE'])
y = df['HOMELESS_RATE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.6f}")
print(f"R² Score: {r2:.4f}")

rf = RandomForestRegressor(n_estimators=20000, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X, y)

test_preds = rf.predict(test)
submission = pd.DataFrame({'ID': ID, 'HOMELESS_RATE': test_preds})
submission.to_csv('r_submission.csv', index=False)
print("\nSubmission saved!")




