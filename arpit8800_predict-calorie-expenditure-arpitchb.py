import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")



print(train[:4])


from sklearn.preprocessing import LabelEncoder
import numpy as np

# Encode 'Sex' as 0/1
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# Feature Engineering: BMI
train['BMI'] = train['Weight'] / (train['Height'] / 100)**2
test['BMI'] = test['Weight'] / (test['Height'] / 100)**2



print(train[:4])


import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()


sns.scatterplot(data=train, x='Duration', y='Calories', hue='Sex')
plt.title("Calories Burned vs Duration")
plt.show()



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split


features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
X = train[features]
y = train['Calories']
X_test = test[features]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# RMSLE Scoring function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_val)
print("XGBoost RMSLE:", rmsle(y_val, xgb_preds))


from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9, 1],
    'colsample_bytree': [0.7, 0.9, 1]
}

search = RandomizedSearchCV(XGBRegressor(), param_grid, scoring='neg_root_mean_squared_error', cv=3)
search.fit(X_train, np.log1p(y_train))  # use log1p target
best_model = search.best_estimator_


best_model.fit(X, np.log1p(y))  # log1p for better RMSLE


log_preds = best_model.predict(X_test)
final_preds = np.expm1(log_preds)  # Inverse of log1p


submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_preds
})

submission.to_csv('final_submission.csv', index=False)
print("Submission file saved.")

