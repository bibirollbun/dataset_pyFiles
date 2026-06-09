import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_copy, test_copy = train_data, test_data


train_data.sample(5)


test_data.sample(5)


train_data.shape


test_data.shape


train_data.isnull().sum()


test_data.isnull().sum()


gender_map = {"male" : 0,"female" : 1}
train_data['Sex'] = train_data['Sex'].replace(gender_map)

train_data=train_data.drop(["id"],axis=1)


corr_matrix = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


def feature_engineering(df):
    df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)    
    df['exercise_intensity'] = df['Heart_Rate'] / df['Duration']
    df['heart_rate_duration'] = df['Heart_Rate'] * df['Duration']
    df['temp_duration'] = df['Body_Temp'] * df['Duration']
    df['hr_to_temp'] = df['Heart_Rate'] / df['Body_Temp']
    df['hr_to_age'] = df['Heart_Rate'] / df['Age']
    df['age_bmi'] = df['Age'] * df['bmi']
    df['max_heart_rate'] = 220 - df['Age']
    df['heart_rate_intensity'] = df['Heart_Rate'] / df['max_heart_rate']
    return df


train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)


train_data.drop(['Sex','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'], axis=1)


X = train_data.drop(['Calories'], axis=1)
y = train_data['Calories']


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_params = {'n_jobs': -1,
             'n_estimators': 500,
             'min_samples_split': 2,
             'min_samples_leaf': 2,
             'max_features': 'log2',
             'max_depth': 20}


rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
y_pred = rf_model.predict(X_test)
print(f"RandomForest → R²: {r2_score(y_test, y_pred):.4f}, RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")


test_data['Sex'] = test_data['Sex'].replace(gender_map)
test_ids = test_data['id']
test_data.drop(['Sex','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'], axis=1)
X_test_final = test_data.drop(["id"], axis=1)


test_predictions = rf_model.predict(X_test_final)

submission = pd.DataFrame({
    'id': test_ids,
    'Calories': test_predictions
})
submission.to_csv('submission.csv', index=False)


submission

