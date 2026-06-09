import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load & preprocess
data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
test['Sex'] = LabelEncoder().fit_transform(test['Sex'])

# Feature engineering
data['BMI'] = data['Weight'] / (data['Height'] ** 2)
test['BMI'] = test['Weight'] / (test['Height'] ** 2)
data['Heart_Rate_x_Duration'] = data['Heart_Rate'] * data['Duration']
test['Heart_Rate_x_Duration'] = test['Heart_Rate'] * test['Duration']

features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Heart_Rate_x_Duration']
X = data[features]
Y = data['Calories']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Optimized Random Forest
reg = RandomForestRegressor(
    n_estimators=80,
    max_depth=None,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
reg.fit(X_train, Y_train)

# Evaluate
pred = reg.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(Y_test, pred)) * 100
print(f'RMSLE: {rmsle:.3f}%')

# Submission
submission = pd.DataFrame({'id': test['id'], 'Calories': reg.predict(test[features])})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Done!")

