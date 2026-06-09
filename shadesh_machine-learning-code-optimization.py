# ğŸ“š Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ğŸ“¥ Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# âœ… Drop ID
train.drop('id', axis=1, inplace=True)
test_id = test['id']
test.drop('id', axis=1, inplace=True)

# ğŸ”„ Encode 'Sex' (was previously 'Gender')
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# ğŸ�¯ Split Features and Target
X = train.drop('Calories', axis=1)
y = train['Calories']

# ğŸ”ª Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸ“� Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)

# ğŸ§  Train Model (Random Forest works well here)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ğŸ”� Evaluate
preds = model.predict(X_val_scaled)

mse = mean_squared_error(y_val, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, preds)

print("ğŸ“Š RMSE:", rmse)
print("ğŸ“ˆ RÂ² Score:", r2)

# âœ… Predict on Test Set
test_preds = model.predict(X_test_scaled)

# ğŸ“� Create Submission
submission = pd.DataFrame({'id': test_id, 'Calories': test_preds})
submission.to_csv('submission.csv', index=False)

