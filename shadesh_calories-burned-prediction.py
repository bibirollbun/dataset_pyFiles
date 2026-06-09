import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

# ğŸ“¥ Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# âœ… Rename 'Sex' to 'Gender' for compatibility
train.rename(columns={'Sex': 'Gender'}, inplace=True)
test.rename(columns={'Sex': 'Gender'}, inplace=True)

# ğŸ”„ Encode Gender
train['Gender'] = train['Gender'].map({'male': 0, 'female': 1})
test['Gender'] = test['Gender'].map({'male': 0, 'female': 1})

# ğŸ§¹ Drop ID
train.drop('id', axis=1, inplace=True)
test_ids = test['id']
test.drop('id', axis=1, inplace=True)

# ğŸ�¯ Split
X = train.drop('Calories', axis=1)
y = train['Calories']

# ğŸ“� Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# ğŸ§ª Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# âœ… Base Models
xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
ridge = Ridge(alpha=1.0)

# âœ… Stacking Model
stack = StackingRegressor(
    estimators=[('xgb', xgb), ('rf', rf), ('ridge', ridge)],
    final_estimator=Ridge(),
    cv=5
)

# ğŸ�‹ï¸� Train
stack.fit(X_train, y_train)

# ğŸ“ˆ Evaluate
y_pred = stack.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"âœ… R2 Score: {r2:.4f}")
print(f"ğŸ“‰ RMSE: {np.sqrt(mse):.2f}")


# âœ… Predict on Test Set
test_preds = stack.predict(test_scaled)

# ğŸ“� Create Submission File
submission = pd.DataFrame({'id': test_ids, 'Calories': test_preds})
submission.to_csv("submission.csv", index=False)

