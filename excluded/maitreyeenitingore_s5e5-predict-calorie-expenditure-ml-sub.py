import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.head()


test=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test.head()


#visualization of the data
# Plot histograms for numerical features
num_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
train[num_features].hist(bins=20, figsize=(15, 10), layout=(3, 3))
plt.tight_layout()
plt.show()


# Bar plot for categorical feature 'Sex'
train['Sex'].value_counts().plot(kind='bar')
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# Convert 'Sex' to numeric
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

# Select features and target
X = train[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
y = train['Calories']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
print('Linear Regression Mean Squared Error:', mean_squared_error(y_test, y_pred_lr, squared=False))
print('Linear Regression R^2 Score:', r2_score(y_test, y_pred_lr))


!pip install tqdm
from tqdm import tqdm

rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print('Random Forest Mean Squared Error:', mean_squared_error(y_test, y_pred_rf, squared=False))
print('Random Forest R^2 Score:', r2_score(y_test, y_pred_rf))


from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print('XGBoost Mean Squared Error:', mean_squared_error(y_test, y_pred_xgb , squared=False))
print('XGBoost R^2 Score:', r2_score(y_test, y_pred_xgb))


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MSE': [
        mean_squared_error(y_test, y_pred_lr,squared=False),
        mean_squared_error(y_test, y_pred_rf, squared=False),
        mean_squared_error(y_test, y_pred_xgb, squared=False),
    ],
    'R2 Score': [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
    ]
})
results


# Load test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
X_test_submission = test_df[['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]

# Make predictions using XGBoost model (best performing model)
test_predictions = xgb_model.predict(X_test_submission)

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Calories': test_predictions
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Display first few rows of submission file
print("Submission file preview:")
submission_df.head()

