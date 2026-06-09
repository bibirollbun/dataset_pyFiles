import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")

df.head()
df.shape
df.info()


# Selecting features and target
features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
X = df[features]
y = df['Listening_Time_minutes']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Mean imputation using only training data
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# Training Decision Tree Model
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_train, y_train)


# Predictions
y_pred = dt.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))

# Visualizing Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(dt, feature_names=features, filled=True, rounded=True)
plt.show()


# Preparing submission
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
X_submission = df_test[features]
X_submission = imputer.transform(X_submission)
y_submission_pred = dt.predict(X_submission)

submission = pd.DataFrame({'id': df_test['id'], 'Listening_Time_minutes': y_submission_pred})
submission.to_csv("submission.csv", index=False)

