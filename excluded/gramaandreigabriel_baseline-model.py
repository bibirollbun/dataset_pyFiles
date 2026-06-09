import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

train.shape, test.shape



train.head()



train.isnull().sum()



plt.figure(figsize=(6,4))
sns.histplot(train['Price'], kde=True, bins=30)
plt.title("Distribution of Target (Price)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()




cat_cols = train.select_dtypes(include='object').columns

le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])



X = train.drop(['id', 'Price'], axis=1)  # Drop ID and target column
y = train['Price']  # Target is Price

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Pipeline: impute missing values with median, then train RandomForest
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation RMSE: {rmse:.4f}')



X_test = test.drop(['id'], axis=1)

# Use pipeline (not the bare model) to predict, so imputer runs on test data
test_preds = pipeline.predict(X_test)

submission['Price'] = test_preds  # Replace 'target' with 'Price' if needed
submission.to_csv('submission.csv', index=False)


