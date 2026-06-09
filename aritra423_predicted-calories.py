# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Load the files
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Preview the data
train_df.head()


train_df = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)
train_df['Sex_male']=train_df['Sex_male'].astype(int)
train_df.head()


import seaborn as sns
correlation=train_df.corr()
sns.heatmap(correlation,cmap='coolwarm',annot=True)


y=train_df['Calories']
x = train_df.drop(columns=['Calories','id'])


x.describe()


x.info()


features_to_check = x.drop(columns=['Sex_male'])  


import matplotlib.pyplot as plt
num_features = features_to_check.shape[1]
rows = (num_features + 2) // 2
cols=2
fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()  

for i, col in enumerate(features_to_check.columns):
    axes[i].boxplot(features_to_check[col])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel(col)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


Q1 = features_to_check.quantile(0.25)
Q3 = features_to_check.quantile(0.75)
IQR = Q3 - Q1

# Outlier condition
outlier_mask = (features_to_check < (Q1 - 1.5 * IQR)) | (features_to_check > (Q3 + 1.5 * IQR))
outlier_counts = outlier_mask.sum()
print("Outlier count per feature:")
print(outlier_counts)



sns.histplot(features_to_check['Body_Temp'], bins=50, kde=True)
plt.title("Distribution of Body_Temp")
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


import cupy as cp
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error

X_train_gpu = cp.array(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
y_train_gpu = cp.array(y_train.values if isinstance(y_train, pd.Series) else y_train)

model_xg = XGBRegressor(
    objective='reg:squaredlogerror',
    tree_method='hist',       # Use histogram-based algorithm
    device='cuda',            # Use GPU explicitly
    random_state=42,
    verbosity=0
)

# Hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2],
}

# Custom MSLE scorer
def msle(y_true, y_pred):
    y_true_np = cp.asnumpy(y_true) if isinstance(y_true, cp.ndarray) else y_true
    y_pred_np = cp.asnumpy(y_pred) if isinstance(y_pred, cp.ndarray) else y_pred
    return mean_squared_log_error(y_true_np, y_pred_np)

msle_scorer = make_scorer(msle, greater_is_better=False)

random_search = RandomizedSearchCV(
    estimator=model_xg,
    param_distributions=param_dist,
    n_iter=50,
    scoring=msle_scorer,
    cv=3,
    verbose=0,
    
    random_state=42,
    n_jobs=-1
)

# Fit the model using GPU arrays
random_search.fit(X_train_gpu, y_train_gpu)


from sklearn.metrics import mean_squared_log_error

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
msle_value = mean_squared_log_error(y_test, y_pred)

print(f"Mean Squared Log Error (MSLE) on validation set: {msle_value:.6f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('XGBoost: Actual vs Predicted')




test_df.head()


test_df = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)
test_df['Sex_male']=test_df['Sex_male'].astype(int)
x_final=test_df.drop(columns=['id'])
x_final.head()


y_finalxg = best_model.predict(x_final)
y_finalxg = [round(val, 3) for val in y_finalxg]


final=pd.DataFrame({
    'id':test_df['id'],
    'Calories':y_finalxg
})


final.head()


submission_df.head()


final.to_csv('mysubmission.csv',index=False)


import os
print(os.getcwd())


import os
os.listdir('/kaggle/working')

