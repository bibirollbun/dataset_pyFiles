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


import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error


Train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
Train_df.head()


Test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
Test_df.head()


Train_df.info()


Test_df.info()


Train_df.duplicated().sum() , Test_df.duplicated().sum()


(Train_df['Calories'] < 0).any()



Train_df['Sex'].unique()


Train_df.describe()


Q1 = Train_df['Calories'].quantile(0.25)
Q3 = Train_df['Calories'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = Train_df[(Train_df['Calories'] < lower) | (Train_df['Calories'] > upper)]
outliers



Train_df = Train_df[(Train_df['Calories'] >= lower) & (Train_df['Calories'] <= upper)]


sns.boxplot(x='Sex', y='Calories', data=Train_df)


# Label Encoding
Train_df['Sex'] = Train_df['Sex'].map({'male': 0, 'female': 1})
Test_df['Sex'] = Test_df['Sex'].map({'male': 0, 'female': 1})



#Correlation Heatmap
sns.heatmap(Train_df.corr(),annot=True, fmt=".2f", cmap="coolwarm")


Train_df["BMI"] = Train_df["Weight"] / (Train_df["Height"] / 100) ** 2

Test_df["BMI"] = Test_df["Weight"] / (Test_df["Height"] / 100) ** 2


sns.heatmap(Train_df.corr(),annot=True, fmt=".2f", cmap="coolwarm")


# Drop Height and Weight columns
Train_df = Train_df.drop(['Weight'], axis=1)
Test_df = Test_df.drop(['Weight'], axis=1)


plt.figure(figsize=(15,  12))

for i, col in enumerate(Train_df.columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(Train_df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



Train_df['Age_Trans'] = np.log1p(Train_df['Age'])
Test_df['Age_Trans'] = np.log1p(Test_df['Age'])


Train_df['Body_Temp_trans'] = np.log1p(Train_df['Body_Temp'].max() - Train_df['Body_Temp'])
Test_df['Body_Temp_trans'] = np.log1p(Test_df['Body_Temp'].max() - Test_df['Body_Temp'])



sns.histplot(Train_df['Age'], kde=True)
plt.title("Age Before Transformation")
plt.show()

sns.histplot(Train_df['Age_Trans'], kde=True)
plt.title("Age After Transformation")
plt.show()



sns.histplot(Train_df['Body_Temp'], kde=True)
plt.title("Body_Temp Before Transformation")
plt.show()

sns.histplot(Train_df['Body_Temp_trans'], kde=True)
plt.title("Body_Temp After Transformation")
plt.show()


Train_df.drop(['Age', 'Body_Temp'],axis = 1, inplace = True)
Train_df.head()


Test_df.drop(['Age', 'Body_Temp'],axis = 1, inplace = True)           
Test_df.head()


X = Train_df.drop(columns=['Calories', 'id'])
y = Train_df['Calories']


# reserving 20% of the training data to validation 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.shape , X_val.shape , y_train.shape , y_val.shape


lr = LinearRegression()
lr.fit(X_train, y_train)


lasso = Lasso(alpha=0.05) # Alpha is the regulariztion strength
lasso.fit(X_train, y_train)


ridge = Ridge(alpha=0.5) 
ridge.fit(X_train, y_train)


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)



rf.get_params()                        


# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

# # Define distributions instead of fixed lists
# param_dist = {
#     'n_estimators': randint(100, 300),        # integers between 100 and 300
#     'max_depth': [10, 20, None],
#     'min_samples_split': randint(2, 10),
# }

# # Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=RandomForestRegressor(),
#     param_distributions=param_dist,
#     n_iter=10,  # Number of random combinations to try
#     scoring='neg_mean_squared_log_error',
#     cv=3,
#     random_state=42,
#     n_jobs=-1  # Use all cores for faster processing

# # Run search
# random_search.fit(X_train, y_train)

# # Best model
# best_model = random_search.best_estimator_





# best_model.get_params()                        



rf_tuned = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)

rf_tuned.fit(X_train, y_train)



from xgboost import XGBRegressor
xgb = XGBRegressor( n_jobs=-1)

xgb.fit(X_train, y_train)


xgb.get_params()


xgb_tuned = XGBRegressor(
    n_estimators=900,        # More boosting rounds
    max_depth=10,             # deeper trees
    learning_rate=0.03,      # Lower for better generalization
    subsample=0.95,          # More randomness
    colsample_bytree=0.95,   # More randomness
    min_child_weight=2,      # Slightly more flexible
    reg_alpha=0.05,           # L1 regularization
    reg_lambda=1,          # L2 regularization
    random_state=42,
    n_jobs=-1
)

xgb_tuned.fit(X_train , y_train)


def evaluate(model):
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)  # Replace negatives with 0
    rmsle = mean_squared_log_error(y_val, y_pred, squared=False)
    print(f"{model.__class__.__name__} RMSlE: {rmsle:.5f}")


evaluate(lr) 
evaluate(lasso)
evaluate(ridge)
evaluate(rf)


# evaluate(best_model)


evaluate(rf_tuned)


evaluate(xgb_tuned)


Test_df.head()


#Predicting calories for the test data
X_test = Test_df.drop('id', axis = 1)
y_pred_test = xgb_tuned.predict(X_test)
y_pred_test = np.clip(y_pred_test, 0, None)  # Replace negatives with 0
y_pred_test.shape


#preparing submission file
submission = pd.DataFrame({
    'id': Test_df['id'],
    'Calories': y_pred_test
})

submission.head()


# Save it to a CSV file
submission.to_csv('submission.csv', index=False)




