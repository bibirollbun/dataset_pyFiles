import catboost


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print("Train df shape:", train_df.shape)
print("Test df shape:", test_df.shape)


train_df.head()


test_df.head()


train_df.isna().mean()*100


test_df.isna().mean()*100


train_df['Sex']=train_df['Sex'].apply(lambda x: 1 if x=="male" else 0)
test_df['Sex']=test_df['Sex'].apply(lambda x: 1 if x=="male" else 0)


train_df.info()


test_ids=test_df['id']


train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)


numerical_cols = train_df.select_dtypes(include=['number']).columns

plt.figure(figsize=(15, len(numerical_cols) * 4))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(len(numerical_cols), 1, i)
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f"Distribution with KDE: {col}")
    plt.xlabel(col)

plt.tight_layout()
plt.show()


def detecting_outliers(val):
    sns.boxplot(x=f'{val}', data=train_df)
    plt.title("Box plot")
    plt.show()


detecting_outliers('Age')
detecting_outliers('Height')
detecting_outliers('Weight')
detecting_outliers('Duration')
detecting_outliers('Heart_Rate')
detecting_outliers('Body_Temp')


def get_iqr_bounds(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def cap_outliers_with_bounds(df, column, lower_bound, upper_bound):
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
for col in numerical_cols:
    lower, upper = get_iqr_bounds(train_df, col)
    train_df = cap_outliers_with_bounds(train_df, col, lower, upper)
    test_df = cap_outliers_with_bounds(test_df, col, lower, upper)


detecting_outliers('Age')
detecting_outliers('Height')
detecting_outliers('Weight')
detecting_outliers('Duration')
detecting_outliers('Heart_Rate')
detecting_outliers('Body_Temp')


X=train_df.drop(columns=['Calories'])
y=train_df['Calories']


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


X_train_scaled=pd.DataFrame(X_train_scaled)
X_test_scaled=pd.DataFrame(X_test_scaled)


test_df_scaled=scaler.transform(test_df)
test_df_scaled=pd.DataFrame(test_df_scaled)


linear_model=LinearRegression()
linear_model.fit(X_train_scaled, y_train)


linear_pred=linear_model.predict(X_test_scaled)


linear_pred_clipped=np.maximum(0, linear_pred)


print("Mean Squared Log Error:", mean_squared_log_error(y_test, linear_pred_clipped))
print("Root Mean Squared Log Error:", np.sqrt(mean_squared_log_error(y_test, linear_pred_clipped)))


predictions1=linear_model.predict(test_df)


predictions1=np.maximum(0, predictions1)


submission1 = pd.DataFrame({'id': test_ids, 'Calories': predictions1})
submission1.to_csv('submission1.csv', index=False)
print("Submission1 file created")


dt_model=DecisionTreeRegressor()

dt_model.fit(X_train_scaled, y_train)


dt_pred=dt_model.predict(X_test_scaled)


dt_pred_clipped=np.maximum(0, dt_pred)


print("Mean Squared Log Error:", mean_squared_log_error(y_test, dt_pred_clipped))
print("Root Mean Squared Log Error:", np.sqrt(mean_squared_log_error(y_test, dt_pred_clipped)))


predictions2=dt_model.predict(test_df_scaled)


predictions2=np.maximum(0, predictions2)


submission2=pd.DataFrame({'id': test_ids, 'Calories': predictions2})
submission2.to_csv('submission2.csv', index=False)
print("Submission2 file created")


random_model=RandomForestRegressor()

random_model.fit(X_train_scaled, y_train)


random_pred=random_model.predict(X_test_scaled)


random_pred_clipped=np.maximum(0, random_pred)


print("Mean Squared Log Error:", mean_squared_log_error(y_test, random_pred_clipped))
print("Root Mean Squared Log Error:", np.sqrt(mean_squared_log_error(y_test, random_pred_clipped)))


predictions3=random_model.predict(test_df_scaled)


predictions3=np.maximum(0, predictions3)


submission3=pd.DataFrame({'id': test_ids, 'Calories': predictions3})
submission3.to_csv('submission3.csv', index=False)
print("Submission3 file created")


train_df7=train_df.copy()
test_df7=test_df.copy()


train_df7['Duration^2']=train_df7['Duration']**2
test_df7['Duration^2']=test_df7['Duration']**2


X7=train_df7.drop(columns=['Calories'])
y7=train_df7['Calories']


X7_train, X7_test, y7_train, y7_test=train_test_split(X7, y7, test_size=0.2)


scaler=StandardScaler()

X7_train_scaled=scaler.fit_transform(X7_train)
X7_test_scaled=scaler.transform(X7_test)

test_df7_scaled=scaler.transform(test_df7)


y7_train_log = np.log1p(y7_train)
y7_test_log = np.log1p(y7_test)


cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_model.fit(X7_train_scaled, y7_train_log)


y_pred_log = cat_model.predict(X7_test_scaled)
y_pred = np.expm1(y_pred_log)


print("Root Mean Squared Log Error:", np.sqrt(mean_squared_log_error(y7_test, y_pred)))


predictions8=cat_model.predict(test_df7_scaled)
predictions8=np.expm1(predictions8)


submission8=pd.DataFrame({'id': test_ids, 'Calories': predictions8})
submission8.to_csv('submission8.csv', index=False)
print("Submission8 file created")




