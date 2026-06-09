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


data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data.head()


data.info()


data = data.rename(columns = {"Laptop Compartment": "Laptop_compartment", "Weight Capacity (kg)":"Weight_Capacity"})


import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('Price ~ C(Color)', data=data).fit()

anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)


#Find Correlation by price
numerical_df = data[['Price', 'Weight_Capacity', 'Compartments']]
correlation_matrix = numerical_df.corr()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidth=1)
plt.title('Correlation House Price')
plt.show()


data.dropna(inplace=True)
data.info()


from sklearn.model_selection import train_test_split


y = data["Price"]
features = ["Brand", "Material", "Color","Weight_Capacity"]
X = pd.get_dummies(data[features], dtype = int)

#X_test = pd.get_dummies(test[features], dtype = int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




y_train


X_train


from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)
y_pred_lr = regr.predict(X_val)


from sklearn.metrics import mean_squared_error, r2_score
def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"ðŸ“ˆ {model_name} - RMSE: {rmse:.4f}, RÂ² Score: {r2:.4f}")


evaluate_model(y_val, y_pred_lr, "Linear Regression")


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test = test.rename(columns = {"Laptop Compartment": "Laptop_compartment", "Weight Capacity (kg)":"Weight_Capacity"})
test.info()


test.head()


test_encode = pd.get_dummies(test[["Brand", "Material", "Color"]], dtype = int)
test_encode


test_encode.isna().sum()


df_combined = pd.concat([test, test_encode], axis=1)
df_combined.isna().sum()


df_test = df_combined.drop(['id','Brand', 'Material','Size','Compartments','Waterproof','Style','Color','Laptop_compartment'], axis = 1)


df_test['Weight_Capacity'] = df_test['Weight_Capacity'].fillna(0)


df_test.isna().sum()


test_prediction = regr.predict(df_test)


test_prediction


# Create Submission File
submission = pd.DataFrame({'id': test.id, 'Price': test_prediction})  # Use Linear regression predictions
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission file saved: submission.csv")


display(submission)

