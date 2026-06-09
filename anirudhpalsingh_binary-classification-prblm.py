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


df = pd.read_csv('/kaggle/input/playground-series-s4e7/train.csv')
df.head()


df.info()


df.isnull().sum() ### no missing values are observed 


df.describe()


df.head()


import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns

sns.countplot(x='Gender', data=df, ax=axs[0])
axs[0].set_title("Gender")

sns.countplot(x='Vehicle_Damage', data=df, ax=axs[1])
axs[1].set_title("Vehicle Damage")

sns.countplot(x='Vehicle_Age', data=df, ax=axs[2])
axs[2].set_title("Vehicle Age")

plt.tight_layout()
plt.show()



### Looking at Numerical datasets ###

sns.distplot(df['Age'])


### Looking at Numerical datasets ###

plt.hist(df['Region_Code'])



# Select only numeric columns (excluding ID if needed)
numeric_cols = df.select_dtypes(include=['number']).columns

# Compute correlation matrix
corr_matrix = df[numeric_cols].corr()

# Get correlation of each numeric feature with the target 'Response'
corr_with_response = corr_matrix['Response'].drop('Response')  # drop self-correlation

# Sort by absolute correlation
corr_with_response = corr_with_response.abs().sort_values(ascending=False)

print("Correlation with Response:\n")
print(corr_with_response)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.iloc[: , 1:-1], df.iloc[: , -1], test_size = 0.2 , random_state = 42)


x_train.head()


y_train.head()


from sklearn.preprocessing import OneHotEncoder

# Initialize the encoder
ohe_gender = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe_vehicle = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit on training data and transform
x_train_gender = ohe_gender.fit_transform(x_train[['Gender']])
x_train_vehicle = ohe_vehicle.fit_transform(x_train[['Vehicle_Damage']])

# Transform test data using the fitted encoder
x_test_gender = ohe_gender.transform(x_test[['Gender']])
x_test_vehicle = ohe_vehicle.transform(x_test[['Vehicle_Damage']])



print(x_train_gender)


import numpy as np

# Drop the original categorical columns
x_train_rest = x_train.drop(['Gender', 'Vehicle_Damage','Vehicle_Age'], axis=1)
x_test_rest = x_test.drop(['Gender', 'Vehicle_Damage','Vehicle_Age'], axis=1)

# Concatenate encoded and numerical features
x_train_final = np.hstack([x_train_rest.values, x_train_gender, x_train_vehicle])
x_test_final = np.hstack([x_test_rest.values, x_test_gender, x_test_vehicle])

### dropping one var 



x_train_final


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()


clf.fit(x_train_final,y_train)


y_pred = clf.predict(x_test_final)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


test_vf = pd.read_csv('/kaggle/input/playground-series-s4e7/test.csv')

print(test_vf.head())


from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Assume encoders were already fit on training data
# If not, use .fit() on training data and then .transform() on test data

# Transform test data using fitted encoders

x_gender_vf = ohe_gender.transform(test_vf[['Gender']])
x_vehicle_vf = ohe_vehicle.transform(test_vf[['Vehicle_Damage']])

#x_age_vf = ohe_age.transform(test_vf[['Vehicle_Age']])  # If you encoded this too

# Drop original categorical columns from test_vf
x_test_rest = test_vf.drop(['Gender', 'Vehicle_Damage', 'Vehicle_Age','id'], axis=1)

# Concatenate encoded features and numeric columns
x_test_final = np.hstack([x_test_rest.values, x_gender_vf, x_vehicle_vf])



print(x_test_final)


# Predict probabilities
y_proba = clf.predict_proba(x_test_final)[:, 1]  # Probability of class 1





# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_vf['id'],
    'Response': y_proba
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

