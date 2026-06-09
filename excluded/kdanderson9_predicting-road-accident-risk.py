
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
train_data.head()


train_data.describe()


print(train_data.shape)
print(test_data.shape)


train_data.info()


# convert boolean values to int

bool_cols = ['road_signs_present', 'public_road','holiday', 'school_season']
for col in bool_cols:
    train_data[f'{col}'] = train_data[f'{col}'].astype(int)
    test_data[f'{col}'] = test_data[f'{col}'].astype(int)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

categorical_cols = ['road_type','lighting','weather','time_of_day']
for col in categorical_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])


train_data=train_data.select_dtypes(include='number')
test_data=test_data.select_dtypes(include='number')


train_data.head()


num_attributes = train_data.select_dtypes(include=["number"]).copy()
corr_matrix = num_attributes.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# columns with the highest correlation to accident_risk
accident_correlations = corr_matrix['accident_risk'].abs().sort_values(ascending=False)[:6].index.tolist()

train_data = train_data[accident_correlations]
train_data.head()



# Splitting the dataset
X = train_data.drop('accident_risk', axis=1)
y = train_data['accident_risk']
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(['accident_risk'], axis = 1), y, test_size = 0.2, random_state = 6)





from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)




# calculate score on train and test sets
y_pred=linear_model.predict(X_val)
mse = mean_squared_error(y_pred, y_val)
print("Test mean squared error (MSE): {:.2f}".format(mse))

print("R square score:", linear_model.score(X_val,y_val))


test_data_copy = test_data.copy()

irrelevant_cols = ['id','num_lanes', 'public_road', "time_of_day", "school_season", "road_signs_present"]

test_data_copy = test_data_copy.drop(columns=irrelevant_cols)


test_data_copy.isna().sum().sort_values(ascending=False).head(10)


#test_X = pd.get_dummies(test_data_copy)


submission = pd.DataFrame({
    'id': test_data['id'],
    'accident_risk': 0
})

submission.to_csv('submission.csv', index=False)

print(submission.head())

