import pandas as pd
import warnings
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print(train.head())
train.info()
train.isnull().sum()
train.describe()


print(test.head())
test.info()
test.isnull().sum()
test.describe()


train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})


#Cleaning the train data


q1, q3 = np.percentile(train['Age'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = train[(train['Age'] >= lower_bound) 
                & (train['Age'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Height'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = clean_data[(clean_data['Height'] >= lower_bound) 
                        & (clean_data['Height'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Age'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = clean_data[(clean_data['Age'] >= lower_bound) 
                        & (clean_data['Age'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Weight'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = clean_data[(clean_data['Weight'] >= lower_bound) 
                        & (clean_data['Weight'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Duration'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (0.75 * iqr)
upper_bound = q3 + (0.75 * iqr)
clean_data = clean_data[(clean_data['Duration'] >= lower_bound) 
                        & (clean_data['Duration'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Heart_Rate'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = clean_data[(clean_data['Heart_Rate'] >= lower_bound) 
                        & (clean_data['Heart_Rate'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Body_Temp'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = clean_data[(clean_data['Body_Temp'] >= lower_bound) 
                        & (clean_data['Body_Temp'] <= upper_bound)]


q1, q3 = np.percentile(clean_data['Calories'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = clean_data[(clean_data['Calories'] >= lower_bound) 
                        & (clean_data['Calories'] <= upper_bound)]


#correlation
corr = train.corr()

plt.figure(dpi=130)
sns.heatmap(train.corr(), annot=True, fmt= '.2f')
plt.show()


#Dropping id, because it has no correlation to Calroies
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


# BMR Calculation for test
test['BMR']=0


i=0
for index, row in test.iterrows():
    if(row['Sex']==0):
        BMR = (10*row['Weight']) + (6.25*row['Height']) - (5*row['Age']) + 5
        test['BMR'][i] = BMR
        i+=1
    else:
        BMR = (10*row['Weight']) + (6.25*row['Height']) - (5*row['Age']) - 161
        test['BMR'][i] = BMR
        i+=1


#Calculating the other features
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)
test['HR_Duration'] = test['Heart_Rate'] * test['Duration']
test['Temp_Duration'] = test['Body_Temp'] * test['Duration']
test['Age_Group'] = pd.cut(test['Age'], bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3])


print(test.head())


#BMR Calculation for train
train['BMR']=0


i=0
for index, row in train.iterrows():
    if(row['Sex']==0):
        BMR = (10*row['Weight']) + (6.25*row['Height']) - (5*row['Age']) + 5
        train['BMR'][i] = BMR
        i+=1
    else:
        BMR = (10*row['Weight']) + (6.25*row['Height']) - (5*row['Age']) - 161
        train['BMR'][i] = BMR
        i+=1


#Calculating the other features

train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
train['HR_Duration'] = train['Heart_Rate'] * train['Duration']
train['Temp_Duration'] = train['Body_Temp'] * train['Duration']
train['Age_Group'] = pd.cut(train['Age'], bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3])


print(train.head())


#correlation
corr = train.corr()

plt.figure(dpi=130)
sns.heatmap(train.corr(), annot=True, fmt= '.2f')
plt.show()


corr['Calories'].sort_values(ascending = False)


X = train.drop(columns =(['Calories']))
Y = train['Calories']


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
regr_2 = AdaBoostRegressor(
   DecisionTreeRegressor(max_depth=7,criterion='squared_error',splitter='best'), n_estimators=120, learning_rate=0.1, random_state=1
)

regr_2.fit(X, Y)

y_2 = regr_2.predict(test)
print(y_2)

