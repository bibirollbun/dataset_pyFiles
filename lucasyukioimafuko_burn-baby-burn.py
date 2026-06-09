# EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import warnings

# preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# metrics
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.sample(10)


test.sample(10)


train.drop(['Sex','id'],axis=1).describe().loc[['mean','std','min','50%','max']].transpose()


cal_count = train['Calories'].value_counts().sort_values()
cal_count


high_calories = train[train['Calories'] >= 280]
print('\n High Calorie Spenders \n')
high_calories


print('\n Summary on High Calorie Spenders \n')
high_calories.drop(['id','Sex'],axis=1).describe().loc[['mean','std','min','max']].transpose()


high_calories['Sex'].value_counts()


fig,ax = plt.subplots(1,3, figsize=(13,4))

sea.boxplot(data=high_calories, x='Age', ax=ax[0])
sea.boxplot(data=high_calories, x='Heart_Rate', ax=ax[1])
sea.boxplot(data=high_calories, x='Body_Temp', ax=ax[2])

plt.tight_layout()


train['Cal_per_Min'] = round(train['Calories'] / train['Duration'], 2)


train.head()


train['Cal_per_Min'].sort_values(ascending=False).head(10)


cal_min = train['Cal_per_Min'].sort_values(ascending=False).head(10)

train.loc[cal_min.index]


train = train.drop('Cal_per_Min',axis=1)


train.drop(['id','Sex'],axis=1).corr()['Calories'].sort_values()


corr_matrix = train.drop(['id','Sex'],axis=1).corr()

plt.figure(dpi=150)
sea.heatmap(data=corr_matrix, linewidths=2, square=True, annot=True, fmt = '.2f', cmap='icefire')


print(f'Train NAs:\n {train.isna().sum()}')
print(f'\nTrain NULLs:\n {train.isnull().sum()}')


print(f'Test NAs:\n {test.isna().sum()}')
print(f'\nTest NULLs:\n {test.isnull().sum()}')


# sampling for computational purposes
train_sample = train.sample(frac=.1)
train_sample = train_sample.drop('id',axis=1)


cols = ['Age','Heart_Rate','Body_Temp','Calories']

for i in cols:
    plt.figure(figsize=(8,4))
    sea.histplot(data=train_sample, x=i, hue='Sex', kde=True, common_norm=False, stat='density', element='step')
    plt.title(f'{i} Density Distribution')
    plt.xlabel(i)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()


sea.set_style('whitegrid')

box_cols = ['Age','Height','Weight','Duration']

for i in box_cols:
    plt.figure(figsize=(8,4))
    sea.boxplot(data=train, x=i, width=.5)
    plt.title(f'{i} Boxplot Distribution')
    plt.tight_layout()
    plt.show()


f, ax = plt.subplots(1, 4, figsize=(14,4))

sea.boxplot(ax=ax[0], data=train_sample, x='Sex', y='Calories')
sea.boxplot(ax=ax[1], data=train_sample, x='Sex', y='Body_Temp')
sea.boxplot(ax=ax[2], data=train_sample, x='Sex', y='Duration')
sea.boxplot(ax=ax[3], data=train_sample, x='Sex', y='Heart_Rate')

f.tight_layout()


corrplots = sea.PairGrid(train_sample, x_vars=['Duration','Heart_Rate','Body_Temp'], y_vars='Calories', height=4, hue='Sex')
corrplots.map(sea.scatterplot, alpha=.3, s=4)
corrplots.add_legend()


label_encoder = LabelEncoder()
scaler = StandardScaler()


# encoding 'Sex' variable

train['Sex'] = label_encoder.fit_transform(train['Sex'])
train = train.drop('id',axis=1)


X = train.drop('Calories', axis=1)
y = train['Calories']


xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=.3, random_state=13)


print(f'Xtrain: {xtrain.shape}')
print(f'Xtest: {xtest.shape}')
print(f'ytrain: {ytrain.shape}')
print(f'ytest: {ytest.shape}')


# scaling

xtrain = scaler.fit_transform(xtrain)
xtest = scaler.fit_transform(xtest)


print(f'xtrain standard deviation: {xtrain.std()}')
print(f'xtest standard deviation: {xtest.std()}')
print(f'xtrain mean: {xtrain.mean()}')
print(f'xtest mean: {xtest.mean()}')


models = [
    ('Linear Regression',LinearRegression()),
    ('ElasticNet',ElasticNet()),
    ('Decision Tree Regressor',DecisionTreeRegressor()),
    ('Gradient Boost Regressor',GradientBoostingRegressor())
]

scores = []


for spec,model in models:
    # fitting
    model.fit(xtrain,ytrain)
    pred = model.predict(xtrain)

    # scores
    r2 = r2_score(ytrain,pred)
    mse = mean_squared_error(ytrain,pred)
    rmse = np.sqrt(mse)

    scores.append([spec,r2,mse,rmse])


print('Training Set Results')
pd.DataFrame(columns=['Model','R2','Mean Squared Error','Root Mean Square Error'], data=scores)


models = [
    ('Linear Regression',LinearRegression()),
    ('ElasticNet',ElasticNet()),
    ('Decision Tree Regressor',DecisionTreeRegressor()),
    ('Gradient Boost Regressor',GradientBoostingRegressor())
]

scores_test = []


for spec,model in models:
    # fitting
    model.fit(xtest,ytest)
    test_pred = model.predict(xtest)

    # scores
    r2 = r2_score(ytest,test_pred)
    mse = mean_squared_error(ytest,test_pred)
    rmse = np.sqrt(mse)

    scores_test.append([spec,r2,mse,rmse])


print('Testing Set Results')
pd.DataFrame(columns=['Model','R2','Mean Squared Error','Root Mean Squared Error'], data=scores_test)


final_model = DecisionTreeRegressor()
final_model.fit(X,y)


# removing and isolating 'id' from test set, so we can put it back on submission data

test_id = test['id']

final_test_df = test.drop('id', axis=1)


# encoding gender column in final test set

final_test_df['Sex'] = label_encoder.fit_transform(final_test_df['Sex'])


final_test_df.head(10)


final_pred = final_model.predict(final_test_df)


submission = pd.DataFrame({'id':test_id,'Calories':final_pred})
submission.head(10)


submission.to_csv('submission.csv', index=False)

