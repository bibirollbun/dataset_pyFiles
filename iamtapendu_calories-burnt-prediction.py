import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import cross_validate
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id')


data.info()


data.describe()[1:].style.background_gradient()


test.describe()[1:].style.background_gradient()


data.isna().sum()


test.isna().sum()


data.head()


data['Sex'] = data['Sex'].replace({'male':2,'female':1})
test['Sex'] = test['Sex'].replace({'male':2,'female':1})


data['BMI'] = data['Weight']/(data['Height']/100)**2
test['BMI'] = test['Weight']/(test['Height']/100)**2

data['Intensity'] = data['Heart_Rate']/data['Duration']
test['Intensity'] = test['Heart_Rate']/test['Duration']

data['Heart_Rate_X_Duration'] = data['Heart_Rate']*data['Duration']
test['Heart_Rate_X_Duration'] = test['Heart_Rate']*test['Duration']

data['Body_Temp_X_Duration'] = data['Body_Temp']*data['Duration']
test['Body_Temp_X_Duration'] = test['Body_Temp']*test['Duration']

data['Heart_Rate_X_Body_Temp'] = data['Heart_Rate']*data['Body_Temp']
test['Heart_Rate_X_Body_Temp'] = test['Heart_Rate']*test['Body_Temp']

data['Heart_Rate_X_Body_Temp_X_Duration'] = data['Heart_Rate']*data['Body_Temp']*data['Duration']
test['Heart_Rate_X_Body_Temp_X_Duration'] = test['Heart_Rate']*test['Body_Temp']*test['Duration']

data['Weight_X_Duration'] = data['Weight']*data['Duration']
test['Weight_X_Duration'] = test['Weight']*test['Duration']

data['Weight_Duration'] = data['Duration']/data['Weight']
test['Weight_Duration'] = test['Duration']/test['Weight']


cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Intensity', 'Heart_Rate_X_Duration', 'Heart_Rate_X_Body_Temp', 'Weight_X_Duration']
fig, ax = plt.subplots(6,2,figsize=(8,18))
ax = ax.flatten()
for i,col in enumerate(cols):
    sns.kdeplot(data=data,x=col,ax=ax[i])
    sns.kdeplot(data=test,x=col,color='r',ax=ax[i])
    ax[i].set_yticks([])
    ax[i].set_title(col)

sns.kdeplot(data=data,x='Calories',ax=ax[-1])
ax[-1].set_yticks([])
ax[-1].set_title('Calories')

plt.suptitle('Distributions')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(6,2,figsize=(8,18))
ax = ax.flatten()
for i,col in enumerate(cols):
    sns.kdeplot(data=data.sample(frac=0.01),x=col,y='Calories',fill=True,levels=20,ax=ax[i])
    ax[i].set_title(f'{col}_X_Calories')
plt.tight_layout()
plt.show()


data['Body_Temp2'] = data['Body_Temp']**0.5
test['Body_Temp2'] = test['Body_Temp']**0.5


plt.figure(figsize=(8,8))
sns.heatmap(data.corr(method='spearman'),annot=True,fmt='.2f',cbar=False,cmap='coolwarm')
plt.show()


print('Total Duplicate Records(Including Target):',data.duplicated().sum())

data.drop_duplicates(inplace=True)

print('Total Duplicate Records(Excluding Target):',data.drop('Calories',axis=1).duplicated().sum())

feats = list(data.columns)
feats.remove('Calories')

data = data.groupby(feats)['Calories'].median().reset_index()


# Function to calculate mean cross-validation scores
def mean_cross_validation_scores(score):
    # Calculate the mean score for each metric
    mean_scores = {metric: np.mean(values) for metric, values in score.items()}
    
    # Return a DataFrame with mean scores, indexed by 'Score'
    return pd.DataFrame(mean_scores, index=['Score'])

X, y = data.drop('Calories',axis=1), np.log1p(data['Calories'])

model = LGBMRegressor(n_estimators=10000,
                      learning_rate=0.02,
                      max_depth=10,
                      subsample= 0.8,
                      colsample_bytree= 0.8,
                      random_state=1,verbose=-1)

score = cross_validate(model,X,y,cv=10,
                       scoring=['r2','neg_mean_squared_log_error','neg_mean_absolute_error','neg_root_mean_squared_error'],
                       return_train_score=True,
                       return_estimator=True)
estimators = score.pop('estimator')
mean_cross_validation_scores(score)


pred = [np.expm1(i.predict(test)) for i in estimators]
pred = np.mean(pred,axis=0)
test['Calories'] = np.clip(pred,1,314)
test['Calories'].to_csv(f'submission.csv')

