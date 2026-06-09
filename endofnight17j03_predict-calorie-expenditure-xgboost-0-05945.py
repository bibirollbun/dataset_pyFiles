import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')


rc = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4,
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
mgt = Style.BRIGHT + Fore.MAGENTA
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL

plt.style.use('fivethirtyeight')


train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.sample(10).style.background_gradient()


test.sample(10).style.background_gradient()


train.info()


test.info()


train.describe().T.style.background_gradient()


test.describe().T.style.background_gradient()


sns.displot(data = train.isnull().melt(value_name='missing'),
           y = 'variable',
           hue = 'missing',multiple = "fill",height = 8,aspect = 1.6)
plt.axvline(0.4,color = 'r')
plt.title("Null values in train data",fontsize = 13)
plt.show()


sns.displot(data = test.isnull().melt(value_name='missing'),
           y = 'variable',
           hue = 'missing',multiple = "fill",height = 8,aspect = 1.6)
plt.axvline(0.4,color = 'r')
plt.title("Null values in train data",fontsize = 13)
plt.show()


# Weight_per_Age
train['Weight_per_Age'] = train['Weight'] / (train['Age'] + 1)
test['Weight_per_Age'] = test['Weight'] / (test['Age'] + 1)

# HeartRate per Weight
train['HeartRate_per_kg'] = train['Heart_Rate'] / train['Weight']
test['HeartRate_per_kg'] = test['Heart_Rate'] / test['Weight']
# Duration Per Age
train['Duration_per_age'] = train['Duration'] / (train['Age'] + 1)
test['Duration_per_age'] = test['Duration'] / (test['Age'] + 1 )

# Duration * Heart Rate
train['Duration_heart_rate']=train['Duration']*train['Heart_Rate']
test['Duration_heart_rate']=test['Duration']*test['Heart_Rate']

# Intensity
train['Duration_per_weight']=train['Duration']/train['Weight']
test['Duration_per_weight']=test['Duration']/test['Weight']

# All Durations add and multi
train['duration_sum']=train['Duration_per_weight']+train['Duration_heart_rate']+train['Duration_per_age']
test['duration_sum']=test['Duration_per_weight']+test['Duration_heart_rate']+test['Duration_per_age']

train['duration_multi']=train['Duration_per_weight']*train['Duration_heart_rate']*train['Duration_per_age']
test['duration_multi']=test['Duration_per_weight']*test['Duration_heart_rate']*test['Duration_per_age']

# Converting Height in Meters
train['Height']=train['Height']/100
test['Height']=test['Height']/100

# Creating new column 'BMI'
train['BMI']=train['Weight']/(train['Height'] ** 2)
train['BMI']=train['BMI'].round(2)
test['BMI']=test['Weight']/(test['Height'] ** 2)
test['BMI']=test['BMI'].round(2)

# Mapping Genders
map={'male':0,'female':1}
train['Sex']=train['Sex'].map(map)
test['Sex']=test['Sex'].map(map)



from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error

x = train.drop(columns=['Calories', 'id'])
y = train['Calories']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,        
    reg_lambda=1.0,            
    reg_alpha=0.5,            
    subsample=0.8,           
    colsample_bytree=0.8,    
    min_child_weight=10,      
    gamma=0.2,               
    max_delta_step=0,      
    grow_policy='depthwise',
    tree_method='hist',        
    objective='reg:squarederror',
    random_state=42,
    verbosity=0,
    n_jobs=-1   
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_clipped = np.maximum(0, y_pred)

print(f"MSE: {mean_squared_error(y_test, y_pred_clipped)}")
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred_clipped))
print(f"RMSLE: {rmsle}")



feature_names = x.columns
raw_importance = model.feature_importances_
normalized_importance = raw_importance / raw_importance.sum()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'IMP': normalized_importance
}).sort_values(by='IMP', ascending=False)
importance_df = importance_df.round(6)


importance_df.style.background_gradient()


import joblib
joblib.dump(model,'model.joblib')
print('Model saved Successfully')
model_lk=joblib.load('model.joblib')
print('Model loaded Successfully')




import joblib
import pandas as pd
import numpy as np
import os

model = joblib.load('model.joblib')
print('Model loaded Successfully')

test = test.rename(columns={'duration_multu': 'duration_multi'})

X_test = test.drop(columns=['id'])

y_pred = model.predict(X_test)
y_pred_clipped = np.maximum(0, y_pred)

submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_pred_clipped
})

submission.to_csv('./submission.csv', index=False)

if os.path.exists('./submission.csv'):
    print('File saved successfully!')
else:
    print('Error: File was not saved.')




submission.head().style.background_gradient()


!head submission.csv

