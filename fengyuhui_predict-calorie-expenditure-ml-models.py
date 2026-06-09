import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# Load Data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)


# Feature Engineering

# 1. One-hot encoding
train_df['female'] = 0
train_df.loc[train_df.Sex=='female','female'] = 1

test_df['female'] = 0
test_df.loc[test_df.Sex=='female','female'] = 1

# 2. Log Transformation
train_df['log_Calories'] = np.log(train_df.Calories)
train_df['Calories_per_Duration'] = train_df.Calories/train_df.Duration

train_df['log_Heart_Rate'] = np.log(train_df.Heart_Rate)
test_df['log_Heart_Rate'] = np.log(test_df.Heart_Rate)

train_df['Heart_Rate_2'] = train_df.Heart_Rate ** 2
test_df['Heart_Rate_2'] = test_df.Heart_Rate ** 2

train_df['log_Body_Temp'] = np.log(train_df.Body_Temp)
test_df['log_Body_Temp'] = np.log(test_df.Body_Temp)

train_df['Body_Temp_2'] = train_df.Body_Temp ** 2
test_df['Body_Temp_2'] = test_df.Body_Temp ** 2

train_df['log_Duration'] = np.log(train_df.Duration)
test_df['log_Duration'] = np.log(test_df.Duration)


# 3. Ratios
train_df['Weight_by_Height'] = train_df.Weight/train_df.Height
test_df['Weight_by_Height'] = test_df.Weight/test_df.Height

train_df['Temp_by_Duration'] = train_df.Body_Temp/train_df.Duration
test_df['Temp_by_Duration'] = test_df.Body_Temp/test_df.Duration

train_df['Heart_by_Duration'] = train_df.Heart_Rate/train_df.Duration
test_df['Heart_by_Duration'] = test_df.Heart_Rate/test_df.Duration


# 4. PCA
from sklearn.decomposition import PCA

pca_features = ['Age','Height','Weight','Duration','Heart_Rate', 'Body_Temp']

# - female
X_female = train_df.loc[train_df.female==1, pca_features]
X_female_scaled = (X_female - X_female.mean(axis=0))/X_female.std(axis=0)

X_female_test = test_df.loc[test_df.female==1, pca_features]
test_X_female_scaled = (X_female_test - X_female_test.mean(axis=0))/X_female_test.std(axis=0)

pca_female = PCA()
X_female_pca = pca_female.fit_transform(X_female_scaled)
test_X_female_pca = pca_female.transform(test_X_female_scaled)

# - male
X_male = train_df.loc[train_df.female==0, pca_features]
X_male_scaled = (X_male - X_male.mean(axis=0))/X_male.std(axis=0)

X_male_test = test_df.loc[test_df.female==0, pca_features]
test_X_male_scaled = (X_male_test - X_male_test.mean(axis=0))/X_male_test.std(axis=0)

pca_male = PCA()
X_male_pca = pca_male.fit_transform(X_male_scaled)
test_X_male_pca = pca_male.transform(test_X_male_scaled)


component_names = [f'PC{i+1}' for i in range(X_female.shape[1])]

X_female_pca = pd.DataFrame(
    X_female_pca, 
    columns=component_names,
    index=train_df.index[train_df.female==1]
)
test_X_female_pca = pd.DataFrame(
    test_X_female_pca, 
    columns=component_names,
    index=test_df.index[test_df.female==1]
)

X_male_pca = pd.DataFrame(
    X_male_pca,
    columns=component_names,
    index=train_df.index[train_df.female==0]
)
test_X_male_pca = pd.DataFrame(
    test_X_male_pca, 
    columns=component_names,
    index=test_df.index[test_df.female==0]
)


train_df = train_df.merge(
    pd.concat([X_female_pca,X_male_pca], axis=0),
    on='id',how='left'
)

test_df = test_df.merge(
    pd.concat([test_X_female_pca, test_X_male_pca], axis=0),
    on='id', how='left'
)


# Train Validation Split
from sklearn.model_selection import train_test_split

X = train_df.copy()
y = X.pop('Calories_per_Duration')
y_target = X.pop('Calories')
test_X = test_df.copy()

train_X,valid_X, train_y,valid_y, train_y_target,valid_y_target = train_test_split(X, y, y_target, random_state = 1)


# Calculate Error
from sklearn.metrics import mean_squared_log_error

def calculate_error(
    train_pred, train_y,
    valid_pred, valid_y
):
    mslg_train = np.sqrt(
        mean_squared_log_error(train_pred, train_y)
    )
    mslg_valid = np.sqrt(
        mean_squared_log_error(valid_pred, valid_y)
    )
    
    print(f'train error is {mslg_train}')
    print(f'valid error is {mslg_valid}')


# Random Forest
from sklearn.ensemble import RandomForestRegressor

model_features = [
    'Age','Height','Weight','Duration','Heart_Rate', 'Body_Temp',
    'log_Duration','Temp_by_Duration', 'Heart_by_Duration',
    'log_Heart_Rate','log_Body_Temp',
    'Heart_Rate_2','Body_Temp_2',
    'Weight_by_Height','PC1','PC2','PC3']


# Female Model
model_female = RandomForestRegressor(
    n_estimators = 100,
    min_samples_split =15,
    max_depth = 20
)
model_female.fit(
    train_X.loc[train_X.female==1, model_features], 
    train_y.loc[train_X.female==1])

# Prediction
pred_train_female = model_female.predict(
    train_X.loc[train_X.female==1,model_features]
)

pred_valid_female = model_female.predict(
    valid_X.loc[valid_X.female==1,model_features]
)

# Calculate Error
calculate_error(
    train_pred=pred_train_female, 
    train_y=train_y.loc[train_X.female==1],
    valid_pred=pred_valid_female,
    valid_y=valid_y.loc[valid_X.female==1]  
)


# Male model
model_male = RandomForestRegressor(
    n_estimators = 100,
    min_samples_split = 15,
    max_depth = 25
)
model_male.fit(
    train_X.loc[train_X.female==0, model_features], 
    train_y.loc[train_X.female==0])

# prediction
pred_train_male = model_male.predict(
    train_X.loc[train_X.female==0,model_features]
)
pred_valid_male = model_male.predict(
    valid_X.loc[valid_X.female==0,model_features]
)

# Calculate Error
calculate_error(
    train_pred=pred_train_male, 
    train_y=train_y.loc[train_X.female==0],
    valid_pred=pred_valid_male,
    valid_y=valid_y.loc[valid_X.female==0]  
)


pred_train_sep = pd.DataFrame(
    {'Duration':train_X.Duration},
    index=train_X.index).merge(
    pd.concat([
        pd.DataFrame(
            {'pred':pred_train_male},
            index=train_X.index[train_X.female==0]),
        pd.DataFrame(
            {'pred':pred_train_female},
            index=train_X.index[train_X.female==1]),
    ]),
    on='id',how='left'
)
pred_train_sep['Calories'] = pred_train_sep.Duration*pred_train_sep.pred
pred_train_sep.loc[pred_train_sep.Calories<=0,'Calories']=0

pred_valid_sep = pd.DataFrame(
    {'Duration':valid_X.Duration},
    index=valid_X.index).merge(
    pd.concat([
        pd.DataFrame(
            {'pred':pred_valid_male},
            index=valid_X.index[valid_X.female==0]),
        pd.DataFrame(
            {'pred':pred_valid_female},
            index=valid_X.index[valid_X.female==1]),
    ]),
    on='id',how='left'
)
pred_valid_sep['Calories'] = pred_valid_sep.Duration*pred_valid_sep.pred
pred_valid_sep.loc[pred_valid_sep.Calories<=0,'Calories']=0

calculate_error(
    train_pred=pred_train_sep.Calories, 
    train_y=train_y_target,
    valid_pred=pred_valid_sep.Calories,
    valid_y=valid_y_target   
)


# Submission
pred_test_female = model_female.predict(
    test_X.loc[test_X.female==1,model_features]
)

pred_test_male = model_male.predict(
    test_X.loc[test_X.female==0,model_features]
)

pred_test_sep = pd.DataFrame(
    {'Duration':test_X.Duration},
    index=test_X.index).merge(
    pd.concat([
        pd.DataFrame(
            {'pred':pred_test_male},
            index=test_X.index[test_X.female==0]),
        pd.DataFrame(
            {'pred':pred_test_female},
            index=test_X.index[test_X.female==1]),
    ]),
    on='id',how='left'
)
pred_test_sep['Calories'] = pred_test_sep.Duration*pred_test_sep.pred
pred_test_sep.loc[pred_test_sep.Calories<=0,'Calories']=0

submission = pd.DataFrame({'id':test_df.index,
                           'Calories':pred_test_sep.Calories})

submission.to_csv('submission.csv', index=False)

