# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#import models
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


road_accident = df.groupby('road_type')['num_reported_accidents'].mean().reset_index()

colors = sb.color_palette('dark')
road_accident.plot.pie(
    y='num_reported_accidents',
    labels=road_accident['road_type'],
    autopct='%1.1f%%',
    legend = False,
    colors = colors,
    ylabel='',  # removes default y-axis label
    title='Number of Reported Accidents by Road Type',
    #figsize=(9, 11)
);


df.groupby('road_signs_present')['num_reported_accidents'].mean().reset_index()


weather_accident = df.groupby('weather')['accident_risk'].mean().reset_index()

sb.barplot(weather_accident,x='weather',y='accident_risk');


def feature_engineer(df):
    df_copy = df.copy()
    
    df_copy['road_signs_present'] = (df_copy['road_signs_present'] == True).astype(int)
    df_copy['public_road'] = (df_copy['public_road'] == True).astype(int)
    df_copy['holiday'] = (df_copy['holiday'] == True).astype(int)
    df_copy['school_season'] = (df_copy['school_season'] == True).astype(int)

    #create new features
    df_copy['road_complexity'] = df_copy['curvature'] * df_copy['num_lanes']#curvature × num_lanes
    df_copy['lane_density'] = df_copy['speed_limit'] / df_copy['num_lanes']
    
    def label(x):
        if x == 'morning':
            return 1
        if x == 'afternoon':
            return 2
        if x == 'evening':
            return 3

    df_copy['time_of_day'] = df_copy['time_of_day'].map(label)

    # Create encoder with unknown handling
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit and transform
    result = encoder.fit_transform(df_copy[['road_type','lighting','weather']])

    # Convert to DataFrame with readable column names
    encoded_df = pd.DataFrame(result, columns=encoder.get_feature_names_out(
    ['road_type','lighting','weather']), index=df_copy.index) 

    df_copy= pd.concat([df_copy,encoded_df], axis=1)

    df_copy = df_copy.drop(['road_type','lighting','weather'],axis=1)

    return df_copy


train = feature_engineer(df)


first_ = train.iloc[:,:11].columns.tolist()

plt.figure(figsize=(10,6))
sb.heatmap(train[first_].corr(numeric_only = True),annot=True);


last_= train.iloc[:,11:].columns.tolist()

plt.figure(figsize=(10,6))
sb.heatmap(train[last_].corr(numeric_only = True),annot=True);


X = train.drop('accident_risk',axis=1)
y = train['accident_risk']


# performing preprocessing part

sc = StandardScaler()

X_scaled = sc.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size= 0.2, random_state=42)


#model

model = LGBMRegressor(
       n_estimators = 100,
        n_leaves = 31,
        reg_alpha = 0.0,
        reg_lambda = 0.0,
        max_depth = -1,
        learning_rate = 0.1, 
)
model.fit(X_train,y_train)

y_pred = model.predict(X_val)

np.sqrt(mean_squared_error(y_val,y_pred))/y.mean()


#lg.get_params()  0.1599169463631135


def custom_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.mean(y_true)

scorer = make_scorer(custom_rmse, greater_is_better=False)

# Run cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring=scorer)

# Print results
print("RMSE Scores for each fold:", scores)
print("Average RMSE Score:", scores.mean())


#match importance to feature name

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

#plot
feature_importances.plot.barh(x='Feature',y='Importance',figsize=(10,10));


#we can also use permutation importance method

from sklearn.inspection import permutation_importance


import shap


# Create SHAP explainer for LGBM
explainer = shap.Explainer(model, X_val)

# Calculate SHAP values
shap_values = explainer(X_val)

# Global summary plot: feature importance + direction
shap.summary_plot(shap_values, X_val)

# Optional: single prediction explanation
shap.plots.waterfall(shap_values[0])  # shows first sample











test = feature_engineer(test_df)


predictions = lg.predict(test)


submission = test[['id']]


submission['accident_risk'] = predictions


submission['accident_risk'] = submission['accident_risk'].apply(lambda x:(round(x,2)))


submission = submission.reset_index(drop=True)


submission.to_csv('submission.csv', index=False)





































