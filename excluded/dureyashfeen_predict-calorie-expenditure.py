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


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train.head()


# Calories distribution
fig = px.histogram(train, x='Calories', nbins=50, title="Calories Distribution", marginal='box')
fig.show()

# Sex distribution
fig = px.histogram(train, x='Sex', color='Sex', title='Sex Distribution')
fig.show()

# Correlation heatmap (after encoding Sex temporarily)
corr_df = train.copy()
corr_df['Sex'] = corr_df['Sex'].map({'female': 0, 'male': 1})
corr = corr_df.drop(['id'], axis=1).corr()

fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu', zmin=-1, zmax=1
))
fig.update_layout(title='Correlation Heatmap', height=600)
fig.show()



# One-hot encode Sex
train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)

# Features & target
X = train.drop(['id', 'Calories'], axis=1)
y = train['Calories']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
results_df


fig = px.bar(results_df, x='Model', y=['MAE', 'RMSE', 'R2'], barmode='group', title='Model Performance')
fig.show()


# Train best model (Random Forest as example)
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_scaled, y)
final_preds = final_model.predict(X_test_scaled)

# Prepare submission
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': final_preds
})
submission.to_csv('submission.csv', index=False)
submission.head()


importances = final_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

fig = px.bar(x=feat_imp.values, y=feat_imp.index, orientation='h', title='Feature Importance')
fig.update_layout(xaxis_title='Importance', yaxis_title='Feature')
fig.show()

