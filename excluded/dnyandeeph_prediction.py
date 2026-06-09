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





import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
import time
import datetime as dt
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_df


test_df= pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_df


def custom_palette(custom_colors):
    customPalette=sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors),size=0.8)
    plt.tick_params(axis='both',labelsize=0,length=0)
    return


#defining colour palette
red = ["#4f000b","#720026","#ce4257","#ff7f51","#ff9b54"]
bo = ["#6930c3","#5e60ce","#0096c7","#48cae4","#ade8f4","#ff7f51","#ff9b54","#ffbf69"]
pink = ["#aa4465","#dd2d4a","#f26a8d","#f49cbb","#ffcbf2","#e2afff","#ff86c8","#ffa3a5","#ffbf81","#e9b827","#f9e576"]
custom_palette(pink)


custom_palette(red)


# set context and style plots
sns.set_context("poster",font_scale=0.6, rc={"grid.linewidth": 0.4})

# set Font Family
sns.set_style({'font.family':'serif'})


# Training Data 
train_df.shape


# Test Data

test_df.shape


train_df.columns


train_df.head()


train_df.info()


train_df.describe().T


(train_df.isnull().sum()/train_df.shape[0])*100


plt.figure(figsize=(12,8))
ax= ((train_df.isnull().sum()/train_df.shape[0])*100).plot(kind='barh',color=pink[0])
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
plt.xlabel('Missing Values %')
plt.ylabel('Columns')
plt.title('Percentage of missing values in train data')
sns.set_style('darkgrid')
plt.show()


train_df['Brand'].value_counts()


colors = ['#4E5166', '#7C90A0', '#B5AA9D', '#B9B7A7','#747274']
# colors = ['#542344','#BFD1E5', '#EBF5EE', '#D8BFAA', '#808080']
# colors = ['#541388','#D90368','#F1E9DA','#2E294E','#FFD400']
labels = train_df['Brand'].value_counts().index
values = train_df['Brand'].value_counts()

# create pie chart 
pie_chart =  go.Figure(go.Pie(labels=labels,values=values))
pie_chart.update_traces(hoverinfo='label+value',textinfo='percent', textfont_size=20,
                        marker=dict(colors=colors,line=dict(color='#000000', width=2)))

# creat bar chart
bar_chart=go.Figure(go.Bar(x=labels, y=values,marker_color=colors))

# create subplots
fig=sp.make_subplots(rows=1,cols=2,column_width=[0.5,0.5], specs = [[{'type':'bar'},{'type':'pie'}]],
                    subplot_titles=('Bar Chart','Pie Chart'))

# Add charts to subplots
fig.add_trace(bar_chart.data[0], row=1, col=1)
fig.add_trace(pie_chart.data[0], row=1, col=2)

# update layout
fig.update_layout(showlegend=False,title_text='Distribution of Brand',
                  xaxis=dict(title='Brand', titlefont_size=16, tickfont_size=14),
                  yaxis=dict(title='Count', titlefont_size=16, tickfont_size=14),
                 )
# show subplots
fig.show()


colors = ['#F8C6C3', '#F2B79F', '#E5B769', '#D8CC34']

labels = train_df['Material'].value_counts().index
values = train_df['Material'].value_counts()

# create pie chart 
pie_chart =  go.Figure(go.Pie(labels=labels,values=values))
pie_chart.update_traces(hoverinfo='label+value',textinfo='percent', textfont_size=20,
                        marker=dict(colors=colors,line=dict(color='#000000', width=2)))

# creat bar chart
bar_chart=go.Figure(go.Bar(x=labels, y=values,marker_color=colors))

# create subplots
fig=sp.make_subplots(rows=1,cols=2,column_width=[0.5,0.5], specs = [[{'type':'bar'},{'type':'pie'}]],
                    subplot_titles=('Bar Chart','Pie Chart'))

# Add charts to subplots
fig.add_trace(bar_chart.data[0], row=1, col=1)
fig.add_trace(pie_chart.data[0], row=1, col=2)

# update layout
fig.update_layout(showlegend=False,title_text='Distribution of Material',
                  xaxis=dict(title='Material', titlefont_size=16, tickfont_size=14),
                  yaxis=dict(title='Count', titlefont_size=16, tickfont_size=14),
                 )
# show subplots
fig.show()



custom_palette(bo)


colors = ['#F9DBBD','#FCA17D','#DA627D']
labels = train_df['Size'].value_counts().index
values = train_df['Size'].value_counts()

# Create Pie Chart
pie_chart = go.Figure(go.Pie(labels=labels, values = values))
pie_chart.update_traces(hoverinfo = 'label+value', textinfo = 'percent', textfont_size = 20,
                       marker=dict(colors=colors, line = dict(color='#000000', width=2)))

# Create Bar Chart
bar_chart = go.Figure(go.Bar(x = labels,y = values,marker_color=colors))

# Create Subplots

fig = sp.make_subplots(rows = 1,cols = 2, column_width = [0.5,0.5], specs = [[{'type':'bar'},{'type':'pie'}]],
                      subplot_titles = ('Bar Chart', 'Pie Chart'))

# Add charts to the subplot

fig.add_trace(bar_chart.data[0],row = 1,col = 1)
fig.add_trace(pie_chart.data[0],row = 1,col = 2)

# Update Layout

fig.update_layout(showlegend = False, title_text = 'Distribution of Size',
                 xaxis = dict(title = 'Size', titlefont_size = 16, tickfont_size = 14),
                 yaxis = dict(title = 'Size', titlefont_size = 16, tickfont_size = 14),
                 )

# show subplots
fig.show()


sns.set_style('darkgrid')
plt.figure(figsize=(16,8))
plt.title('Distribution of compartments', size=20, color=red[0])
sns.countplot(train_df,x='Compartments', palette=pink[1:])
plt.ylabel('Compartments')
plt.show()


colors = ['#016FB9','#FF9505',]

labels = train_df['Laptop Compartment'].value_counts().index
values = train_df['Laptop Compartment'].value_counts()

# create pie chart 
pie_chart =  go.Figure(go.Pie(labels=labels,values=values))
pie_chart.update_traces(hoverinfo='label+value',textinfo='percent', textfont_size=20,
                        marker=dict(colors=colors,line=dict(color='#ffffff', width=2)))

# creat bar chart
bar_chart=go.Figure(go.Bar(x=labels, y=values,marker_color=colors))

# create subplots
fig=sp.make_subplots(rows=1,cols=2,column_width=[0.5,0.5], specs = [[{'type':'bar'},{'type':'pie'}]],
                    subplot_titles=('Bar Chart','Pie Chart'))

# Add charts to subplots
fig.add_trace(bar_chart.data[0], row=1, col=1)
fig.add_trace(pie_chart.data[0], row=1, col=2)

# update layout
fig.update_layout(showlegend=False,title_text='Distribution of Laptop',
                  xaxis=dict(title='Laptop Compartment', titlefont_size=16, tickfont_size=14),
                  yaxis=dict(title='Count', titlefont_size=16, tickfont_size=14),
                 )
# show subplots
fig.show()


colors = ['#313B72','#62A87C']

labels = train_df['Waterproof'].value_counts().index
values = train_df['Waterproof'].value_counts()

# create pie chart 
pie_chart =  go.Figure(go.Pie(labels=labels,values=values))
pie_chart.update_traces(hoverinfo='label+value',textinfo='percent', textfont_size=20,
                        marker=dict(colors=colors,line=dict(color='#ffffff', width=2)))

# creat bar chart
bar_chart=go.Figure(go.Bar(x=labels, y=values,marker_color=colors))

# create subplots
fig=sp.make_subplots(rows=1,cols=2,column_width=[0.5,0.5], specs = [[{'type':'bar'},{'type':'pie'}]],
                    subplot_titles=('Bar Chart','Pie Chart'))

# Add charts to subplots
fig.add_trace(bar_chart.data[0], row=1, col=1)
fig.add_trace(pie_chart.data[0], row=1, col=2)

# update layout
fig.update_layout(showlegend=False,title_text='Does Bags are Waterproof',
                  xaxis=dict(title='Waterproof', titlefont_size=16, tickfont_size=14),
                  yaxis=dict(title='Count', titlefont_size=16, tickfont_size=14),
                 )
# show subplots
fig.show()


fig,ax = plt.subplots(figsize=(16,8))
fig.suptitle('Distribution of Style',size=20 , font = 'Serif')
labels = train_df['Style'].value_counts().index
sizes = train_df['Style'].value_counts()
explode = (0.05,0.05,0.05)
ax.pie(sizes , explode = explode,startangle = 60 , labels=labels , autopct = '%1.0f%%' , pctdistance = 0.7 , 
      colors = ["#ff228a","#20b1fd","#ffb703"])
ax.add_artist(plt.Circle((0,0),0.4,fc='white'))
plt.show()


train_df.columns


plt.figure(figsize=(16,8))

# Hist plot

plt.subplot(1,2,1)
sns.histplot(data = train_df, x='Weight Capacity (kg)',kde=True, color='#DB3069')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Count')

# Box plot

plt.subplot(1,2,2)
sns.boxplot(data = train_df , x = 'Weight Capacity (kg)', color = '#DB3069')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Count')

plt.suptitle('Distribution of Weight')
plt.show()


plt.figure(figsize=(16,8))

# Hist plot

plt.subplot(1,2,1)
sns.histplot(data = train_df, x='Price',kde=True, color='mediumseagreen')
plt.xlabel('Price')
plt.ylabel('Count')

# Box plot

plt.subplot(1,2,2)
sns.boxplot(data = train_df , x = 'Price', color = 'mediumseagreen')
plt.xlabel('Price')
plt.ylabel('Count')

plt.suptitle('Distribution of Target Variable')
plt.show()


plt.figure(figsize=(12,8))
ax= ((train_df.isnull().sum()/train_df.shape[0])*100).plot(kind='barh',color=red[2])
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
plt.xlabel('Missing Values %')
plt.ylabel('Columns')
plt.title('Percentage of missing values in train data')
sns.set_style('darkgrid')
plt.show()


train_df.columns


cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
num_cols = ["Weight Capacity (kg)"]


# Numeric Columns
train_df["Weight Capacity (kg)"].fillna(train_df["Weight Capacity (kg)"].median(), inplace=True)
test_df["Weight Capacity (kg)"].fillna(test_df["Weight Capacity (kg)"].median(), inplace=True)


train_df['Style'].mode()[0]


# Categorical columns
for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)


train_df.isnull().sum()


train_df


def feature_engineering(df):
    # Brand & Material Interaction - Some brands may use only specific materials
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']

    # Brand & Size Interaction - Some brands may use only specific sizes
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']

    # Encode Laptop compartment to 1/0
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes':1 , 'No':0})

    # Encode Waterproof to 1/0
    df['Waterproof'] = df['Waterproof'].map({'Yes':1, 'No': 0})

    # Compartment Bining
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins = [0,2,5,10, np.inf], labels = ['Few', 'Moderate', 'Many', 'Very Many'])

    # Weight Capacity Ratio - Normalize Weight Capacity using the max value
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()

    # Interaction Feature: Weight vs. Compartments - Some bags may hold more with less compartments
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments']+1)

    #  Style and Size Interaction - Certain styles may correlate with sizes
    df['Style_Size'] = df['Style'] + '_' + df['Size']

    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


train_df


X = train_df.drop(columns=['id','Price'])
y = train_df['Price']


X


y


test = test_df.drop(columns=['id'])
test_id = test_df['id']


test


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np


import pandas as pd
import numpy as np

# Load the datasets (already done in your notebook)
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Reapply your feature engineering function (copying from your earlier code)
def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

# Apply feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Define X and y
X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']

# Define test and test_id
test = test_df.drop(columns=['id'])
test_id = test_df['id']

# Verify data
print("X shape:", X.shape)
print("y shape:", y.shape)
print("test shape:", test.shape)
print("test_id shape:", test_id.shape)


# Identify categorical columns in X and test
cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']

# Apply one-hot encoding to both train and test sets
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)

# Align test_encoded with X_encoded to ensure same columns
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Verify shapes
print("X_encoded shape:", X_encoded.shape)
print("test_encoded shape:", test_encoded.shape)


from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.impute import SimpleImputer

# Create an imputer to fill NaNs with the median
imputer = SimpleImputer(strategy='median')

# Impute NaNs in X_encoded and test_encoded
X_encoded_imputed = imputer.fit_transform(X_encoded)
test_encoded_imputed = imputer.transform(test_encoded)

# Initialize Linear Regression model
lr_model = LinearRegression(n_jobs=-1)

# Train the model
print("Training Linear Regression...")
lr_model.fit(X_encoded_imputed, y)
print("Training complete!")

# Predict on test set
test_predictions = lr_model.predict(test_encoded_imputed)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)  # Ensure no negative prices

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_id,
    'Price': test_predictions
})

# Save to CSV
submission_df.to_csv('/kaggle/working/submission_lr.csv', index=False)
print("Submission saved as '/kaggle/working/submission_lr.csv'")
print(submission_df.head())


from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

# Initialize HistGradientBoostingRegressor
hgb_model = HistGradientBoostingRegressor(random_state=42)

# Train the model (no need to impute NaNs)
print("Training HistGradientBoostingRegressor...")
hgb_model.fit(X_encoded, y)
print("Training complete!")

# Predict on test set
test_predictions = hgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)  # Ensure no negative prices

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_id,
    'Price': test_predictions
})

# Save to CSV
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission saved as '/kaggle/working/submission.csv'")
print(submission_df.head())


from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# Assuming X_encoded, y, test_encoded, and test_id are still in memory
# If not, redefine them from your last working state:
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
test = test_df.drop(columns=['id'])
test_id = test_df['id']

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Initialize and train HistGradientBoostingRegressor
hgb_model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
print("Training HistGradientBoostingRegressor...")
hgb_model.fit(X_encoded, y)
print("Training complete!")

# Predict and ensure non-negative prices
test_predictions = hgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

# Create submission
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission_hgb.csv', index=False)
print("Submission saved as '/kaggle/working/submission_hgb.csv'")
print(submission_df.head())


hgb_model = HistGradientBoostingRegressor(
    max_iter=200,        # More iterations
    max_depth=8,         # Deeper trees
    learning_rate=0.05,  # Slower learning for stability
    random_state=42
)
hgb_model.fit(X_encoded, y)
test_predictions = hgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission_hgb_tuned.csv', index=False)
print("Tuned submission saved!")


# Redo Linear Regression with imputed data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_encoded_imputed = imputer.fit_transform(X_encoded)
test_encoded_imputed = imputer.transform(test_encoded)

lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_encoded_imputed, y)
lr_predictions = lr_model.predict(test_encoded_imputed)

# Blend predictions (50% Linear, 50% HGB)
blend_predictions = 0.5 * lr_predictions + 0.5 * test_predictions
blend_predictions = np.clip(blend_predictions, a_min=0, a_max=None)

submission_df = pd.DataFrame({'id': test_id, 'Price': blend_predictions})
submission_df.to_csv('/kaggle/working/submission_blend.csv', index=False)
print("Blended submission saved!")


from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# Re-run preprocessing if needed
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
test = test_df.drop(columns=['id'])
test_id = test_df['id']

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Train and predict
hgb_model = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
hgb_model.fit(X_encoded, y)
test_predictions = hgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

# Save submission
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission saved! Submit this now!")


import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Re-run preprocessing to ensure everythingâ€™s fresh
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
test = test_df.drop(columns=['id'])
test_id = test_df['id']

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Initialize XGBoost with tuned parameters
xgb_model = XGBRegressor(
    n_estimators=200,      # More trees for better fit
    max_depth=6,           # Depth for complexity
    learning_rate=0.05,    # Slower learning for generalization
    random_state=42,
    n_jobs=-1              # Use all cores
)

# Train the model
print("Training XGBoost...")
xgb_model.fit(X_encoded, y)
print("Training complete!")

# Predict and clip
test_predictions = xgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

# Save submission
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission_xgb.csv', index=False)
print("Submission saved as '/kaggle/working/submission_xgb.csv'")
print(submission_df.head())


# Enhanced feature engineering
def feature_engineering_enhanced(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    # New features
    df['Compartments_per_Weight'] = df['Compartments'] / (df['Weight Capacity (kg)'] + 1)  # Avoid division by zero
    df['Quality_Score'] = df['Waterproof'] + df['Laptop Compartment']  # Simple quality indicator
    return df

# Apply enhanced features
train_df = feature_engineering_enhanced(train_df)
test_df = feature_engineering_enhanced(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = np.log1p(train_df['Price'])  # Log-transform target for better fit
test = test_df.drop(columns=['id'])

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Train XGBoost with log-transformed target
xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_model.fit(X_encoded, y)

# Predict and reverse log-transform
test_predictions = np.expm1(xgb_model.predict(test_encoded))  # Reverse log1p with expm1
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

# Save submission
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission_xgb_enhanced.csv', index=False)
print("Enhanced submission saved!")


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Impute NaNs for Linear Regression
imputer = SimpleImputer(strategy='median')
X_encoded_imputed = imputer.fit_transform(X_encoded)
test_encoded_imputed = imputer.transform(test_encoded)

# Train Linear Regression (no log-transform here for simplicity)
lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_encoded_imputed, train_df['Price'])  # Use original Price
lr_predictions = lr_model.predict(test_encoded_imputed)
lr_predictions = np.clip(lr_predictions, a_min=0, a_max=None)

# Blend with XGBoost predictions (70% XGBoost, 30% Linear)
blend_predictions = 0.7 * test_predictions + 0.3 * lr_predictions
blend_predictions = np.clip(blend_predictions, a_min=0, a_max=None)

# Save blended submission
submission_df = pd.DataFrame({'id': test_id, 'Price': blend_predictions})
submission_df.to_csv('/kaggle/working/submission_blend.csv', index=False)
print("Blended submission saved!")


import pandas as pd
import numpy as np
from xgboost import XGBRegressor

train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
test = test_df.drop(columns=['id'])
test_id = test_df['id']

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_model.fit(X_encoded, y)
test_predictions = xgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission saved! Submit this now!")




