# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.tail()


train.info()


test.info()


train.describe().loc[['mean','min','max']].T


train['accident_risk'].value_counts()


#By Vishnupriya https://www.kaggle.com/code/vishnupriyagarige/predict-the-introverts-from-the-extroverts/notebook

# Categorical columns to plot
cat_cols = ['road_type', 'lighting','weather','road_signs_present','public_road','time_of_day', 'holiday','school_season']

# Set up 4x2 grid for subplots
fig, axes = plt.subplots(4, 2, figsize=(20, 20))#Original 10,6
axes = axes.flatten()  # Flatten to iterate easily

# Generate pie charts
for i, col in enumerate(cat_cols):
    train[col].value_counts().plot.pie(
        ax=axes[i],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        shadow=True
    )
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")  # Remove y-label for cleaner plot

plt.tight_layout()
plt.show()


#By H-Z-Ning  https://www.kaggle.com/code/hzning/top-10-solution-0-97525-esay-is-all-you

#categorical_columns = ["shelter_name", "city", "state","season", "notes"]

plt.figure(figsize=(14, 12))
for i, column in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=column, data=train, palette='Set2')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


#By OutlierPandas https://www.kaggle.com/code/abhyudaya456/s5e6-eda-for-predicting-optimal-fertilizers/notebook
numerical_cols = ['id', 'num_lanes', 'curvature', 'num_reported_accidents', 'speed_limit', 'accident_risk']


train[numerical_cols].hist(figsize=(15,10), bins=30, color='green', edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()


# OutlierPandas https://www.kaggle.com/code/abhyudaya456/s5e6-eda-for-predicting-optimal-fertilizers/notebook 
plt.figure(figsize=(10,6))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='summer')
plt.title("Correlation Between Numerical Features")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='accident_risk', y='lighting', data=train)
plt.xticks(rotation=45)
plt.title("Lighting vs Accident risk")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='accident_risk', y='weather', data=train)
plt.xticks(rotation=45)
plt.title("Weather vs Accident risk")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='accident_risk', y='time_of_day', data=train)
plt.xticks(rotation=45)
plt.title("Time of Day vs Accident risk")
plt.show()


import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


#Code by Kohei-mu https://www.kaggle.com/koheimuramatsu/industrial-accident-causal-analysis/notebook

reported_cnt = np.round(train['num_reported_accidents'].value_counts(normalize=True) * 100)
hv.Bars(reported_cnt).opts(title="Number of reported accidents", color="red", xlabel="num_reported_accidents", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=500, height=300,tools=['hover'],show_grid=True))


#http://holoviews.org/user_guide/Customizing_Plots.html

#Code by Kohei-mu https://www.kaggle.com/koheimuramatsu/industrial-accident-causal-analysis/notebook

light_cnt = np.round(train['lighting'].value_counts(normalize=True) * 100)
hv.Bars(light_cnt).opts(title="Light Conditions-Related Accident Risk", color="purple", xlabel="Light Conditions", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True)).opts(xrotation=45)


#http://holoviews.org/user_guide/Customizing_Plots.html

#Code by Kohei-mu https://www.kaggle.com/koheimuramatsu/industrial-accident-causal-analysis/notebook

curvature_cnt = np.round(train['curvature'].value_counts(normalize=True) * 100)
hv.Bars(curvature_cnt).opts(title="Road Curvature", color="green", xlabel="Percentage", ylabel="Road Curvature", xformatter='%d%%')\
                .opts(opts.Bars(invert_axes=True, width=500, height=300,tools=['hover'],show_grid=True))


#Code by Kohei-mu https://www.kaggle.com/koheimuramatsu/industrial-accident-causal-analysis/notebook

speed_cnt = np.round(train['speed_limit'].value_counts(normalize=True) * 100)
hv.Bars(speed_cnt).opts(title="Speed Limit", color="cyan", xlabel="Road Owner", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True)).opts(xrotation=45)


#Code by Kohei-mu https://www.kaggle.com/koheimuramatsu/industrial-accident-causal-analysis/notebook

accident_cnt = np.round(train['accident_risk'].value_counts(normalize=True) * 100)
hv.Bars(accident_cnt).opts(title="Accident Risk", color="DarkKhaki", xlabel="Accident risk", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=800, height=500,tools=['hover'],show_grid=True)).opts(xrotation=45)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB,BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
%matplotlib inline


from sklearn.preprocessing import LabelEncoder

#fill in mean for floats
for c in train.columns:
    if train[c].dtype=='float16' or  train[c].dtype=='float32' or  train[c].dtype=='float64':
        train[c].fillna(train[c].mean())

#fill in -999 for categoricals
train = train.fillna(-999)
# Label Encoding
for f in train.columns:
    if train[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))
        
print('Labelling done.')


#Linear regression, first create test and train dataset
x=train.loc[:,['num_lanes', 'curvature', 'num_reported_accidents', 'speed_limit']].values
y=train.loc[:,'accident_risk'].values


# Creating a test and training dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


#Code by Arpita Gupta https://www.kaggle.com/arpita28/analysis-of-spotify-trends

# Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)


#Displaying the difference between the actual and the predicted
y_pred = regressor.predict(X_test)
train_output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(train_output)


#Checking the accuracy of Linear Regression

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

