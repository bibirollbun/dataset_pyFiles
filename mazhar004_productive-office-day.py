# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import plotly.io as pio
pio.renderers.default = "iframe"

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_filename = '/kaggle/input/data-guild-will-you-be-productive-today/train.csv'
df = pd.read_csv(train_filename)
df.head()


df.describe()


## Data Exploration
# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print('Missing values:')
print(missing_values)
print(5 * '-' + '\n')


# Check for duplicates
duplicates = df.duplicated().sum()
print('Number of duplicate samples:', duplicates)
print(5 * '-' + '\n')

# Check for unique values
unique_values = df.nunique()
print('Unique values:')
print(unique_values)
print(5 * '-' + '\n')

# Check for data types
data_types = df.dtypes
print('Data types:')
print(data_types)
print(5 * '-' + '\n')

# Check for distribution of target variable
target_distribution = df['productive_day'].value_counts()
print('Target distribution:')
print(target_distribution)
print(5 * '-' + '\n')



fig = sp.make_subplots(rows=3, cols=3, subplot_titles=df.columns)

for index, column in enumerate(df.columns):
    row , col = divmod(index,3)
    row += 1
    col += 1
    hist = go.Histogram(x=df[column], nbinsx=30, marker=dict(line=dict(color='white', width=1)))
    fig.add_trace(hist, row=row, col=col)

fig.update_layout(height=900, width=1200, title_text="Distribution of Features", showlegend=False, bargap=0.1)

pio.show(fig) 


frequency = df['productive_day'].value_counts().reset_index()
frequency.columns = ['productive_day', 'count']

fig = px.pie(frequency, names='productive_day', values='count')

fig.update_layout(title='Distribution of Productive Days with Frequencies')

pio.show(fig) 


fig = go.Figure()
for column in df.columns:
    fig.add_trace(go.Box(y=df[column], name=column))

fig.update_layout(
    title="Boxplot to Detect Outliers",
    yaxis_title="Value",
    xaxis_title="Feature",
    width=1300,
    height=500
)

pio.show(fig) 


# df[col] = df[col].fillna(median_value)


df.drop_duplicates(inplace=True)


def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs(stats.zscore(df))
    filtered_entries = z_scores < threshold
    return filtered_entries

mask = remove_outliers_zscore(df['lines_of_code_written'])
df_cleaned = df[mask]

# fig = go.Figure()
# for column in ['lines_of_code_written']:
#     fig.add_trace(go.Box(y=df_cleaned[column], name=column))

# fig.update_layout(
#     title="Boxplot to Detect Outliers",
#     yaxis_title="Value",
#     xaxis_title="Feature",
#     width=1300,
#     height=500
# )

# pio.show(fig) 


correlation_matrix = df_cleaned.corr()

fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
pio.show(fig) 


target_correlation = correlation_matrix['productive_day'].abs().sort_values(ascending=False)
print(10*'-')
print("Correlation with target variable")
print(target_correlation)


fig = px.bar(target_correlation.drop('productive_day'), title='Importance order of features')
fig.update_layout(showlegend=False)
pio.show(fig)  


target = 'productive_day'
X = df_cleaned.drop(columns=[target])
y = df_cleaned[target]

preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_transformed = preprocessing_pipeline.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

print("Main Data     :", df_cleaned.shape)
print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)
print("X_test shape. :", X_test.shape)
print("y_test shape  :", y_test.shape)


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier(),
    'XGBoost Classifier': XGBClassifier(),
    'Support Vector Classifier': SVC(),
    'MLP Classifier': MLPClassifier(max_iter=500, hidden_layer_sizes=(100,), solver='adam', learning_rate_init=0.001, random_state=42)
}

model_accuracies = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy


accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

fig = px.bar(accuracy_df, x='Model', y='Accuracy', color='Model', title='Model Accuracy Comparison')
fig.update_layout(xaxis_title='Model', yaxis_title='Accuracy')
pio.show(fig)


classifier = models['Logistic Regression']

test_df = pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/test.csv")

X_new_transformed = preprocessing_pipeline.transform(test_df)

y_pred_new = classifier.predict(X_new_transformed)


y_pred_new

