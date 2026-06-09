import pandas as pd


import numpy as np


train_df=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


print(train_df.shape)
train_df.head()


print(test_df.shape)
test_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.dtypes


train_df.describe()


print(train_df.duplicated().sum())


obj_column=train_df.select_dtypes(include=['object']).apply(pd.Series.nunique)
print(obj_column)


num_column=train_df.select_dtypes(include=['number']).apply(pd.Series.nunique)
print(num_column)


import matplotlib.pyplot as plt
import seaborn as sns


numerical_cols_to_plot = num_column[num_column.index != 'id'].index
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols_to_plot):
    plt.subplot(3, 2, i + 1)
    sns.histplot(x=col, data=train_df, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


for col in obj_column.index:
  print(col,train_df[col].unique())


import matplotlib.pyplot as plt
import seaborn as sns


categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 2, i + 1)
    sns.countplot(x=col, data=train_df, hue=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()


obj_column=train_df.select_dtypes(include=['object']).apply(pd.Series.nunique)
print(obj_column)


categorical_cols = train_df.select_dtypes(include=['object', 'bool']).columns
print("Categorical and Boolean Columns:")
print(categorical_cols)


fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    if i < len(axes):
        train_df[col].value_counts().plot.pie(
            ax=axes[i],
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            legend=False
        )
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_ylabel('')
    else:
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler


for col in train_df.columns:
  if train_df[col].dtype=='object':
    le=LabelEncoder()
    train_df[col]=le.fit_transform(train_df[col])
  if train_df[col].dtype=='bool':
    train_df[col]=train_df[col].astype(int)


train_df.head()


X=train_df.loc[:,['num_lanes', 'curvature', 'num_reported_accidents', 'speed_limit']].values
y=train_df.loc[:,'accident_risk'].values


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
output=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(output)


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

