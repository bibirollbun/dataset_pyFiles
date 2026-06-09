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


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# adjust display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_theme(style="whitegrid")

# load data
train = pd.read_csv("/kaggle/input/allstate-claims-severity/train.csv")     # path to training dataset
test  = pd.read_csv("/kaggle/input/allstate-claims-severity/test.csv")      # if provided


train.head()


train.info()


train.describe()


desc_table = pd.DataFrame({
    'Column Name': train.columns,
    'Data Type': train.dtypes.astype(str),
    'Example Value': [train[col].iloc[0] for col in train.columns]
})
desc_table.head(132)


missing = train.isnull().sum()
missing = missing[missing > 0]
print("Missing Values per Column:")
print(missing)



# Histogram + density plot
plt.figure(figsize=(10,6))
sns.histplot(train['loss'], bins=50, kde=True)
plt.title('Distribution of Loss')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.show()


cont_features = [f'cont{i}' for i in range(1,15)]
train[cont_features].hist(bins=50, figsize=(15,10))
plt.suptitle('Distribution of Continuous Features')
plt.show()


# Example for cat1
plt.figure(figsize=(12,6))
sns.countplot(x='cat1', data=train, order=train['cat1'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Category Counts for cat1')
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(x='cat1', y='loss', data=train)
plt.xticks(rotation=45)
plt.title('Loss by cat1')
plt.show()


plt.figure(figsize=(12,6))
sns.violinplot(x='cat1', y='loss', data=train)
plt.xticks(rotation=45)
plt.title('Loss Distribution by cat1 (Violin Plot)')
plt.show()



corr = train[cont_features + ['loss']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, cbar_kws={'shrink':0.8})
plt.title('Correlation Heatmap of Continuous Features')
plt.show()



corr_target = corr['loss'].sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=corr_target.index, y=corr_target.values)
plt.xticks(rotation=45)
plt.title('Correlation of Features with Loss')
plt.show()




high_loss = train[train['loss'] > train['loss'].quantile(0.99)]
plt.figure(figsize=(12,6))
sns.countplot(x='cat1', data=high_loss, order=high_loss['cat1'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Categories in Top 1% of Losses')
plt.show()



### Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X = train.drop(['id','loss'], axis=1)
y = train['loss']

cat_features = [col for col in X.columns if 'cat' in col]
num_features = [col for col in X.columns if 'cont' in col]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100,  
        max_depth=12,     
        random_state=42,
        n_jobs=-1           
    ))
])

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Random Forest MAE: {mae:.2f}, RMSE: {rmse:.2f}")


### Feature Engineering
features = num_features + list(rf_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())

importances = rf_model.named_steps['model'].feature_importances_
top15_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=top15_df.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

