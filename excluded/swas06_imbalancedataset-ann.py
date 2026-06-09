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


df_train= pd.read_csv("/kaggle/input/churn-challenge-ai/train.csv")
df_test= pd.read_csv("/kaggle/input/churn-challenge-ai/test.csv")


df_train.shape,df_test.shape


df_train.info()


df_train.Exited.value_counts()


df_train.head(3)


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.head(3)


df_train=df_train.drop(['id','customerid','surname'],axis=1)


df_test=df_test.drop(['id','customerid','surname'],axis=1)


numerical_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)
print(f"\nTotal number of numerical columns: {len(numerical_columns)}")

print("\nCategorical Columns:")
print(categorical_columns)
print(f"\nTotal number of categorical columns: {len(categorical_columns)}")


df_train[numerical_columns].describe()


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))  
fig.suptitle('Boxplots of Selected Features', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(numerical_columns):
    sns.boxplot(y=df_train[col], ax=axes[i])
    axes[i].set_title(col)
fig.delaxes(axes[-1])

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


import warnings
warnings.filterwarnings("ignore")
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))  
fig.suptitle('Boxplots of Selected Features', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(numerical_columns):
    sns.histplot(x=df_train[col],kde=True, ax=axes[i])
    axes[i].set_title(col)
fig.delaxes(axes[-1])

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


sns.pairplot(df_train)
plt.show()


numeric_columns = df_train.select_dtypes(include=['number'])
corr_matrix = numeric_columns.corr()

plt.figure(figsize=(10, 8))  # Adjust the size as necessary
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


for col in categorical_columns:
    print(col, "-->", df_train[col].unique())


import math
n_features = len(categorical_columns)
ncols = 2  # You can adjust the number of columns here
nrows = math.ceil(n_features / ncols)  # Calculate rows needed to fit all features

# Plot the countplot for each categorical feature
plt.figure(figsize=(10, nrows * 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(nrows, ncols, i)  # Create subplots with dynamic grid size
    sns.countplot(x=df_train[col], palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()

for column in df_train.columns:
    if df_train[column].dtype == "object":
        df_train[column] = le.fit_transform(df_train[column])
        
df_train.head(3)  


df_train.exited.value_counts()


from sklearn.model_selection import train_test_split, cross_val_score
X = df_train.drop("exited", axis=1).values
y = df_train["exited"].values



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state=9)


import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

weights_assigned={0:1,1:4}

model = Sequential()
# define first hidden layer and visible layer
model.add(Dense(50, input_dim=10, activation='relu', kernel_initializer='he_uniform'))
# define output layer
model.add(Dense(1, activation='sigmoid'))
# define loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train,y_train,class_weight=weights_assigned,epochs=100)


y_pred=model.predict(X_test)


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)

