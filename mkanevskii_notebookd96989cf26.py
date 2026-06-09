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


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
pd.set_option('display.max_columns', None)


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from scipy.stats import pointbiserialr
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
test.head(3)
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


train = train.rename(columns={'Temparature':'Temperature'})
train.head(3)


train.columns


train.size


train.shape


train.shape[0]*train.shape[1]


train.isnull().sum()


total_fields = train.size
missing_values_count = train.isnull().sum().sum()

missing_values_percentage = (missing_values_count / total_fields) * 100
print(f"Percentage of missing values: {missing_values_percentage:.2f} %")


train.columns


total_fields_1 = train.size
missing_values_count_1 = train.isnull().sum().sum()

missing_values_percentage_1 = (missing_values_count_1 / total_fields_1) * 100
print(f"Percentage of missing values ​​for required fields: {missing_values_percentage_1:.2f}%")


# Number of records with all required fields filled in
complete_records_count = train.dropna().shape[0]

# Total number of records
total_records_count = train.shape[0]

# Completion rate
completion_rate = (complete_records_count / total_records_count) * 100
print(f"Record count ratio: {completion_rate:.2f}%")


train.dtypes


train['Temperature'].unique()


train['Temperature'].nunique()


train['Temperature'].max()


train['Temperature'].min()


train_info = pd.DataFrame({
    "DataType": train.dtypes,
    "MissingValues": train.isnull().sum(),
    "UniqueValues": train.nunique()
}).sort_values(by="MissingValues", ascending=False)
train_info


train.head(5)


train.columns


numeric_variables_graf = []

for column in train.columns:
    if train[column].dtype in ['float64', 'int64'] and train[column].nunique() > 2:
                numeric_variables_graf.append(column)
numeric_variables_graf


categorical = []
for column in train.columns:
    if train[column].nunique() <= 12 and train[column].nunique() > 2 :
                categorical.append(column)
categorical


train['Soil Type'].value_counts()


train['Soil Type'].nunique()


train['Crop Type'].nunique()


import seaborn as sns


selected_palette = sns.color_palette("RdYlGn", n_colors=20)

sns.palplot(selected_palette)
plt.show()


# Plotting a boxplot for the 'Temperature' column
plt.figure(figsize=(16, 6))
sns.boxplot(x=train['Temperature'], color=selected_palette[17])

plt.title("Boxplot")
plt.xlabel('Temperature')

plt.show()


Me = train['Temperature'].quantile(0.5)
Q1 = train['Temperature'].quantile(0.25)
Q3 = train['Temperature'].quantile(0.75)

IQR = Q3 - Q1

lower_bound_predict = Q1 - 1.5 * IQR
minimum = train['Temperature'].min()
lower_bound = max(minimum, lower_bound_predict)
upper_bound_predict = Q3 + 1.5 * IQR
maximum = train['Temperature'].max()
upper_bound = min(maximum, upper_bound_predict)


print(f"Calculated lower mustache {lower_bound_predict}")
print(f"Min {minimum}")
print(f"Lower mustache {lower_bound}")

print(f"Estimated upper whisker {upper_bound_predict}")
print(f"Upper mustache {upper_bound}")
print(f"Max {maximum}")

print(f"Median {Me}")
print(f"First quartile {Q1}")
print(f"Third quartile {Q3}")
print(f"Interquartile range {IQR}")


# Creating a canvas with subgroups for boxplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  

# We expand the matrix of graphs and sort through the columns
for ax, column in zip(axes.flat, numeric_variables_graf):
    sns.boxplot(x=train[column], ax=ax, color=selected_palette[17])
    ax.set_title(f"Boxplot {column}")

plt.show()


import pandas as pd
df = pd.DataFrame(train)
correlation_matrix = df[['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']].corr()

print(correlation_matrix)


df = pd.DataFrame(train)

# Numeric columns only
num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Correlation matrix
corr = df[num_cols].corr()

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Correlation between features (Pearson)")
plt.tight_layout()
plt.show()


train


train.boxplot(
    column=["Temperature"], by="Fertilizer Name", fontsize=18, figsize=(20, 5)
);


# Average values of elements by soil type:
df.groupby('Soil Type')[['Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']].mean().round(2)


# Number of records by soil type:
df['Soil Type'].value_counts()


soil_crop = df.groupby(['Soil Type', 'Crop Type']).size().unstack().fillna(0)
soil_crop


# New column - "Soil_Crop_Combo" 
train['Soil_Crop_Combo'] = (train['Soil Type'] + '_' + train['Crop Type']).str.lower()
train


df.columns = df.columns.str.strip()


df = pd.DataFrame(train)
df.columns.tolist()


soil_fert = df.groupby(['Soil Type', 'Fertilizer Name']).size().unstack(fill_value=0)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(soil_fert, cmap="YlOrBr", annot=True, fmt="d")
plt.title("Frequency of Fertilizer Use by Soil Type")
plt.tight_layout()
plt.show()


# What kind of soil retains moisture is an important sign for crop yield
df.groupby('Soil Type')['Moisture'].mean().sort_values(ascending=False)


# Let's divide the temperature into 3 fixed groups
train['Temperature_Group'] = pd.cut(train['Temperature'], 
                                 bins=[0, 28, 38, 50], 
                                 labels=['Low', 'Medium', 'High'])
train


# Fertility Index (Simple Sum of NPK)
train['Fertility_Index'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
train


# Nutrient Ratios
train['N_P_ratio'] = df['Nitrogen'] / df['Phosphorous']
train['N_K_ratio'] = df['Nitrogen'] / df['Potassium']
train['P_K_ratio'] = df['Phosphorous'] / df['Potassium']
train


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

Cramer_features = ['Soil Type', 'Crop Type', 'Fertilizer Name']

result = pd.DataFrame(index=Cramer_features, columns=Cramer_features)

for col1 in Cramer_features:
    for col2 in Cramer_features:
        if col1 == col2:
            result.loc[col1, col2] = 1.0  # идеальная зависимость с самой собой
        else:
            result.loc[col1, col2] = cramers_v(df[col1], df[col2])

result = result.astype(float).round(2)

result


from sklearn.preprocessing import LabelEncoder

df.columns = df.columns.str.strip()    

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])  # это y

print(df.dtypes)


from sklearn.model_selection import train_test_split

features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

X = df[features]
y = df[target]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


df.to_csv('output.csv', index=False)

