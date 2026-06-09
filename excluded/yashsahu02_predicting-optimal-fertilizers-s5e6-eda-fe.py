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
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.head() ## top 5 rows in train


test.head() ## top 5 rows of test data


print(f"Shape of train data: {train.shape}")
print(f"Shape of test data: {test.shape}")


sample_submission


print("Information of train data")
train.info()


train.describe()


## checking is there any missing values
train.isnull().sum()


## is there any duplicate values
train.duplicated().sum()


numerical_features = [feature for feature in train.columns if train[feature].dtype!='O']
categorical_features = [feature for feature in train.columns if train[feature].dtype=='O']





print("List of Numerical Features: ",numerical_features)
print()
print("List of Categorical Features",categorical_features)


train[numerical_features].corr()


def plot_graphs(df,feature):
    if df[feature].dtype=='O':
        print(f"--------- {feature} is a CATEGORICAL feature --------")
        print(f"Total Missing Values: {df[feature].isnull().sum()}")
        print(f"Total Unique Categories: {df[feature].nunique()}\n")
        print("Unique Values:\n", df[feature].unique())
        print()

        ## bar plot for top 10 categoris in categorical feature
        plt.figure(figsize=(15,6))

        plt.subplot(1,2,1)
        plt.title("Bar Plot for {}".format(feature))
        plt.ylabel("Count")
        df[feature].value_counts().plot(kind='bar')

        plt.subplot(1,2,2)
        plt.title("Pie Chart for {}".format(feature))
        df[feature].value_counts().plot(kind='pie', autopct='%.2f%%')
        plt.show()


    elif df[feature].dtype!='O':
        print(f"------- {feature} is a NUMERICAL feature ------")
        print(f"Total Missing Values: {df[feature].isnull().sum()}")
        print(f"Summary Statistics:\n{df[feature].describe()}\n")
        df[feature].describe()
        
        plt.figure(figsize=(18,25))

        plt.subplot(3,2,1)
        plt.title("Histogram for '{}'".format(feature))
        df[feature].plot(kind='hist')

        plt.subplot(3,2,2)
        plt.title("KDE plot for '{}'".format(feature))
        df[feature].plot(kind='kde')

        plt.subplot(3,2,3)
        plt.title("Box Plot for '{}'".format(feature))
        df[feature].plot(kind='box')

        plt.subplot(3,2,4)
        plt.title("Distplot for '{}'".format(feature))
        sns.distplot(df[feature])

        plt.subplot(3,2,5)
        plt.title("Lineplot for '{}'".format(feature))
        df[feature].value_counts().sort_index().plot.line()

        plt.show()


    else:
        print("Datatype of feature is neither numerical nor categorical...")

    print()


numerical_features


for i in numerical_features:
    plot_graphs(train,i)


for i in categorical_features:
    plot_graphs(train,i)


numerical_features.remove('id') ## removing id feature from numerical_features list 


train.head()


sns.barplot(x=train['Fertilizer Name'],y=train['Temparature'])
plt.title("Relationship between Fertilizer Name and Temparature")
plt.show()


sns.barplot(x=train['Fertilizer Name'],y=train['Humidity'])
plt.title("Relationship between Fertilizer Name and Humidity")
plt.show()


sns.barplot(x=train['Soil Type'],y=train['Temparature'])
plt.title("Relationship between Soil Type and Temparature")
plt.show()


sns.barplot(x=train['Soil Type'],y=train['Humidity'])
plt.title("Relationship between Soil Type and Humidity")
plt.show()


sns.barplot(x=train['Soil Type'],y=train['Moisture'])
plt.title("Relationship between Soil Type and Moisture")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=train['Soil Type'],y=train['Potassium'])
plt.title("Relationship between Soil Type and Potassium")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=train['Soil Type'],y=train['Nitrogen'])
plt.title("Relationship between Soil Type and Nitrogen")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=train['Soil Type'],y=train['Phosphorous'])
plt.title("Relationship between Soil Type and Phosphorous")
plt.show()





plt.figure(figsize=(12,6))
sns.barplot(x=train['Crop Type'],y=train['Temparature'])
plt.title("Relationship between Crop Type and Temparature")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=train['Crop Type'],y=train['Temparature'], hue=train['Fertilizer Name'])
plt.title("Relationship between Crop Type and Temparature")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=train['Crop Type'],y=train['Humidity'])
plt.title("Relationship between Crop Type and Humidity")
plt.show()





plt.figure(figsize=(12,6))
sns.barplot(x=train['Crop Type'],y=train['Humidity'], hue=train['Fertilizer Name'])
plt.title("Relationship between Crop Type and Humidity")
plt.show()


plt.figure(figsize=(12,6))
sns.barplot(x=train['Crop Type'],y=train['Moisture'])
plt.title("Relationship between Crop Type and Moisture")
plt.show()


sns.scatterplot(x=train['Temparature'],y=train['Humidity'])


sns.scatterplot(x=train['Temparature'],y=train['Humidity'],hue=train['Fertilizer Name'])


# count = 1
# plt.figure(figsize=(15,12))
# for feature in numerical_features:
#     plt.subplot(2, 3, count)
#     plt.title(f"Temperature vs. {feature}") 
#     sns.scatterplot(x=train['Temparature'], y=train[feature])
#     plt.xlabel('Temperature') 
#     plt.ylabel(feature)       
#     count += 1

# plt.tight_layout() # This is the key to fixing the conjestion!
# plt.show()


numerical_features


count = 1
plt.figure(figsize=(15,12))
for feature in numerical_features:
    plt.subplot(3, 2, count)
    sns.boxplot(x=train['Fertilizer Name'],y=train[feature])     
    count += 1

plt.tight_layout() # This is the key to fixing the conjestion!
plt.show()


count = 1
plt.figure(figsize=(15,14))
for feature in numerical_features:
    plt.subplot(3, 2, count)
    sns.distplot(train[feature],hist=False)   
    count += 1

plt.subplots_adjust(wspace=0.3, hspace=0.4) # This is the key to fixing the conjestion!
plt.show()


pd.crosstab(train['Soil Type'],train['Crop Type'])


plt.figure(figsize=(12,6))
sns.heatmap(pd.crosstab(train['Soil Type'],train['Crop Type']))
plt.title("Observations for each combination of 'Soil Type' and 'Crop Type'")
plt.show()


pd.crosstab(train['Soil Type'],train['Fertilizer Name'])


plt.figure(figsize=(12,6))
sns.heatmap(pd.crosstab(train['Soil Type'],train['Fertilizer Name']),annot=True)
plt.title("Observations for each combination of 'Soil Type' and 'Fertilizer Name'")
plt.show()


sns.pairplot(train) ## pair plot


sns.heatmap(train[numerical_features].corr(),annot=True)


sns.heatmap(train[numerical_features].corr(method='spearman'),annot=True)


## function to find and print all the rows where outlier is present
def check_outlier(df,feature):
    if df[feature].dtype!='O':
        print("Feature Name : {}".format(feature))
        df_col_mean = df[feature].mean()
        df_col_std = df[feature].std()

        df_col_lower_limit = df_col_mean - 3*df_col_std 
        df_col_upper_limit = df_col_mean + 3*df_col_std 
        
        outliers = df[(df[feature]<df_col_lower_limit) | (df[feature]>df_col_upper_limit)]
        if outliers.shape[0]==0:
            print("There are no any outliers")
        else:
            display(outliers)
            print(f"Total outlier containing rows: {outliers.shape}")
    else:
        print("Feature Name : {}".format(feature))
        print("This is a categorical Feature...")

    print()


for i in train.columns:
    check_outlier(train,i)


train.drop('id',axis=1,inplace=True)


X = train.drop('Fertilizer Name',axis=1)
y = train['Fertilizer Name']


X


y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


X_train


from sklearn.preprocessing import LabelEncoder


fertilizer_encoder = LabelEncoder()
y_train = fertilizer_encoder.fit_transform(y_train)
y_test = fertilizer_encoder.transform(y_test)


## applying label encoder on soil type ad crop type features
soil_type_le = LabelEncoder()

X_train['Soil Type'] = soil_type_le.fit_transform(X_train['Soil Type'])
X_test['Soil Type'] = soil_type_le.transform(X_test['Soil Type'])

crop_type_le = LabelEncoder()
X_train['Crop Type'] = crop_type_le.fit_transform(X_train['Crop Type'])
X_test['Crop Type'] = crop_type_le.transform(X_test['Crop Type'])


test.isnull().sum()


test.head()


### doing same preprocessing with test data
test.drop('id',axis=1,inplace=True)
test['Soil Type'] = soil_type_le.transform(test['Soil Type'])
test['Crop Type'] = crop_type_le.transform(test['Crop Type'])


from xgboost import XGBClassifier


model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_train)),
    n_estimators=3200,
    learning_rate=0.045,         
    max_depth=7,                
    colsample_bytree=0.6,       
    colsample_bylevel=0.8,      
    subsample=0.8,
)
model.fit(X_train, y_train)


y_test_pred_probs = model.predict_proba(X_test)
y_test_pred = model.predict(X_test)


# MAP@3 function
def mapk(true_labels, predicted_labels, k=3):
    map_total = 0.0
    for true, preds in zip(true_labels, predicted_labels):
        score = 0.0
        for i, pred in enumerate(preds[:k]):
            if pred == true:
                score = 1.0 / (i + 1)
                break
        map_total += score
    map_score = map_total / len(true_labels)
    print(f"MAP@{k} Score: {map_score:.4f}")
    return map_score


# Get top 3 predictions
top_3_indices = np.argsort(y_test_pred_probs, axis=1)[:, -3:][:, ::-1]

# Flatten -> inverse transform -> reshape
flat_indices = top_3_indices.flatten()
flat_labels = fertilizer_encoder.inverse_transform(flat_indices)
top_3_labels = flat_labels.reshape(top_3_indices.shape)

# Prepare predictions
predicted_labels = [list(row) for row in top_3_labels]

# Get true labels
true_labels = fertilizer_encoder.inverse_transform(y_test)

# Evaluating the prediction for X_test
mapk(true_labels, predicted_labels, k=3)


sample_submission


## prediction for test data
test_pred_probs = model.predict_proba(test)
test_pred = model.predict(test)

# Predict class probabilities for test set
test_pred_probs = model.predict_proba(test)

# Get top 3 predicted indices
top_3_test_indices = np.argsort(test_pred_probs, axis=1)[:, -3:][:, ::-1]

# Flatten â†’ inverse_transform â†’ reshape to get original fertilizer names
flat_test_indices = top_3_test_indices.flatten()
flat_test_labels = fertilizer_encoder.inverse_transform(flat_test_indices)
top_3_test_labels = flat_test_labels.reshape(top_3_test_indices.shape)

# Join top 3 fertilizer names with space for each test sample
top_3_test_joined = [' '.join(row) for row in top_3_test_labels]

# Assign predictions
sample_submission['Fertilizer Name'] = top_3_test_joined

# Save to CSV
sample_submission.to_csv('XGBClassifier_prediction.csv', index=False)

print("âœ… Submission file saved as XGBClassifier_prediction.csv")




