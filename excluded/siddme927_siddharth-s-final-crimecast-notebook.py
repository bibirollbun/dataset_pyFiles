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
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, chi2
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier 
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/crime-cast-forecasting-crime-categories/train.csv')
test_data = pd.read_csv('/kaggle/input/crime-cast-forecasting-crime-categories/test.csv')
sample_data = pd.read_csv('/kaggle/input/crime-cast-forecasting-crime-categories/sample.csv')


'''y = train_data['Crime_Category']
X = train_data.drop('Crime_Category', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

dc = DummyClassifier(strategy = "most_frequent")
dc.fit(X_train, y_train)
dc.predict(y_test)
dc.score(X_test, y_test)'''


#Analysing the shape of train data
train_data.shape


#Analysing the shape of test data
test_data.shape


train_data.head(n=2)


test_data.head(n=2)


train_data.info()


test_data.info()


train_data.describe().T


test_data.describe().T


train_data.isna().sum()/len(train_data)*100


test_data.isna().sum()/len(test_data)*100


print(train_data['Crime_Category'].nunique())
print(train_data['Crime_Category'].unique())


train_data['Crime_Category'].value_counts()/len(train_data)*100


crime_counts = train_data['Crime_Category'].value_counts()
sns.set(style="whitegrid")
colors = sns.color_palette('pastel')
plt.figure(figsize=(12, 12))
plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=180, colors=colors)
plt.title('Crime Category Distribution for Train Data', fontsize=16)
plt.show()


correlation = train_data.corr(numeric_only=True)
correlation


plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True,
            linewidths=.5, annot_kws={'size': 10, 'weight': 'bold'})
plt.title('Correlation of Numerical Features', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


train_data['Location'].unique


train_data['Location'].value_counts()/len(train_data)*100


def clean_location_column(x):
    x = re.sub(" +", " ", x) # Removes extra spaces
    x = re.sub("[^A-Za-z ]", "", x)     # Remove numbers and special characters
    return x


# Applying the function clean_location_column on the Training Data
train_data['Location'] = train_data['Location'].apply(clean_location_column)

# Applying the same function on the Testing Data
test_data["Location"] = test_data["Location"].apply(clean_location_column)


Lat_Long = train_data.iloc[:,[2,3]]


print(Lat_Long)


Lat_Long.describe().T


train_data.columns


# Converting Data_Reported, Date_Occurred into DateTime format
train_data['Date_Reported'] = pd.to_datetime(train_data['Date_Reported'])
train_data['Date_Occurred'] = pd.to_datetime(train_data['Date_Occurred'])

# Now with Testing Data now 
test_data['Date_Reported'] = pd.to_datetime(test_data['Date_Reported'])
test_data['Date_Occurred'] = pd.to_datetime(test_data['Date_Occurred'])


# Creating New Columns for Date, month, Year respectively in Training Data. 

train_data['Reported_day'] = train_data['Date_Reported'].dt.day
train_data['Reported_month'] = train_data['Date_Reported'].dt.month
train_data['Reported_year'] = train_data['Date_Reported'].dt.year

# Creating New Columns for Date, month, Year respectively in Testing Data.
test_data['Reported_day'] = test_data['Date_Reported'].dt.day
test_data['Reported_month'] = test_data['Date_Reported'].dt.month
test_data['Reported_year'] = test_data['Date_Reported'].dt.year


# Extracting the Year, Month and Day in separate columns
train_data['Occurred_day'] = train_data['Date_Occurred'].dt.day
train_data['Occurred_month'] = train_data['Date_Occurred'].dt.month
train_data['Occurred_year'] = train_data['Date_Occurred'].dt.year

# Doing the same with Testing Data too
test_data['Occurred_day'] = test_data['Date_Occurred'].dt.day
test_data['Occurred_month'] = test_data['Date_Occurred'].dt.month
test_data['Occurred_year'] = test_data['Date_Occurred'].dt.year


train_data.drop(columns = ['Date_Reported','Date_Occurred'], inplace = True)

test_data.drop(columns = ['Date_Reported','Date_Occurred'], inplace = True)


hour = lambda x: int(x//100)

train_data["Hour_Occurred"] = train_data['Time_Occurred'].apply(hour)

test_data["Hour_Occurred"] = test_data['Time_Occurred'].apply(hour)


# After doing all this, let's have a look at the Training and Testin Data
train_data


train_data.shape


avar = ['Reported_year', 'Reported_month', 'Reported_day', 'Occurred_year',
       'Occurred_month', 'Occurred_day', 'Hour_Occurred']
for i in avar:
    print("-------------------------------")
    print(train_data[i].value_counts()/len(train_data)*100)


correlation2 = train_data.corr(numeric_only = True)


plt.figure(figsize = (12, 12))
sns.heatmap(correlation2, annot = True, fmt = '.2f', cmap = 'coolwarm', cbar = True, square = True,
            linewidths = 0.5, annot_kws = {'size': 10, 'weight': 'bold'})
plt.title('Correlation of Numerical Features', fontsize = 16, fontweight = 'bold')
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.tight_layout()
plt.show()


sns.countplot(data = train_data, x = 'Reported_year')


train_data.drop(columns = ['Reported_month', 'Reported_day', 'Occurred_year',
       'Occurred_month', 'Occurred_day', 'Hour_Occurred','Time_Occurred'], inplace = True)

test_data.drop(columns = ['Reported_month', 'Reported_day', 'Occurred_year',
       'Occurred_month', 'Occurred_day', 'Hour_Occurred','Time_Occurred'], inplace = True)


print(train_data.shape)

print(test_data.shape)


train_data.head(n=2)


print(train_data.Area_ID.nunique())
print(train_data.Area_Name.nunique())


train_data[train_data['Area_ID'] == 15.0]['Area_Name']


train_data['Area_ID'].value_counts()/len(train_data)*100


train_data['Area_Name'].value_counts()/len(train_data)*100


train_data['Reporting_District_no'].nunique()


train_data['Part 1-2'].nunique()


train_data['Part 1-2'].value_counts()/len(train_data)*100


part1_train = train_data['Part 1-2'].value_counts().index
part1_count_train = train_data['Part 1-2'].value_counts().values
sns.set(style="whitegrid")
colors = sns.color_palette('pastel')
plt.figure(figsize=(10, 10))
plt.pie(part1_count_train, labels = part1_train, autopct = '%1.1f%%', startangle=140, colors=colors)
plt.title('Part 1-2  Distribution for Train Data', fontsize=16)
plt.show()


train_data['Modus_Operandi'].nunique()


train_data['Modus_Operandi'].value_counts()


print(train_data.Victim_Sex.nunique())
print(train_data.Victim_Sex.unique())



train_data['Victim_Sex'].value_counts()/len(train_data)*100


train_data.Victim_Sex.isna().sum()/len(train_data)


train_data['Victim_Descent'].nunique()


train_data['Victim_Descent'].value_counts()/len(train_data)*100


print(train_data['Premise_Code'].nunique())

print(train_data['Premise_Description'].nunique())


train_data['Premise_Code']


train_data[train_data['Premise_Code']==102.0]['Premise_Description']


print(train_data['Weapon_Used_Code'].nunique())

print(train_data['Weapon_Description'].nunique())


train_data[train_data['Weapon_Used_Code']==400.0]['Weapon_Description']


print(train_data['Status'].nunique())
print(train_data['Status_Description'].nunique())


train_data['Status']


train_data[train_data['Status']=='IC']['Status_Description']


label_encoder = LabelEncoder()
label_encoder.fit(['Crime_Category'])


#train_data['Crime_Category'] = label_encoder.inverse_transform(train_data['Crime_Category'])


train_data.groupby('Victim_Age')['Crime_Category'].value_counts(normalize = True).to_frame()*100


train_data.groupby('Victim_Sex')['Crime_Category'].value_counts(normalize = True).to_frame()*100


sns.set_style("whitegrid")
plt.subplots(figsize = (18, 7))
sns.countplot(
    x = "Victim_Sex",
    hue = "Crime_Category",
    data = train_data,
    edgecolor = "black",  
    palette = "Accent")
plt.title("Victim_Sex Vs Crime_Category", weight = "bold", fontsize = 40, pad = 20)
plt.ylabel("Count", weight = "bold", fontsize = 16)  # Increased fontsize for consistency
plt.xlabel("Victim_Sex", weight = "bold", fontsize = 16)
plt.legend(title = "Crime Category", fancybox = True, title_fontsize = 14, fontsize = 12)  # Added font sizes
plt.grid(True)
plt.tight_layout()
plt.show()


train_data.groupby('Status_Description')['Crime_Category'].value_counts(normalize=True).to_frame()*100



# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure and a set of subplots with the specified size
plt.subplots(figsize = (18, 7))

# Create a count plot with the specified data and style elements
sns.countplot(
    x = "Status_Description",
    hue = "Crime_Category",
    data = train_data,
    edgecolor = "black",  # Changed 'ec' to 'edgecolor' for clarity
    palette = "Accent"
)

# Set the title with bold font, specific size, and padding
plt.title("Status_Description Vs Crime_Category", weight = "bold", fontsize = 40, pad = 20)

# Set the labels for the y-axis and x-axis with bold font and specific size
plt.xlabel("Status_Description", weight = "bold", fontsize =  20)
plt.ylabel("Count", weight = "bold", fontsize =  20)



train_data.groupby('Reporting_District_no')['Crime_Category'].value_counts(normalize = True).to_frame()*100





train_data.columns


train_data.shape


y = train_data['Crime_Category']
X = train_data.drop(columns = ['Crime_Category'])


X.shape


X.drop(columns = ['Cross_Street', 'Area_Name', 'Premise_Description', 'Weapon_Description', 'Status_Description'], inplace = True)

test_data.drop(columns = ['Cross_Street', 'Area_Name', 'Premise_Description', 'Weapon_Description', 'Status_Description'], inplace = True)


X.shape # Just to be sure


columns_X = X.columns
columns_test_data = test_data.columns


si = SimpleImputer(strategy = 'most_frequent')

X = si.fit_transform(X)

test_data = si.transform(test_data)

X = pd.DataFrame(X, columns = columns_X)

test_data = pd.DataFrame(test_data, columns = columns_test_data)


test_data.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


X_test.columns





text_columns = ['Location', 'Modus_Operandi']
numerical_columns = ['Latitude', 'Longitude', 'Victim_Age']
categorical_columns = ['Area_ID', 'Reporting_District_no', 'Part 1-2', 'Victim_Sex', 'Victim_Descent', 'Weapon_Used_Code', 'Status', 'Reported_year']


Column_transformer = ColumnTransformer([('num', StandardScaler(), numerical_columns),('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_columns), ('text1', TfidfVectorizer(), 'Location'),
                                       ('text2', TfidfVectorizer(), 'Modus_Operandi')])


X_train_preprocessed = Column_transformer.fit_transform(X_train, y_train)
X_test_preprocessed = Column_transformer.transform(X_test)





label = LabelEncoder()
y_train_le = label.fit_transform(y_train)
y_test_le = label.transform(y_test)





kbest = SelectKBest(f_regression, k = 720)
kbest.fit(X_train_preprocessed, y_train_le)


pipeline1 = Pipeline([('column_trans', Column_transformer), ('kbests', SelectKBest(f_regression, k = 840)), ("LGBM", LGBMClassifier())])
pipeline1.fit(X_train, y_train_le)
pipeline1.score(X_test, y_test_le)



pipeline2 = Pipeline([('column_trans', Column_transformer), ('kbests', SelectKBest(f_regression, k = 3911)), ("XGB", XGBClassifier())])
pipeline2.fit(X_train, y_train_le)
pipeline2.score(X_test, y_test_le)



# def select_best_K_lgbm(k_input, X_train, y_train_le, X_test, y_test_le):
#     scores = {}
#     for k in k_input:
#         pipe_lgbm = Pipeline([('transformer', Column_transformer), ('k_best',SelectKBest(f_regression, k=k)) ,('model_lgbm', LGBMClassifier())])
#         pipe_lgbm.fit(X_train, y_train_le)
#         scores[k] = pipe_lgbm.score(X_test, y_test_le)
#     return scores

# lgbm_scores = select_best_K_lgbm([700, 730, 760, 790, 820, 850, 880, 890], X_train, y_train_le, X_test, y_test_le)


# print(lgbm_scores)


# max_key = max(lgbm_scores, key = lgbm_scores.get)
# max_value = lgbm_scores[max_key]
# print(f"{max_key}: {max_value}")


# x = list(lgbm_scores.keys())
# y = list(lgbm_scores.values())
# sns.set(style = "darkgrid")
# plt.figure(figsize=(10, 6), facecolor='black')
# ax = plt.gca()
# ax.set_facecolor('black')
# sns.lineplot(x = x, y = y, marker = 'o', color='cyan', linewidth = 2, markersize = 10)
# plt.title('Select_K_Best_features Vs Accuracy with LGBMClassifier', color = 'white')
# plt.xlabel('Features', color = 'white')
# plt.ylabel('Score', color = 'white')
# plt.xticks(color = 'white')
# plt.yticks(color = 'white')
# plt.show()


# def select_best_K_xgb(k_input, X_train, y_train_le, X_test, y_test_le):
#     scores = {}
#     for k in k_input:
#         k_best_pipeline = Pipeline([('transformer', Column_transformer), ('k_best',SelectKBest(f_regression, k=k)) ,('model_xgb', XGBClassifier())])
#         k_best_pipeline.fit(X_train, y_train_le)
#         scores[k] = k_best_pipeline.score(X_test, y_test_le)
#     return scores

# xgbscores = select_best_K_xgb([500, 850, 1000, 2000, 3000, 3911], X_train, y_train_le, X_test, y_test_le)


# print(xgbscores)


# max_key = max(xgbscores, key = xgbscores.get)
# max_value = xgbscores[max_key]
# print(f"{max_key}: {max_value}")


# x = list(xgbscores.keys())
# y = list(xgbscores.values())
# sns.set(style = "darkgrid")
# plt.figure(figsize=(10, 6), facecolor='black')
# ax = plt.gca()
# ax.set_facecolor('black')
# sns.lineplot(x = x, y = y, marker = 'o', color='cyan', linewidth = 2, markersize = 10)
# plt.title('Select_K_Best_features Vs Accuracy with LGBMClassifier', color = 'white')
# plt.xlabel('Features', color = 'white')
# plt.ylabel('Score', color = 'white')
# plt.xticks(color = 'white')
# plt.yticks(color = 'white')
# plt.show()


# pipe_svd1 = Pipeline([('transformer', Column_transformer), ('Truncatedsvd', TruncatedSVD(n_components = 100)), ('model_lgb', LGBMClassifier())])
# pipe_svd1.fit(X_train, y_train) 
# pipe_svd1.score(X_test, y_test)


# pipe_svd2 = Pipeline([('transformer', Column_transformer), ('Truncatedsvd', TruncatedSVD(n_components = 100)), ('model_lgb', XGBClassifier())])
# pipe_svd2.fit(X_train, y_train_le) 
# pipe_svd2.score(X_test, y_test_le)


# PCA with LightBGM Classifier
# pca_transform_column = ColumnTransformer([('num', StandardScaler(), numerical_columns),('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_columns), ('text1', TfidfVectorizer(), 'Location'),
#                                        ('text2', TfidfVectorizer(), 'Modus_Operandi')], sparse_threshold = 0)

# pca_pipe = Pipeline([('tranform', pca_transform_column), ('PCA', PCA(n_components = 200)), ('lgbm_model', LGBMClassifier())])

# pca_pipe.fit(X_train, y_train)
# pca_pipe.score(X_test, y_test)


# Trying the same with XGBoost Classifier
# pca_transform_column = ColumnTransformer([('num', StandardScaler(), numerical_columns),('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_columns), ('text1', TfidfVectorizer(), 'Location'),
#                                        ('text2', TfidfVectorizer(), 'Modus_Operandi')], sparse_threshold = 0)

# pca_pipe = Pipeline([('tranform', pca_transform_column), ('PCA', PCA(n_components = 200)), ('xgb_model', XGBClassifier())])

# pca_pipe.fit(X_train, y_train_le)
# pca_pipe.score(X_test, y_test_le)


lbgm = LGBMClassifier(class_weight = None, learning_rate = 0.1, n_estimators = 100, num_leaves = 40, reg_alpha = 0)
lbgm.fit(X_train_preprocessed, y_train)
lbgm.score(X_test_preprocessed, y_test)


logitregression = LogisticRegression()
logitregression.fit(X_train_preprocessed, y_train)
logitregression.score(X_test_preprocessed, y_test)


dt = DecisionTreeClassifier()
dt.fit(X_train_preprocessed, y_train)
dt.score(X_test_preprocessed, y_test)
# Low score


rf = RandomForestClassifier()
rf.fit(X_train_preprocessed, y_train)
rf.score(X_test_preprocessed, y_test)
# Low score


gb = GradientBoostingClassifier()
gb.fit(X_train_preprocessed, y_train)
gb.score(X_test_preprocessed, y_test)
# Good score comparatively


le = LabelEncoder()
y_t_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)


xgb = XGBClassifier()
xgb.fit(X_train_preprocessed, y_t_encoded)
xgb.score(X_test_preprocessed, y_test_encoded)
# Best Score 


new_pipeline = Pipeline([('ColumnTransformer', Column_transformer), ('SelectKBest', SelectKBest(f_regression, k = 3911))])
X_train_new = new_pipeline.fit_transform(X_train, y_train_le)
X_test_new = new_pipeline.transform(X_test)


y = train_data['Crime_Category']


le1 = LabelEncoder()
y_encoded = le1.fit_transform(y)


print(f"{X.shape}, {y_encoded.shape}")


X_train_final = new_pipeline.fit_transform(X, y_encoded)
X_test_final = new_pipeline.transform(test_data)


X_test_final.shape


xgbsubm = XGBClassifier(learning_rate = 0.3, max_depth = 10)
xgbsubm.fit(X_train_final, y_encoded)
y_pred = le1.inverse_transform(xgbsubm.predict(X_test_final))


knn = KNeighborsClassifier()
knn.fit(X_train_preprocessed, y_train)
knn.score(X_test_preprocessed, y_test)


mlp = MLPClassifier()
mlp.fit(X_train_preprocessed, y_train)
mlp.score(X_test_preprocessed, y_test)


# cat = CatBoostClassifier()
# cat.fit(X_train_preprocessed, y_train)
# cat.score(X_test_preprocessed, y_test)





# param_logit = {'penalty': ['l1', 'l2', 'elasticnet'],
#               'C': [0.1, 0.5, 1.0],
#               'solver': ['lbfgs', 'liblinear', 'saga']}

# grid_logit = GridSearchCV(LogisticRegression(), param_grid = param_logit, cv = 3, verbose = 3, n_jobs = -1)

# grid_logit.fit(X_train_preprocessed, y_train)


# grid_logit.best_params_


logitparams = LogisticRegression(C = 1.0, penalty = 'l1', solver = 'liblinear')
logitparams.fit(X_train_preprocessed, y_train)
logitparams.score(X_test_preprocessed, y_test)


# param_dt = {'max_depth': [1, 10, 100, 1000, 10000],
#            'min_samples_split': [1, 2, 3, 4, 5], 
#            'min_samples_leaf': [1, 2, 3, 4, 5]
#            }

# grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid = param_dt, cv = 3, verbose = 3, n_jobs = -1)

# grid_dt.fit(X_train_preprocessed, y_train)  


# grid_dt.best_params_


dt_new = DecisionTreeClassifier(max_depth = 10000, min_samples_leaf = 1, min_samples_split = 2)
dt_new.fit(X_train_preprocessed, y_train)
dt_new.score(X_test_preprocessed, y_test)


# param_xgb = {'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.5, 0.8, 0.9], 
#             'max_depth': [1, 10, 50, 100, 200, 500, 1000], 
#             'min_child_weight': [1, 10, 100, 1000, 10000, 100000], 
#             'subsample': [0.00001, 0.0001, 0.001, 0.01, 0.5, 0.8, 0.9], 
#             'reg_alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 0.9], 
#             'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 0.9]}

# grid_xgb = GridSearchCV(XGBClassifier(), param_grid = param_xgb, cv = 5, verbose = 5, n_jobs = -1)
# grid_xgb.fit(X_train_preprocessed, y_t_encoded)
# grid_xgb.score(X_test_preprocessed, y_test_encoded)


le2 = LabelEncoder()
y_encoded_final = le2.fit_transform(y)


new_pipeline = Pipeline([('ColumnTransformer', Column_transformer), ('SelectKBest', SelectKBest(f_regression, k = 3911))])
X_train_new_s = new_pipeline.fit_transform(X_train, y_train_le)
X_test_new_s = new_pipeline.transform(X_test)


X_train_final_s = new_pipeline.fit_transform(X, y_encoded)
X_test_final_s = new_pipeline.transform(test_data)


xg_sum = XGBClassifier(learning_rate = 0.3, max_depth = 10)
xg_sum.fit(X_train_final_s, y_encoded_final)
y_predd = le2.inverse_transform(xg_sum.predict(X_test_final_s)) 


submission = pd.DataFrame(columns = ["ID","Crime_Category"])
submission["ID"] = [i for i in range(1, len(y_pred) + 1)]
submission["Crime_Category"] = y_predd
submission.to_csv('submission.csv', index = False)

# Original Submission Code Block














import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace this with your actual data)
crime_category_train = ['Theft', 'Assault', 'Burglary', 'Fraud', 'Vandalism']
crime_count_train = [350, 280, 200, 180, 120]

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))  # Larger figure size
plt.pie(crime_count_train, 
        labels=crime_category_train, 
        autopct='%1.1f%%', 
        startangle=90,  # Start the pie chart from 90 degrees (top of the chart)
        colors=sns.color_palette('pastel'), 
        explode=(0.1, 0, 0, 0, 0.1),  # Explode the first and last slice
        shadow=True)  # Add shadow to the pie chart

plt.title('Crime Category Distribution for Train Dataset', fontsize=20)





















lgmc = LGBMClassifier(class_weight = None, learning_rate = 0.1, n_estimators = 100, num_leaves = 40, reg_alpha = 0)
lgmc.fit(X_transformed_train, y_encoded)
lgmc.score(X_transformed_test, y_encoded_t)









# submission = pd.DataFrame(columns = ["ID", "Crime_Category"])
# submission["ID"] = [i for i in range(1, len(y_pred)+1)]
# submission["Crime_Category"] = y_pred
# submission.to_csv('submission.csv', index = False)

