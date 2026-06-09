# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


training = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

 # prepare the data for combined processing 

training['train_test'] = 1
test['train_test'] = 0 
test['Personality'] = np.NaN
data = pd.concat([training ,test])
%matplotlib inline




# look at our data types and null counts 
training.info()



# to understand the numeric data , we use the describe() method 
# which gives us basic descriptive statistics for all of our numeric data
training.describe()


training.columns


# seperating the numeric columns 
training.describe().columns


# we seperate the data into numeric and categorical data 
data_num = training[['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']]
data_cat = training[['Stage_fear','Drained_after_socializing']]


#Summary statistics for categorical features
data_cat.describe()


# Function to create scrollable table within a small window
def create_scrollable_table(df, table_id, title):
    html = f'<h3>{title}</h3>'
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += df.to_html()
    html += '</div>'
    return html


from IPython.display import display, HTML
# Null values in the dataset
null_values = training.isnull().sum()
html_null_values = create_scrollable_table(null_values.to_frame(), 'null_values', 'Null values in the dataset')

# Percentage of missing values for each feature
missing_percentage = (null_values / len(training)) * 100
html_missing_percentage = create_scrollable_table(missing_percentage.to_frame(), 'missing_percentage', 'Percentage of missing values for each feature')

display(HTML(html_null_values + html_missing_percentage))


training['Personality'].value_counts(normalize=True).plot(kind='bar',title='Personality Class Distribution')



#distribution for all numeric variables
for i in data_num.columns :
    sns.histplot(data_num[i], kde=True)
    plt.title(i)
    plt.show()



plt.figure(figsize=(10, 8))
corr = data_num.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


#compare Personality across numeric features 
pd.pivot_table(training , index = 'Personality' , values =['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency'])


for i in data_cat :
    plt.figure(figsize=(6, 4))
    sns.countplot(x=i, data=data_cat)
    plt.title(f"Count of {i}")
    plt.show()


pd.pivot_table(training , index = 'Personality' , values =['Drained_after_socializing', 'Stage_fear'], aggfunc = 'count')


cat_cols = data.select_dtypes(include=['object']).columns.drop('Personality')

# Fit encoder only on train set
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(data.loc[data['train_test'] == 1, cat_cols])


data[cat_cols] = encoder.transform(data[cat_cols])



#Restore the train data and the test data
train = data[data['train_test'] == 1].copy()
test = data[data['train_test'] == 0].copy()
test_ids = test['id'].values
# Drop columns not needed for imputation
cols_to_drop = ['Personality', 'train_test', 'id']

X_train_raw = train.drop(columns=cols_to_drop)
X_test_raw = test.drop(columns=cols_to_drop)

# Impute Missing Values using the iterative imputer via RandomForst
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=20, random_state=42),
    max_iter=10,
    random_state=42
)



# Fit imputer only on training data
imputer.fit(X_train_raw)

# Transform train and test separately
X_train_imputed = pd.DataFrame(imputer.transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)




X_train_imputed['Personality'] = train['Personality'].values




features = [
     'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]


X = X_train_imputed[features]
y = X_train_imputed['Personality']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/validation split 
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)






my_model = XGBClassifier(n_estimators = 1000 ,learning_rate = 0.05) 
my_model.fit(X_train , y_train , early_stopping_rounds = 5 , eval_set = [(X_val , y_val)] , verbose = False)




from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


y_pred_val = my_model.predict(X_val)
print(confusion_matrix(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val, target_names=label_encoder.classes_))


import matplotlib.pyplot as plt

from xgboost import plot_importance

plot_importance(my_model, max_num_features=20)
plt.tight_layout()
plt.show()


my_model.fit(X, y_encoded)




# predict on test data
X_test = X_test_imputed[features]
test_preds_encoded = my_model.predict(X_test)



test_preds = label_encoder.inverse_transform(test_preds_encoded)



submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission = pd.DataFrame({
    'id': test_ids,               # The saved ids
    'Personality': test_preds     # The predictions
})

submission.to_csv("submission.csv", index=False)


