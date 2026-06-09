import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score ,accuracy_score
from imblearn.over_sampling import SMOTE  # Synthetic Minority Oversampling Technique

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
 

# Model Tuning 
from sklearn.model_selection import GridSearchCV

# Warnings (to keep notebook clean)
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)


plt.style.available  # viewing styling catogries for visualizing 


plt.style.use('seaborn-v0_8-darkgrid')  # applying style 


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
sub = pd.read_csv("sample_submission.csv")


# basic eda function
def explore_data(dataset):
    print(f'Shape       : {dataset.shape}')
    print('x'*90)
    print(f'Columns     : {dataset.columns}')
    print('x'*90)
    print(f'Data types  :{dataset.dtypes}')
    print('x'*90)
    dataset.info()
    print('x'*90)


# models in a dictionary
models = {
    "LightGBM": LGBMClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0)  # silent mode
}

# function to evaluate models
def fit_and_score(models, X_train, X_test, y_train, y_test, seed=42):
    np.random.seed(seed)
    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)

    return model_scores


train.head(10)


explore_data(train)


train.describe()


train.isnull().sum() # checking missing values


train.duplicated().sum()  # checking for duplicate values


sns.countplot(x='y', data=train)
plt.title("Target Variable Distribution")
plt.show()


train['y'].value_counts()   # frequency table of y


cat_cols = train.select_dtypes(include='object').columns
print("Categorical Features:", cat_cols.tolist())


test.head(10)


explore_data(test)


test.describe()


test.isnull().sum() # checking for null


test.duplicated().sum()  # checking for duplicates


cat_cols = train.select_dtypes(include='object').columns
print("Categorical Features:", cat_cols.tolist())


num_col = train.select_dtypes(include=np.number).columns.tolist()
num_col.remove('y') # target removed

train[num_col].hist(figsize=(15,12),bins=30 , edgecolor='black')
plt.suptitle("Distribution of Numerical Features",fontsize= 16)
plt.tight_layout()
plt.show()


num_col = train.select_dtypes(include=np.number).columns.tolist() # with target

corr = train[num_col].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True)
plt.title("Correlation Heatmap")
plt.show()

# Correlation with target
corr_target = train[num_col].corr()['y'].sort_values(ascending=False)
print(corr_target)


# drop unwanted columns
train.drop('id',axis=1,inplace=True)


train.select_dtypes(include='object').columns


for col in cat_cols:
    print(f"{col}: {train[col].nunique()} unique values")


le = LabelEncoder()
#  For Binary Columns:

binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    train[col] = le.fit_transform(train[col])


#For Ordinal Column (education, optionally month):

# Manual mapping
education_order = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
train['education'] = train['education'].map(education_order)

# Optional month mapping
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
train['month'] = train['month'].map(month_order)


# For Non-Ordinal Columns (multi-category):

multi_cat_cols = ['job', 'marital', 'contact', 'poutcome']
train = pd.get_dummies(train, columns=multi_cat_cols, drop_first=True , dtype=int)


train.head()


cat_cols = test.select_dtypes(include='object').columns


for col in cat_cols:
    print(f"{col}: {test[col].nunique()} unique values")


le = LabelEncoder()
#  For Binary Columns:

binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    test[col] = le.fit_transform(test[col])


#For Ordinal Column (education, optionally month):

# Manual mapping
education_order = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
test['education'] = test['education'].map(education_order)

# Optional month mapping
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
test['month'] =test['month'].map(month_order)


# For Non-Ordinal Columns (multi-category):

multi_cat_cols = ['job', 'marital', 'contact', 'poutcome']
test = pd.get_dummies(test, columns=multi_cat_cols, drop_first=True , dtype=int)


test.head(10)


train.head()


# spliting  the data in to target and features
X = train.drop('y',axis=1)
y = train['y']


# spltting into test and train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


scores = fit_and_score(models,X_train,X_test,y_train,y_test)


scores


model_compare = pd.DataFrame(scores,index=['Accuracy'])
model_compare.plot.bar()
plt.show()


# Base model
model = CatBoostClassifier(verbose=0, random_state=42)

# Reduced and balanced grid for faster and optimal search
param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1],       # Removed 0.01 to speed up
    'depth': [4, 6],                    # Removed 8 to reduce combinations
    'l2_leaf_reg': [3, 5],
    'border_count': [32, 64]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1  # Show progress
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output
print("✅ Best Parameters:", grid_search.best_params_)
print("✅ Test Accuracy after GridSearchCV tuning:", accuracy)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

final_model = CatBoostClassifier(
    border_count=64,
    depth=6,
    iterations=200,
    l2_leaf_reg=5,
    learning_rate=0.1,
    random_state=42,
    verbose=0  # Keeps output clean
)

# Train the model
final_model.fit(X_train, y_train)

# Make predictions
y_pred = final_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Final Test Accuracy:", accuracy)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
model = rfc.fit(X_train, y_train)
model.score(X_test, y_test)


# splitting data into  train 
X_train = train.drop('y',axis=1)
y_train = train['y']
X_test = test.drop('id',axis=1)

cb =  CatBoostClassifier(verbose=0,random_state=42)
model = cb.fit(X_train,y_train)

y_pred = model.predict(X_test)


sub.head()


test.head()


result = pd.DataFrame({'id' : test['id'],
                      'y' : y_pred})


result.head(10)


result.to_csv('submission.csv',index=False)

