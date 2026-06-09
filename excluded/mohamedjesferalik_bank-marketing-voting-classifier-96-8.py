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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score,roc_auc_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping,log_evaluation
from catboost import CatBoostClassifier
import random,warnings
warnings.filterwarnings('ignore')


#Importing Dataset
data_path = '/kaggle/input'
train_df = pd.read_csv(data_path + '/playground-series-s5e8/train.csv')
bank_full = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv',sep = ";")
train_full = pd.concat([train_df,bank_full], axis = 0, ignore_index=True).drop_duplicates()
test_df = pd.read_csv(data_path + '/playground-series-s5e8/test.csv')


# Basic Checks
def Basic_checks(df):
    print('\n============ First 10 rows ==============')
    display(df.head(10))

    print('\n============ Shape ================')
    print(df.shape)

    print('\n============ Info ==================')
    display(df.info())

    print('\n============= Descriptive Statistics (Numerical Column) =============')
    display(df.describe())

    categorical_df = train_df.select_dtypes(include=['object', 'category'])
    print('\n======== Descriptive Statistics (Categorical Columns) ===========')
    if not categorical_df.empty:
        display(categorical_df.describe())
    else:
        print("No categorical columns found.")

    print('\n================= Checking null values ================')
    display(df.isnull().sum())

    print('\n================== Checking duplicate values ================')
    display(df.duplicated().sum())


Basic_checks(train_full)


Basic_checks(test_df)


train_df


plt.figure(figsize = (8,5))
sns.histplot(data = train_df,x = 'age',color = 'green',kde = True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Customers')
plt.show()


plt.figure(figsize = (5,5))
sns.countplot(data = train_df, x = 'education',color = 'blue')
plt.xlabel('Education')
plt.ylabel('Count')
plt.title('Education level of Customers')
plt.show()


plt.figure(figsize = (10,5))
sns.countplot(data = train_df, x = 'job',palette = 'rainbow')
plt.xlabel('Jobs')
plt.xticks(rotation = 45)
plt.ylabel('Count')
plt.title('Job Distribution')
plt.show()


category = train_df['marital'].value_counts()
plt.pie(category , labels = category.index,colors = ['red','green','blue'],autopct='%1.1f%%')
plt.title('Marital Status')
plt.axis('equal')
plt.show()



plt.figure(figsize= (8,5))
sns.countplot(data = train_df , x = 'education',hue = 'y',palette = 'Set1')
plt.xlabel('Education')
plt.xticks(rotation = 45)
plt.ylabel('Count')
plt.title('Education level effect in subcription')
plt.show()


plt.figure(figsize= (5,5))
sns.countplot(data = train_df , x = 'loan',hue = 'y',palette = 'Set2')
plt.xlabel('Loan')
plt.xticks(rotation = 45)
plt.ylabel('Count')
plt.title('Loan affecting in subcription')
plt.show()


plt.figure(figsize = (5,5))
sns.violinplot(data = train_df,x = 'y',y = 'age')
plt.xlabel('Subscription')
plt.ylabel('Age')
plt.title('Age groups subscription tendency')
plt.show()


#Scatter
sns.scatterplot(data = train_df, x = 'campaign',y = 'duration',hue = 'y')
plt.xlabel('Campaign')
plt.title('Relation Between Campaign and Duration')
plt.show()



num = train_df.select_dtypes(exclude = 'object')
plt.figure(figsize = (12,12))
sns.heatmap(num.corr(),annot = True,cmap = 'coolwarm')
plt.show()


#Splitting Independent and Dependent Features
X = train_full.drop(['id','y'],axis = 1)
y = train_full['y'].replace({'yes':1,'no':0}).astype(int)


#Splitting Dataset for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Defining Column groups

binary_yn = ["default", "housing", "loan"]  # binary yes/no columns
ordinal_cols = {  # ordinal categorical variables with natural order
    "education": ["unknown", "primary", "secondary", "tertiary"],
    "month": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
}
nominal_cat = ["job", "marital", "contact", "poutcome"]  # nominal categories
numeric_cols = ["age","balance","day","duration","campaign","pdays","previous"]  # numeric features

ordinal_list = list(ordinal_cols.keys())                     # list of ordinal columns
ordinal_categories = [ordinal_cols[c] for c in ordinal_list] # their category orders


#Transformer for yes/no → 0/1
def map_yes_no(df):
    mapping = {"yes": 1, "no": 0}
    out = df.copy()
    for c in binary_yn:
        out[c] = out[c].map(mapping).astype(int)   # map yes/no to 1/0
    return out

yn_mapper = FunctionTransformer(map_yes_no, feature_names_out="one-to-one")


#Build preprocessing pipelines

binary_pipe = Pipeline([
    ("yn_map", yn_mapper),   # convert yes/no into 1/0
])

ordinal_pipe = Pipeline([
    ("ord", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1)),
])

nominal_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),  # one-hot encode nominal
])

numeric_pipe_scaled = Pipeline([
    ("scaler", StandardScaler()),  # scale numeric columns
])

# Combine all preprocessing into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("bin", binary_pipe, binary_yn),       # process binary yes/no cols
        ("ord", ordinal_pipe, ordinal_list),   # process ordinal cols
        ("nom", nominal_pipe, nominal_cat),    # process nominal cols
        ("num", numeric_pipe_scaled, numeric_cols),  # process numeric cols
    ],
    remainder="drop"  # drop other columns if any
)


#Transforming Training and validation data

X_train_full = preprocessor.fit_transform(X)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
val_prep = preprocessor.transform(test_df)


# Models (XGBoost,LightGBM,CatBoost and Voting ensemble)

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 300

assert 'X_train' in globals() and 'y_train' in globals()
assert 'X_test' in globals() and 'y_test' in globals()

#validation split from training data for early stopping 
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
)

# XGBoost 
xgb = XGBClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    tree_method="hist",  
    device="cuda"       
)
xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose=False
)

# LightGBM 
lgb = LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=RANDOM_STATE,
    device="gpu"
)

lgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
        log_evaluation(period=0) 
    ]
)

# CatBoost 
cat = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    task_type="GPU",
    verbose=False
)
cat.fit(
    X_tr, y_tr,
    eval_set=(X_val, y_val),
    early_stopping_rounds=EARLY_STOPPING_ROUNDS
)

# ENSEMBLE 
ensemble = VotingClassifier(
    estimators=[("xgb", xgb), ("lgb", lgb), ("cat", cat)],
    voting="soft", n_jobs=-1
)
ensemble.fit(X_train, y_train)


#EVALUATION 
def eval_model(name, model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    auc = roc_auc_score(y, y_proba)
    f1 = f1_score(y, y_pred)
    print(f"{name} | Test AUC: {auc:.4f} | Test F1: {f1:.4f}")

print("\n========== TEST RESULTS ==========")
eval_model("XGBoost", xgb, X_test, y_test)
eval_model("LightGBM", lgb, X_test, y_test)
eval_model("CatBoost", cat, X_test, y_test)
eval_model("Ensemble", ensemble, X_test, y_test)


print(X_train_full.shape)
print(y.shape)


best_model = ensemble   # choose best based on auc
best_model.fit(X_train_full, y)     # retrain on full train data
test_proba = best_model.predict_proba(val_prep)[:,1]
print('Completed')# predict test probabilities


# save submission
sub = pd.DataFrame({'id': test_df['id'], 'y': test_proba})
sub.to_csv("submission.csv", index=False)
print("submission.csv saved!")
sub.head()




