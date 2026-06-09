# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from warnings import filterwarnings
filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv').set_index('id')
print('train:',train.shape,'test:',test.shape)
TARGET='y'
omit_cols = [TARGET]


original = pd.read_csv('/kaggle/input/bank-full/bank-full.csv',  delimiter=';')
original['y'] = original.y.map({'yes':1,'no':0})
original.index.name='id'
original = original.reset_index()

original.head(1)


from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler

df_all = pd.concat([train,original,test],axis=0).reset_index(drop=True)  # create the base table

poly_base = ['age','duration','campaign', 'balance','pdays','previous']
poly = PolynomialFeatures(degree=2, include_bias=False)

# Step 1: fit and transform the base features into polynomial values
poly_values = poly.fit_transform(df_all[poly_base])

# Step 2: get the feature names from the fitted model (will include the features in poly_base)
new_features = poly.get_feature_names_out()

poly_features = ['p_' + x for x in new_features if x not in poly_base] # list of polynomial feature names

# Step 3: Create a DataFrame for the polynomial values created in Step 1
dfi = pd.DataFrame(poly_values, columns=poly_base + poly_features)
dfi = dfi.drop(columns=poly_base,axis=1)   # remove base columns for concatenation

# Step 4: Combine the new features with the base table
df_all = pd.concat([df_all, dfi], axis=1).reset_index(drop=True)


# encode categorical columns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer as ct

le = LabelEncoder()

CATS =  df_all.select_dtypes(include=['object']).columns

for c in CATS:
    df_all[c] = le.fit_transform(df_all[c])    



test = df_all[len(train)+len(original):]
train = df_all[:len(train)+len(original)]
test = test.drop(TARGET,axis=1)


features = [x for x in test.columns if x != 'id']
features



X_train, X_test, y_train, y_test = train_test_split(train, train[TARGET], test_size=0.2, random_state=42)


# get a basic reading which should go up as you add feature engineering to the above

model = XGBClassifier(objective='binary:logistic', random_state=42)

model.fit(
    X_train[features], 
    X_train[TARGET]
)

test_preds = model.predict(X_test[features])
score = roc_auc_score(y_test, test_preds)
print(score) 

# 0.8135741610028036


# take a look at feature importance

import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][0:20], importance_df["Importance"][0:20])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBClassifier Feature Importance")
plt.gca().invert_yaxis()  
plt.show()



# now make submission predictions using split processing

from sklearn.model_selection import StratifiedKFold
N_SPLITS = 7

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_sub_preds = []
all_test_scores = []

model = XGBClassifier(objective='binary:logistic', random_state=42,n_estimators=500)

for fold, (train_idx, test_idx) in enumerate(skf.split(train, train[TARGET])):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    X_train = train.iloc[train_idx]
    y_train = train[TARGET][train_idx]

    X_test = train.iloc[test_idx]
    y_test = train[TARGET][test_idx]

    model.fit(X_train[features],y_train)
    
    test_preds = model.predict(X_test[features])    
    score = roc_auc_score(y_test, test_preds)
    all_test_scores.append(score)
    print('fold:',fold,score)    
    
    sub_preds = model.predict_proba(test[features])
    all_sub_preds.append(sub_preds) 

print()
print('Average score', np.mean(all_test_scores))


sub_preds = sum(all_sub_preds)/len(all_sub_preds)


sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
sub[TARGET] = sub_preds[:,1]
sub.head()


sub.to_csv('submission.csv', index=False)

