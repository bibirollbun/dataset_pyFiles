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
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

train["Weight/Compartment"] = train["Weight Capacity (kg)"]/(train["Compartments"]+1)
test["Weight/Compartment"] = test["Weight Capacity (kg)"]/(test["Compartments"]+1)

y = train["Price"]
features = [
    "Brand",
    "Material",
    "Size",
    "Compartments",
    "Laptop Compartment",
    "Waterproof",
    "Style",
    "Color",
    "Weight Capacity (kg)",
    "Weight/Compartment"
]

X = train[features]
X_test = test[features]

categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat',categorical_transformer,categorical_cols),
        ('num',numerical_transformer,numerical_cols)
    ]
)

model = LGBMRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,random_state=42)

pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                          ('model',model)])

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

pipeline.fit(X_train,y_train)
preds = pipeline.predict(X_val)
print("Baseline RMSE:",mean_squared_error(y_val,preds,squared=False))

param_grid = {
    'model__n_estimators':[100,200,300],
    'model__max_depth':[5,10,15,None],
    'model__min_samples_split':[2,5,10]
}

pipeline.fit(X,y)

final_preds = pipeline.predict(X_test)

submission = pd.DataFrame({'id':test["id"], "Price":final_preds})
submission.to_csv('submission.csv',index=False)
print("Improved model submission")

