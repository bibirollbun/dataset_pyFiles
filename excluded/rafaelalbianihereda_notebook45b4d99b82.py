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


## Loading data
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.shape


train.head()


train.info()


train.describe()


# Salvar os índices originais (ou ids) para separação posterior
train_ids = train['id']
test_ids = test['id']

# Adicionar um marcador de origem
train['source'] = 'train'
test['source'] = 'test'
test['Calories'] = np.nan  # Adiciona target ausente ao test

# Combinar os dois
df = pd.concat([train, test], ignore_index=True)


df.head()


# cálculo do imc

df['BMI'] = df['Weight']/(df['Height']/100)**2

df.head()


# esforço estimado

df['Effort'] = df['Duration'] * df['Heart_Rate']

df.head()


# potencial metabólico

df["Metabolic_Potential"] = (df["Heart_Rate"] * df["Duration"]) / (df['Age'] + 1)

df.head()


df['Temp_by_Duration'] = df['Body_Temp'] / df['Duration']



train_final = df[df['source'] == 'train'].drop(columns=['source'])
test_final = df[df['source'] == 'test'].drop(columns=['source', 'Calories']) 

train_final = train_final.set_index('id').loc[train_ids].reset_index()
test_final = test_final.set_index('id').loc[test_ids].reset_index()


train_final.head()


X = train_final.drop(columns=['id', 'Calories'])
y = train_final.Calories


X.head()


y.head()


X.shape, y.shape


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler



categorical_feature = ['Sex']
numerical_feature = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Effort', 'Metabolic_Potential', 'Temp_by_Duration']

transformations = [
    ('ohe', OneHotEncoder(drop='first'), categorical_feature)
]
preprocessor = ColumnTransformer(transformers=transformations, remainder='passthrough')

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import numpy as np

model = LGBMRegressor(
    objective='regression',
    random_state=42,
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=12,
    num_leaves=70,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_alpha=0.3,
    reg_lambda=0.3,
    min_child_samples=10,
    force_col_wise=True
)

model.fit(
    X_train_processed, y_train,
    eval_set=[(X_test_processed, y_test)],
    eval_metric='rmse',
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
)

y_pred = model.predict(X_test_processed)
y_pred = np.maximum(y_pred, 0)  # evita erro no MSLE

msle = mean_squared_log_error(y_test, y_pred)
rmsle = np.sqrt(msle)

print(f'RMSLE: {rmsle:.4f}')


