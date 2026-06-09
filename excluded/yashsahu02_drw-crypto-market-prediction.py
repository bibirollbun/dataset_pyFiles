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
%matplotlib inline 


train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
sample_submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")


##top 5 rows in train df
train.head()


test.head()


sample_submission


train.isnull().sum()


test.isnull().sum()


numerical_features = [feature for feature in train.columns if train[feature].dtype!='O']
categorical_features = [feature for feature in train.columns if train[feature].dtype=='O']


numerical_features


print("Total Numerical Features:",len(numerical_features))
print("Total Categorical Features:",len(categorical_features))


categorical_features


# # with the following function we can select highly correlated features
# # it will remove the first feature that is correlated with anything other feature

# def correlation(dataset, threshold):
#     col_corr = set()  # Set of all the names of correlated columns
#     corr_matrix = dataset.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
#                 colname = corr_matrix.columns[i]  # getting the name of column
#                 col_corr.add(colname)
#     return col_corr


# highly_correlated_feature = correlation(train,0.9)


train.head()


print("Shape of train:",train.shape)


train.replace([np.inf, -np.inf], np.nan, inplace=True)


train.isnull().sum().sort_values(ascending=False)


train.isnull().sum().sort_values(ascending=False)[lambda x: x > 0]


cols_to_drop = train.isnull().sum().sort_values(ascending=False)[lambda x: x > 0].index


cols_to_drop


train.drop(columns=cols_to_drop, inplace=True)


train.shape


X = train.drop(columns=['label']) ## Independent features
y = train['label']


from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=30)  # choose top 30
X_selected = selector.fit_transform(X, y)

mask = selector.get_support()

# Get names of selected features
selected_features = X.columns[mask]
print(selected_features)


## Previous Selected Features
# cols_to_keep = ['bid_qty','ask_qty','buy_qty','sell_qty','volume',
#         'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X26', 'X27', 'X28', 'X29',
#        'X30', 'X175', 'X181', 'X217', 'X218', 'X219', 'X225', 'X226', 'X281',
#        'X283', 'X285', 'X286', 'X287', 'X288', 'X289', 'X290', 'X291', 'X292',
#        'X293', 'X294', 'X295', 'X296', 'X297', 'X298', 'X299', 'X300', 'X301',
#        'X302', 'X303', 'X465', 'X466', 'X524', 'X531', 'X598', 'X856', 'X857',
#        'X858', 'X860', 'X861', 'X863'] ## keeping only these features for training 

# cols_to_keep = ['bid_qty','ask_qty','buy_qty','sell_qty','volume',
#         'X19', 'X20', 'X21', 'X22', 'X27', 'X28', 'X29', 'X218', 'X219', 'X287',
#        'X289', 'X291', 'X293', 'X295', 'X531', 'X598', 'X857', 'X858', 'X860',
#        'X863'] ## keeping only these features for training 

# cols_to_keep = ['X19', 'X20', 'X21', 'X22', 'X27', 'X28', 'X29', 'X218', 'X219', 'X287',
#        'X289', 'X291', 'X293', 'X295', 'X531', 'X598', 'X857', 'X858', 'X860',
#        'X863'] ## keeping only these features for training 

## keeping top 
X = X[selected_features]


X.head()


y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


print("Shape of X_train:",X_train.shape)
print("Shape of X_test:",X_test.shape)


print("Shape of y_train:",y_train.shape)
print("Shape of y_test:",y_test.shape)


from sklearn.preprocessing import StandardScaler


from sklearn.compose import ColumnTransformer


scaler = StandardScaler()


numerical_features_list = [feature for feature in X.columns if X[feature].dtype!='O']


transformer = ColumnTransformer(transformers=[
    ('standard_scalling', scaler, numerical_features_list),
], remainder='passthrough')  # Keeps other columns as they are


X_train_trf = transformer.fit_transform(X_train)
X_test_trf = transformer.transform(X_test)


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error


## Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(true,predicted)

    r = np.corrcoef(true, predicted)[0, 1]
    print()
    print(f"Pearson Correlation Coefficient: {r}")
    print("R2 Score:{:.4f}".format(r2))
    print("MAE:{:.4f}".format(mae))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    
    # ---------
    return 0


test=test.drop(columns=['label']) ## dropping target feature from test dataframe
test = test[selected_features]
test_trf = transformer.transform(test) ## scalling





sample_submission


X_train_trf.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(32, input_dim=30, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="linear"))


model.compile(loss='mse', optimizer='adam', metrics=['mse'])


history = model.fit(X_train_trf, y_train, epochs=50,
                    validation_data = (X_test_trf, y_test),
)


y_train_pred.shape


type(y_train)


type(y_train_pred)


print(f"Evaluating Model on training data")
y_train_pred = model.predict(X_train_trf)

y_train_pred = np.reshape(y_train_pred,y_train.shape)
evaluate_model(y_train,y_train_pred)


print(f"Evaluating Model on test(validation) data")
y_test_pred = model.predict(X_test_trf)
y_test_pred = np.reshape(y_test_pred,y_test.shape)
evaluate_model(y_test,y_test_pred)


import matplotlib.pyplot as plt


plt.plot(history.history['loss'],color='green',label='train')
plt.plot(history.history['val_loss'],color='black',label='validation')
plt.title("Loss vs Validation Loss")
plt.legend()
plt.show()


plt.plot(history.history['mse'],color='green',label='train')
plt.plot(history.history['val_mse'],color='black',label='validation')
plt.title("mse vs val mse")
plt.legend()
plt.show()


model.predict(test_trf)


prediction = model.predict(test_trf)


sample_submission['prediction'] = prediction
sample_submission.to_csv('submission.csv',index=False)
display(sample_submission.head())
print(f"File saved as submission.csv")




