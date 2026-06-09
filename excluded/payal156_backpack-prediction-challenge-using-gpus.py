import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from cuml.preprocessing import TargetEncoder
from xgboost import XGBRegressor, plot_importance
import warnings
warnings.filterwarnings("ignore")


#Importing Train, Test, and Training Extra Datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')


print(train.shape, train.dtypes)


train.head()


print("Missing Values Count for Train Dataset")

train.isnull().sum()


print("Missing Values Count for Train_ex Dataset")

train_ex.isnull().sum()


print("Missing Values Count for Test Dataset")

test.isnull().sum()




train = pd.concat([train, train_ex], axis=0, ignore_index=True)



train.shape


train = train.drop_duplicates()
train.shape


# Impute missing numerical data with the median values from the TRAIN dataset

num_cols = test.select_dtypes(include=['number']).columns
imputation_value = train[num_cols].median()
train[num_cols] = train[num_cols].fillna(imputation_value)
test[num_cols] = test[num_cols].fillna(imputation_value)


# Impute Missing Values in Object Columns with 'None'

obj_cols = train.select_dtypes(include=['object']).columns

train[obj_cols] = train[obj_cols].fillna('uknown')
test[obj_cols] = test[obj_cols].fillna('uknown')


train.isnull().sum()



TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test.columns.tolist()

for col in features:
    TE.fit(train[col], train['Price'])
    train[col] = TE.transform(train[col])
    test[col] = TE.transform(test[col])



X = train.drop(['Price'], axis=1)
y = train['Price']
X_train,X_val,y_train,y_val = train_test_split(X,y, test_size=0.2, random_state=42)


# Define XGBoost model
model = XGBRegressor(
    device="cuda",
    max_depth=5,
    n_estimators=2000,
    learning_rate=0.01,
    random_state=42
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    verbose=400
)



y_test_pred = model.predict(test)


submission = pd.DataFrame({'id': test.index, 'Price': y_test_pred})
submission.to_csv('submissionxgb.csv', index=False)
display(submission)

