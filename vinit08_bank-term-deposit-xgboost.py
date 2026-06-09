import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.sample(5)


test.sample(5)


print(train.shape)
print(test.shape)


train.isnull().sum()


test.isnull().sum()


train.info()


train.describe()


object_columns = train.select_dtypes(include='object').columns.tolist()
numeric_columns = train.select_dtypes(include='int').columns.tolist()


import plotly.express as px

for col in object_columns:
    a = train[col].value_counts().sort_values(ascending=False).reset_index()
    a.columns = [col, 'count'] 

    fig = px.bar(
        a, 
        x=col, 
        y='count',
        hover_data=['count'], 
        color=col,  
        height=400, 
        text='count'
    )
    fig.update_layout(title=f'Value Counts of {col}')
    fig.show()



train_df =  train.copy()
test_df =  test.copy()



def encoder_transform(column):
    encoder = LabelEncoder()
    train_df[column] = encoder.fit_transform(train_df[column])



def encoder_transform_test(column):
    encoder = LabelEncoder()
    test_df[column] = encoder.fit_transform(test_df[column])



encoder_transform("job")
encoder_transform("marital")
encoder_transform("education")
encoder_transform("default")
encoder_transform("housing")
encoder_transform("loan")
encoder_transform("contact")
encoder_transform("poutcome")
encoder_transform("month")


encoder_transform_test("job")
encoder_transform_test("marital")
encoder_transform_test("education")
encoder_transform_test("default")
encoder_transform_test("housing")
encoder_transform_test("loan")
encoder_transform_test("contact")
encoder_transform_test("poutcome")
encoder_transform_test("month")


train_df


test_df


x = train_df.iloc[: , 0:-1]
y= train_df.iloc[: , -1]


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


from xgboost import XGBClassifier


model = XGBClassifier(
    objective =  "binary:logistic",
    eval_metric= "logloss",
    max_depth = 4,
    eta = 0.1,
    subsample =  0.8,
    colsample_bytree = 0.8,
    random_state = 42 , 
    n_estimators = 300
)


model.fit(x_train, y_train)


from sklearn.metrics import roc_auc_score
y_pred = model.predict(x_val)
roc_auc = roc_auc_score(y_val, y_pred)
print("ROC AUC Score:", roc_auc)


y_pred


test_predictions = model.predict(test_df)



output = pd.DataFrame({'id': submission['id'], 'y': test_predictions})
output.to_csv('submission.csv', index=False)

