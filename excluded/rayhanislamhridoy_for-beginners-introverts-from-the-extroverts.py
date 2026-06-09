import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train.drop(columns=["id"],inplace=True)
test.drop(columns=["id"],inplace=True)


train.head()


test.head()


train.describe()


train.info()


train.isnull().sum()


test.isnull().sum()


# Handling missing values

def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
    return df   


train=handle_missing_values(train)
test=handle_missing_values(test)


# Encode the catagorial value
le=LabelEncoder() 
def encode(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])
    return df


train=encode(train)
test=encode(test)


train.isnull().sum()


test.isnull().sum()


train.head()


test.head()


x_train=train.drop(columns="Personality")
y_train=train["Personality"]
x_test=test 


model=RandomForestClassifier(
    n_estimators=113,
    max_depth=11,
    min_samples_split=6,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)
model.fit(x_train,y_train)


y_predict=model.predict(x_test)


# Map predictions to 'Introvert' and 'Extrovert'
label_map = {1: 'Introvert', 0: 'Extrovert'}
y_pred_labels = [label_map[pred] for pred in y_predict]

# Create submission file
submission = sample.copy()
submission['Personality'] = y_pred_labels
submission.to_csv('submission(9).csv', index=False)




