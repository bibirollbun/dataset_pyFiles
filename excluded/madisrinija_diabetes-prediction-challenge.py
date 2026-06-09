import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm  as lgb


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



train.head()


test.head()


train.info()


train.isnull().sum()


train.columns


test.columns


cat_cols = ['ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status','gender']

for col in cat_cols:
    encoder = LabelEncoder()
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.transform(test[col])


X = train.drop(['id','diagnosed_diabetes'],axis=1)
Y = train['diagnosed_diabetes']


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.3)


model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)
model.fit(X,Y)


preds = model.predict(test.drop('id',axis=1))


submission = pd.DataFrame({
    'id':test['id'],
    'diagnosed_diabetes': preds
})


submission.to_csv('submission.csv', index=False)







