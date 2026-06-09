import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


sample_submission


train


train.info()


for col in train.columns:
    print(train[col].unique())


sample_submission['id']=train['id']
sample_submission['y']=train['y']


x = train.select_dtypes(include=['int64', 'float64']).drop(columns=['y'])


x


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,train['y'])



prediction=model.predict(test.select_dtypes(include=['int64', 'float64']))


submission=pd.DataFrame({'id':test['id'],'y':prediction})


submission.to_csv("submission.csv", index=False)

