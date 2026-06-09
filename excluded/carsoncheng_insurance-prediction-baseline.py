import pandas as pd
train_df = pd.read_csv("/kaggle/input/classwork-3-insurance-prediction/train.csv")


train_df.head()
# do some EDA (exploratory data analysis) here


X, y = train_df.drop(columns=['charges']), train_df['charges']
# data splitting...


test_df = pd.read_csv("/kaggle/input/classwork-3-insurance-prediction/test.csv")


test_df.head()


# change the code here
from sklearn.dummy import DummyRegressor
reg = DummyRegressor() # constant prediction predictor to show sklearn format
reg.fit(X, y)
test_preds = reg.predict(test_df)


subm_df = pd.read_csv("/kaggle/input/classwork-3-insurance-prediction/sample_submission.csv")
subm_df.head()


subm_df['charges'] = test_preds
subm_df.to_csv("submission.csv", index=False) # upload this file to submit to competition




