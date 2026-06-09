import pandas as pd
train_df = pd.read_csv("/kaggle/input/query-intent-detection/train.csv")


train_df


from sklearn.dummy import DummyClassifier
X, y = train_df['query'], train_df['label']
model = DummyClassifier() # "fit" a dummy classifier
# a dummy classifier returns constant predictions
# replace this with your NLP pipeline to carry out the classification
model.fit(X, y)


test_df = pd.read_csv("/kaggle/input/query-intent-detection/test.csv")


X_test = test_df['query']
preds = model.predict(X_test)


subm_df = pd.read_csv("/kaggle/input/query-intent-detection/sample_submission.csv")
subm_df['label'] = preds
subm_df.to_csv("submission.csv", index=False)
pd.read_csv("submission.csv") # verify submission contents


from IPython.display import FileLink
FileLink(r'submission.csv') # download your submission file

