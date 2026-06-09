import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


# load dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

train_df.head(5)


# check dataset structure and missing values
train_df.info()


# prepare features and labels from dataset
X_features = train_df.drop(['id','rainfall'], axis=1, inplace=False) # remove unnecessary unique column(id) and label(rainfall)
y_target = train_df['rainfall']

# split train sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=0)


# train
lr = LogisticRegression()
lr.fit(X_train, y_train)

# evaluate
pred1 = lr.predict(X_test)
pred_probs1 = lr.predict_proba(X_test)[:,1]

print('accuracy : {0:.4f}'.format(accuracy_score(y_test ,pred1)))
print('ROC-AUC : {0:.4f}'.format(roc_auc_score(y_test, pred_probs1)))


# check dataset structure and missing values
test_df.info()


# prepare data and predict
X_result = test_df.drop(['id'], axis=1, inplace=False)
X_result.fillna(X_result.mean(),inplace=True) # Fill in the null with the average
test_pred = lr.predict_proba(X_result)[:,1]

# make submission file
test_pd = pd.DataFrame({'id':test_df['id'],'rainfall':test_pred})
test_pd.to_csv("submission.csv",index=False)

test_pd.head(10)

