# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

traindata = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
testdata = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv')


traindata.head(10)



traindata.info()
traindata.describe()



num_vars = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights',
            'lead_time','no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room','no_of_special_requests']
fig, axes = plt.subplots(3,3, figsize = (12,6))
for i in range(len(num_vars)):
    sns.boxplot(traindata[num_vars[i]], ax = axes[i//3, i%3])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3,3, figsize = (12,6))
for i in range(len(num_vars)):
    sns.histplot(traindata[num_vars[i]], ax = axes[i//3, i%3])
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
traindata[num_vars] = scaler.fit_transform(traindata[num_vars])
testdata[num_vars] = scaler.transform(testdata[num_vars])


cat_vars = ['type_of_meal_plan', 'room_type_reserved','market_segment_type']
fig, ax = plt.subplots(1,3, figsize = (15,5))
for i in range(len(cat_vars)):
    sns.histplot(data = traindata, x=cat_vars[i], hue='booking_status', bins = traindata[cat_vars[i]].nunique(), ax = ax[i%3])
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
features = traindata.columns[1:-1]
X = traindata[features]
y = traindata['booking_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 123)


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
lrmodel = LogisticRegression()


# Logistic regression with l1 regularisation
selector = RFE(lrmodel)
selector.fit(X_train, y_train)
print("Selected features: ", selector.get_feature_names_out())


y_predlr = selector.predict(X_test)
print("Accuracy score is:", selector.score(X_test, y_test))
cm_logreg = confusion_matrix(y_test, y_predlr)
rep_logreg = classification_report(y_test, y_predlr)
print(rep_logreg)
print("Confusion matrix is:\n", cm_logreg)


from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(max_depth = 10)
rfmodel.fit(X_train, y_train)
feat_importances = rfmodel.feature_importances_ 
sns.barplot(x= feat_importances, y = X_train.columns)
plt.xlabel("Feature Importance score")
plt.ylabel("Features")
plt.show()


y_predrf = rfmodel.predict(X_test)
print("Accuracy score is:", rfmodel.score(X_test, y_test))
cm_randf = confusion_matrix(y_test, y_predrf)
rep_randf = classification_report(y_test, y_predrf)
print(rep_randf)
print("Confusion matrix is:\n", cm_randf)


X_pred = testdata[features]
y_out = rfmodel.predict_proba(X_pred)
submission = pd.DataFrame({'id': testdata['id'], 'booking_status': y_out[:,1]})
submission.to_csv("submission.csv", index=False)

