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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_train.head(5)


def feature_enginnering_for_train(df):
    df=df.dropna()
    df["Stage_fear"]=df["Stage_fear"].apply(lambda x:1 if x=="Yes" else 0)
    df["Drained_after_socializing"]=df["Drained_after_socializing"].apply(lambda x:1 if x=="Yes" else 0)
    df["Personality"]=df["Personality"].apply(lambda x:1 if x=="Extrovert" else 0)
    return df

df_train.info()


df_train_x=feature_enginnering_for_train(df_train)
df_train_x.info()


df_test["Stage_fear"]=df_test["Stage_fear"].apply(lambda x:1 if x=="Yes" else 0)
df_test["Drained_after_socializing"]=df_test["Drained_after_socializing"].apply(lambda x:1 if x=="Yes" else 0)



df_test.head(5)


import numpy as np

def eveluation(column_name, max_data, min_data):
    n_missing = df_test[column_name].isna().sum()
    random_values = np.random.randint(min_data, max_data, size=n_missing)
    return random_values

df_test.loc[df_test["Time_spent_Alone"].isna(), "Time_spent_Alone"] = eveluation("Time_spent_Alone", 12, 0)
df_test.loc[df_test["Social_event_attendance"].isna(), "Social_event_attendance"] = eveluation("Social_event_attendance", 10, 0)
df_test.loc[df_test["Going_outside"].isna(), "Going_outside"] = eveluation("Going_outside", 7, 0)
df_test.loc[df_test["Friends_circle_size"].isna(), "Friends_circle_size"] = eveluation("Friends_circle_size", 15, 0)
df_test.loc[df_test["Post_frequency"].isna(), "Post_frequency"] = eveluation("Post_frequency", 10, 0)



X=df_train_x.drop("Personality",axis=1)
y=df_train_x["Personality"]


from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Get all classifiers
classifiers = [cls for cls in all_estimators(type_filter='classifier') if issubclass(cls[1], ClassifierMixin)]

# Loop through classifiers and evaluate using cross-validation
results = []

for name, Classifier in classifiers:
    try:
        model = Classifier()
        scores = cross_val_score(model, X, y, cv=5)
        results.append((name, scores.mean()))
    except Exception as e:
        results.append((name, f"Error: {str(e)}"))

# Print top 5 performing models
sorted_results = sorted([r for r in results if not isinstance(r[1], str)], key=lambda x: x[1], reverse=True)




for name, score in sorted_results[:5]:
    print(f"{name}: {score:.4f}")


df_test_x=df_test


from sklearn.linear_model import LogisticRegression
predictive_model=LogisticRegression()
predictive_model.fit(X,y)
df_test_x["Personality"]=predictive_model.predict(df_test_x)
df_test_x=df_test_x[["id","Personality"]]
df_test_x["Personality"]=df_test_x["Personality"].apply(lambda x:"Extrovert" if x==1 else "Introvert")
df_test_x.head(5)



Submission = df_test_x
Submission.to_csv("Submission.csv", index=False)










