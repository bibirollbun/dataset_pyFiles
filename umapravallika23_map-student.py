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


import pandas as pd
import numpy as np


df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")


df.head()


df.columns


df.info()


df.describe()


df.value_counts()


df.shape


df.head()


df.isnull().sum()


X=df["StudentExplanation"].fillna("")
Y=df["Misconception"].fillna("None")


X,Y


Y.unique()


df.isnull().sum()


from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer=TfidfVectorizer(max_features=5000)
X_vect=vectorizer.fit_transform(X)


from sklearn.preprocessing import LabelEncoder


# Combine Category and Misconception as a single label
df['target_label'] = df['Category'] + ":" + df['Misconception']

le = LabelEncoder()
y_train = le.fit_transform(df['target_label'])



from sklearn.ensemble import RandomForestClassifier


model=RandomForestClassifier(n_estimators=100,random_state=42)


model.fit(X_vect,y_train)


test_df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


test_df.head()


test_df_vect=vectorizer.transform(test_df["StudentExplanation"])


probs = model.predict_proba(test_df_vect)


print(probs)



# Get top 3 predictions for each row
top_n = 3
top_preds_idx = np.argsort(probs, axis=1)[:, -top_n:][:, ::-1]  # top 3 indices
top_preds_idx = top_preds_idx.astype(int)  # convert to int

top_preds_labels = le.inverse_transform(top_preds_idx.flatten()).reshape(probs.shape[0], top_n)


import pandas as pd

# Suppose top_preds_labels is a list of lists of strings like:
# [["True_Correct:NA"], ["False_Neither:NA"], ["False_Misconception:Additive", "True_Correct:NA"]]

submission_list = []

for i, preds in enumerate(top_preds_labels):
    # Ensure each pred is string
    preds_str = " ".join([str(p) for p in preds])
    submission_list.append({
        "row_id": test_df.loc[i, "row_id"],
        "Category:Misconception": preds_str
    })

submission_df = pd.DataFrame(submission_list)
submission_df.to_csv("submission.csv", index=False)




print("submission.csv created successfully!")



















