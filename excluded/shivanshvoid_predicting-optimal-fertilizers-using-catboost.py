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
df=pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head(5)





df.info()





for x,y in df.items():
    print(x)



!pip install catboost


from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df["Fertilizer Name"]=le.fit_transform(df["Fertilizer Name"])




df.head()



x=df.drop(["Fertilizer Name","id"],axis=1)
y=df["Fertilizer Name"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42 ,shuffle=True)






model = CatBoostClassifier(cat_features=["Soil Type", "Crop Type"], verbose=0)
model.fit(x_train,y_train)




from sklearn.metrics import accuracy_score

# Predict
y_pred = model.predict(x_test)

re=accuracy_score(y_test,y_pred)

print("the accuracy of the model is : ",re)






tf=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
tf.head()








import numpy as np

x_test=tf.drop("id",axis=1)
test_ids=tf["id"]
probs = model.predict_proba(x_test)
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
# top3_preds is shape (250000, 3)
# Step 1: Flatten to 1D
flat_preds = top3_preds.flatten()  # shape (750000,)

# Step 2: Convert to original labels
flat_labels = le.inverse_transform(flat_preds)  # shape (750000,)

# Step 3: Reshape back to (250000, 3)
top3_fertilizers = flat_labels.reshape(top3_preds.shape)

top3_joined=[" ".join(row) for row in top3_fertilizers]









re=pd.DataFrame({
    'id':test_ids,
    'Fertilizer Name':top3_joined
})


re.head()

re.to_csv("fertilizer_predictions9.csv", index=False)







