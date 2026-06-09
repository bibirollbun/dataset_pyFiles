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


train=pd.read_csv("/kaggle/input/predict-who-is-more-influential-in-a-social-network/train.csv")
test=pd.read_csv("/kaggle/input/predict-who-is-more-influential-in-a-social-network/test.csv")
train


x_train=train.drop(columns="Choice")
y_train=train['Choice']




from sklearn.linear_model import LogisticRegression


model=LogisticRegression(max_iter=6000).fit(x_train,y_train)


y_predict=model.predict(test)


print(pd.Series(y_predict))


submission={
    "Id":range(len(y_predict)),
    "Choice":y_predict
}



submission=pd.DataFrame(submission)
submission


# 1. Pehle Sample file load karein taake humein sahi IDs mil jayein
sample_df = pd.read_csv("/kaggle/input/predict-who-is-more-influential-in-a-social-network/sample_predictions.csv")

# 2. Submission Dataframe banayein
submission = pd.DataFrame()

# 3. 'Id' column wesa hi rakhein jesa Sample mein hai (1 se 5952 tak)
submission['Id'] = sample_df['Id']  

# 4. 'Choice' column mein apni predictions dalein
submission['Choice'] = y_predict

# 5. Check karein (Optional printing)
print("First 5 rows check:")
print(submission.head())
print("\nLast 5 rows check:")
print(submission.tail())

# 6. File save karein
submission.to_csv("submission.csv", index=False)




