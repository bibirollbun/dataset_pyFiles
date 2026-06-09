# My first Kaggle Notebook. I tried a simple baseline where females = survived.  




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

train = pd.read_csv("/kaggle/input/titanic-machine-learning-u-lima/train.csv")
test  = pd.read_csv("/kaggle/input/titanic-machine-learning-u-lima/test.csv")

train.head()



# Simple baseline: predict all females survived
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (test["Sex"] == "female").astype(int)
})

submission.to_csv("submission.csv", index=False)
submission.head()


