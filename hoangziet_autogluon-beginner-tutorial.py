


!pip install autogluon


from autogluon.tabular import TabularDataset, TabularPredictor


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


target = "Calories"
df[target] = df[target].astype(np.float32)


LE = LabelEncoder()
df["Sex"] = LE.fit_transform(df["Sex"])
df.drop(columns = ["id"], axis = 1, inplace = True)
df.head()


train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)
train_data.shape, test_data.shape


predictor = TabularPredictor(label = target, problem_type = "regression")


predictor.fit(train_data)


y_pred = predictor.predict(test_data.drop(columns=[target]))


y_test = test_data[target]
print(y_pred[:5])
print(y_test[:5])


predictor.evaluate(test_data)


predictor.leaderboard(test_data)


test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
test_data


submission_dict = {
    "id": test_data["id"],
    target : predictor.predict(test_data.drop(columns = ["id"], axis = 1))
}


submission_df = pd.DataFrame(submission_dict)
submission_df.to_csv("submission.csv", index = False)


submission_df[target] = np.round(submission_df[target])
submission_df.to_csv("submission.csv", index = False)

