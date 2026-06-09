import pandas as pd
import numpy as np
import pickle as pkl


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test.head()


model = pkl.load(open("/kaggle/input/road-accident-risk-prediction-model/scikitlearn/default/1/Road_Accident_Risk_Prediction_Model.pkl", "rb"))


pred = model.predict(test)


submission = {"id":test["id"], "accident_risk":pred}


submission = pd.DataFrame(submission)
submission.head()


submission.to_csv("submission.csv", index=False)

