!pip install mambular -q


import pandas as pd 
from mambular.models import MambularRegressor


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')
target = "Listening_Time_minutes"


# Train a Gradient Boosted Trees model
model = MambularRegressor()

model.fit(train.drop(target, axis=1), train[target], max_epochs=50, lr=1e-03)
y_preds = model.predict(test)

submission[target] = y_preds
submission.to_csv("submission.csv", index=False)
submission

