!pip install bluecast -q


import numpy as np 
import pandas as pd 
from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV


train = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e3/sample_submission.csv')
target = "rainfall"


automl = BlueCastCV(class_problem="binary")
automl.fit(train.copy(), target_col=target)
y_probs, y_classes = automl.predict(test)
submission[target] = y_probs
submission.to_csv("submission.csv", index=False)
submission

