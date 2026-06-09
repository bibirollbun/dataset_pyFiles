import pandas as pd
import numpy as np
p1 = pd.read_csv('/kaggle/input/calorie-predictions/submission (1).csv')
p2 = pd.read_csv('/kaggle/input/calorie-predictions/submission.csv')
p3 = pd.read_csv('/kaggle/input/calorie-predictions/pred73.csv')

sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
avg = np.mean([p1.Calories, p2.Calories, p3.Calories], axis = 0)
sub.Calories = avg
sub.to_csv('avg_pred.csv', index = False)

