import numpy as np
import pandas as pd

dtest = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
# it is how the original dataset generate targets
# https://www.kaggle.com/code/ianktoo/simulated-road-accident-data-generator
data = dtest
base_risk = (
    0.3 * data["curvature"] + 
    0.2 * (data["lighting"] == "night").astype(int) + 
    0.1 * (data["weather"] != "clear").astype(int) + 
    0.2 * (data["speed_limit"] >= 60).astype(int) + 
    0.1 * (np.array(data["num_reported_accidents"]) > 2).astype(int)
)


sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sub['accident_risk'] = base_risk.values

sub.to_csv('submission.csv', index=False)

