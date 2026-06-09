!pip install autogluon
!pip install h2o


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.metrics import mean_squared_log_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


train  = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


test.head()


train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
colors = ["red","blue","gold","green","black","gray","purple"]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 16))
axes = axes.flatten()

for idx, (col, color) in enumerate(zip(cols, colors)):
    sns.histplot(train[col], color=color, ax=axes[idx])
    axes[idx].set_title(col)
    
for i in range(len(cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


import h2o
from h2o.automl import H2OAutoML
h2o.init()


train_data = h2o.H2OFrame(train)


test_data = h2o.H2OFrame(test)


model = H2OAutoML(max_runtime_secs=60,seed=100,
    sort_metric="RMSLE",distribution="AUTO",
    nfolds=10,exclude_algos=["DeepLearning"],
    verbosity="info")


y = "Calories"
x = [col for col in train.columns if col != y]


model.train(x=x, y=y, training_frame=train_data)


model


model.leaderboard


model.leader


h2o_predictions = model.leader.predict(test_data)
h2o_predictions_df = h2o_predictions.as_data_frame()
h2o_predictions_df


from autogluon.tabular import TabularPredictor

train  = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


from sklearn.metrics import mean_squared_log_error

label = 'Calories'

predictor = TabularPredictor(label=label, problem_type='regression').fit(train_data=train)


predictions = predictor.predict(test)

y_true_train = train[label]

train_predictions = predictor.predict(train)


rmsle_train = np.sqrt(mean_squared_log_error(y_true_train, train_predictions))
print(f'RMSLE on Training Set: {rmsle_train}')


print(predictions.head())
predictions.to_csv("test_predictions_with_rmsle.csv", index=False)

