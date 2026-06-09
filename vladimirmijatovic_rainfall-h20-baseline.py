# import libraries
import numpy as np
import pandas as pd

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import h2o
import h2o
from h2o.automl import H2OAutoML



train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# create already submission file 
submission = pd.DataFrame()

# add id column to submission file
submission['id'] = test_df['id']


train_df.info()


h2o.init() #initialize



train_df = train_df.drop(columns = ['id'])
test_df = test_df.drop(columns = ['id'])





# put train and test into H2OFrame

h2o_train = h2o.H2OFrame(train_df)
h2o_test = h2o.H2OFrame(test_df)


# Change target variable to factor

h2o_train['rainfall'] = h2o_train['rainfall'].asfactor()


splits = h2o_train.split_frame(ratios=[0.75],seed=42)
train = splits[0]
test = splits[1]





y = "rainfall" 
x = h2o_train.columns 
x.remove(y)


aml = H2OAutoML(max_runtime_secs=3600, seed=15, balance_classes=True)
aml.train(x=x,y=y, training_frame=train)


# leaderboard

lb = aml.leaderboard
lb.head()


# get all model_ids
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])


# Get the "All Models" Stacked Ensemble model
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])



# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se.metalearner()['name'])



metalearner.std_coef_plot()



pred = aml.predict(h2o_test)

pred.head()


pred = pred.as_data_frame()


pred.tail()


# add prediction column to submission file 

submission["rainfall"] = pred['p1']


submission.isnull().sum()


submission.shape


# last check of submission.csv
submission.tail()


h2o.shutdown(prompt = False)


# write to csv
submission.to_csv("submission.csv", index = False)

