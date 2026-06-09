import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h2o


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# create already now an empty submission file 
submission = pd.DataFrame()

# dd id column to submission file
submission['id'] = test['id']


train.head()


train.info()


print("Shape of train: ", train.shape)

print("Shape of test: ", test.shape)


train.describe()


h2o.init() #initialize


# remove ID columns, as they are not relevant for modeling

train = train.drop(columns = ['id'])
test = test.drop(columns = ['id'])





# put train and test into H2OFrame

h2o_train = h2o.H2OFrame(train)
h2o_test = h2o.H2OFrame(test)


splits = h2o_train.split_frame(ratios=[0.6],seed=42)
train = splits[0]
test = splits[1]


y = "Calories" 
x = h2o_train.columns 
x.remove(y)




# train AutoML

aml = h2o.automl.H2OAutoML(
    max_runtime_secs = 7200,  # adjust - longer training would bring better results
    seed = 25,
    sort_metric = "RMSLE"  # optimizing for RMSLE metric (on which the competition is scored)

)

aml.train(
    x = x,
    y = y, 
    training_frame = train
)


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


pred = aml.predict(h2o_test).as_data_frame(use_multi_thread=True)

pred.head(5)


# add prediction column to submission file 

submission["Calories"] = pred['predict']


h2o.shutdown(prompt = False)


# write to csv
submission.to_csv("submission.csv", index = False)  # remove index, otherwise submission will fail

