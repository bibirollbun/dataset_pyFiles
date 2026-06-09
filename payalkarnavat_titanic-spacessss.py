# This Python 3 environment comes with many helpful analytics libraries installed
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset_df=pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
print("Full data set shape is {}".format(dataset_df.shape))

dataset_df.describe()
dataset_df.info()

plot_df=dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")

fig,ax=plt.subplots(5,1,figsize=(10,10))
plt.subplots_adjust(top=2)
sns.histplot(dataset_df=["Age"],color='b',bins=50,ax=ax[0])
sns.histplot(dataset_df=["FoodCourt"],color='b',bins=50,ax=ax[1])
sns.histplot(dataset_df=["ShoppingMall"],color='b',bins=50,ax=ax[2])
sns.histplot(dataset_df=["Spa"],color='b',bins=50,ax=ax[3])
sns.histplot(dataset_df=["VRDeck"],color='b',bins=50,ax=ax[4])
dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)
dataset_df.head(5)

dataset_df.isnull().sum().sort_values(ascending=False)
dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = \
    dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)

dataset_df.isnull().sum().sort_values(ascending=False)
label="Transported"
dataset_df[label]=dataset_df[label].astype(int)
dataset_df['VIP']=dataset_df['VIP'].astype(int)
dataset_df['CryoSleep']=dataset_df['CryoSleep'].astype(int)
dataset_df[["Deck","Cabin","Side"]]=dataset_df["Cabin"].str.split("/",expand=True)

try:
    dataset_df = dataset_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

dataset_df.head(5)

def split_dataset(dataset,test_ratio=0.20):
    test_indices=np.random.rand(len(dataset))<test_ratio
    return dataset[~test_indices],dataset[test_indices]
train_ds_pd,valid_ds_pd=split_dataset(dataset_df)

print("{}examples in training,{} examples in testing".format(len(train_ds_pd),len(valid_ds_pd)))
train_ds=tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd,label=label)
valid_ds=tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd,label=label)

rf=tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"])

rf.fit(x=train_ds)
tfdf.model_plotter.plot_model_in_colab(rf,tree_idx=0,max_depth=3)
logs=rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs],[log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy(out of bag)")
plt.show()

rf.fit(x=train_ds)

tfdf.model_plotter.plot_model_in_colab(rf,tree_idx=0,max_depth=3)


inspector = rf.make_inspector()
inspector.evaluation()

evaluation= rf.evaluate(x=valid_ds,return_dict = True)
for name , value in evaluation.items():
    print(f"{name} : {value:.4f}")

print(f"Available  variable importances")
for importance in inspector.variable_importances().keys():
    print("\t",importance)

inspector.variable_importances()["NUM_AS_ROOT"]

test_df=pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
submission_id=test_df.PassengerId
test_df[['VIP','CryoSleep']]=test_df[['VIP','CryoSleep']].fillna(value=0)

test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

test_df['VIP']=test_df['VIP'].astype(int)
test_df['CryoSleep']=test_df['CryoSleep'].astype(int)

test_ds=tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

predictions = rf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})

output.head()

sample_submission_df = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()


import os
os.listdir("/kaggle/input")

import os 
os.listdir("/kaggle/working")




