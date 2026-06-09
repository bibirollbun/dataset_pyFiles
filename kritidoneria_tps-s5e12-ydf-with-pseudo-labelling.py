import ydf


import numpy as np
import pandas as pd


trainoriginal = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


trainsubmission = pd.read_csv('/kaggle/input/ps-s5e12-blend/submission.csv')


#Here is the magic. we use labels from the public submission to label test, and then concat train and train+submission before training


trainoriginal.shape


trainsubmission.shape


test.shape


trainsubmission['id']


trainfinal = test.merge(trainsubmission,on=['id'])
trainfinal.shape


trainfinal.head()





train = pd.concat([trainoriginal,trainfinal])
train.shape


is_test = np.random.rand(len(train)) < 0.1

train_dataset = train[~is_test]
test_dataset = train[is_test]



train.head()



model = ydf.GradientBoostedTreesLearner(label='diagnosed_diabetes',task=ydf.Task.REGRESSION).train(train_dataset)


model.plot_tree() 
# ( this is the magic )


submission1 = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
submission1['diagnosed_diabetes']=model.predict(test).round(2).clip(0,1)
submission1.head()


submission1.to_csv("final_submission.csv", index=False)




