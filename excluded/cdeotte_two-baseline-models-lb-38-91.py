import pandas as pd, numpy as np

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
print("Train shape",train.shape)
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print("Extra Train shape",train_extra.shape)
train = pd.concat([train,train_extra],axis=0,ignore_index=True)
print("Combined Train shape",train.shape)
train.head()


train_mean = train.Price.mean()
train['pred'] = train_mean
s = np.sqrt(np.mean( (train.Price-train.pred)**2.0 ) )
print(f"Validation RMSE using Train Mean = {s}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
print('Submission shape', sub.shape)
sub['Price'] = train_mean
sub.to_csv("submission_mean.csv",index=False)
sub.head()


from cuml.preprocessing import TargetEncoder

TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
train['pred'] = TE.fit_transform(train['Weight Capacity (kg)'],train.Price)
s = np.sqrt(np.mean( (train.Price-train.pred)**2.0 ) )
print(f"Validation RSME using Target Encode Weight Capacity = {s}")


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sub['Price'] = TE.transform(test['Weight Capacity (kg)'])
sub.to_csv("submission_TE_weight_capacity.csv",index=False)
sub.head()

