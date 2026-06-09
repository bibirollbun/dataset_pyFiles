import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# path_train = '/kaggle/input/try-to-calculate-math-expectation/train.csv'
# data_train = pd.read_csv(path_train,sep=";")

# print(data_train.shape)
# data_train.head(10)


path_test = '/kaggle/input/try-to-calculate-math-expectation/test.csv'
data_test = pd.read_csv(path_test,sep=";")

print(data_test.shape)
data_test.head(20)


path_sample = '/kaggle/input/try-to-calculate-math-expectation/sample_submission.csv'
data_sample = pd.read_csv(path_sample)
data_sample.head(20)
# data_sample.to_csv('submission.csv', index=False)




data_test['target_feature']=data_test.iloc[:, 1:17].mean(axis=1).round(2)

result = data_test[['ID', 'target_feature']]
result.columns = ['id', 'target_feature']
print(result)
result.to_csv('submission.csv', index=False)




path_my_sample = '/kaggle/working/submission.csv'
data_my_sample = pd.read_csv(path_my_sample)
data_my_sample

