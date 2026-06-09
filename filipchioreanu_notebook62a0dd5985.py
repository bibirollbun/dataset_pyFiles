import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

train.head()


#train.dropna(inplace=True)
#for col in test:
#    if test[col].dtype == 'object':
        #if test[col].isnull().any():
            #print(test[col])
#        test[col] = test[col].fillna('not listed')
#    if test[col].dtype == 'int' or test[col].dtype == 'float':
#        test[col] = test[col].fillna(-1)
        
plt.figure(figsize=(8, 4))
sns.histplot(train['Price'], kde=True)
plt.title('Price Distribution in Training Data')
plt.xlabel('Price')
plt.ylabel('Count')
plt.grid(True)
plt.show()


mean_price = train['Price'].mean()
print("Mean Price (Baseline Prediction):", mean_price)

baseline_preds = np.full(len(test), mean_price)

submission['Price'] = baseline_preds
print(submission['Price'])
submission.to_csv('baseline_mean_prediction.csv', index=False) 

