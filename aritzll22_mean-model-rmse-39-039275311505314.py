import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')


train.head()


train.info()


test.info()


train.describe()


train.hist()


# Closer look to the target feature
plt.figure(figsize=(10, 8))

plt.scatter(train.index, train['Price'], alpha=0.01, s=4)

plt.xlabel('Backpacks')
plt.ylabel('Price')
plt.title('Backpack general price')
plt.grid()


# Generate mean modelpredictions
model = pd.Series(np.array([train['Price'].mean()]*300000))
model


# Mean model performance
mean_squared_error(train['Price'], model, squared=False)


# Plot the model
plt.figure(figsize=(10, 8))

plt.scatter(train.index, train['Price'], alpha=0.01, s=4)
plt.plot(train.index, model, c='r', label='Mean model')

plt.xlabel('Backpacks')
plt.ylabel('Price')
plt.title('Backpack general price')
plt.grid()
plt.legend(loc='lower center')
plt.text(90000, 100, 'Mean model (All 81.411107)', fontsize = 15)
plt.text(90000, 90, 'RMSE = 39.039275311505314', fontsize = 15)

plt.savefig('Mean model')


data = {'id': test.index,
        'Price': pd.Series(np.array([train['Price'].mean()]*200000))}

submission_df = pd.DataFrame(data)
submission_df


# Save Submission file
submission_df.to_csv('submission.csv', index=False)

