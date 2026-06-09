# import
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma


# load the data
train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra_data = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


all_train_data = pd.concat([train_data, train_extra_data], ignore_index=True)
all_train_data = all_train_data.dropna(subset=['Price'])
all_train_data.head()


all_train_data['log_price'] = np.log1p(all_train_data['Price'])


plt.figure(figsize=(8, 6))
sns.histplot(all_train_data['Price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(all_train_data['log_price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.xlabel('log-Price')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=all_train_data['Price'])
plt.title('Price Boxplot')
plt.xlabel('Price')
plt.show()


all_train_data['Price'].describe()


mean_price = all_train_data['Price'].mean()
std_price = all_train_data['Price'].std()

generated_normal = np.random.normal(mean_price, std_price, size=len(all_train_data))

min_price = all_train_data['Price'].min()
max_price = all_train_data['Price'].max()
generated_uniform = np.random.uniform(min_price, max_price, size=len(all_train_data))

shape, loc, scale = gamma.fit(all_train_data['Price'].dropna())
generated_gamma = gamma.rvs(shape, loc=loc, scale=scale, size=len(all_train_data))

generated_resample = np.random.choice(all_train_data['Price'].dropna(), size=len(all_train_data))

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(generated_normal, bins=30, kde=True, color='skyblue', stat='density')
plt.title('Normal Distribution')

plt.subplot(2, 2, 2)
sns.histplot(generated_uniform, bins=30, kde=True, color='lightgreen', stat='density')
plt.title('Uniform Distribution')

plt.subplot(2, 2, 3)
sns.histplot(generated_gamma, bins=30, kde=True, color='salmon', stat='density')
plt.title('Gamma Distribution')

plt.subplot(2, 2, 4)
sns.histplot(generated_resample, bins=30, kde=True, color='purple', stat='density')
plt.title('Resampling Distribution')

plt.tight_layout()
plt.show()


generated_normal_test = np.random.normal(mean_price, std_price, size=test_data.shape[0])

min_price = all_train_data['Price'].min()
max_price = all_train_data['Price'].max()
generated_uniform_test = np.random.uniform(min_price, max_price, size=test_data.shape[0])

shape, loc, scale = gamma.fit(all_train_data['Price'].dropna())
generated_gamma_test = gamma.rvs(shape, loc=loc, scale=scale, size=test_data.shape[0])

generated_resample_test = np.random.choice(all_train_data['Price'].dropna(), size=test_data.shape[0])



# submission
def submission(test_pred, var_name=0):
    submission = pd.DataFrame({
        'id': test_data['id'],
        'Price': test_pred
    })
    
    submission.to_csv(f"submission_{var_name}.csv", index=False)
    print(f"Submission file saved as submission_{var_name}.csv")

submission(generated_normal_test, 1)
submission(generated_uniform_test, 2)
submission(generated_gamma_test, 3)
submission(generated_resample_test, 4)




