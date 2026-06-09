import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install vaex


import vaex


%%time

df = vaex.open("/kaggle/input/ga-customer-revenue-prediction/train_v2.csv")


import pandas as pd


%%time

df_p = pd.read_csv("/kaggle/input/ga-customer-revenue-prediction/train_v2.csv")


df


def extract_revenue(s):
    """Parse transactionRevenue from JSON-like string"""
    if 'transactionRevenue' in s:
        try:
            # Extract value after "transactionRevenue": 
            value_part = s.split('transactionRevenue":')[1].split(',')[0]
            return float(value_part.strip(' "'))  # Remove quotes/whitespace
        except:
            return 0.0
    return 0.0

# Apply function to totals column (lazily)
df['transactionRevenue'] = df.totals.apply(extract_revenue)


import numpy as np

df['tx_log'] = np.log(df.transactionRevenue + 1)

df['visitStartTime_dt'] = df.visitStartTime.astype('datetime64[s]')  # 's' for epoch seconds
df['visit_hour'] = df.visitStartTime_dt.dt.hour  # Extract hour

print("Virtual columns created:")
print(df.tx_log.expression)
print(df.visit_hour.expression)


df['is_high_value'] = df.transactionRevenue > 1000


df_ex = vaex.example()
limits = [-10, 10]

delayed_count = df_ex.count(df_ex.y, binby=df_ex.x, limits=limits, shape=4, delay=True)
delayed_sum = df_ex.sum(df_ex.y, binby=df_ex.x, limits=limits, shape=4, delay=True)

@vaex.delayed
def calculate_mean(sums, counts):
    print('calculating mean')
    return sums / counts

print('before calling mean')
# since calculate_mean is decorated with vaex.delayed
# this now also returns a 'delayed' object (a promise)
delayed_mean = calculate_mean(delayed_sum, delayed_count)

@vaex.delayed
def print_mean(means):
    print('means', means)
print_mean(delayed_mean)

print('before calling execute')
df_ex.execute()  # Execute all delayed computations

# Retrieve the result after execution
means = delayed_mean.get()
print('same means', means)


df_ex.viz.histogram(df_ex.x, limits='99.7%')


import vaex.jupyter
import matplotlib.pyplot as plt


df_ex


vaex.multithreading.thread_count_default = 8
import vaex.ml


df_iris = vaex.datasets.iris()
features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
scaler = vaex.ml.StandardScaler(features=features, prefix='scaled_')
scaler.fit(df_iris)
df_trans = scaler.transform(df_iris)
df_trans

