"""
まずはどういうデータを観察しよう
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
!pip install japanize-matplotlib
import japanize_matplotlib


train_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv')
train_df.describe()


!pip install ydata-profiling
import ydata_profiling
ydata_profiling.ProfileReport(train_df)


test_df = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv')
test_df.describe()


ydata_profiling.ProfileReport(test_df)


# datetime と e_usersの推移
x_sampled = train_df['datetime']
y_sampled = train_df['e_users']

num_labels = 1000
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('人')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::100]
plt.xticks(x[::100], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のe_usersの推移')
plt.show()


# datetime と promotion_1の推移
x_sampled = train_df['datetime']
y_sampled = train_df['promotion_1']

num_labels = 1000
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('value')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::100]
plt.xticks(x[::100], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のpromotion_1の推移')
plt.show()


# datetime と promotion_2の推移
x_sampled = train_df['datetime']
y_sampled = train_df['promotion_2']

num_labels = 1000
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('value')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::100]
plt.xticks(x[::100], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のpromotion_2の推移')
plt.show()


# datetime と promotion_3の推移
x_sampled = train_df['datetime']
y_sampled = train_df['promotion_3']

num_labels = 1000
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('value')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::100]
plt.xticks(x[::100], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のpromotion_3の推移')
plt.show()


# datetime と promotion_1の推移
x_sampled = test_df['datetime']
y_sampled = test_df['promotion_1']

num_labels = 125
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('value')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::10]
plt.xticks(x[::10], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のpromotion_1の推移')
plt.show()


# datetime と promotion_2の推移
x_sampled = test_df['datetime']
y_sampled = test_df['promotion_2']

num_labels = 125
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('value')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::10]
plt.xticks(x[::10], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のpromotion_2の推移')
plt.show()


# datetime と promotion_3の推移
x_sampled = test_df['datetime']
y_sampled = test_df['promotion_3']

num_labels = 125
x = x_sampled[::len(x_sampled)//num_labels]
y = y_sampled[::len(y_sampled)//num_labels]

plt.xlabel('年-月-日')
plt.ylabel('value')
plt.plot(x, y)

formatted_labels = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in x]
formatted_labels = formatted_labels[::10]
plt.xticks(x[::10], formatted_labels, rotation=90)
start_time = x_sampled.iloc[0].split(' ')[0]
end_time = x_sampled.iloc[-1].split(' ')[0]
plt.title(f'{start_time} 〜 {end_time} のpromotion_3推移')
plt.show()




