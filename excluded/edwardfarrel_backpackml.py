import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_decision_forests as tfdf


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


train_data


num_cols = train_data.select_dtypes(include='number').columns.tolist()

num_plots = len(num_cols)
num_rows = (num_plots + 1) // 2
num_cols_subplot = 2

fig, axes = plt.subplots(num_rows, num_cols_subplot, figsize=(16, 6*num_rows))
fig.suptitle('Boxplots for Numeric Columns', fontsize=20, color='white', y=0.95)

for col, ax in zip(num_cols, axes.flat):
    sns.boxplot(data=train_data, x=col, ax=ax, orient='h', 
                boxprops=dict(facecolor='purple', color='white'),
                whiskerprops=dict(color='white'),
                capprops=dict(color='white'),
                flierprops=dict(markerfacecolor='white', marker='o', markersize=5))
    ax.set_title(f'Boxplot of {col}', fontsize=12, color='white', pad=10) 
    ax.set_xlabel(col, color='white')
    ax.set_ylabel('')

for ax in axes.flat[num_plots:]:
    ax.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


train_data = train_data.drop(columns=['id'])


train_data


train_data.columns


column_to_label = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment',
       'Waterproof']
label_encoders = {} 

for col in column_to_label:
    label_encoders[col] = LabelEncoder()
    train_data[col] = label_encoders[col].fit_transform(train_data[col])


train_data


train_data.columns


column_to_distribution = ['Weight Capacity (kg)', 'Price', 'Compartments']

plt.figure(figsize=(20, 12))
num_columns = 4  
num_rows = (len(column_to_distribution) + num_columns - 1) // num_columns 

for i, column in enumerate(column_to_distribution, 1):
    plt.subplot(num_rows, num_columns, i)  
    sns.histplot(train_data[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()

plt.show()


column_to_scaler = ['Weight Capacity (kg)', 'Price', 'Compartments']
scaler = MinMaxScaler(feature_range=(0,1))
train_data[column_to_scaler] = scaler.fit_transform(train_data[column_to_scaler])


train_data


X = train_data.drop(columns=['Price'])
y = train_data[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=48)


data_train = pd.concat([X_train,y_train],axis=1)
data_test = pd.concat([X_test, y_test], axis=1)


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data_train, task=tfdf.keras.Task.REGRESSION, label='Price')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(data_test, task=tfdf.keras.Task.REGRESSION, label='Price')


model_tree_tensor = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model_tree_tensor.fit(train_ds)


tfdf.model_plotter.plot_model_in_colab(model_tree_tensor, tree_idx=0, max_depth=3)


mse_loss_tensor = tf.keras.losses.MeanSquaredError()
mae_loss_tensor = tf.keras.losses.MeanAbsoluteError()

total_mse = 0
total_mae = 0
total_samples = 0

for batch in test_ds:
    X_batch, y_batch = batch  
    y_pred_batch = model_tree_tensor.predict(X_batch)

    mse_batch = mse_loss_tensor(y_batch, y_pred_batch)
    mae_batch = mae_loss_tensor(y_batch, y_pred_batch)

    total_mse += mse_batch.numpy() * len(y_batch) 
    total_mae += mae_batch.numpy() * len(y_batch)  
    total_samples += len(y_batch)  

average_mse = total_mse / total_samples
average_mae = total_mae / total_samples

print(f"Mean Squared Error: {average_mse}")
print(f"Mean Absolute Error: {average_mae}")


for batch in test_ds:
    X_batch, y_batch = batch  
    y_pred_batch = model_tree_tensor.predict(X_batch)  
    
    price_min = scaler.data_min_[1]  
    price_max = scaler.data_max_[1]
    
    y_pred_batch_orig = y_pred_batch * (price_max - price_min) + price_min
    
    print("Prediksi (skala asli):", y_pred_batch_orig[:5])
    break  


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


test_data


test_data = test_data.drop(columns=['id'])


test_data


features_to_scale = ['Compartments', 'Weight Capacity (kg)']
test_data[features_to_scale] = scaler.fit_transform(test_data[features_to_scale])


test_data.columns


column_to_label_test = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment',
       'Waterproof']
label_encoders = {} 

for col in column_to_label_test:
    label_encoders[col] = LabelEncoder()
    test_data[col] = label_encoders[col].fit_transform(test_data[col])


test_data


test_data.rename(columns={
    "Laptop Compartment": "Laptop_Compartment",
    "Weight Capacity (kg)": "Weight_Capacity_(kg)"
}, inplace=True)


test_ds_submission = tf.data.Dataset.from_tensor_slices(dict(test_data)).batch(32)

predictions = []  

price_min = scaler.data_min_[1]  
price_max = scaler.data_max_[1]

for X_batch in test_ds_submission:
    y_pred_batch = model_tree_tensor.predict(X_batch)
    y_pred_batch_orig = y_pred_batch * (price_max - price_min) + price_min
    predictions.append(y_pred_batch_orig)
    
import numpy as np
predictions = np.concatenate(predictions, axis=0)
predictions = predictions.flatten()

sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

if len(sample_submission) != len(predictions):
    raise ValueError("Jumlah prediksi tidak sesuai dengan jumlah baris di sample submission!")

sample_submission['Price'] = predictions
sample_submission.to_csv("submission.csv", index=False)

print("File submission berhasil disimpan sebagai submission.csv")

