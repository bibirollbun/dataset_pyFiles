# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers, regularizers



# Loading the training data
df_train = pd.read_csv('../input/higgs-boson/training.zip')





df_train.columns


df_train.info()


df_train.describe()


df_train = df_train.drop(["EventId","Weight"], axis = 1)





df_train['Label'] = df_train['Label'].replace({'s':0 , 'b':1})

# Replacing -999 with nan
df_train.replace(to_replace = -999, value = np.nan, inplace = True)



# Columns with missing values with respective proportions
(df_train.isna().sum()[df_train.isna().sum() > 0] / len(df_train)).sort_values(ascending = False)




# List of columns to remove
columns_to_remove = [
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_lep_eta_centrality", "PRI_jet_subleading_pt", "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi", "PRI_jet_leading_pt", "PRI_jet_leading_eta",
    "PRI_jet_leading_phi"
]

# Drop the columns except 'DER_mass_MMC'
df_train = df_train.drop(columns=columns_to_remove)


# Median imputation
df_train['DER_mass_MMC'] = df_train['DER_mass_MMC'].fillna(df_train['DER_mass_MMC'].median())





df_train.shape


classes = df_train['Label'].unique()
classes_count = df_train["Label"].value_counts()
plt.bar(classes,classes_count,color=["blue",'orange'])
plt.show()






df_train.head()














corr= df_train.corr()
plt.figure(figsize=(20, 16)) 

subset = corr.iloc[:-1,:-1]

sns.heatmap(
    subset, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    annot_kws={"size": 10},  # Adjust annotation font size
    linewidths=0.5  
)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate and align labels
plt.yticks(fontsize=12)  # Adjust y-axis label font size

# Show plot
plt.tight_layout()  # Ensure everything fits without overlapping
plt.show()




mask0 = df_train['Label'] == 0 
mask1 = df_train['Label'] == 1


df_train0 = df_train[mask0] 
df_train1 = df_train[mask1]


print(f"The shape of s data is : {df_train0.shape}")
print(f"The shape of b data is : {df_train1.shape}")





df_train.hist(figsize=(15,15))
plt.tight_layout()
plt.show();


cols = df_train.columns



# Define the number of rows and columns for the subplot grid
n_rows = 4 # Number of rows
n_cols = 5  # Number of columns

# Create a figure and a grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))  # Adjust figsize as needed

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the columns and plot histograms
for i, column in enumerate(cols):
    if i >= n_rows * n_cols:  # Stop if we exceed the number of subplots
        break
    ax = axes[i]
    ax.hist(df_train0[column], label="s", histtype='step', bins=30)
    ax.hist(df_train1[column], label="b", histtype='step', bins=30)
    ax.set_title(column)
    ax.legend()

# Hide any unused subplots
for j in range(i + 1, n_rows * n_cols):
    axes[j].axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()


## Here We are very Sad as the data is not that good, look how overlapped everything is. Sadness is in my eyes. Buts lets cut off few parameters which are totally the same and remove them and work on other. 


df_train.columns


columns_we_want = ['DER_mass_MMC','DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltar_tau_lep', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt',
       'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt','PRI_lep_phi', 'PRI_met_phi', 'PRI_met_sumet','PRI_jet_all_pt']
    


df_train_final = df_train[columns_we_want]
Y=df_train['Label']
df_train_final


X_train,X_test,Y_train,Y_test = train_test_split(df_train_final,Y,test_size=0.4,random_state=0)




scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test =  scaler.transform(X_test)




rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)


prediction = rfc.predict(X_test)

rfc_acc = accuracy_score (Y_test,prediction)

print(f"logistic accuracy is :{rfc_acc}")



cm = confusion_matrix(Y_test,prediction)
sns.heatmap(cm,annot= True, cmap='coolwarm',fmt='.2f')




X_train_ten = tf.convert_to_tensor(X_train)
Y_train_ten = tf.convert_to_tensor(Y_train)

X_test_ten = tf.convert_to_tensor(X_test)
Y_test_ten = tf.convert_to_tensor(Y_test)





model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'), 
])


model.compile (optimizer= 'adam' , loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])


model.summary()


model_seq = model.fit (X_train_ten,Y_train_ten,
          epochs= 10 ,
          batch_size = 32,
          validation_data=(X_test_ten,Y_test_ten))


loss, accuracy = model.evaluate(X_test_ten, Y_test_ten)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")



# Plot Training & Validation Loss
plt.plot(model_seq.history['loss'], label='Training Loss')
plt.plot(model_seq.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()


