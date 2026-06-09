import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # for scaling

# plotting
import matplotlib.pyplot as plt
import seaborn as sns


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.shape


train.head()


train.info()


train.isnull().sum()


test.isnull().sum()


train.describe().T


plt.figure(figsize = (14, 8))
sns.heatmap(train.corr(), annot = True)


# histograms

sns.histplot(data = train, x = "maxtemp",kde = True, color = "darkorchid")


sns.histplot(data = train, x = "mintemp",kde = True, color = "hotpink")


sns.histplot(data = train, x = "dewpoint",kde = True, color = 'steelblue')


# note that temperature is wrongly spelled as 'temparature'

sns.histplot(data = train, x = "temparature",kde = True, color = 'lavender')


columns = train.columns


columns


# let's remove ID, and target variable



variables = [column for column in columns if not column in ['id', 'rainfall']]


variables


# idea from @cdeotte

for c in variables:

    # PLOT TRAIN DISTRIBUTION COMPARED WITH TEST DISTRIBUTION
    custom_palette = ['aquamarine', 'darkorchid'] 
    sns.set_palette(custom_palette)
    plt.figure(figsize=(12,6))
    # plt.subplot(1,2,1)
    sns.distplot(train[c],label='train')
    sns.distplot(test[c],label='test')
    plt.legend()
    plt.title(f"{c}")

    plt.show()


for c in variables:

    # PLOT TARGET RELATIONSHIP WITH BINNED NUMERIC FEATURES
    plt.figure(figsize=(12,3))
    train['bucket'], bin_edges = pd.cut(train[c], bins=10, retbins=True, labels=False)
    bucket_means = train.groupby('bucket')['rainfall'].mean()
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_midpoints, bucket_means, marker='o', linestyle='dashed')
    plt.xlabel(f'{c} (Binned)')
    plt.ylabel('Mean Rainfall')
    plt.title(f'Mean Rainfall per {c} (train)')
    plt.xticks(bin_midpoints, rotation=45)
    plt.grid()
    
    plt.show()


train.drop(columns = ['bucket'], axis = 1)





X_train = train.copy()
X_test = test.copy()

# remove ID column from the dataset

X_train = X_train.drop("id", axis = 1)
X_test = X_test.drop('id', axis = 1)


columns = X_train.columns
columns


# prepare X_train and Y_train


# drop last column (target variable) in X_train to leave only predictors

X_train = X_train.drop("rainfall", axis = 1)


X_train.head()


Y_train = train['rainfall']


Y_train.head()





# add more features (feature engineering)

def feature_engineering(df):
    df = df.copy()

        
    # add time series features
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)

    # add interactions 
    df['hci'] = df['humidity'] * df['cloud']
    df['hsi'] = df['humidity'] * df['sunshine']
    df['csr'] = df['cloud'] / (df['sunshine'] + 1e-5)
    df['rd'] = 100 - df['humidity']
    df['sp'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['wi'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])

    # drop day
    

    
    return df

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)





# dropping 'day' column

X_train = X_train.drop(columns=['day'])
X_test = X_test.drop(columns=['day'])

X_train = X_train.drop(columns = ['bucket'])





X_train.head()



X_test.head()






# scaling the input

scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





from sklearn.model_selection import train_test_split


# divide into train and validation set

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)





X_train.shape


X_test.shape


# reshaping 

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_scaled = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


X_train.shape


X_test.shape


X_test_scaled.shape


model = tf.keras.models.Sequential()

# input layer
model.add(tf.keras.layers.Dense(128, 
                                activation = 'relu', 
                                kernel_initializer = 'he_normal',
                                input_dim = X_train.shape[1]))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))


# second layer
model.add(tf.keras.layers.Dense(256, 
                                kernel_initializer = 'he_normal',
                                activation = 'relu')
         )
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))


# Third layer
model.add(tf.keras.layers.Dense(128, 
                                activation = 'relu', 
                                kernel_initializer = 'he_normal'
                               )
         )
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))



# output layer is only one neuron
# as this is binary classification problem

model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

# compile
model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = 0.001
    ), 
              loss = "binary_crossentropy", 
              metrics = ["accuracy", "AUC"]
)


from keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=20, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.0001)

# set early stopping

early_stopping = EarlyStopping(monitor='val_acc', 
                               patience=10, 
                               mode = 'max',
                               restore_best_weights=True)


history = model.fit(X_train, Y_train,
         epochs = 55,
         batch_size = 64,
         callbacks = [learning_rate_reduction, early_stopping],
         verbose = 1,
            validation_data = (X_val, Y_val)
        
                   )





train_auc = history.history['AUC']
val_auc = history.history['val_AUC']




plt.figure(figsize=(10, 6))
plt.plot(train_auc, label='Training AUC', color='b', lw=2)
plt.plot(val_auc, label='Validation AUC', color='r', lw=2)

plt.title('Training AUC vs Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()


# let's use model to predict

X_test_scaled.shape


X_train.shape


# let's use model to predict

predictions = model.predict(X_test_scaled)


test.head()


submission = pd.DataFrame()

submission['id'] = test['id']
submission["rainfall"] = predictions


# sort NaN value

submission[submission['id'] == 2707]


# change that value to 1

submission.loc[submission['id'] == 2707, 'rainfall'] = 0.5


submission.head()


# write to csv
submission.to_csv("submission.csv", index = False)

