import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")


df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
df_extra=pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


df_extra.columns = df_extra.columns.str.replace(' ', '')
df_extra = df_extra[df_extra.columns].copy()
df_extra['rainfall'] = df_extra['rainfall'].map({'no': 0, 'yes': 1})
df_extra['humidity']=df_extra['humidity'].astype(float)
df_extra['cloud']=df_extra['cloud'].astype(float)
df_train_features=list(df_train)
df_extra=df_extra[df_train_features]


df_train = pd.concat([df_train, df_extra], axis=0, ignore_index=True)


df_train = df_train.drop_duplicates()
df_train.shape


df_train.info()


df_train.isnull().sum()


df_train.fillna(method='ffill', inplace=True)


for col in df_train.columns[:-1]:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_train, y=col, x='rainfall',palette='mako')
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_train, x=col, hue='rainfall', palette='mako',kde=True)
    plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(data=df_train.corr(), annot=True, cmap='flare')


sns.countplot(df_train,x='rainfall',palette='flare')


# from imblearn.over_sampling import SMOTE

# def sampling(df):
#     smote=SMOTE(sampling_strategy='minority')
#     x=df.drop(['rainfall','id','days_cnt'],axis=1)
#     y=df[['rainfall']]
#     x_sm,y_sm=smote.fit_resample(x,y)
#     df= pd.concat([x_sm,y_sm],axis=1)
#     return df

# df_train=sampling(df_train)


# X=df_train.drop(['rainfall','day'],axis=1)
X=df_train[['humidity']]
y=df_train['rainfall']


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_scaled=sc.fit_transform(X)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential([
        Dense(128, activation='relu', kernel_initializer='he_normal',input_shape=(X_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, kernel_initializer='he_normal',activation='relu'),
        Dropout(0.3),
        # Dense(64, kernel_initializer='he_normal',activation='relu'),
        # Dropout(0.3),
        Dense(32, kernel_initializer='he_normal',activation='relu'),
        Dropout(0.2),
        Dense(16, kernel_initializer='he_normal',activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model



model=create_model()
model.summary()


checkpoint_callback = ModelCheckpoint(
    filepath='best_model.keras',  # Filepath to save the model
    monitor='val_loss',        # Metric to monitor ('val_accuracy' or 'val_loss')
    save_best_only=True,       # Save only the best model
    mode='min',                # 'min' for loss, 'max' for accuracy
    verbose=1                  # Print messages when saving
)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model.fit(X_scaled,y, epochs=200, batch_size=32, validation_split=0.2,callbacks=[checkpoint_callback])


# from sklearn.model_selection import KFold

# k=4
# kf = KFold(n_splits=k, shuffle=True, random_state=42)
# accuracy_scores = []

# for train_index, val_index in kf.split(X_train):
#     X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
#     y_train_fold, y_val_fold = y[train_index], y[val_index]
    
#     model = create_model()
#     model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0, validation_data=(X_val_fold, y_val_fold), callbacks=[checkpoint_callback])
    
#     _, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
#     accuracy_scores.append(accuracy)

# print(f'Average accuracy across {k} folds: {np.mean(accuracy_scores):.4f}')



# from tensorflow.keras.optimizers import Adam
# import keras_tuner as kt

# def build_model(hp):
#     model = Sequential()
    
#     model.add(Dense(hp.Int('units_1', min_value=64, max_value=256, step=32), 
#                     activation='relu', input_shape=(X_train.shape[1],)))
#     if hp.Boolean('batch_norm_1'):
#         model.add(BatchNormalization())
#     model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
#     model.add(Dense(hp.Int('units_2', min_value=32, max_value=128, step=16), activation='relu'))
#     if hp.Boolean('batch_norm_2'):
#         model.add(BatchNormalization())
#     model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    
#     model.add(Dense(hp.Int('units_3', min_value=32, max_value=128, step=16), activation='relu'))
#     if hp.Boolean('batch_norm_3'):
#         model.add(BatchNormalization())
#     model.add(Dropout(hp.Float('dropout_3', 0.1, 0.5, step=0.1)))
    
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    
#     model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
    
#     return model

# # Instantiate tuner
# tuner = kt.RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=10,
#     executions_per_trial=2,
#     directory='hyperparam_tuning',
#     project_name='tune_nn_model'
# )

# # Perform hyperparameter tuning
# tuner.search(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32, verbose=1)

# # Get the best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# model = tuner.hypermodel.build(best_hps)
# model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32,callbacks=[checkpoint_callback])


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_scaled,y)


df_test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_test.fillna(method='ffill', inplace=True)
ID=df_test['id']


# df_test_scaled=sc.transform(df_test.drop(['id','day'],axis=1))
df_test_scaled=sc.transform(df_test[['humidity']])


preds=model.predict_proba(df_test_scaled)[:,:1]
preds=preds.ravel()
submission={'id':ID,'rainfall':preds}


submission=pd.DataFrame(submission)
submission.to_csv('submission.csv',index=False)


submission




