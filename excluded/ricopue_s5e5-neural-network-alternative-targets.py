# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import gc, os,random
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold,  StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import time


import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import tensorflow.keras.backend as K
print('TensorFlow version =',tf.__version__)


class CFG:
    SEED  = 42
    FOLDS = 5
    verbose=0
    epochs=100
    
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(seed=CFG.SEED)


train=pd.read_csv('//kaggle/input/playground-series-s5e5/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train_ori=pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')
train_ori = train_ori.rename({"Gender":"Sex"},axis=1)
train_ori = train_ori.rename({"User_ID":"id"},axis=1)


print("#"*25)
print("Data Info:")
print("#"*25)
train.info()

print('\n')
print("#"*25)
print("Features Description:")
print("#"*25)
display(train.describe())

print('\n')
print("#"*25)
print("Unique Elements per Feature:")
print("#"*25)
train.nunique()


train_ori_tmp_0=train_ori.copy()
train_ori_tmp_1=train_ori.copy()
for k in [.99,1.01]:   
    train_ori_tmp_0['Calories']=train_ori_tmp_1['Calories']*k
    train_ori_tmp_0['Body_Temp']=train_ori_tmp_1['Body_Temp']*k
    train_ori_tmp_0['Calories']=train_ori_tmp_1['Calories']*k
    train_ori = pd.concat([train_ori,train_ori_tmp_0],axis=0)  

train_ori['id'] = np.arange( len(train_ori) ) + 1_000_000
train_ori['Calories']=np.clip(train_ori['Calories'], 1, 314)
train_ori['is_train']=2
del train_ori_tmp_0,train_ori_tmp_1


test['Calories']=0
train['is_train']=1
test['is_train']=0
df_combi=pd.concat((train,test,train_ori)).reset_index(drop=True)

sex_map={'male':0,'female':1}
df_combi['Sex']=df_combi['Sex'].map(sex_map).astype('uint8')
df_combi['is_train']=df_combi['is_train'].astype('uint8')
df_combi['Age']=df_combi['Age'].astype('float32')
df_combi['Height']=df_combi['Height'].astype('float32')
df_combi['Weight']=df_combi['Weight'].astype('float32')
df_combi['Duration']=df_combi['Duration'].astype('float32')
df_combi['Duration_ori']=df_combi['Duration'].astype('float32')
df_combi['Heart_Rate']=df_combi['Heart_Rate'].astype('float32')
df_combi['Calories']=df_combi['Calories'].astype('float32')
df_combi['Body_Temp']=df_combi['Body_Temp'].astype('float32')

df_combi['BMR']=0
df_combi.loc[df_combi.Sex==0,'BMR'] = df_combi['Weight'] * 9.65 + (df_combi['Height'] / 100) * 573 - df_combi['Age'] * 5.08 + 260
df_combi.loc[df_combi.Sex==1,'BMR'] = df_combi['Weight'] * 7.38 + (df_combi['Height'] / 100) * 607 - df_combi['Age'] * 2.31 + 43
df_combi['BMR']=df_combi['BMR'].astype('float32')

df_combi['Intensity'] = df_combi['Heart_Rate']/np.log(df_combi.Duration+10)
df_combi['Intensity'] = df_combi['Intensity'].astype('float32')
df_combi['Intensity_ori'] = df_combi['Intensity'].astype('float32')


df_combi['Calories_log']=np.log1p(df_combi['Calories'])
dd=df_combi[:len(train)]
sns.histplot(data=dd, x='Calories_log', hue='Sex', kde=True, bins=30, multiple='layer')
plt.title(f'Distribution of Calories_log by Sex')
plt.xlabel('Calories_log')
plt.ylabel('Frequency')
print('\nCalories_log skew ' ,df_combi[:len(train)]['Calories_log'].skew())


df_combi['Calories_Minute']=df_combi['Calories']/df_combi['Duration']
df_combi['Calories_Minute']=df_combi['Calories_Minute'].astype('float32')

dd=df_combi[:len(train)]
h_clip=len(dd[dd.Calories_Minute >11])
l_clip=len(dd[dd.Calories_Minute <0.1])
dd=dd[dd.Calories_Minute <12]
sns.histplot(data=dd, x='Calories_Minute', hue='Sex', kde=True, bins=30, multiple='layer')
plt.title(f'Distribution of Calories_Minute by Sex')
plt.xlabel('Calories_Minute')
plt.ylabel('Frequency')

print('\nCalories_Minute Median:',dd['Calories_Minute'].median())
print('Num low Clip rows (Calories_Minute < 0.1): ',l_clip)
print('Num hight Clip rows (Calories_Minute > 11):',h_clip)


print('\nCalories_Minute skew before clip' ,df_combi[:len(train)]['Calories_Minute'].skew())
df_combi['Calories_Minute_c'] = np.clip(df_combi['Calories_Minute'], 0.1, 11)
print('Calories_Minute skew after clip',df_combi[:len(train)]['Calories_Minute_c'].skew(),'\n')


df_combi['Calories_Minute_log']=np.log(df_combi['Calories']/np.sqrt((df_combi['Duration'])))
df_combi['Calories_Minute_log']=df_combi['Calories_Minute_log'].astype('float32')

dd=df_combi[:len(train)]
h_clip=len(dd[dd.Calories_Minute_log >4.5])
l_clip=len(dd[dd.Calories_Minute_log <-1])
dd=dd[dd.Calories_Minute_log <4.5]
sns.histplot(data=dd, x='Calories_Minute_log', hue='Sex', kde=True, bins=30, multiple='layer')
plt.title(f'Distribution of Calories_Minute by Sex')
plt.xlabel('Calories_Minute_log')
plt.ylabel('Frequency')

print('\nCalories_Minute Median:',dd['Calories_Minute_log'].median())
print('Num low Clip rows (Calories_Minute_log < 0.1): ',l_clip)
print('Num hight Clip rows (Calories_Minute_log > 11):',h_clip)


print('\nCalories_Minute skew before clip' ,df_combi[:len(train)]['Calories_Minute_log'].skew())
df_combi['Calories_Minute_log_c'] = np.clip(df_combi['Calories_Minute_log'], 0, 4.5)
print('Calories_Minute_log skew after clip',df_combi[:len(train)]['Calories_Minute_log_c'].skew(),'\n')


Encode_cols=['Height','Weight']

FOLDS=5
train=df_combi[:len(train)]
for col in Encode_cols: 
    df_combi[f'{col}_Encode']=0
    for tr_idx, tst_idx in KFold(n_splits=FOLDS, shuffle=True, random_state=42).split(train):
        Target_Encoder = TargetEncoder(cols=col)        
        Target_Encoder.fit(train.loc[tr_idx, Encode_cols],train.loc[tr_idx, 'Heart_Rate'])
        col_df=Target_Encoder.transform(df_combi[Encode_cols])[col]/FOLDS
        df_combi[f'{col}_Encode']= df_combi[f'{col}_Encode']+col_df  
        
    df_combi[f'{col}_Encode']=df_combi[f'{col}_Encode'].astype('float32')

#To help NN some skew features and converted with log.
skew_cols=['Age','Body_Temp','Weight_Encode']
df_combi[skew_cols]=np.log(df_combi[skew_cols])

Num_cols=['Sex','Age', 'Heart_Rate','Intensity','Body_Temp', 'Height_Encode','BMR','Weight_Encode','Duration','Weight','Height']

#Data scale to help NN.
scaler = StandardScaler()
df_combi[Num_cols] = scaler.fit_transform(df_combi[Num_cols])    
        
df_combi.info()

train=df_combi[df_combi.is_train==1]
test=df_combi[df_combi.is_train==0]
train_ori=df_combi[df_combi.is_train==2]


features_keras=['Sex', 'Age','Intensity','Body_Temp','Heart_Rate','Duration','Height_Encode','BMR', 'Weight_Encode' ]
targets=['Calories_Minute']


#Simple NN https://www.kaggle.com/code/cdeotte/nn-mlp-starter-cv-0-0608
def base_network():
    x_input_num= tf.keras.Input(shape=(len(features_keras),)) 
    x = tf.keras.layers.Dense(32, activation='swish')(x_input_num)      
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    x = tf.keras.layers.Dense(32, activation='swish')(x) 
    x_otput= tf.keras.layers.Dense(len(targets), activation='linear',name='target')(x) 

    model = tf.keras.Model(x_input_num,x_otput)
    
    return model


model = base_network() 
model.summary()


tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
from IPython.display import Image
Image(retina=True, filename='model.png')


bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='kmeans')
hrd_bins = bins.fit_transform(train[['Intensity']]).astype(int).flatten()
kf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)


features_keras=['Sex', 'Age','Intensity','Body_Temp','Heart_Rate','Duration','Height_Encode','BMR', 'Weight_Encode' ]

for target in ['Calories_log','Calories_Minute_log','Calories_Minute']:

    oof_xgb = np.zeros(len(train))
    pred_xgb = np.zeros(len(test))

    for i, (train_index, valid_index) in enumerate(kf.split(train,hrd_bins)):
    
        tf.keras.backend.clear_session()
        model=base_network()
        
        if target == 'Calories_log':
            batch_size=256            
            monitor='val_root_mean_squared_error'
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                          loss='mse', 
                          metrics=[tf.keras.metrics.RootMeanSquaredError()])

        if target == 'Calories_Minute_log':
            batch_size=256            
            monitor='val_root_mean_squared_error'
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-3),
                          loss='mse', 
                          metrics=[tf.keras.metrics.RootMeanSquaredError()])
            
        if target == 'Calories_Minute':
            batch_size=256
            monitor='val_mean_squared_logarithmic_error'
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=4e-3),
                          loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                          metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])

        print("#"*25)
        print(f"### Fold {i+1} for {target}")
        #print("#"*25)

        X_tr = train.loc[train_index,features_keras].copy()
        y_tr = train.loc[train_index,target]
        
        X_val = train.loc[valid_index,features_keras].copy()
        y_val = train.loc[valid_index,target]

        X_tr=pd.concat((X_tr,train_ori[features_keras]), axis=0)
        y_tr=pd.concat((y_tr,train_ori[target]),axis=0)   


        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,     
            factor=0.8,              
            patience=2,              
            verbose=0,
            min_lr=1e-6,
            mode="min")
    
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, 
            patience=10,            
            restore_best_weights=True,
            mode="min",
            verbose=1)

        history = model.fit(X_tr,y_tr,validation_data=(X_val,y_val),callbacks=[lr_callback,early_stop],verbose=CFG.verbose,epochs=CFG.epochs,batch_size=batch_size)

        A=train.loc[valid_index,'Duration_ori']
        oof_xgb[valid_index]=model.predict(X_val,batch_size=1024*2,verbose=0).ravel()
        pred_xgb+=model.predict(test[features_keras],batch_size=1024*2,verbose=0).ravel()/CFG.FOLDS  


    if target == 'Calories_log':
        oof_xgb=np.expm1(oof_xgb)
        oof_xgb=np.clip(oof_xgb, 1, 314)
        train['preds_Calories_log']=oof_xgb

        pred_xgb=np.expm1(pred_xgb)
        pred_xgb=np.clip(pred_xgb, 1, 314)
        test['test_Calories_log']=pred_xgb

    if target == 'Calories_Minute_log':
        oof_xgb=np.exp(oof_xgb)*np.sqrt(train.Duration_ori.values)
        oof_xgb=np.clip(oof_xgb, 1, 314)
        train['preds_Calories_Minute_log']=oof_xgb

        pred_xgb=np.exp(pred_xgb)*np.sqrt(test.Duration_ori.values)
        pred_xgb=np.clip(pred_xgb, 1, 314)
        test['test_Calories_Minute_log']=pred_xgb  

    if target == 'Calories_Minute':
        oof_xgb=oof_xgb*train.Duration_ori.values
        oof_xgb=np.clip(oof_xgb, 1, 314)
        train['preds_Calories_Minute']=oof_xgb

        pred_xgb=pred_xgb*test.Duration_ori.values
        pred_xgb=np.clip(pred_xgb, 1, 314)
        test['test_Calories_Minute']=pred_xgb

    print("#"*25)
    print(f' score for {target}: ',np.sqrt(mean_squared_log_error(train.Calories,oof_xgb)))
    print("\n")


print('preds_Calories_Minute_log: ',np.sqrt(mean_squared_log_error(train.Calories,train.preds_Calories_Minute_log)))
print('preds_Calories_log: ',np.sqrt(mean_squared_log_error(train.Calories,train.preds_Calories_log)))
print('preds_Calories_Minute: ',np.sqrt(mean_squared_log_error(train.Calories,train.preds_Calories_Minute)))
A=train.preds_Calories_Minute_log*.5
B=train.preds_Calories_log*.25
C=train.preds_Calories_Minute*.25
print('preds_Calories_All: ',np.sqrt(mean_squared_log_error(train.Calories,A+B+C)))

train['train_preds']=A+B+C
test['test_preds']=test.test_Calories_Minute_log.values*.5+test.test_Calories_log.values*.25+test.test_Calories_Minute.values*.25


def NumPySLE(y_true: list, y_pred: list) -> float:
    n = len(y_true)
    sle =np.square(np.log1p(y_pred) - np.log1p(y_true))
    return sle


train['Calories_Minute_log_err']=NumPySLE(train.Calories.values,train.preds_Calories_Minute_log.values)
train['Calories_log_err']=NumPySLE(train.Calories.values,train.preds_Calories_log.values)
train['Calories_Minute_err']=NumPySLE(train.Calories.values,train.preds_Calories_Minute.values)


plot_cols = ['Age','Heart_Rate','Body_Temp','Duration','Sex']

for col in plot_cols:
    
    plt.figure(figsize=(20, 5)) 

    if col != 'Sex':
        plt.subplot(1, 4, 1)
        data = np.sqrt(train.groupby([col,'Sex'])[['Calories_Minute_log_err']].mean())
        sns.scatterplot(x=col, y='Calories_Minute_log_err', data=data,  hue='Sex')
        plt.title(f'Calories_Minute_log_xgb_err  Vs {col}')
        plt.ylabel('Calories_Minute_log_err')

        plt.subplot(1, 4, 2)
        data = np.sqrt(train.groupby([col,'Sex'])[['Calories_log_err']].mean())
        sns.scatterplot(x=col, y='Calories_log_err', data=data,hue='Sex')
        plt.title(f'Calories_log_xgb_err Vs {col}')
        plt.ylabel('Calories_log_err')


        plt.subplot(1, 4, 3)
        data = np.sqrt(train.groupby([col,'Sex'])[['Calories_Minute_err']].mean())
        sns.scatterplot(x=col, y='Calories_Minute_err', data=data,hue='Sex')
        plt.title(f'Calories_log_cat_err Vs {col}')
        plt.ylabel('Calories_Minute_err')


    if col == 'Sex':
        plt.subplot(1, 4, 1)
        data = np.sqrt(train.groupby('Sex')[['Calories_Minute_log_err']].mean()).reset_index()
        sns.barplot(x=col, y='Calories_Minute_log_err', data=data)
        plt.title(f'Calories_Minute_log_xgb_err Vs Sex')
        plt.ylabel('Calories_Minute_log_xgb_err')

        plt.subplot(1, 4, 2)
        data = np.sqrt(train.groupby('Sex')[['Calories_log_err']].mean()).reset_index()
        sns.barplot(x=col, y='Calories_log_err', data=data)
        plt.title(f'Calories_log_xgb_err Vs Sex')
        plt.ylabel('Calories_log_err')

        plt.subplot(1, 4, 3)
        data = np.sqrt(train.groupby('Sex')[['Calories_Minute_err']].mean()).reset_index()
        sns.barplot(x=col, y='Calories_Minute_err', data=data)
        plt.title(f'Calories_log_cat_err Vs Sex')
        plt.ylabel('Calories_Minute_err') 



    plt.tight_layout()
    plt.show()


train['train_error']=NumPySLE(train.Calories.values,train.train_preds.values)
train.loc[train.Calories>train.train_preds,'train_error']=train.train_error*-1.
train['train_error_abs']=abs(train['train_error'])
print(train.train_error.max())
print(train.train_error.min())
print(train.train_error.median())


train['Calories_Minute_c']=round(train['Calories_Minute_c'],2)

plt.figure(figsize=(20, 5)) 
data = train.groupby(['Calories_Minute_c','Sex'])[['train_error_abs']].mean()
sns.scatterplot(x='Calories_Minute_c', y='train_error_abs', data=data,  hue='Sex')

plt.figure(figsize=(20, 5)) 
data = train.groupby(['Calories_Minute_c','Sex'])[['train_error']].mean()
sns.scatterplot(x='Calories_Minute_c', y='train_error', data=data,  hue='Sex')


train['Intensity_c']=round(train['Intensity_ori'],1)

print('Low Intensity rows:', train[train.Intensity_ori<25]['train_error_abs'].count(),' - Error:',train[train.Intensity_ori<25]['train_error_abs'].mean())
print('High Intensity rows:',train[train.Intensity_ori>37]['train_error_abs'].count(),'- Error:',train[train.Intensity_ori>37]['train_error_abs'].mean())
print('Intensity tails Error:',train[(train.Intensity_ori<25) | (train.Intensity_ori>37)]['train_error_abs'].mean())
print('Intensity body Error:',train[(train.Intensity_ori>25) & (train.Intensity_ori<37)]['train_error_abs'].mean())

plt.figure(figsize=(20, 5)) 
sns.histplot(data=train, x='Intensity_c', hue='Sex', kde=True, bins=30, multiple='layer')
plt.title(f'Distribution of Intensity by Sex')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.figure(figsize=(20, 5)) 
data = train.groupby(['Intensity_c','Sex'])[['train_error_abs']].mean()
sns.scatterplot(x='Intensity_c', y='train_error_abs', data=data,  hue='Sex')

plt.figure(figsize=(20, 5)) 
data = train.groupby(['Intensity_c','Sex'])[['train_error']].mean()
sns.scatterplot(x='Intensity_c', y='train_error', data=data,  hue='Sex')


train.to_csv('train_keras.csv', index=False)
test.to_csv('test_keras.csv', index=False)


submission=pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories']=test.test_preds.values
submission.to_csv('submission.csv', index=False)
submission.head()

