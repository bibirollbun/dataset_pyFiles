import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc, os,random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import tensorflow as tf
import tensorflow.keras.backend as K
print('TensorFlow version =',tf.__version__)

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:# Invalid device or cannot modify virtual devices once initialized.
        pass    
    print(f'Using {len(gpus)} GPU')
    
else: 
    strategy = tf.distribute.MirroredStrategy()
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except:# Invalid device or cannot modify virtual devices once initialized.
        pass    
    print(f'Using {len(gpus)} GPUs')


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


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.rename(columns={'Temparature':'Temperature','Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer_Name'}, inplace=True)
test.rename(columns={'Temparature':'Temperature','Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer_Name'}, inplace=True)
train.head()


features=['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type','Nitrogen', 'Potassium', 'Phosphorous']
target=['Fertilizer_Name']

Fertilizer_Name_dict_inv=dict(enumerate(train.Fertilizer_Name.unique()))
Fertilizer_Name_dict= {v: k for k, v in Fertilizer_Name_dict_inv.items()}
train['Fertilizer_Name']=train['Fertilizer_Name'].map(Fertilizer_Name_dict)
print(Fertilizer_Name_dict)

Soil_Type_dict=dict(enumerate(train.Soil_Type.unique()))
Soil_Type_dict= {v: k for k, v in Soil_Type_dict.items()}
print('Soil_Type_dict :',Soil_Type_dict)

Crop_Type_dict=dict(enumerate(train.Crop_Type.unique()))
Crop_Type_dict= {v: k for k, v in Crop_Type_dict.items()}
print('Crop_Type_dict :',Crop_Type_dict)



from keras.utils import to_categorical
y_ori=train.Fertilizer_Name.values
y = to_categorical(train.Fertilizer_Name)
Fertilizer_Name_classes = [key for key in Fertilizer_Name_dict]
N_CLASSES = len(Fertilizer_Name_classes)
train.drop(['Fertilizer_Name'], axis=1, inplace=True)


df_combi=pd.concat((train,test)).reset_index(drop=True)
df_combi['Soil_Type']=df_combi['Soil_Type'].map(Soil_Type_dict)
df_combi['Crop_Type']=df_combi['Crop_Type'].map(Crop_Type_dict)
df_combi.drop(['id'], axis=1, inplace=True)

for col in ['Temperature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']:
    df_combi[col]=df_combi[col]-df_combi[col].min()

Binary_cols=[]
for col in features:
    col_max=df_combi[col].max()+1
    b_l=col_max.bit_length()
    bin_cols=[f'Bin_{col}_col_{i}' for i in range(6)]
    print(col_max,b_l,bin_cols)
    df_combi[bin_cols]=np.unpackbits(df_combi[col].values.astype('uint8').byteswap().view('uint8')).reshape(-1,8)[:,2:]
    Binary_cols+=bin_cols

train=df_combi[:len(train)]
test=df_combi[len(train):]


col_bin_num=[col for col in Binary_cols if 'Soil' not in col ]
col_bin_num=[col for col in col_bin_num if 'Crop' not in col ]
col_bin_Soil=[col for col in Binary_cols if 'Soil' in col ]*6
col_bin_Crop=[col for col in Binary_cols if 'Crop' in col ]*6
col_bin=col_bin_Soil+col_bin_num+col_bin_Crop

img_rows, img_cols = 3, 36
channnels=1
binary_df=train[col_bin]
for i in range(3):
    fig = plt.figure()
    plt.imshow(binary_df.iloc[i].values.reshape(img_rows, img_cols))
    plt.title('row_{} target_{} '.format(i,y[i]))
    plt.show()


# Reshape image in 3 dimensions.
def data_prep(raw,channnels):
    num_images = raw.shape[0]
    shaped_array = raw.values.reshape(num_images, img_rows, img_cols, channnels)
    return shaped_array

train_bin=data_prep(train[col_bin],channnels)
test_bin=data_prep(test[col_bin],channnels)


def top_3_accuracy(y_true, y_pred):
    dd=tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)*.5
    dd+=tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)*.17
    dd+=tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)*.33
    return dd


def base_network():    
    input_model = tf.keras.layers.Input(shape=(img_rows,img_cols,channnels))
    x = tf.keras.layers.Conv2D(32, kernel_size=3,padding='same', activation='relu')(input_model)  
    x = tf.keras.layers.Conv2D(32, kernel_size=3,strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='swish')(x)
    x_otput= tf.keras.layers.Dense(N_CLASSES, activation='softmax',name='target')(x)   
    model = tf.keras.Model(input_model,x_otput)
    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
            loss='categorical_crossentropy',
            metrics=[top_3_accuracy])
    
    return model

model=base_network()
model.summary()


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

train_oof = np.zeros((len(train),N_CLASSES))
test_pred= np.zeros((len(test),N_CLASSES))


for i, (train_index, valid_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    X_tr = train_bin[train_index]
    y_tr = y[train_index]
        
    X_val = train_bin[valid_index]
    y_val = y[valid_index]

  
    
    tf.keras.backend.clear_session()
    with strategy.scope():
        model=base_network()


    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_top_3_accuracy',     
            factor=0.5,              
            patience=1,              
            verbose=CFG.verbose,
            min_lr=1e-6,
            mode="max")
    
    early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_top_3_accuracy', 
            patience=5,            
            restore_best_weights=True,
            mode="max",
            verbose=CFG.verbose)

    history = model.fit(X_tr,y_tr,validation_data=(X_val,y_val),verbose=CFG.verbose,epochs=CFG.epochs,batch_size=1024*2,callbacks=[lr_callback,early_stop])
         
    train_oof[valid_index] = model.predict(X_val,verbose=0,batch_size=1024*8)
    test_pred += model.predict(test_bin,verbose=0,batch_size=1024*8)
    
 

oof_map=top_3_accuracy(tf.convert_to_tensor(y, dtype=tf.float32) ,tf.convert_to_tensor(train_oof, dtype=tf.float32)).numpy().mean()
print('The oof map3 score of the keras model (top_3_accuracy): ', oof_map)




top_3_preds = pd.DataFrame(np.argsort(test_pred, axis=1)[:, -3:][:, ::-1] ).replace(Fertilizer_Name_dict_inv)
sample_submission['Fertilizer Name']=top_3_preds.agg(' '.join, axis=1)
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()

