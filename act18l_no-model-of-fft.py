# VERSION
VER=2
# WHICH OF 5 PRODUCTS 0,1,2,3,4 TO TRAIN AND PREDICT
PROD = 4
# LENGTH OF TRAIN FEATURES
LEN = 1440

# TRAIN OR LOAD MODEL
TRAIN_MODEL = True
PATH = "/kaggle/input/kaggle-sticker-comp-sub-v1/"
USE_INTERNET = False


import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train.date = pd.to_datetime(train.date)
print("Train shape:", train.shape )
train.head()


import requests 
def get_gdp_per_capita(alpha3, year):
    url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
    response = requests.get(url.format(alpha3,year)).json()
    return response[1][0]['value']

alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
train['alpha3'] = train['country'].map(dict(zip(
    np.sort(train['country'].unique()), alpha3s)))
years = np.sort(train['date'].dt.year.unique())
train['year'] = train['date'].dt.year
if USE_INTERNET:
    gdp = np.array([
        [get_gdp_per_capita(alpha3, year) for year in years]
        for alpha3 in alpha3s
    ])
    gdp = pd.DataFrame(gdp, index=alpha3s, columns=years)
else:
    gdp = pd.read_csv(f"{PATH}gdp0.csv")
    gdp = gdp.set_index("Unnamed: 0")
    gdp = gdp.rename(columns=lambda x: int(x))
train['GDP'] = train.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)
train = train.drop(['alpha3','year'],axis=1)
train.head()


train["num_sold"] /= train["GDP"]
store_ratio = train.groupby("store").num_sold.mean().to_dict()
train["store_ratio"] = train.store.map(store_ratio)
train["num_sold"] /= train["store_ratio"]


C = list( train.country.unique() )
S = list( train.store.unique() )
P = list( train["product"].unique() )
print("Countries:", C )
print("Stores:", S )
print("Products:", P)

# DATA IS PRODUCT X 7 YEARS X STORE+COUNTRY
data = np.zeros( (5,2557,18) )
for i in range(5):
    for j in range(3):
        for k in range(6):
            f = 1 
            if k==3: f=1.15 # FUDGE FACTOR FOR KENYA
            df = train.loc[(train.country==C[k])&(train.store==S[j])&(train["product"]==P[i])].copy()
            data[i,:,j*6+k] = df["num_sold"].values*f

# COMPUTE MEANS AND STDS
means = {}; stds = {}
for k in range(5):
    m = np.nanmean( data[k,:,:] )
    s = np.nanstd( data[k,:,:] )
    means[k]=m; stds[k]=s

# PLOT ALL TIME SERIES DATA
for i in range(5):
    plt.figure(figsize=(10,5))
    for j in range(3):
        for k in range(6):
            f = 1
            if k==3: f=1.15
            df = train.loc[(train.country==C[k])&(train.store==S[j])&(train["product"]==P[i])].copy()
            df["smooth_sold"] = df["num_sold"].rolling(window=180).mean()
            m = means[i]; s = stds[i]
            plt.plot(df["date"], (df["smooth_sold"]*f-m)/s )
            data[i,:,j*6+k] = (df["num_sold"].values*f-m)/s
    plt.title(f"Product = {P[i]}, All 18 Country Store Pairs smoothed and standardized.",size=10)
    plt.show()


import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, shuffle=False, product=0, f_length=768, t_length=32): 

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.product = product
        self.f_length = f_length
        self.t_length = t_length
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = int(np.ceil(32*1024/self.batch_size))
        return ct

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( 32*1024 )
        if self.shuffle: np.random.shuffle(self.indexes)
                        
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
    
        SIZE = self.f_length
        TARGET = self.t_length
        X = np.zeros((len(indexes),SIZE,1),dtype='float32')
        y = np.zeros((len(indexes),TARGET),dtype='float32')
        
        for k in range(len(indexes)):
            r = np.random.randint(0,self.data.shape[2])
            a = np.random.randint(0,self.data.shape[1]-SIZE-TARGET)
            X[k,:,0] = self.data[self.product,a:a+SIZE,r]
            y[k,:] = self.data[self.product,a+SIZE:a+SIZE+TARGET,r]
        return np.nan_to_num(X),np.nan_to_num(y)


# DISPLAY DATA LOADER
gen = DataGenerator(data, shuffle=False, f_length=LEN, product=PROD)
for x,y in gen:
    for k in range(4):
        plt.figure(figsize=(20,5))
        LN = x.shape[1]
        LN2 = y.shape[1]
        plt.plot(np.arange(LN),x[k,:,0],label='features')
        plt.plot(np.arange(LN2)+LN,y[k,:],label='target')
        plt.legend()
        plt.title(f"Product = {P[PROD]}. Sample Dataloader.",size=14)
        plt.show()
    break


gen = DataGenerator(data, shuffle=False, f_length=LEN, product=PROD)
for x,y in gen:
    plt.figure(figsize=(20, 10))
    plt.subplot(2,1,1)
    fft_result = np.fft.rfft(x[0].squeeze(axis=-1))
    plt.plot( np.abs(fft_result))
    plt.axvline(x=206, color='r', linestyle='--', label='x = 3') 
    plt.axvline(x=411, color='r', linestyle='--', label='x = 3')
    plt.axvline(x=617, color='r', linestyle='--', label='x = 3')
    plt.title('Fourier Transform')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    x = np.concatenate((x.squeeze(axis=-1), y), axis=1)
    x= x[0]

    fft_result = np.fft.rfft(x)
    plt.subplot(2,1,2)
    plt.plot( np.abs(fft_result))
    plt.axvline(x=210, color='r', linestyle='--', label='x = 3') 
    plt.axvline(x=421, color='r', linestyle='--', label='x = 3')
    plt.axvline(x=631, color='r', linestyle='--', label='x = 3')
    plt.title('Fourier Transform')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    break


# TRAIN SCHEDULE
def lrfn(epoch):
        return [1e-3,1e-3,1e-4,1e-4,1e-5][epoch]
LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
EPOCHS = 1


def apply_mask(x,k=10):
    seq_len = tf.shape(x)[2]
    indices = tf.range(seq_len, dtype=tf.int32) 
    
    cond1 = tf.logical_and(indices >= 0, indices <= 7)
    cond2 = tf.logical_and(indices >= 206-k, indices <= 206+k)
    cond3 = tf.logical_and(indices >= 411-k, indices <= 411+k)
    cond4 = tf.logical_and(indices >= 617-k, indices <= 617+k)
    mask = tf.logical_or(cond1, cond2)
    mask = tf.logical_or(mask,cond3)
    mask = tf.logical_or(mask,cond4)
    
    mask = tf.cast(mask, x.dtype)
    mask = tf.reshape(mask, [1, 1, -1])
    
    return x * mask

def fft_shift(x,k=10):
    elements_to_move = x[:, :, 206-k:206+k+1]
    tensor_without_elements = tf.concat([
        x[:, :, :206-k],
        x[:, :, 206+k+1:] 
    ], axis=2)
    new_start_idx_in_new_tensor = 210-k - (2*k+1)
    new_tensor = tf.concat([
        tensor_without_elements[:, :, :new_start_idx_in_new_tensor],
        elements_to_move,                                           
        tensor_without_elements[:, :, new_start_idx_in_new_tensor:]
    ], axis=2)
    ###
    elements_to_move = new_tensor[:, :, 411-k:411+k+1]
    tensor_without_elements = tf.concat([
        new_tensor[:, :, :411-k],
        new_tensor[:, :, 411+k+1:] 
    ], axis=2)
    new_start_idx_in_new_tensor = 421-k - (2*k+1)
    new_tensor = tf.concat([
        tensor_without_elements[:, :, :new_start_idx_in_new_tensor],
        elements_to_move,                                           
        tensor_without_elements[:, :, new_start_idx_in_new_tensor:]
    ], axis=2)
    ###
    elements_to_move = new_tensor[:, :, 617-k:617+k+1]
    tensor_without_elements = tf.concat([
        new_tensor[:, :, :617-k],
        new_tensor[:, :, 617+k+1:] 
    ], axis=2)
    new_start_idx_in_new_tensor = 631-k - (2*k+1)
    new_tensor = tf.concat([
        tensor_without_elements[:, :, :new_start_idx_in_new_tensor],
        elements_to_move,                                           
        tensor_without_elements[:, :, new_start_idx_in_new_tensor:]
    ], axis=2)
    return new_tensor



import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np

class NoModel(Model):
    def __init__(self):
        super(NoModel, self).__init__()
        self.seq_len = 1440
        self.pred_len = 32
        self.channels = 1
        self.cut_freq = 721
        
        self.total_len = self.seq_len + self.pred_len  
        self.length_ratio = self.total_len / self.seq_len
        self.extended_freq = int(self.cut_freq * self.length_ratio)

    def call(self, x):   
        # [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = tf.transpose(x, perm=[0, 2, 1])
        # [batch, channels, fft_bins]
        spectrum = tf.signal.rfft(x)
        spectrum = apply_mask(spectrum, 5)
        spectrum = fft_shift(spectrum, 5)
        full_spectrum = tf.pad(spectrum, paddings=[[0, 0], [0, 0], [0, 737-721]], mode="CONSTANT", constant_values=0)
        reconstruction = tf.signal.irfft(full_spectrum, 
                                       fft_length=[self.total_len])
        
        # [batch, channels, total_len] -> [batch, total_len, channels]
        output = tf.transpose(reconstruction, perm=[0, 2, 1])

        output_scaled = output
        final_prediction = output_scaled[:, -self.pred_len:, :]
        
        return tf.squeeze(final_prediction,axis=-1)



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
print('TensorFlow version =',tf.__version__)

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else: 
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')


# USE MIXED PRECISION
MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')


train_gen = DataGenerator(data, shuffle=True, batch_size=64, f_length=LEN, product=PROD)
valid_gen = DataGenerator(data, shuffle=False, batch_size=128, f_length=LEN, product=PROD)

with strategy.scope():
    model = NoModel()
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
if TRAIN_MODEL:
    model.fit(train_gen, verbose=1,
          validation_data = valid_gen,
          epochs=EPOCHS, callbacks = [LR])
    model.save_weights(f'model_v{VER}_p{PROD}.h5')
else:
    model.load_weights(f'{PATH}model_v{VER}_p{PROD}.h5')


preds = np.zeros((18,32*35))

# ITERATE OVER ALL COMBINATIONS OF COUNTRY AND STORE FOR SPECIFIC PRODUCT
# PREDICT 3 YEARS INTO THE FUTURE

bad_rows = []
for jj in range(18):
    ddd0 = data[PROD:PROD+1,-LEN:,jj:jj+1].copy()
    if np.isnan(ddd0).sum()==LEN:
        bad_rows.append(jj)
    
    pp = []
    for j in range(0,35):
        print(j,", ",end="")
        if j==0: dd2 = ddd0
        else: dd2 = np.concatenate([ddd0[:,32*j:,:]]+[z.reshape((1,32,1)) for z in pp],axis=1) 
        p2 = model.predict( np.nan_to_num(dd2[:,-LEN:,:]) ,verbose=0)
        pp.append(p2)
        if j==34:
            print()
            plt.figure(figsize=(20,5))
            plt.plot(np.arange(LEN), np.nan_to_num(ddd0[0,:,0]) )
            for k in range(j+1):
                plt.plot(np.arange(32)+LEN+32*k,pp[k][0,:])
            cc = C[jj%6]
            ss = S[jj//6]
            plt.title(f"Product={P[PROD]}, Country={cc}, Store={ss}",size=16)
            plt.show()
            
    preds[jj,:] = np.concatenate([z.reshape((1,32,1)) for z in pp],axis=1).flatten() 


# FILLNAN PREDS
FILLNAN = np.nanmean(preds,axis=0)
for r in bad_rows:
    preds[r,:] = FILLNAN

# REVERSE STANDARIZE PREDICTIONS AND FIX KENYA
preds = (preds*stds[PROD])+means[PROD]
for i in [3,9,15]: preds[i,:] = preds[i,:]/1.15


if PROD==0:
    test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
    test.date = pd.to_datetime(test.date)
    test['alpha3'] = test['country'].map(dict(zip(
        np.sort(test['country'].unique()), alpha3s)))
    years = np.sort(test['date'].dt.year.unique())
    test['year'] = test['date'].dt.year
    if USE_INTERNET:
        gdp = np.array([
            [get_gdp_per_capita(alpha3, year) for year in years]
            for alpha3 in alpha3s
        ])
        gdp1 = pd.DataFrame(gdp, index=alpha3s, columns=years)
    else:
        gdp1 = pd.read_csv(f"{PATH}gdp1.csv")
        gdp1 = gdp1.set_index("Unnamed: 0")
        gdp1 = gdp1.rename(columns=lambda x: int(x))
    test['GDP'] = test.apply(lambda s: gdp1.loc[s['alpha3'], s['year']], axis=1)
    test["num_sold"] = 0.0
else:
    test = pd.read_csv(f"test_v{VER}_p{PROD-1}.csv")
    test.date = pd.to_datetime(test.date)


for i in range(3):
    for j in range(6):
        test.loc[(test['product']==P[PROD])&(test.store==S[i])&(test.country==C[j]),'num_sold'] =\
            preds[i*6+j,:1095]
test["store_ratio"] = test.store.map(store_ratio)
test.loc[test['product']==P[PROD],"num_sold"] =\
    test.loc[test['product']==P[PROD],"num_sold"] * test.loc[test['product']==P[PROD],"GDP"]
test.loc[test['product']==P[PROD],"num_sold"] =\
    test.loc[test['product']==P[PROD],"num_sold"] * test.loc[test['product']==P[PROD],"store_ratio"]

print( test.shape )
display( test.head() )


for ss in range(3):
    for cc in range(6):
        df1 = train.loc[(train.country==C[cc])&(train['product']==P[PROD])&(train.store==S[ss])]
        df2 = test.loc[(test.country==C[cc])&(test['product']==P[PROD])&(test.store==S[ss])]

        plt.figure(figsize=(20,5))
        tmp = df1.num_sold * df1.GDP * df1.store_ratio
        tmp.iloc[0] = np.nan_to_num( tmp.iloc[0] )
        plt.plot(np.arange(len(df1)), tmp )
        plt.plot(np.arange(len(df2))+len(df1),df2.num_sold)
        plt.title(f"Product={P[PROD]}, Country={cc}, Store={ss}",size=14)
        plt.show()


# DELETE THIS IF YOU RUN THIS NOTEBOOK 5 TIMES AND MAKE YOUR OWN SUBMISSION.CSV
if PROD!=4:
    os.system(f"cp {PATH}submission_v2.csv submission_v2.csv")


if PROD<4:
    test.to_csv(f"test_v{VER}_p{PROD}.csv",index=False)
    print(f"Saved partial predictions for product {P[PROD]} (PROD = {PROD})")
    print(f"Now run this notebook again with PROD = {PROD+1} to make more predictions.")
else:
    test[['id','num_sold']].to_csv(f"submission_v{VER}.csv",index=False)
    print(f"Wrote submission_v{VER}.csv, now submit to comp!")
test[['id','num_sold']].head()

