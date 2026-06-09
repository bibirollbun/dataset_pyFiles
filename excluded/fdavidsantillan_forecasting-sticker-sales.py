import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


Data=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
Data.head(2)


Data['date'] = pd.to_datetime(Data['date'], format='%Y-%m-%d')

#Replace names for shorter ones
Data['store'] = Data['store'].replace({
    'Discount Stickers': 'Discount',
    'Stickers for Less': 'Mid price',
    'Premium Sticker Mart': 'Premium'
})

Data['product'] = Data['product'].replace({
    'Holographic Goose': 'HG',
    'Kaggle': 'Kg',
    'Kaggle Tiers': 'KgT',
    'Kerneler':'Kn',
    'Kerneler Dark Mode':'KnD',
})

df = Data.copy().dropna(ignore_index=True)

columns = ['country','product','store']

# Convert each column in the list to categorical
for col in columns:
    df[col] = df[col].astype('category')


#Generate new features

df['day']  = df['date'].dt.dayofyear
df['week']  = df['date'].dt.isocalendar().week
df['month'] = df['date'].dt.month



import matplotlib.pyplot as plt
import seaborn as sns
def seasonp(col,period):
    plt.figsize=(16, 8)
    df_fil=(df.groupby([period,col])["num_sold"].sum()/df.groupby([period])["num_sold"].sum() ).reset_index()
    sns.lineplot(data=df_fil, x=period, y="num_sold", hue=col)
    plt.tight_layout()
    plt.show()


seasonp('country','date')


#There is a tendency of grow over time exepct for Norway that reduce whit it


seasonp('store','date')


seasonp('product','date')


# Feature engeniering
df['biennial_cycle'] = (df['date'].dt.isocalendar().year % 2)
df['quartile'] = pd.qcut(df['num_sold'], q=5, labels=['Q1', 'Q2', 'Q3','Q4','Q5'])
df['Place_store'] = df['country'].astype(str) + '_' + df['store'].astype(str)
df['Product_store'] = df['product'].astype(str) + '_' + df['store'].astype(str)
df['Product_year']=df['product'].astype(str) + '_' + df['store'].astype(str)

cat_col=['country', 'store', 'product', 'day', 'week', 'month','biennial_cycle',
       'Place_store','Product_store','Product_year']
for col in cat_col:
    df[col]=df[col].astype("category")


!pip install Seq2Pat
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sequential.seq2pat import Seq2Pat
from sequential.pat2feat import Pat2Feat
import time
tim=time.time() 

seq_fil = df.drop(columns=['id','date','num_sold','quartile']).apply(lambda row: row.astype(str).tolist(),axis=1).tolist()
seq2pat = Seq2Pat(sequences=seq_fil)

patterns = seq2pat.get_patterns(min_frequency=0.15)
pat2feat = Pat2Feat()

encodings = pat2feat.get_features(seq_fil, patterns,
                                 drop_pattern_frequency=True)

print(f'There are {len(patterns)} patterns')
print(f'tardo {(time.time()-tim):.2f}')


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_encoded = df.drop(columns=['id','date','num_sold','quartile'])
label_encoders = {}
for column in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le
    
X=df_encoded.apply(lambda row: row.astype(str).tolist(),axis=1).tolist()
X=np.array(X).astype('int')

y = pd.get_dummies(df['quartile'], prefix='', prefix_sep='')
y = np.array(y).astype('int')

X_train_seq, X_test_seq, X_train_pat,X_test_pat,y_train, y_test = train_test_split(X,encodings.drop(columns=['sequence']).values, y, test_size=0.2, random_state=42)


args={}
args['embedding_dim'] = df.drop(columns=['id','date','num_sold','quartile']).nunique().sum()+1
args['num_lstm_units'] = [128, 256]
args['input_len'] = X_train_seq.shape[1]
args['layer_nodes'] = [[256],[128]]

args['NFOLDS'] = 3
args['num_classes'] = 5
args['batch_size'] =1000
args['max_epochs'] = 200
args['early_stop_start'] = 20
args['verbose'] = 0



import keras
import keras.backend as K
from keras import layers
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Concatenate
from keras.models import Model, Sequential
from keras import initializers, regularizers, optimizers, layers
from sklearn.model_selection import KFold
import gc
import keras.backend as K
from sklearn.metrics import roc_auc_score

def pat_LSTM(vocab_size, embedding_dim, num_lstm_units, input_len, num_pattern_features,
                 num_classes, layer_nodes=[512], optimizer='adam'):
    
    input_lstm = Input(shape=(input_len,))
    input_pat_feat = Input(shape=(num_pattern_features))
    
    lstm = Embedding(vocab_size, embedding_dim)(input_lstm)
    output_lstm = LSTM(units=num_lstm_units)(lstm)
    
    merged_nodes = Concatenate(axis=-1)([output_lstm, input_pat_feat])
    
    for i in range(len(layer_nodes)):
        merged_nodes = Dense(layer_nodes[i], activation="sigmoid")(merged_nodes)
        
    output = Dense(num_classes, activation="softmax")(merged_nodes)
    
    model = Model(inputs=[input_lstm, input_pat_feat], outputs=output)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    return model

def keras_categorical(y, num_class):
    return keras.utils.to_categorical(y, num_classes=num_class)

# Customize early stopper
class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', 
                 start_epoch=40, restore_best_weights=False):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose,
                         mode=mode, restore_best_weights=restore_best_weights)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


folds = KFold(n_splits=args['NFOLDS'], shuffle=True, random_state=42)

vocab_size =np.max(X_train_seq)+1 
num_pat_features =(X_train_pat.shape[1],)
tim=time.time() 
results = []

for i in range(len(args['num_lstm_units'])):
    
    for j in range(len(args['layer_nodes'])):

        score = 0
        
        splits = folds.split(X_train_seq,y_train)
        
        for fold_n, (train_index, valid_index) in enumerate(splits):

            X_train_seq_cv, X_valid_seq = X_train_seq[train_index], X_train_seq[valid_index]
            X_train_pat_cv, X_valid_pat = X_train_pat[train_index], X_train_pat[valid_index]
            y_train_cv, y_valid = y_train[train_index], y_train[valid_index]
            
            K.clear_session()
            gc.collect()

            model = pat_LSTM(vocab_size, args['embedding_dim'], 
                             args['num_lstm_units'][i], 
                             args['input_len'], 
                             num_pat_features,
                             args['num_classes'], 
                             layer_nodes=args['layer_nodes'][j])

            early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', 
                                       start_epoch=args['early_stop_start'],
                                       restore_best_weights=True)

            y_train_cv_categorical = y_train_cv#keras_categorical(y_train_cv, args['num_classes'])
            y_valid_categorical = y_valid#keras_categorical(y_valid, args['num_classes'])

            model.fit([X_train_seq_cv, X_train_pat_cv], y_train_cv_categorical, 
                      batch_size=args['batch_size'], 
                      epochs=args['max_epochs'], 
                      validation_data=([X_valid_seq, X_valid_pat], y_valid_categorical),
                      verbose=args['verbose'],
                      callbacks=[early_stop])

            y_pred_valid = model.predict([X_valid_seq, X_valid_pat])

            score += roc_auc_score(y_valid, y_pred_valid, multi_class='ovr', average='macro') / args['NFOLDS']

            del X_train_seq_cv, X_valid_seq, X_train_pat_cv, X_valid_pat
            del y_train_cv, y_train_cv_categorical, y_valid, y_valid_categorical
            del model, y_pred_valid, early_stop
        
        del splits
        
        results.append({'num_lstm_units' : args['num_lstm_units'][i],
                       'layer_nodes' : args['layer_nodes'][j],
                       'score': score})
        print(f'>>> {(time.time()-tim ):.0f} seg later')
        print(f">>> num_lstm_units={args['num_lstm_units'][i]} layer_nodes={args['layer_nodes'][j]} Mean AUC = {score}")
        tim=time.time() 
results_df = pd.DataFrame(results)
results_df.head()

# results_df.to_csv(args['cv_results'], index=False)


best_par=results_df.loc[results_df['score'].idxmax()]

args['final_params'] = {'num_lstm_units': best_par['num_lstm_units'],
                        'layer_nodes': best_par['layer_nodes']}


X_train_final_seq, X_valid_seq,X_train_final_pat,X_valid_pat,y_train_final_categorical, y_valid_categorical = train_test_split(X,encodings.drop(columns=['sequence']).values,y,test_size=0.2,random_state=42)
tim=time.time() 

vocab_size =np.max(X_train_final_seq)+1 
num_pat_features =(X_train_final_pat.shape[1],)

K.clear_session()
gc.collect()

model_LSTM = pat_LSTM(vocab_size, args['embedding_dim'], 
                         int(args['final_params']['num_lstm_units']), 
                         args['input_len'],
                         num_pat_features,
                         args['num_classes'], 
                         layer_nodes=args['final_params']['layer_nodes'])
    
early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', 
                           start_epoch=args['early_stop_start'],
                           restore_best_weights=True)

model_LSTM.fit([X_train_final_seq, X_train_final_pat], y_train_final_categorical, 
          batch_size=args['batch_size'], 
          epochs=args['max_epochs'], 
          validation_data=([X_valid_seq, X_valid_pat], y_valid_categorical),
          verbose=args['verbose'],
          callbacks=[early_stop])
print(f'training ready, it took {(time.time() -tim):.0f} seg')


quartile_pred=model_LSTM.predict([X,encodings.drop(columns=['sequence']).values])


from sklearn.metrics import accuracy_score, log_loss
y = pd.get_dummies(df['quartile'], prefix='', prefix_sep='')
y = np.array(y).astype('int')

log_loss_score_ovr = log_loss(y, quartile_pred)

y_pred_bool =  np.zeros_like(quartile_pred, dtype=bool)
y_pred_bool[np.arange(quartile_pred.shape[0]), quartile_pred.argmax(axis=1)] = True

y_real_bool =  np.zeros_like(y, dtype=bool)
y_real_bool[np.arange(y.shape[0]), y.argmax(axis=1)] = True

accuracy_ovr = accuracy_score(y_real_bool, y_pred_bool)

print(f'Accuracy (OvR): {accuracy_ovr:.4f}')
print(f'Log Loss (OvR): {log_loss_score_ovr:.4f}')



import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import lightgbm as lgb
import numpy as np

tim = time.time()

X=df.drop(columns=['id','date','num_sold'])
y=df['num_sold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log-transform the target variable
y_train = y_train.apply(lambda x: np.log1p(x))

# Parameter grid for RandomizedSearchCV
param_dist = {
    'learning_rate': [0.01, 0.5, 1],   
    'max_depth': [10, 20, 50],  
    'num_leaves': [50, 150, 300],   
    'min_data_in_leaf': [10, 30, 50],  
}

# Initialize LGBMRegressor with GPU support
model = lgb.LGBMRegressor(
    device='gpu',         
    verbose=-1           
)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_dist, 
    n_iter=30,  
    cv=3,  
    scoring='neg_mean_absolute_error', 
    random_state=42,
    n_jobs=-1  
)

eval_set = [(X_test, y_test)]
eval_metric = 'mae' 
callbacks = [lgb.early_stopping(stopping_rounds=10)]  

random_search.fit(X_train, y_train, eval_set=eval_set, eval_metric=eval_metric, callbacks=callbacks)

# Print best score and parameters
print(f'Best score {-random_search.best_score_}, for the parameter: {random_search.best_params_}')
print(f'Training took {(time.time() - tim):.0f} seconds')


train_data = lgb.Dataset(X_train, label=y_train,categorical_feature=cat_col, free_raw_data=False)
valid_data = lgb.Dataset(X_test, label=y_test,categorical_feature=cat_col, free_raw_data=False)

best_params=random_search.best_params_

callbacks=[
        lgb.early_stopping(stopping_rounds=5)
    ]

model_lgb = lgb.train(
    best_params,   
    train_data,   
    valid_sets=[valid_data], 
    callbacks=callbacks,
    num_boost_round=1000
)

y_pred_test = model_lgb.predict(X_test)
y_pred_test=np.expm1(y_pred_test)

y_pred_train= model_lgb.predict(X_train)
y_pred_train=np.expm1(y_pred_train)

from sklearn.metrics import mean_absolute_percentage_error

MAPE_val=mean_absolute_percentage_error(y_pred_test,y_test)

MAPE_train = mean_absolute_percentage_error(y_pred_train,y_train)

print(f'LightGBM Model MAPE validation score: {MAPE_val:.4f}')

print(f'LightGBM Model MAPE training score: {MAPE_train:.4f}')


test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
test.head()

test['store'] = test['store'].replace({
    'Discount Stickers': 'Discount',
    'Stickers for Less': 'Mid price',
    'Premium Sticker Mart': 'Premium'
})

test['product'] = test['product'].replace({
    'Holographic Goose': 'HG',
    'Kaggle': 'Kg',
    'Kaggle Tiers': 'KgT',
    'Kerneler':'Kn',
    'Kerneler Dark Mode':'KnD',
})

def featuring(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['day']  = df['date'].dt.dayofyear
    df['week']  = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['biennial_cycle'] = (df['date'].dt.isocalendar().year % 2)
    df['Place_store'] = df['country'].astype(str) + '_' + df['store'].astype(str)
    df['Product_store'] = df['product'].astype(str) + '_' + df['store'].astype(str)
    df['Product_year']=df['product'].astype(str) + '_' + df['store'].astype(str)

    cat_col=['country', 'store', 'product', 'day', 'week', 'month','biennial_cycle',
       'Place_store','Product_store','Product_year']
    for col in cat_col:
        df[col]=df[col].astype("category")
    return df

test=featuring(test)

seq_fil = test.drop(columns=['id','date']).apply(lambda row: row.astype(str).tolist(),axis=1).tolist()

encodings = pat2feat.get_features(seq_fil, patterns,
                                 drop_pattern_frequency=True)

df_encoded=test.drop(columns=['id','date'])

for column in df_encoded.columns:
    df_encoded[column] = label_encoders[column].transform(df_encoded[column])
    
X=df_encoded.apply(lambda row: row.tolist(),axis=1).tolist()
X=np.array(X)

quartile_pred=model_LSTM.predict([X,encodings.drop(columns=['sequence']).values])

y_pred_bool =  np.zeros_like(quartile_pred, dtype=bool)
y_pred_bool[np.arange(quartile_pred.shape[0]), quartile_pred.argmax(axis=1)] = True

ordinal_categories = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

y_pred_label_encoding = y_pred_bool.argmax(axis=1)

test['quartile'] = pd.Categorical.from_codes(y_pred_label_encoding, categories=ordinal_categories, ordered=True)

test['quartile'] = test['quartile'].astype('category')

X=test.drop(columns=['id','date'])


X.head() 


test_pred=model_lgb.predict(X)
test_pred=np.expm1(test_pred)

df_submission = pd.DataFrame({
    'id': test['id'],  # Mantener la columna 'id'
    'num_sold': test_pred  # Colocar las predicciones en 'num_sold'
})

df_submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv is successfully generated!")


df_submission.head()

