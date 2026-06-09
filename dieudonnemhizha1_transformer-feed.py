import pandas as pd
pd.set_option("max_colwidth", None)

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re

import spacy
from spacy import displacy

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import os
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import warnings # Supress warnings
warnings.filterwarnings("ignore")



# Delete train that was modified for EDA purposes and start with a fresh train dataframe
train = pd.read_csv("../input/feedback-prize-effectiveness/train.csv")
test = pd.read_csv("../input/feedback-prize-effectiveness/test.csv")


train["label"] = train["discourse_effectiveness"].replace({"Ineffective": 0, "Adequate": 1, "Effective": 2})


train_df = train[['discourse_text','discourse_type','label']]
test_df = test.drop(['discourse_id','essay_id'],axis=1)


stop_words = stopwords.words('english')
data_cleaning_re = 'https:\S*|http:\S*|[^a-zA-Z0-9]'


stop_words[:10]


# Removing stopwords from texts
def rmv_stopwords(text):
    word_list = text.split()
    tokens =[]
    for word in word_list:
        if word not in stop_words:
            tokens.append(word)
    return " ".join(tokens)


def standardizer(text):
    text = text.lower()
    text = re.sub(data_cleaning_re," ",text)
    text = text.strip()
    return text


# cleaning the data
train_df['discourse_text'] = train_df['discourse_text'].apply(standardizer).apply(rmv_stopwords)
test_df['discourse_text'] = test_df['discourse_text'].apply(standardizer).apply(rmv_stopwords)


train_df.head()


def concat(df):
    discourse_text = []
    for text,Type in zip(df['discourse_text'],df['discourse_type']):
        text = f"[{Type.upper()}]"+" "+text
        discourse_text.append(text)
    return discourse_text


train_discourse_text = concat(train_df)
test_discourse_text = concat(test_df)


train_df['discourse_text']=train_discourse_text
test_df['discourse_text']=test_discourse_text


# dropping discourse type
train_df.drop(['discourse_type'],axis=1,inplace=True)
test_df.drop(['discourse_type'],axis=1,inplace=True)


VOCAB_SIZE = 100000
SEQUENCE_LENGTH = 300

tokenizer = tf.keras.layers.TextVectorization(
    standardize = None,
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = SEQUENCE_LENGTH
)


adapting_ds = tf.data.Dataset.from_tensor_slices(train_df['discourse_text'])
tokenizer.adapt(adapting_ds)


print("length of vocabulary of input data :",len(tokenizer.get_vocabulary())+1)


def get_angles(pos,d_model,i):
    return (pos) * 1 / (np.power(10000, (2 * (i//2)/np.float32(d_model))))

def positional_encoding(SEQUENCE_LENGTH,EMBEDDING_DIM):
    angles = get_angles(pos=np.arange(SEQUENCE_LENGTH)[...,np.newaxis],
                       d_model=EMBEDDING_DIM,
                       i=np.arange(EMBEDDING_DIM)[np.newaxis,...])
    
    angles[:,0::2] = np.sin(angles[:,0::2])
    angles[:,1::2] = np.cos(angles[:,1::2])

    pos_encoding = angles[np.newaxis,...]
    return tf.cast(pos_encoding,dtype=tf.float32) #(1,30,30)

positional_encoding(30,100).shape


n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


def scaled_dot_product_attention(query,key,value):
    matmul_qk = tf.matmul(a=query,b=key,transpose_b=True)
    
    # scale the matmul_qk
    dk = tf.cast(tf.shape(query)[-1],dtype=tf.float32)
    scaled_attention_logits = matmul_qk/ tf.sqrt(dk)
    
    # we are not using either padding_mask or look_ahead_mask
    # padding_mask will be applied in Embedding layer as mask_zero=True
    # look_ahead_mask only applied in decoder and we don't need any decoder
    
    attention_weights = tf.nn.softmax(scaled_attention_logits,axis=-1)
    
    output = tf.matmul(attention_weights,value) # (...,seq_len,emb_dim)
    
    return output,attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,num_heads,d_model):
        super(MultiHeadAttention,self).__init__()
        self.num_heads= num_heads
        self.d_model = d_model
        self.depth = self.d_model // self.num_heads
        
        # multiplying x with these three outputs from three dense layer
        # we will get three thing 
        # query = matmul(x, trans(wq)) --> (seq_len,embedding-dim)
        # key = matmul(x, trans(wk))
        # value = matmul(x, trans(wv))
        self.wq = tf.keras.layers.Dense(d_model) # --> (emb_dim X emb_dim)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        # this is the linear layer in the above picture
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self,x,batch_size):
        """ this is a logical split of q,k,v we are not actually splitting"""
        
        # we are reshaping the q,k,v (seq_len,seq_len) mean(30,30)
        # to shape of ()
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth)) 
        # shape --> TensorShape([16, 30, 4, 25]) # (batch_size,seq_len,heads,depth)
        
        # we are just bringing 4 inplace of 30 
        # so we want to do (16,4,30,25)
        # forget about batch_size 16
        # we have (30,4,25)
        # just assume we are feeding (seq_len,depth) feeding to heads(like batches)
        # so we write the batch_size(here num_heads) in first place
        # so now we get (4,30,25)
        
        x = tf.transpose(x,perm=[0,2,1,3]) # see we just swapped 1 and 2
        # shape --> TensorShape([16, 4, 30, 25])
        return x
    
    def call(self,q,k,v):
        batch_size = tf.shape(q)[0]
        q = self.wq(q) # shape (batch_size,seq_len,embedding_dim)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q,batch_size) # (batch_size,num_heads,seq_len,depth)
        k = self.split_heads(k,batch_size)
        v = self.split_heads(v,batch_size)
        
        scaled_output,attention_weights = scaled_dot_product_attention(query=q,key=k,value=v)
        # again swap back to previous state (16,30,4,25)
        scaled_output = tf.transpose(scaled_output,perm=[0,2,1,3])
        #concatenating outputs of all heads
        concat_output = tf.reshape(scaled_output,shape=(batch_size,-1,self.d_model))
        
        output = self.dense(concat_output)
        
        return output, attention_weights


BATCH_SIZE =32


embedding_dim = 100
mha = MultiHeadAttention(4,100)
x = tf.random.uniform((BATCH_SIZE,SEQUENCE_LENGTH,embedding_dim),minval=0,maxval=100)
split_output = mha.split_heads(x,BATCH_SIZE)
print('Shape of output of split head fuction :',split_output .shape)

# when output is passed through scaled_dot_product_attention
output,attention_weights = scaled_dot_product_attention(split_output ,split_output ,split_output )
print('Shape of output of scaled dot attention :',output.shape)

# output of multi-head-attention
mha_output,_ = mha.call(x,x,x)
print('Shape of attention output from mha :',mha_output.shape)


## point wise feed forward network
def point_wise_feed_forward_network(d_model,dff):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(dff,activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayers(tf.keras.layers.Layer):
    def __init__(self,num_heads,d_model,dff,rate=0.1):
        super(EncoderLayers,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.mha = MultiHeadAttention(num_heads=self.num_heads,d_model=self.d_model)
        self.fnn = point_wise_feed_forward_network(d_model=self.d_model,dff=self.dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self,x,training):
        attention_output,attention_weights = self.mha(x,x,x)
        attention_output = self.dropout1(attention_output,training=training)
        out1 = self.layernorm1(x + attention_output)
        
        fnn_output = self.fnn(out1)
        fnn_output = self.dropout2(fnn_output,training=training)
        out2 = self.layernorm2(out1 + fnn_output)
        
        return out1,attention_weights


test_encoder_layer = EncoderLayers(num_heads=4,d_model=embedding_dim,dff=2048)
x = tf.random.uniform((BATCH_SIZE,SEQUENCE_LENGTH,embedding_dim),minval=0,maxval=100)

encoder_output,_ = test_encoder_layer.call(x,training=False)
print('Shape of encoder output : ',encoder_output.shape)


class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,rate=0.1):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size,d_model,mask_zero=True)
        self.pos_encoding = positional_encoding(SEQUENCE_LENGTH=SEQUENCE_LENGTH,EMBEDDING_DIM=d_model)
        self.enc_layers = [EncoderLayers(num_heads=num_heads,d_model=self.d_model,dff=dff) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self,x,training):
        
        x = self.embedding(x)
        x *= tf.sqrt(tf.cast(self.d_model,dtype=tf.float32))
        x += self.pos_encoding[:,:tf.shape(x)[1],:] # tf.shape(x)[1] gives sequence length
        
        x = self.dropout(x,training=training)
        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x,training)
        return x # (batch_size,seq_len,d_model)


sample_encoder = Encoder(num_layers=8,d_model=100,num_heads=4,dff=512,input_vocab_size=10000,rate=0.2)
x = tf.random.uniform((BATCH_SIZE,SEQUENCE_LENGTH),minval=0,maxval=100)
sample_enc_output= sample_encoder.call(x,training=False)
print('Shape of sample encoder output : ',sample_enc_output.shape)


X = train_df['discourse_text']
y = train_df['label']


test_vec = tokenizer(np.array(test_df)).numpy()
test_dataset = (
tf.data.Dataset
.from_tensor_slices(test_vec)
.shuffle(10000)
.batch(BATCH_SIZE)
.cache()
.prefetch(tf.data.AUTOTUNE))


from sklearn.model_selection import train_test_split


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=43)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)


BATCH_SIZE=128
X_train_vec = tokenizer(np.array(X_train)).numpy()
X_val_vec = tokenizer(np.array(X_val)).numpy()
y_train_fold = np.array(y_train)
y_val_fold = np.array(y_val)

train_dataset = (
tf.data.Dataset
.from_tensor_slices((X_train_vec, y_train_fold))
.shuffle(10000)
.batch(BATCH_SIZE)
.cache()
.prefetch(tf.data.AUTOTUNE))

val_dataset = (
tf.data.Dataset
.from_tensor_slices((X_val_vec, y_val_fold))
.shuffle(10000)
.batch(BATCH_SIZE)
.cache()
.prefetch(tf.data.AUTOTUNE))


def LogLoss(y_true,y_pred):
    return log_loss(y_true,y_pred)



EMBEDDING_DIM = 64
NUM_HEADS = 8 ## this have to be a factor of 100 else it will through reshape error
NUM_LAYERS =2
DFF = 512
MAX_SEQ_LEN = 300
INP_VOCAB_SIZE = len(tokenizer.get_vocabulary())+1

#Encoder layer
encoder = Encoder(num_layers=NUM_LAYERS,d_model=EMBEDDING_DIM,
           num_heads = NUM_HEADS,dff=DFF,input_vocab_size=INP_VOCAB_SIZE,rate=0.25)

# Building model
Inputs = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,))
x = encoder.call(Inputs,training=True)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(32,activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.Dropout(0.25)(x)
Outputs = tf.keras.layers.Dense(3,activation='softmax')(x)

model = tf.keras.models.Model(inputs=Inputs,outputs=Outputs)


# compiling model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adamax(beta_1=0.5,beta_2=0.555),
              metrics=['accuracy'])


model.fit(train_dataset,validation_data=val_dataset,epochs=4)


y_val_pred = model.predict(X_val_vec)


print('log loss - ',log_loss(y_val_fold,y_val_pred))


# Predict
y_pred = model.predict(test_vec, verbose=1)
y_pred


# Load submission template
submission = pd.read_csv("../input/feedback-prize-effectiveness/sample_submission.csv")

# Replace template with predictions
submission['Ineffective'] = y_pred[:,0]
submission['Adequate'] = y_pred[:,1]
submission['Effective'] = y_pred[:,2]

# Save submission file
submission.to_csv("submission.csv", index=False)

submission.head()







