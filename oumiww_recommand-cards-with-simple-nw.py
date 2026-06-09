# basic library to process the data
import numpy as np
import pandas as pd

# transform data to one-hot form and processing
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from scipy import sparse

# build nerual network
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# monitoring train process
from tqdm import tqdm

# memory management
import gc


# read the data
folder = '/kaggle/input/what-card-should-i-select-next/'

cards_df = pd.read_csv(folder + 'cards.csv')

train_df = pd.read_csv(folder + 'train.csv')

test_df = pd.read_csv(folder + 'test.csv')

# do not need the submission.csv


# drop the standard decks for they are not present in test.csv
# shuffle the data because I will break the data to many part and study inside for loop
train_df = train_df[train_df["is_wild"] == True].sample(frac=1, random_state=0)
train_df.head()


# extract hero from train.csv
heros_df = pd.DataFrame(train_df.hero.unique(),columns=["hero"])
heros_df


# confirm the cards information
# analyze the cards infromation like attack, health and cost is a standard way, 
#   but in these method I skip that process and just let network to learn it
df_len = len(cards_df)
print(df_len)
cards_df.head(1)


# define the common function

# 1. creat dictionary from id/hero name to index
def extract_dict(df,colname):
    porecess_df = df.reset_index()[[colname,"index"]]
    porecess_df.columns = ["index",colname]
    porecess_df = porecess_df.set_index("index")
    
    return porecess_df[colname].to_dict()

# 2.transform cards column to numpy array
def card_to_numpy(card_series):
    card_cleand = card_series.str.replace(r'[\[\],]', ' ', regex=True)
    card_numpy_flat = np.fromstring(' '.join(card_cleand), sep=' ', dtype=np.int32)
    
    m = len(card_series)
    n = len(card_numpy_flat) // m
    
    return card_numpy_flat.reshape(m, n)

# 3.transfrom key to value(process that in 2-d array)
def dict_transform(x,trans_dict):
    transformer = np.vectorize(lambda x: trans_dict.get(x, x))
    transformed = transformer(x)
    return transformed

# 4.The trick to sum the sparse matrix. I will explan later
#    I hard coding the number in these task
def compress_x_card(mtrix, df_len, times=30):
    x_coo = mtrix.tocoo()

    new_row = x_coo.row // 29

    aggregated = coo_matrix(
        (x_coo.data, (new_row, x_coo.col)),
        shape=(times * df_len, 2628)
    ).tocsr()

    return aggregated

# convert saparse array to SparseTensor
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


# dict to transfrom hero to index, And you don't need to get back
hero_dict = extract_dict(heros_df,"hero")
hero_dict


# at normal, you need a card_dict. But when you use OneHotEncoder,
#    the only information you need is the order of card, and the 
#    dict to converse the order to card id back
card_dict_reverse = cards_df.sort_values("id").reset_index().id.to_dict()

# it will be so long, don't print it fully if not necessary
print(str(card_dict_reverse)[:50])


# use OneHotEncoder to transform card/hero to one-hot form
# the data is huge so I used the spare_output
card_encoder = OneHotEncoder(categories=[list(card_dict_reverse.values())],sparse_output=True)
hero_encoder = OneHotEncoder(categories=[list(hero_dict.keys())],sparse_output=True)

card_encoder.fit(cards_df.id.to_numpy().reshape(-1,1))
hero_encoder.fit(heros_df.hero.to_numpy().reshape(-1,1))


# dense matrix
test = np.array([[0,0,0,1],[0,2,1,0],[1,3,0,0],[0,0,0,0]])
test


# sparse matrix: not zero value and there location
print(sparse.csr_matrix(test))


# make 30 version of extract cards
# normal way to do this is use for i in range(30), but it costs time
# so I creat a mask to extactit
mask = np.array([np.concatenate([np.arange(i), np.arange(i+1, 30)]) for i in range(30)])


# how mask work
simple_mask  = np.array([np.concatenate([np.arange(i), np.arange(i+1, 4)]) for i in range(4)])
print("simple_mask")
print(simple_mask)

sample_cards = np.array([[101, 102, 103, 104],
                        [201, 202, 203, 204],
                        [301, 302, 303, 304]])

print("masked cards:")
print(sample_cards[:,simple_mask])

print("y card")
print(sample_cards.reshape(-1))


# how one hot work
cate_sample = [101,102,103,104,201,202,203,204,301,302,303,304]
encoder_sample = OneHotEncoder(categories=[cate_sample],sparse_output=True)
onehot_sampe = encoder_sample.fit_transform(sample_cards.reshape(-1,1))
print(onehot_sampe)
print("one hot from")
onehot_sampe.todense()


card_number_of_decks = 4
row_of_decs = 3
variation_of_cards = len(cate_sample)

# it should be 29 in realtask because we masked the card,
# but for explan, we use card befor maske, so let it be 1
masked_num = 1

# change it from csr form to coo from matrix so we can get row attribute
x_coo = onehot_sampe.tocoo()

# get de floor division of x_coo.row by the card_number_of_decks
new_row = x_coo.row // card_number_of_decks

print("before:", x_coo.row)
print("after: ", new_row)


# use the new_row to reform the matrix, and set the shape by row_of_decs and masked_num
aggregated = coo_matrix(
    (x_coo.data, (new_row, x_coo.col)),
    shape=(row_of_decs * masked_num, variation_of_cards)
).tocsr()
print(aggregated)
print(aggregated.todense())


class SparseToDenseLayer(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):

        dense = tf.sparse.to_dense(inputs)
        dense.set_shape((None, 2637))
        return dense

    def compute_output_shape(self, input_shape):
        return input_shape


# build model

# another error massage appear when I run this model, It was solved by use this option
# it worked
# https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio
tf.config.run_functions_eagerly(True)

# set sparse=True to feed the spares matrix, and set batch_size to let the model know the shape
inputs = tf.keras.Input(batch_size=512, shape=(2637,), sparse=True)
x = SparseToDenseLayer()(inputs)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# use softmax to transform the output to probability
y = tf.keras.layers.Dense(2628, activation='softmax')(x)


model = tf.keras.Model(inputs, y)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# seprate the data by step
step = 20000
# set model inital_epoch
initial_epoch = 0

for i in tqdm(range(0, df_len, step)):

    # I done this work in other notebook by mistake so this time I just read the trained weight
    break
    
    chunk_df = train_df[i:min(i+step, df_len)]
    tmp_len = len(chunk_df)
    
    # repeat the hero 30 times for match the masked data
    x_hero_one_hot = hero_encoder.transform(chunk_df.hero.to_numpy().repeat(30).reshape(-1,1))
    
    # transform cards columns to numpy
    x_cards_numpy = card_to_numpy(chunk_df.cards)

    # transfrom numpy array to one hot form and compress it
    x_card_one_hot = card_encoder.transform(x_cards_numpy[:, mask].reshape(-1, 1))
    x_card_one_hot = compress_x_card(x_card_one_hot,tmp_len)

    # get one-hot from y and trans it back to dense
    y_train = card_encoder.transform(x_cards_numpy.reshape(-1,1)).todense()

    # connect hero feature and 29 cards
    X_train = hstack([x_hero_one_hot, x_card_one_hot])

    # clean the tmp data befor train
    del x_hero_one_hot, x_card_one_hot, x_cards_numpy
    
    # SparseTensor need a sorted sparse matrix
    X_train = X_train.tocsr()
    X_train.sort_indices()

    # convert csr to sparetensor
    X_train = convert_sparse_matrix_to_sparse_tensor(X_train)

    # train the model 3 epoch each iter
    model.fit(X_train, y_train, epochs=initial_epoch + 3, batch_size=512, initial_epoch=initial_epoch, verbose=1)
    initial_epoch += 3

    # clean the train data
    del X_train, y_train
    gc.collect()


# trained weights
model.load_weights("/kaggle/input/trained-para/model_shuffled_20250216.hdf5")


# get the len of test
df_len = len(test_df)

# get one-hot hero without repeat
x_hero_one_hot = hero_encoder.transform(test_df.hero.to_numpy().reshape(-1,1))

# get cards numpy
x_cards_numpy = card_to_numpy(test_df.cards_incomplete)

# get one-hot cards without mask, and compress it
x_card_one_hot = card_encoder.transform(x_cards_numpy.reshape(-1, 1))
x_card_one_hot = compress_x_card(x_card_one_hot, df_len, 1)

# connect the hero and cards
X_train = hstack([x_hero_one_hot, x_card_one_hot])

del x_hero_one_hot, x_card_one_hot, x_cards_numpy

# transform to csr and sort
X_train = X_train.tocsr()
X_train.sort_indices()


# predict
predictions_sub = model.predict(X_train)


# get top three(k=3) max values from each predected row
top_k = tf.math.top_k(predictions_sub, k=3)

# get the indices(column) of the best three
top_column_numbers = top_k.indices.numpy()

# transform indices to card id
transformed = dict_transform(top_column_numbers, card_dict_reverse)
transformed


# change the data to submission form by join it by ' '
joined_data = np.apply_along_axis(lambda row: ' '.join(map(str, row)), axis=1, arr=transformed)
joined_data


# connect the recommendations to deck id, make it to DataFrame
sub_df = pd.DataFrame( joined_data, columns=["recommendations"])
sub_df["deckid"] = test_df.deckid
sub_df = sub_df[["deckid","recommendations"]]
sub_df


# output
sub_df.to_csv("/kaggle/working/submission.csv.csv",index=False)

