import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Embedding, Dropout, SpatialDropout1D, 
                                     BatchNormalization, LSTM, Dense, Concatenate,                                                                        
                                     MultiHeadAttention, LayerNormalization, Add, 
                                     GlobalAveragePooling1D)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import regularizers

np.random.seed(42)


def mapK(pred_items, target_item, K=12):
    """
    pred_items : list of K recommended item_id (order from 1 to K)
    target_item : correct item_id
    """
    score = 0.0
    for k in range(1, K + 1):
        if pred_items[k-1] == target_item:
            score = 1.0 / k
            break
    return score


def map12_score(preds, targets):
    total_score = 0
    for p, t in zip(preds, targets):
        if t in p:
            # Знаходимо позицію (1-based index)
            rank = np.where(p == t)[0][0] + 1
            total_score += 1 / rank
    return total_score / len(targets)


def model_predict_last_item(history):
    return history[-1]

def model_predict_last_12_item(history):
    return history[-12:][::-1]


def get_top_k_predictions(model, dataset, k=12):
    all_predicted_indices = []
    all_actual_indices = []

    for x_batch, y_batch in dataset:
        # 1. Get probability distribution from model
        preds = model.predict(x_batch, verbose=0)
        
        # 2. Extract indices of the top K probabilities
        # values: the probabilities, indices: the item integer IDs
        values, indices = tf.math.top_k(preds, k=k)
        
        all_predicted_indices.extend(indices.numpy().tolist())
        all_actual_indices.extend(y_batch.numpy().tolist())
        
    return all_actual_indices, all_predicted_indices


def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['top_12_accuracy'], label='Train Top 12 Acc', marker='o')
    plt.plot(history.history['val_top_12_accuracy'], label='Val Top 12 Acc', marker='o')
    plt.title('Model Top 12 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def create_predict(model, test_loader, model_type='single_input'):
    all_top_12 = []
    
    # Iterate through batches with DataLoader
    for x_batch, _ in tqdm(test_loader, desc="Predicting batches"):
        # Probabilities for current batch
        if model_type == 'multi_input':
            batch_preds = model.predict_on_batch(x_batch)
        else: 
            batch_preds = model.predict_on_batch(x_batch['input_art'])
        
        # Top-12 indexes (greatest probabilities)
        # np.argsort - indexes from smallest to biggest
        # [:, -12:] takes 12 best, [:, ::-1] reverts them in correct order
        batch_top_12 = np.argsort(batch_preds, axis=1)[:, -12:][:, ::-1]
        del batch_preds
        
        # Save only these 12 numbers
        all_top_12.append(batch_top_12)
        del batch_top_12
        
        gc.collect()
    
    # 3. Join all batches
    top_12_indices = np.vstack(all_top_12)
    del all_top_12  
    
    print(f"\nReady! Calculated predictions for {top_12_indices.shape[0]} users.")
    print(f"Output format: {top_12_indices.shape}") 
    
    return top_12_indices


transactions_train = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                usecols=['t_dat', 'customer_id', 'article_id'])

unique_transactions = transactions_train.drop_duplicates(subset=['t_dat', 'customer_id', 'article_id'])
del transactions_train
purchase_counts = unique_transactions.groupby("customer_id")["article_id"].count()


articles_df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv',
                         usecols=['article_id', 'product_group_name', 'product_type_name', 'colour_group_code'])
articles_df


N_USERS_TRAIN_TEST = 50000 # Number of top users that we select to speed up comparison of models


purchase_counts.shape


top_users = purchase_counts.sort_values(ascending=False).head(N_USERS_TRAIN_TEST).index.tolist()


df_top = unique_transactions[unique_transactions.customer_id.isin(top_users)].reset_index(drop=True)
df_top = df_top.merge(articles_df, how='left', on=['article_id'], validate='m:1', )

def encode_column(df, column_name):
    unique_values = df[column_name].unique()
    
    val2idx = {val: i + 1 for i, val in enumerate(unique_values)}
    idx2val = {i + 1: val for i, val in enumerate(unique_values)}
    
    encoded_series = df[column_name].map(val2idx)
    
    vocab_size = len(val2idx) + 1
    
    return encoded_series, val2idx, idx2val, vocab_size

df_top['article'], art2idx, idx2art, article_vocab_size = encode_column(df_top, 'article_id')
df_top['cust'], cust2idx, idx2cust, customer_vocab_size = encode_column(df_top, 'customer_id')
df_top['product_type'], prod_type2idx, idx2prod_type, prod_type_vocab_size = encode_column(df_top, 'product_type_name')
df_top['product_group'], prod_group2idx, idx2prod_group, prod_group_vocab_size = encode_column(df_top, 'product_group_name')
df_top['colour_group'], color2idx, idx2color, color_vocab_size = encode_column(df_top, 'colour_group_code')

df_top = df_top.drop(columns=['customer_id', 'article_id', 'product_type_name', 'product_group_name'], errors='ignore')
df_top


df_top.shape


top_users = np.array(top_users)
top_users_le = pd.Series(top_users).map(cust2idx).values
top_users_le[:2]

np.random.shuffle(top_users_le)
train_size = int(0.99 * len(top_users_le))

train_users = set(top_users_le[:train_size])
test_users  = set(top_users_le[train_size:])
print('Users train / test:', len(train_users), len(test_users))

train_df = df_top[df_top['cust'].isin(train_users)].copy()
test_df  = df_top[df_top['cust'].isin(test_users)].copy()

print('Samples train / test:', train_df.shape, test_df.shape)

train_df = train_df.sort_values(['cust', 't_dat'])
test_df  = test_df.sort_values(['cust', 't_dat'])


MAX_LEN = 50
def prepare_multi_input_data(df, max_len, is_train=True, max_examples_per_user=30):
    # Групуємо дані по юзерах один раз
    # Порядок у списках відповідатиме t_dat, якщо df відсортовано
    grouped = df.sort_values(['cust', 't_dat']).groupby('cust')
    
    X_art, X_type, X_group, X_col = [], [], [], []
    y = []

    for cust_id, group in tqdm(grouped, desc="Processing Users"):
        # Перетворюємо колонки групи в масиви для швидкості
        articles = group['article'].values
        types = group['product_type'].values
        groups = group['product_group'].values
        colors = group['colour_group'].values
        
        if len(articles) < 2:
            continue
            
        if is_train:
            # Логіка "свіжості": беремо лише хвіст історії для створення прикладів
            # Нам потрібно (max_examples + 1) елементів, щоб отримати max_examples пар (X, y)
            start_pos = max(1, len(articles) - max_examples_per_user)
            
            for i in range(start_pos, len(articles)):
                # Історія — все до поточного індексу i
                X_art.append(articles[:i][-max_len:])
                X_type.append(types[:i][-max_len:])
                X_group.append(groups[:i][-max_len:])
                X_col.append(colors[:i][-max_len:])
                # Ціль — елемент на індексі i
                y.append(articles[i])
        else:
            # Для тесту/валідації — тільки один (найсвіжіший) приклад на юзера
            X_art.append(articles[:-1][-max_len:])
            X_type.append(types[:-1][-max_len:])
            X_group.append(groups[:-1][-max_len:])
            X_col.append(colors[:-1][-max_len:])
            y.append(articles[-1])

    # Спільні параметри паддінгу
    pad_cfg = {'maxlen': max_len, 'padding': 'pre', 'dtype': 'int32', 'value': 0}
    
    X_dict = {
        'input_art': pad_sequences(X_art, **pad_cfg),
        'input_type': pad_sequences(X_type, **pad_cfg),
        'input_group': pad_sequences(X_group, **pad_cfg),
        'input_col': pad_sequences(X_col, **pad_cfg)
    }
    
    return X_dict, np.array(y, dtype='int32')

# Виклик прямо з DataFrame
X_train, y_train = prepare_multi_input_data(train_df, MAX_LEN, is_train=True)
X_test, y_test = prepare_multi_input_data(test_df, MAX_LEN, is_train=True)


BATCH_SIZE = 1024*4

def get_dataloader(X, y, batch_size=1024, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(100000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_loader = get_dataloader(X_train, y_train, batch_size=BATCH_SIZE)
test_loader = get_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)


del articles_df


df_top = unique_transactions[unique_transactions.customer_id.isin(top_users)].reset_index(drop=True)
del unique_transactions

def encode_column(df, column_name):
    unique_values = df[column_name].unique()
    
    val2idx = {val: i + 1 for i, val in enumerate(unique_values)}
    idx2val = {i + 1: val for i, val in enumerate(unique_values)}
    
    encoded_series = df[column_name].map(val2idx)
    
    vocab_size = len(val2idx) + 1
    
    return encoded_series, val2idx, idx2val, vocab_size

df_top['article'], art2idx, idx2art, article_vocab_size = encode_column(df_top, 'article_id')
df_top['cust'], cust2idx, idx2cust, customer_vocab_size = encode_column(df_top, 'customer_id')

df_top = df_top.drop(columns=['customer_id', 'article_id'])
df_top


map_scores = []
num_samples = len(X_test['input_art'])

for i in tqdm(range(num_samples), desc="Calculating MAP@12"):
    # Отримуємо історію покупок (тільки артикули) для Baseline
    # Ми беремо i-й рядок з масиву 'input_art'
    history_art = X_test['input_art'][i]
    target = y_test[i]
    
    # Baseline функція: вона зазвичай дивиться на історію артикулів
    # і вибирає останні 12 унікальних значень
    pred = model_predict_last_12_item(history_art)
    
    # Рахуємо MAP@12 для цього юзера
    map_scores.append(mapK(pred, target))

baseline_score = np.mean(map_scores)
print(f"\nBaseline MAP@12 (Last 12 items): {baseline_score:.6f}")


def build_multi_input_model():
    # 1. Визначаємо входи (назви мають збігатися з ключами у нашому X_dict)
    input_art = Input(shape=(MAX_LEN,), name='input_art')
    input_type = Input(shape=(MAX_LEN,), name='input_type')
    input_group = Input(shape=(MAX_LEN,), name='input_group')
    input_col = Input(shape=(MAX_LEN,), name='input_col')
    
    emb_art = Embedding(article_vocab_size, 512, name='emb_art')(input_art)
    emb_type = Embedding(prod_type_vocab_size, 64, name='emb_type')(input_type)
    emb_group = Embedding(prod_group_vocab_size, 64, name='emb_group')(input_group)
    emb_col = Embedding(color_vocab_size, 64, name='emb_col')(input_col)
    
    merged = Concatenate()([emb_art, emb_type, emb_group, emb_col])    
    x = SpatialDropout1D(0.4)(merged) 
    
    lstm_out = LSTM(128)(x)  
    # lstm_out = LSTM(128, return_sequences=True)(x)  # For using attention layer
    x = BatchNormalization()(lstm_out)
    
    # Multi-Head Attention Layer
    # This allows the model to focus on specific important past articles
    #attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    
    # Residual Connection & Layer Normalization (Transformer-style)
    #x = Add()([lstm_out, attention_out]) 
    #x = LayerNormalization()(x)
    
    # Pooling to condense the 50 time-steps into one feature vector
    #x = GlobalAveragePooling1D()(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)    
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 6. Вихідний шар (передбачаємо тільки ID наступного артикула)
    output = Dense(article_vocab_size, activation='softmax', name='output')(x)
    
    model = Model(
        inputs=[input_art, input_type, input_group, input_col], 
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=12, name='top_12_accuracy'),
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    return model


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model_feat = build_multi_input_model() 
    model_feat.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3,           
    restore_best_weights=True # Зберігаємо найкращу версію
)

history_feat = model_feat.fit(
    train_loader, 
    validation_data=test_loader,
    epochs=20,
    callbacks=[early_stop]
)


plot_history(history_feat)


model_feat.save('hm_lstm_multi_input_model.keras')
del model_feat


loaded_model_mi = load_model('hm_lstm_multi_input_model.keras')


tf.keras.utils.plot_model(
    loaded_model_mi, 
    show_shapes=True, 
    show_layer_names=True,
    rankdir='TB', # 'TB' for Top-to-Bottom, 'LR' for Left-to-Right
    expand_nested=True,
    dpi=96
)


top_12_indices_mi = create_predict(loaded_model_mi, test_loader, model_type='multi_input')
lstm_multi_input_score = map12_score(top_12_indices_mi, y_test)
print(lstm_multi_input_score)


print("MAP@12")
print(f"\nBaseline (Last 12 items): {baseline_score:.6f}")
print(f"LSTM with features: {lstm_multi_input_score:.6f}")




