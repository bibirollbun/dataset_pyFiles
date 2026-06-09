import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
np.random.seed(42)

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences


def map12(pred_items, target_item):
    """
    pred_items : список з 12 рекомендованих item_id (в порядку від 1 до 12)
    target_item : правильний item_id
    """
    score = 0.0
    for k in range(1, 13):
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

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Графік Loss (Втрат)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Графік Accuracy (Точності)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def model_predict_last_item(history):
    return history[-1]

def model_predict_last_12_item(history):
    return history[-12:][::-1]


transactions_train = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                                usecols=['t_dat', 'customer_id', 'article_id'])

unique_transactions = transactions_train.drop_duplicates(subset=['t_dat', 'customer_id', 'article_id'])
del transactions_train
purchase_counts = unique_transactions.groupby("customer_id")["article_id"].count()

top_users = purchase_counts.sort_values(ascending=False).head(10000).index.tolist()


unique_transactions


articles_df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv',
                         usecols=['article_id', 'product_group_name', 'product_type_name', 'colour_group_code'])
articles_df


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


top_users = np.array(top_users)
top_users_le = pd.Series(top_users).map(cust2idx).values
top_users_le[:2]

np.random.shuffle(top_users_le)

train_size = int(0.9 * len(top_users_le))

train_users = set(top_users_le[:train_size])
test_users  = set(top_users_le[train_size:])

train_df = df_top[df_top['cust'].isin(train_users)].copy()
test_df  = df_top[df_top['cust'].isin(test_users)].copy()

print(train_df.shape, test_df.shape)

train_df = train_df.sort_values(['cust', 't_dat'])
test_df  = test_df.sort_values(['cust', 't_dat'])


df_cust_info = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')


df_cust_info


df_cust_info['customer_id_mapped'] = df_cust_info['customer_id'].map(cust2idx)


df_train = df_cust_info[df_cust_info["customer_id_mapped"].isin(train_users)]
df_test  = df_cust_info[df_cust_info["customer_id_mapped"].isin(test_users)]

cat_cols = ["Active", "club_member_status", "fashion_news_frequency"]

num_cols = ["age"]

for col in cat_cols:
    print(f"\n=== {col}: TRAIN ===")
    display(df_train[col].value_counts(normalize=True).round(3))

    print(f"\n=== {col}: TEST ===")
    display(df_test[col].value_counts(normalize=True).round(3))


print(df_train[num_cols].describe())
print(df_test[num_cols].describe())


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


import tensorflow as tf

BATCH_SIZE = 1024*4
def get_dataloader(X, y, batch_size=1024, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(100000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_loader = get_dataloader(X_train, y_train, batch_size=BATCH_SIZE)
test_loader = get_dataloader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)


# next(iter(train_loader))


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

def build_multi_input_model():
    # 1. Визначаємо входи (назви мають збігатися з ключами у нашому X_dict)
    input_art = Input(shape=(MAX_LEN,), name='input_art')
    input_type = Input(shape=(MAX_LEN,), name='input_type')
    input_group = Input(shape=(MAX_LEN,), name='input_group')
    input_col = Input(shape=(MAX_LEN,), name='input_col')

    # 2. Embedding шари для кожної фічі
    # Для артикулів залишаємо більше векторів (64), для категорій — менше (16-32)
    emb_art = Embedding(article_vocab_size, 64, name='emb_art')(input_art)
    emb_type = Embedding(prod_type_vocab_size, 8, name='emb_type')(input_type)
    emb_group = Embedding(prod_group_vocab_size, 8, name='emb_group')(input_group)
    emb_col = Embedding(color_vocab_size, 8, name='emb_col')(input_col)

    # 3. Конкатенація (об'єднуємо всі ознаки в один вектор для кожного кроку часу)
    # Тепер на кожному з 50 кроків LSTM бачитиме вектор розмірністю 64+16+16+16 = 112
    merged = Concatenate()([emb_art, emb_type, emb_group, emb_col])

    # 4. Рекурентна частина
    lstm_out = LSTM(128, dropout=0.4)(merged)
    
    # 5. Повнозв'язні шари
    x = BatchNormalization()(lstm_out)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 6. Вихідний шар (передбачаємо тільки ID наступного артикула)
    output = Dense(article_vocab_size, activation='softmax', name='output')(x)

    # Збираємо модель
    model = Model(
        inputs=[input_art, input_type, input_group, input_col], 
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, use_ema=True),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Створюємо модель
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
model = build_multi_input_model() # Твоя функція для Functional API
model.summary()


# Налаштування EarlyStopping, щоб не чекати всі 20 епох, якщо модель перестане вчитися
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3,           # Чекаємо 3 епохи без покращень
    restore_best_weights=True # Зберігаємо найкращу версію
)

# Навчання на розширеному (augmented) датасеті
# Якщо використовуєш DataLoader (рекомендую):
history = model.fit(
    train_loader, # DataLoader сам розбереться з іменами входів
    validation_data=test_loader,
    epochs=20,
    callbacks=[early_stop]
)


model.save('hm_lstm_multi_input_model_use_ema.keras')


# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Вказуємо шлях до другого GPU (індексація починається з 0)

# loaded_model = load_model('hm_lstm_multi_input_model_use_ema.keras')

loaded_model = model


import numpy as np
from tqdm import tqdm
import tensorflow as tf

all_top_12 = []

# Ітеруємося по батчах з DataLoader
for x_batch, _ in tqdm(test_loader, desc="Predicting batches"):
    # Отримуємо ймовірності тільки для поточного батчу (наприклад, 256 юзерів)
    batch_preds = loaded_model.predict_on_batch(x_batch)
    
    # Одразу знаходимо ТОП-12 індексів (найбільші ймовірності)
    # np.argsort дає індекси від найменшого до найбільшого
    # [:, -12:] бере 12 найкращих, [:, ::-1] розвертає їх у правильному порядку
    batch_top_12 = np.argsort(batch_preds, axis=1)[:, -12:][:, ::-1]
    
    # Зберігаємо лише ці 12 чисел на кожного юзера
    all_top_12.append(batch_top_12)

# 3. Об'єднуємо всі батчі в один масив
top_12_indices = np.vstack(all_top_12)

print(f"\nГотово! Отримано прогнозів для {top_12_indices.shape[0]} користувачів.")
print(f"Формат виходу: {top_12_indices.shape}") # Має бути (кількість_юзерів, 12)


plot_history(history)


lstm_score = map12_score(top_12_indices, y_test)
print(f"LSTM Validation MAP@12: {lstm_score:.6f}")


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
    map_scores.append(map12(pred, target))

baseline_score = np.mean(map_scores)
print(f"\nBaseline MAP@12 (Last 12 items): {baseline_score:.6f}")




