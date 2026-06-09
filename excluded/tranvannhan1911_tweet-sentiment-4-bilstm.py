import numpy as np
import pandas as pd


df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df.dropna(inplace=True)


df.head()


print(df['sentiment'].value_counts())


import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)
df['clean_selected_text'] = df['selected_text'].apply(clean_text)

# Check result
display(df[['text','clean_text','selected_text','clean_selected_text']].head())


df_pos = df[df['sentiment']=='positive'].copy()
df_neg = df[df['sentiment']=='negative'].copy()


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


MAX_LEN = 64
VOCAB_SIZE = 20000

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'].tolist())


def encode_example(text, selected_text, tokenizer, max_len):
    words = text.split()
    sel_words = selected_text.split()

    joined_text = " ".join(words)
    joined_sel = " ".join(sel_words)
    start_idx = joined_text.find(joined_sel)
    end_idx = start_idx + len(joined_sel)

    char_targets = np.zeros(len(joined_text))
    if start_idx != -1:
        char_targets[start_idx:end_idx] = 1
    
    enc = tokenizer.texts_to_sequences([words])[0]
    text_offsets, idx = [], 0
    for w in words:
        text_offsets.append((idx, idx + len(w)))
        idx += len(w) + 1

    token_targets = [0]*len(words)
    for i, (s,e) in enumerate(text_offsets):
        if np.mean(char_targets[s:e]) > 0:
            token_targets[i] = 1

    active_idx = [i for i,t in enumerate(token_targets) if t==1]
    if len(active_idx)>0:
        start_label, end_label = active_idx[0], active_idx[-1]
    else:
        start_label, end_label = 0, 0

    enc = pad_sequences([enc], maxlen=max_len, padding='post')[0]
    return enc, start_label, end_label


def prepare_data(df, tokenizer, max_len=64):
    encoded = [encode_example(t, s, tokenizer, max_len)
               for t, s in zip(df['clean_text'], df['clean_selected_text'])]
    
    X = np.array([e[0] for e in encoded])
    y_start = np.array([e[1] for e in encoded])
    y_end = np.array([e[2] for e in encoded])
    
    indices = np.arange(len(df))

    # Chia thÃ nh train / temp
    X_train, X_temp, y_start_train, y_start_temp, y_end_train, y_end_temp, idx_train, idx_temp = train_test_split(
        X, y_start, y_end, indices, test_size=0.3, random_state=42
    )

    # Chia temp thÃ nh val / test
    X_val, X_test, y_start_val, y_start_test, y_end_val, y_end_test, idx_val, idx_test = train_test_split(
        X_temp, y_start_temp, y_end_temp, idx_temp, test_size=2/3, random_state=42
    )

    # âœ… Táº¡o láº¡i df_test tá»« index
    df_test = df.iloc[idx_test].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_train = df.iloc[idx_train].reset_index(drop=True)

    return (
        X_train, X_val, X_test,
        y_start_train, y_start_val, y_start_test,
        y_end_train, y_end_val, y_end_test,
        df_train, df_val, df_test
    )


BATCH_SIZE = 64
EPOCH = 50
LEARNING_RATE = 1e-5


# def build_model(vocab_size=20000, embed_dim=128, lstm_units=64, max_len=64):
#     inp = Input(shape=(max_len,))
#     x = Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
#     x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
#     x = Dropout(0.2)(x)
#     x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
#     x = Dropout(0.2)(x)

#     start_logits = Dense(max_len, activation='softmax', name='start')(x)
#     end_logits = Dense(max_len, activation='softmax', name='end')(x)

#     model = Model(inputs=inp, outputs=[start_logits, end_logits])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy', 'accuracy']
#     )
#     return model


import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense,
    Dropout, LayerNormalization, Add, SpatialDropout1D,
    MultiHeadAttention, GaussianNoise
)
from tensorflow.keras.models import Model

def build_model(vocab_size=20000, embed_dim=128, lstm_units=96, max_len=64, learning_rate=5e-4):
    inp = Input(shape=(max_len,), name='input_ids')

    # 1ï¸�âƒ£ Embedding + GaussianNoise + SpatialDropout
    x = Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = GaussianNoise(0.1)(x)
    x = SpatialDropout1D(0.3)(x)

    # 2ï¸�âƒ£ Bidirectional LSTM stack (vá»›i residual + LayerNorm)
    lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.25))(x)
    lstm1 = LayerNormalization()(lstm1)

    lstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.25))(lstm1)
    lstm2 = Add()([lstm1, lstm2])  # residual
    lstm2 = LayerNormalization()(lstm2)

    # 3ï¸�âƒ£ Multi-head attention (vá»›i dropout)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=embed_dim, dropout=0.3)(lstm2, lstm2)
    x = Add()([lstm2, attn_out])  # residual
    x = LayerNormalization()(x)

    # 4ï¸�âƒ£ Regularization
    x = Dropout(0.4)(x)

    # 5ï¸�âƒ£ Output heads
    start_logits = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='start_dense')(x)
    end_logits = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name='end_dense')(x)

    start_logits = tf.keras.layers.Flatten()(start_logits)
    end_logits = tf.keras.layers.Flatten()(end_logits)

    start_out = tf.keras.layers.Activation('softmax', name='start_out')(start_logits)
    end_out = tf.keras.layers.Activation('softmax', name='end_out')(end_logits)

    # 6ï¸�âƒ£ Compile
    model = Model(inputs=inp, outputs=[start_out, end_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={
            'start_out': 'sparse_categorical_crossentropy',
            'end_out': 'sparse_categorical_crossentropy'
        },
        metrics={
            'start_out': 'accuracy',
            'end_out': 'accuracy'
        }
    )

    model.summary()
    return model



from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

def train_model(df, tokenizer, max_len=64, label="positive"):
    print(f"Training model for {label.upper()} sentiment...")

    # Chuáº©n bá»‹ dá»¯ liá»‡u
    (
        X_train, X_val, X_test,
        y_start_train, y_start_val, y_start_test,
        y_end_train, y_end_val, y_end_test,
        df_train, df_val, df_test
    ) = prepare_data(df, tokenizer, max_len)
    
    # XÃ¢y dá»±ng model
    model = build_model(VOCAB_SIZE, 128, 64, max_len)

    # Táº¡o thÆ° má»¥c lÆ°u checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/best_model_{label}.keras"

    # Thiáº¿t láº­p callback
    callbacks = [
        ModelCheckpoint(
            filepath=f"checkpoints/best_model_{label}.keras",
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # Huáº¥n luyá»‡n model
    history = model.fit(
        X_train,
        {'start_out': y_start_train, 'end_out': y_end_train},
        validation_data=(X_val, {'start_out': y_start_val, 'end_out': y_end_val}),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        verbose=1,
        callbacks=callbacks
    )

    # Load láº¡i best model
    model = tf.keras.models.load_model(checkpoint_path)

    return history, model, df_test



history_pos, model_pos, df_pos_test = train_model(df_pos, tokenizer, label="positive")
history_neg, model_neg, df_neg_test = train_model(df_neg, tokenizer, label="negative")


def plot_training_history(history, label):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    axes[0].plot(history.history['start_out_accuracy'], label='Train Start Acc', color='tab:blue')
    axes[0].plot(history.history['val_start_out_accuracy'], label='Val Start Acc', color='tab:orange', linestyle='--')
    axes[0].set_title(f'Start Accuracy ({label})'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history['end_out_accuracy'], label='Train End Acc', color='tab:green')
    axes[1].plot(history.history['val_end_out_accuracy'], label='Val End Acc', color='tab:red', linestyle='--')
    axes[1].set_title(f'End Accuracy ({label})'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history.history['loss'], label='Train Loss', color='tab:purple')
    axes[2].plot(history.history['val_loss'], label='Val Loss', color='tab:pink', linestyle='--')
    axes[2].set_title(f'Training Loss ({label})'); axes[2].legend(); axes[2].grid(True)

    plt.suptitle(f"ğŸ“ˆ \Training Progress â€” {label.upper()}", fontsize=14)
    plt.tight_layout()
    plt.show()


plot_training_history(history_pos, "positive")


plot_training_history(history_neg, "negative")


!pip install rouge-score


from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# âœ… Jaccard metric
# ==========================
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if len(a.union(b)) == 0:
        return 0
    return len(a.intersection(b)) / len(a.union(b))

# ==========================
# âœ… BLEU metric
# ==========================
def bleu_score(eval_df):
    bleu_scores = []
    for _, row in eval_df.iterrows():
        reference = row['true_selected']
        prediction = row['predicted']
        score = sentence_bleu([reference.split()], prediction.split())
        bleu_scores.append(score)
    return np.mean(bleu_scores)

# ==========================
# âœ… ROUGE metric
# ==========================
def rouge_score(eval_df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for _, row in eval_df.iterrows():
        reference = row['true_selected']
        prediction = row['predicted']
        scores = scorer.score(reference, prediction)
        rouge_scores.append(scores)
    
    rouge1_avg = np.mean([s['rouge1'].fmeasure for s in rouge_scores])
    rouge2_avg = np.mean([s['rouge2'].fmeasure for s in rouge_scores])
    rougeL_avg = np.mean([s['rougeL'].fmeasure for s in rouge_scores])
    return rouge1_avg, rouge2_avg, rougeL_avg

# ==========================
# âœ… Dá»± Ä‘oÃ¡n span tá»« text
# ==========================
def predict_span(text, tokenizer, model, max_len):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')
    start_pred, end_pred = model.predict(seq, verbose=0)
    start_idx = np.argmax(start_pred[0])
    end_idx = np.argmax(end_pred[0])
    words = text.split()
    if start_idx > end_idx:
        end_idx = start_idx
    selected = " ".join(words[start_idx:end_idx+1])
    return selected


def evaluate_samples(df_test, tokenizer, model, max_len=64, show_samples=5):
    """
    Dá»± Ä‘oÃ¡n vÃ  hiá»ƒn thá»‹ má»™t vÃ i máº«u tá»« táº­p test (khÃ´ng cáº§n predict toÃ n bá»™).
    Tráº£ vá»� cÃ¡c metric tÃ­nh trÃªn cÃ¡c máº«u nÃ y.
    """
    results = []

    print(f"\nÄ�ang dá»± Ä‘oÃ¡n {show_samples} máº«u minh há»�a:")

    # Chá»�n ngáº«u nhiÃªn hoáº·c láº¥y cÃ¡c máº«u Ä‘áº§u tiÃªn
    sample_df = df_test.sample(show_samples, random_state=42) if len(df_test) > show_samples else df_test

    for i, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Predicting", ncols=80):
        text = row['clean_text']
        true_sel = row['clean_selected_text']
        pred_sel = predict_span(text, tokenizer, model, max_len)
        results.append({
            'clean_text': text,
            'true_selected': true_sel,
            'predicted': pred_sel,
            'jaccard': jaccard(true_sel, pred_sel)
        })

    eval_df = pd.DataFrame(results)

    # === TÃ­nh metric trung bÃ¬nh ===
    mean_jac = np.mean(eval_df['jaccard'])
    bleu_avg = bleu_score(eval_df)
    rouge1_avg, rouge2_avg, rougeL_avg = rouge_score(eval_df)

    # === In ra máº«u ===
    for i, row in eval_df.iterrows():
        print(f"\nText: {row['clean_text']}")
        print(f"True: {row['true_selected']}")
        print(f"Pred: {row['predicted']}")
        print(f"Jaccard: {row['jaccard']:.3f}")
        print('-'*60)

    # === In káº¿t quáº£ tá»•ng há»£p ===
    print("\nKáº¿t quáº£ trung bÃ¬nh (trÃªn cÃ¡c máº«u hiá»ƒn thá»‹):")
    print(f"Jaccard: {mean_jac:.4f}")
    print(f"BLEU: {bleu_avg:.4f}")
    print(f"ROUGE-1: {rouge1_avg:.4f}")
    print(f"ROUGE-2: {rouge2_avg:.4f}")
    print(f"ROUGE-L: {rougeL_avg:.4f}")



evaluate_samples(df_pos_test, tokenizer, model_pos)


evaluate_samples(df_neg_test, tokenizer, model_neg)


def evaluate_model(df_test, tokenizer, model_pos, model_neg, max_len=64):
    results = []

    print(f"\nÄ�ang Ä‘Ã¡nh giÃ¡ {len(df_test)} máº«u test (POS + NEG)...")
    for i in tqdm(range(len(df_test)), desc="Evaluating", ncols=90):
        text = df_test.iloc[i]['clean_text']
        true_sel = df_test.iloc[i]['clean_selected_text']
        sentiment = df_test.iloc[i]['sentiment']

        # chá»�n model theo sentiment
        model = model_pos if sentiment == "positive" else model_neg

        pred_sel = predict_span(text, tokenizer, model, max_len)
        results.append({
            'sentiment': sentiment,
            'clean_text': text,
            'true_selected': true_sel,
            'predicted': pred_sel,
            'jaccard': jaccard(true_sel, pred_sel)
        })

    eval_df = pd.DataFrame(results)

    # TÃ­nh trung bÃ¬nh theo tá»«ng nhÃ³m sentiment
    mean_jac_pos = eval_df[eval_df.sentiment == "positive"]['jaccard'].mean()
    mean_jac_neg = eval_df[eval_df.sentiment == "negative"]['jaccard'].mean()
    mean_jac_all = eval_df['jaccard'].mean()

    bleu_avg = bleu_score(eval_df)
    rouge1_avg, rouge2_avg, rougeL_avg = rouge_score(eval_df)

    print("\nKáº¿t quáº£ tá»•ng há»£p:")
    print(f"Jaccard POS: {mean_jac_pos:.4f}")
    print(f"Jaccard NEG: {mean_jac_neg:.4f}")
    print(f"Jaccard TRUNG BÃŒNH: {mean_jac_all:.4f}")
    print(f"BLEU trung bÃ¬nh: {bleu_avg:.4f}")
    print(f"ROUGE-1: {rouge1_avg:.4f}")
    print(f"ROUGE-2: {rouge2_avg:.4f}")
    print(f"ROUGE-L: {rougeL_avg:.4f}")

    return eval_df, {
        "jaccard_pos": mean_jac_pos,
        "jaccard_neg": mean_jac_neg,
        "jaccard_all": mean_jac_all,
        "bleu": bleu_avg,
        "rouge1": rouge1_avg,
        "rouge2": rouge2_avg,
        "rougeL": rougeL_avg
    }


df_test_all = pd.concat([df_pos_test, df_neg_test]).reset_index(drop=True)
eval_df, metrics = evaluate_model(df_test_all, tokenizer, model_pos, model_neg, MAX_LEN)

