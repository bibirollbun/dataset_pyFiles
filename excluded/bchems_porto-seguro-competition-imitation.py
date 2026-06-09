# --------------------------------------------------------------
#  Porto Seguro – 1st-place style model (Embeddings + DNN)
#  TensorFlow 2.x | CPU / GPU | No NumPy in metrics
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Concatenate,
    BatchNormalization, Dropout, Activation, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --------------------- 1. Normalized Gini (Pure TF) ---------------------
def normalized_gini_tf(y_true, y_pred):
    """
    Normalized Gini coefficient as a Keras metric.
    Formula: Gini = 2 * AUC - 1 → Normalized = Gini / Gini_max
    Fully TensorFlow — no NumPy allowed in graph mode.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    n = tf.cast(tf.shape(y_true)[0], tf.float32)
    indices = tf.argsort(y_pred, direction='DESCENDING')
    y_true_sorted = tf.gather(y_true, indices)

    # Cumulative sum of true labels
    cum_true = tf.cumsum(y_true_sorted)
    sum_true = tf.reduce_sum(y_true)

    # Random baseline: linear cumulative
    cum_random = tf.range(1.0, n + 1.0) * sum_true / n
    sum_random = tf.reduce_sum(cum_random)

    # Gini
    gini = (tf.reduce_sum(cum_true) - sum_random) / (sum_true + tf.keras.backend.epsilon())

    # Max possible Gini (perfect ordering)
    perfect_cum = tf.cumsum(tf.sort(y_true, direction='DESCENDING'))
    gini_max = (tf.reduce_sum(perfect_cum) - sum_random) / (sum_true + tf.keras.backend.epsilon())

    # Avoid division by zero
    gini_max = tf.where(tf.equal(gini_max, 0), tf.ones_like(gini_max), gini_max)

    return gini / gini_max

# --------------------------------------------------------------------
# 2. Load and preprocess data
# --------------------------------------------------------------------
def load_data(train_path='/kaggle/input/porto-seguro-safe-driver-prediction/train.csv', test_path='/kaggle/input/porto-seguro-safe-driver-prediction/test.csv'):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    y = train['target'].values.astype(np.float32)

    X = train.drop(['id', 'target'], axis=1)
    X_test = test.drop('id', axis=1)
    test_ids = test['id'].values

    # Fill missing values
    X = X.fillna(-1)
    X_test = X_test.fillna(-1)

    # Identify categorical columns (suffix '_cat')
    cat_cols = [c for c in X.columns if c.endswith('_cat')]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Label encode categoricals (train + test together)
    cat_max = {}
    for col in cat_cols:
        combined = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
        uniques = sorted(combined.unique())
        mapping = {val: idx for idx, val in enumerate(uniques)}
        X[col] = X[col].astype(str).map(mapping).astype(np.int32)
        X_test[col] = X_test[col].astype(str).map(mapping).astype(np.int32)
        cat_max[col] = len(uniques)  # input_dim = n_unique + 1 (for padding)

    return X, y, X_test, test_ids, cat_cols, num_cols, cat_max

# --------------------------------------------------------------------
# 3. Build model: Embeddings + Dense Tower
# --------------------------------------------------------------------
def build_model(cat_cols, cat_max, num_cols, emb_dims,
                dense_units=[400, 300, 200], dropout=0.15):
    # Numerical input
    num_inp = Input(shape=(len(num_cols),), name='numerical')

    # Categorical inputs + embeddings
    cat_inputs = []
    cat_embs = []
    for col in cat_cols:
        inp = Input(shape=(1,), name=col)
        emb = Embedding(
            input_dim=cat_max[col] + 1,   # +1 for padding/unknown
            output_dim=emb_dims[col],
            input_length=1
        )(inp)
        emb = Flatten()(emb)
        cat_inputs.append(inp)
        cat_embs.append(emb)

    # Concatenate
    if cat_embs:
        cat_concat = Concatenate()(cat_embs)
        x = Concatenate()([num_inp, cat_concat])
    else:
        x = num_inp

    # Dense tower
    for units in dense_units:
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Dropout(dropout)(x)

    out = Dense(1, activation='sigmoid')(x)

    # Model
    inputs = [num_inp] + cat_inputs
    model = Model(inputs=inputs, outputs=out)
    return model

# --------------------------------------------------------------------
# 4. Main execution
# --------------------------------------------------------------------
if __name__ == '__main__':
    # 4.1 Load data
    X, y, X_test, test_ids, cat_cols, num_cols, cat_max = load_data()

    # 4.2 Embedding dimensions (as in 1st place write-up)
    emb_dims = {}
    for col in cat_cols:
        n = cat_max[col]
        emb_dims[col] = min(50, (n + 1) // 2)

    # 4.3 Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 4.4 Prepare model inputs
    def df_to_inputs(df, cat_cols, num_cols):
        num = df[num_cols].values.astype(np.float32)
        cat = [df[col].values.astype(np.int32).reshape(-1, 1) for col in cat_cols]
        return [num] + cat

    train_inputs = df_to_inputs(X_tr, cat_cols, num_cols)
    val_inputs   = df_to_inputs(X_val, cat_cols, num_cols)
    test_inputs  = df_to_inputs(X_test, cat_cols, num_cols)

    # 4.5 Build and compile model
    model = build_model(cat_cols, cat_max, num_cols, emb_dims)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[normalized_gini_tf]
    )

    # 4.6 Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_normalized_gini_tf',
            patience=12,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_normalized_gini_tf',
            factor=0.5,
            patience=5,
            mode='max'
        )
    ]

    # 4.7 Train
    print("Starting training...")
    model.fit(
        train_inputs, y_tr,
        validation_data=(val_inputs, y_val),
        epochs=200,
        batch_size=1024,
        callbacks=callbacks,
        verbose=2
    )

    # 4.8 Validation score (NumPy version for reporting)
    val_pred = model.predict(val_inputs, batch_size=2048).ravel()
    auc = roc_auc_score(y_val, val_pred)
    gini = 2 * auc - 1
    print(f"\nValidation AUC: {auc:.5f}")
    print(f"Validation Normalized Gini: {gini:.5f}")

    # 4.9 Test prediction
    test_pred = model.predict(test_inputs, batch_size=2048).ravel()
    submission = pd.DataFrame({
        'id': test_ids,
        'target': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved: submission.csv")


# --------------------------------------------------------------
#  Porto Seguro – DAE + Classifier 
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Concatenate, BatchNormalization,
    Dropout, Activation, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --------------------- 1. Normalized Gini (Pure TF) ---------------------
def normalized_gini_tf(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    n = tf.cast(tf.shape(y_true)[0], tf.float32)
    indices = tf.argsort(y_pred, direction='DESCENDING')
    y_true_sorted = tf.gather(y_true, indices)
    cum_true = tf.cumsum(y_true_sorted)
    sum_true = tf.reduce_sum(y_true)
    cum_random = tf.range(1.0, n + 1.0) * sum_true / n
    sum_random = tf.reduce_sum(cum_random)
    gini = (tf.reduce_sum(cum_true) - sum_random) / (sum_true + 1e-8)
    perfect_cum = tf.cumsum(tf.sort(y_true, direction='DESCENDING'))
    gini_max = (tf.reduce_sum(perfect_cum) - sum_random) / (sum_true + 1e-8)
    gini_max = tf.where(tf.equal(gini_max, 0), tf.ones_like(gini_max), gini_max)
    return gini / gini_max

# --------------------------------------------------------------------
# 2. Load and preprocess data
# --------------------------------------------------------------------
def load_data(train_path='/kaggle/input/porto-seguro-safe-driver-prediction/train.csv', test_path='/kaggle/input/porto-seguro-safe-driver-prediction/test.csv'):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    y = train['target'].values.astype(np.float32)
    X = train.drop(['id', 'target'], axis=1)
    X_test = test.drop('id', axis=1)
    test_ids = test['id'].values

    X = X.fillna(-1)
    X_test = X_test.fillna(-1)

    cat_cols = [c for c in X.columns if c.endswith('_cat')]
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_max = {}
    for col in cat_cols:
        combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
        uniques = sorted(combined.unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        X[col] = X[col].astype(str).map(mapping).astype(np.int32)
        X_test[col] = X_test[col].astype(str).map(mapping).astype(np.int32)
        cat_max[col] = len(uniques)

    return X, y, X_test, test_ids, cat_cols, num_cols, cat_max

# --------------------------------------------------------------------
# 3. Build Denoising Autoencoder (with named layers)
# --------------------------------------------------------------------
def build_denoising_autoencoder(cat_cols, cat_max, emb_dims, noise_level=0.3):
    inputs = []
    decoded = []

    for col in cat_cols:
        inp = Input(shape=(1,), name=f'input_{col}')
        emb = Embedding(cat_max[col] + 1, emb_dims[col], name=f'emb_{col}')(inp)  # NAMED
        emb = Flatten()(emb)

        noisy = Dropout(noise_level)(emb, training=True)

        h = Dense(64, activation='elu')(noisy)
        h = BatchNormalization()(h)
        code = Dense(emb_dims[col], activation='linear')(h)

        h = Dense(64, activation='elu')(code)
        h = BatchNormalization()(h)
        out = Dense(emb_dims[col], activation='linear')(h)

        inputs.append(inp)
        decoded.append(out)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')
    return autoencoder

# --------------------------------------------------------------------
# 4. Build classifier with pre-trained weights
# --------------------------------------------------------------------
def build_classifier_with_pretrained_weights(cat_cols, cat_max, num_cols, pretrained_weights, dropout=0.2):
    num_inp = Input(shape=(len(num_cols),), name='numerical')
    cat_inputs = []
    cat_embs = []

    for i, col in enumerate(cat_cols):
        inp = Input(shape=(1,), name=f'cat_{col}')
        emb = Embedding(
            input_dim=cat_max[col] + 1,
            output_dim=pretrained_weights[i].shape[1],
            weights=[pretrained_weights[i]],
            trainable=False,
            name=f'pretrained_emb_{col}'
        )(inp)
        emb = Flatten()(emb)
        cat_inputs.append(inp)
        cat_embs.append(emb)

    x = Concatenate()([num_inp] + cat_embs) if cat_embs else num_inp

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(dropout)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(dropout)(x)

    out = Dense(1, activation='sigmoid')(x)
    model = Model([num_inp] + cat_inputs, out)
    return model

# --------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------
if __name__ == '__main__':
    print("Loading data...")
    X, y, X_test, test_ids, cat_cols, num_cols, cat_max = load_data()

    emb_dims = {col: min(50, (cat_max[col] + 1) // 2) for col in cat_cols}

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def get_cat_inputs(df):
        return [df[col].values.astype(np.int32).reshape(-1, 1) for col in cat_cols]

    cat_train = get_cat_inputs(X_tr)
    cat_val   = get_cat_inputs(X_val)
    cat_test  = get_cat_inputs(X_test)

    # ------------------- DAE Pre-training -------------------
    print("Building DAE...")
    dae = build_denoising_autoencoder(cat_cols, cat_max, emb_dims, noise_level=0.3)

    print("Training DAE...")
    dae.fit(
        cat_train, cat_train,
        epochs=50,
        batch_size=1024,
        validation_data=(cat_val, cat_val),
        verbose=2,
        callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
    )

    # ------------------- Extract Weights BY NAME -------------------
    print("Extracting embedding weights...")
    pretrained_weights = []
    for col in cat_cols:
        layer = dae.get_layer(f'emb_{col}')  # SAFE LOOKUP
        weights = layer.get_weights()[0]
        pretrained_weights.append(weights)

    # ------------------- Final Classifier -------------------
    print("Building final classifier...")
    classifier = build_classifier_with_pretrained_weights(
        cat_cols, cat_max, num_cols, pretrained_weights
    )

    def df_to_inputs(df):
        num = df[num_cols].values.astype(np.float32)
        cat = [df[col].values.astype(np.int32).reshape(-1, 1) for col in cat_cols]
        return [num] + cat

    train_inputs = df_to_inputs(X_tr)
    val_inputs   = df_to_inputs(X_val)
    test_inputs  = df_to_inputs(X_test)

    classifier.compile(
        optimizer=Adam(0.0005),
        loss='binary_crossentropy',
        metrics=[normalized_gini_tf]
    )

    print("Training final model...")
    classifier.fit(
        train_inputs, y_tr,
        validation_data=(val_inputs, y_val),
        epochs=100,
        batch_size=1024,
        verbose=2,
        callbacks=[
            EarlyStopping(monitor='val_normalized_gini_tf', patience=15, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_normalized_gini_tf', factor=0.5, patience=6, mode='max')
        ]
    )

    # ------------------- Predict -------------------
    val_pred = classifier.predict(val_inputs, batch_size=2048).ravel()
    auc = roc_auc_score(y_val, val_pred)
    gini = 2 * auc - 1
    print(f"\nValidation AUC: {auc:.5f} | Gini: {gini:.5f}")

    test_pred = classifier.predict(test_inputs, batch_size=2048).ravel()
    sub = pd.DataFrame({'id': test_ids, 'target': test_pred})
    sub.to_csv('submission_dae.csv', index=False)
    print("Submission saved: submission_dae.csv")

