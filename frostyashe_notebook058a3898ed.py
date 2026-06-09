import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import RootMeanSquaredError



# Táº£i dá»¯ liá»‡u
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# Sá»­ dá»¥ng cÃ¹ng má»™t TfidfVectorizer tá»« notebook gá»‘c
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5), max_features=1000)

print("Ä�ang biáº¿n Ä‘á»•i SMILES thÃ nh vector...")
X = vectorizer.fit_transform(train["SMILES"].fillna(""))
X_test = vectorizer.transform(test["SMILES"].fillna(""))

# Láº¥y 5000 Ä‘áº·c trÆ°ng Ä‘áº§u vÃ o cho Máº¡ng NÆ¡-ron
INPUT_SHAPE = X.shape[1]
print(f"Sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng Ä‘áº§u vÃ o: {INPUT_SHAPE}")


datasets = {}
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for target in target_cols:
    print(f"Ä�ang chuáº©n bá»‹ dá»¯ liá»‡u cho: {target}")
    mask = train[target].notna()
    X_target = X[mask.values]
    y_target = train.loc[mask, target].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_target, y_target, 
        test_size=0.2, 
        random_state=42
    )
    
    # Keras/TF hoáº¡t Ä‘á»™ng tá»‘t nháº¥t vá»›i máº£ng NumPy (dense arrays)
    # ChÃºng ta chuyá»ƒn Ä‘á»•i ma tráº­n thÆ°a (sparse matrix) TF-IDF thÃ nh máº£ng dÃ y (dense array)
    datasets[target] = (
        X_train.toarray(), 
        X_val.toarray(), 
        y_train, 
        y_val
    )
    print(f"-> {target}: {len(y_target)} máº«u -> Train: {len(y_train)}, Val: {len(y_val)}")


def create_nn_model(target, input_shape):
    """Táº¡o má»™t mÃ´ hÃ¬nh Neural Network (MLP) cho má»™t má»¥c tiÃªu cá»¥ thá»ƒ."""
    
    inputs = Input(shape=(input_shape,))
    
    if target == 'FFV':
        # Kiáº¿n trÃºc lá»›n hÆ¡n cho FFV (vÃ¬ cÃ³ nhiá»�u dá»¯ liá»‡u)
        x = Dense(256)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x) # ThÃªm lá»›p má»›i
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0.2)(x) # Dropout tháº¥p hÆ¡n cho lá»›p sÃ¢u hÆ¡n
    else:
        # Kiáº¿n trÃºc nhá»� hÆ¡n + Regularization máº¡nh hÆ¡n cho cÃ¡c má»¥c tiÃªu Ã­t dá»¯ liá»‡u
        x = Dense(64)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0,7)(x) # Dropout cao Ä‘á»ƒ chá»‘ng overfitting
        
        x = Dense(32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = Dropout(0.5)(x)

    # Lá»›p output (Há»“i quy)
    outputs = Dense(1, activation='linear', name=target)(x)
    
    model = Model(inputs, outputs)
    
    # BiÃªn dá»‹ch mÃ´ hÃ¬nh, sá»­ dá»¥ng MAE lÃ m hÃ m máº¥t mÃ¡t (loss) vÃ¬ Ä‘Ã³ lÃ  thÆ°á»›c Ä‘o chÃºng ta quan tÃ¢m
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mean_absolute_error', 
                  metrics=[RootMeanSquaredError(name='rmse')])
    
    return model


models = {}
mae_results = {}

# Callback Ä‘á»ƒ dá»«ng sá»›m náº¿u val_mae khÃ´ng cáº£i thiá»‡n sau 20 epochs
# vÃ  khÃ´i phá»¥c láº¡i trá»�ng sá»‘ tá»‘t nháº¥t.
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20, 
    restore_best_weights=True, 
    verbose=1
)

for target in target_cols:
    print(f"\n--- ğŸš€ Báº¯t Ä‘áº§u huáº¥n luyá»‡n cho {target} ---")
    
    # Láº¥y dá»¯ liá»‡u (Ä‘Ã£ chuyá»ƒn sang .toarray())
    (X_train, X_val, y_train, y_val) = datasets[target]
    
    # Táº¡o mÃ´ hÃ¬nh
    model = create_nn_model(target, INPUT_SHAPE)
    
    if target == target_cols[0]: # Chá»‰ in summary cho mÃ´ hÃ¬nh Ä‘áº§u tiÃªn
        model.summary()
        
    # Huáº¥n luyá»‡n
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,          # Ä�áº·t sá»‘ epoch cao, EarlyStopping sáº½ lo pháº§n cÃ²n láº¡i
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # LÆ°u láº¡i mÃ´ hÃ¬nh vÃ  káº¿t quáº£
    models[target] = model
    best_val_mae = min(history.history['val_loss'])
    best_train_mae = min(history.history['loss'])
    
    mae_results[target] = {'train': best_train_mae, 'val': best_val_mae}
    print(f"--- âœ… HoÃ n thÃ nh {target} ---")
    print(f"{target}: Best Train MAE = {best_train_mae:.4f}, Best Validation MAE = {best_val_mae:.4f}\n")

