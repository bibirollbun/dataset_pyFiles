import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam

train_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')

train_df_clean = train_df.dropna(subset=['sii']).copy()

train_columns = set(train_df_clean.columns)
test_columns = set(test_df.columns)
columns_to_drop = list(train_columns - test_columns - {'sii'})
train_df_clean.drop(columns=columns_to_drop, inplace=True)
test_df.drop(columns=list(test_columns - train_columns), errors='ignore', inplace=True)

label_encoder = LabelEncoder()
categorical_columns = train_df_clean.select_dtypes(include=['object']).columns

for col in categorical_columns:
    train_df_clean[col] = train_df_clean[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    train_df_clean[col] = label_encoder.fit_transform(train_df_clean[col])
    test_df[col] = test_df[col].map(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)

numeric_cols = train_df_clean.select_dtypes(include=['float', 'int']).columns
numeric_cols = [col for col in numeric_cols if col != 'sii']

for col in numeric_cols:
    median_value = train_df_clean[col].median()
    train_df_clean[col] = train_df_clean[col].fillna(median_value)
    test_df[col] = test_df[col].fillna(median_value)

print(f"Taille des données après encodage : {train_df_clean.shape}")

X = train_df_clean.drop(columns=['id', 'sii'])
y = train_df_clean['sii']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df.drop(columns=['id']))

num_classes = len(y.unique())
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_val_encoded = to_categorical(y_val, num_classes=num_classes)

X_train_model = X_train_scaled
X_val_model   = X_val_scaled
X_test_model  = X_test_scaled
classes = np.unique(y_train)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Poids des classes :", class_weights)

def create_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    shortcut = x
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='linear', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    
    shortcut = x
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='linear', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

resnet_model = create_resnet(input_shape=X_train_model.shape[1:], num_classes=num_classes)
resnet_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = resnet_model.fit(
    X_train_model, y_train_encoded,
    validation_data=(X_val_model, y_val_encoded),
    epochs=1000,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
    , class_weight=class_weights
)

test_pred_probs = resnet_model.predict(X_test_model)
test_pred = np.argmax(test_pred_probs, axis=1)

submission = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv")
submission['sii'] = test_pred
submission.to_csv('submission.csv', index=False)

print("✅ Fichier de soumission généré avec succès !")
print(submission)


