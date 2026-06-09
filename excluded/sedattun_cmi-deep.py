import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, optimizers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Sabitler ve Yapılandırma
class Config:
    SEED = 42
    N_SPLITS = 5
    EPOCHS = 100
    BATCH_SIZE = 64
    PATIENCE = 15
    LR = 0.001
    VERBOSE = 1
    MAX_SEQ_LEN = 100  # Sabit sekans uzunluğu
    DROPOUT_RATE = 0.4
    NUM_HEADS = 4  # Dikkat katmanları için

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(Config.SEED)

# GLOBAL INPUT_DIR tanımı
INPUT_DIR = '/kaggle/input/cmi-detect-behavior-with-sensor-data'

# Veri yükleme
def load_data():
    try:
        train = pd.read_parquet(f'{INPUT_DIR}/train.parquet')
        test = pd.read_parquet(f'{INPUT_DIR}/test.parquet')
        print("Parquet dosyaları yüklendi")
    except:
        train = pd.read_csv(f'{INPUT_DIR}/train.csv')
        test = pd.read_csv(f'{INPUT_DIR}/test.csv')
        print("CSV dosyaları yüklendi")
    
    # Demografik verileri yükle
    try:
        train_demo = pd.read_csv(f'{INPUT_DIR}/train_demographics.csv')
        test_demo = pd.read_csv(f'{INPUT_DIR}/test_demographics.csv')
        
        # Demografik verileri birleştir
        train = train.merge(train_demo, on='subject', how='left')
        test = test.merge(test_demo, on='subject', how='left')
        print("Demografik veriler birleştirildi")
    except Exception as e:
        print(f"Demografik veri yükleme hatası: {e}")
    
    return train, test

# Gelişmiş Özellik Mühendisliği
def get_feature_columns(df):
    # IMU özellikleri
    imu_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    
    # Thermopile özellikleri
    thm_features = [col for col in df.columns if col.startswith('thm_')]
    
    # Time-of-flight özellikleri
    tof_features = [col for col in df.columns if col.startswith('tof_')]
    
    # Demografik özellikler
    demo_features = ['age', 'sex', 'handedness', 'height_cm', 
                    'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    
    # ToF sensörlerini ayrı ayrı işle
    tof_sensors = {}
    for i in range(1, 6):
        sensor_cols = [col for col in tof_features if col.startswith(f'tof_{i}_')]
        if sensor_cols:  # Sadece mevcut sensörleri ekle
            tof_sensors[f'tof_{i}'] = sensor_cols
    
    return {
        'imu': imu_features,
        'thm': thm_features,
        'tof': tof_features,
        'tof_sensors': tof_sensors,
        'demo': demo_features
    }

# Gelişmiş Ön İşleme
def preprocess_data(df, feature_config):
    df_processed = df.copy()
    
    # IMU işleme
    for col in feature_config['imu']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Thermopile işleme
    for col in feature_config['thm']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # ToF işleme - özel işlem
    for sensor, cols in feature_config['tof_sensors'].items():
        sensor_cols = [col for col in cols if col in df_processed.columns]
        if sensor_cols:
            # -1 değerlerini NaN ile değiştir
            df_processed[sensor_cols] = df_processed[sensor_cols].replace(-1, np.nan)
            # NaN değerleri sensör medyanı ile doldur
            df_processed[sensor_cols] = df_processed[sensor_cols].fillna(
                df_processed[sensor_cols].median())
    
    # Demografik işleme
    for col in feature_config['demo']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

# Sekans oluşturma (DÜZELTİLDİ - 'behavior' sütunu kontrolü)
def create_sequences(df, feature_config, seq_len=Config.MAX_SEQ_LEN):
    sequences = {
        'imu': [],
        'thm': [],
        'tof': [],
        'demo': []
    }
    sequence_ids = []
    
    grouped = df.groupby('sequence_id')
    
    for seq_id, group in grouped:
        # 'behavior' sütunu varsa gesture fazını seç, yoksa tüm veriyi kullan
        if 'behavior' in group.columns:
            gesture_data = group[group['behavior'].str.contains('gesture', case=False, na=False)]
            if len(gesture_data) == 0:
                gesture_data = group
        else:
            gesture_data = group  # Test seti için tüm veriyi kullan
        
        # IMU verileri
        imu_data = gesture_data[feature_config['imu']].values
        imu_data = pad_or_truncate(imu_data, seq_len)
        sequences['imu'].append(imu_data)
        
        # Thermopile verileri
        if feature_config['thm']:
            thm_data = gesture_data[feature_config['thm']].values
            thm_data = pad_or_truncate(thm_data, seq_len)
            sequences['thm'].append(thm_data)
        
        # ToF verileri - uzamsal format (8x8 grid)
        if feature_config['tof_sensors']:
            tof_data = []
            for sensor, cols in feature_config['tof_sensors'].items():
                sensor_cols = [col for col in cols if col in gesture_data.columns]
                if sensor_cols:
                    # Her zaman adımı için 8x8 grid oluştur
                    sensor_values = gesture_data[sensor_cols].values
                    sensor_values = pad_or_truncate(sensor_values, seq_len)
                    
                    # 8x8 grid haline getir (zaman_adimi, 64) -> (zaman_adimi, 8, 8)
                    sensor_values = sensor_values.reshape(-1, 8, 8)
                    tof_data.append(sensor_values)
            
            if tof_data:
                # Sensörleri kanal boyutunda birleştir
                tof_data = np.stack(tof_data, axis=-1)  # Şekil: (seq_len, 8, 8, num_sensors)
                sequences['tof'].append(tof_data)
        
        # Demografik veriler (tüm zaman adımları için tekrarla)
        if feature_config['demo'] and gesture_data[feature_config['demo']].size > 0:
            demo_data = gesture_data[feature_config['demo']].values
            demo_data = demo_data[0]  # İlk satırı al (hepsi aynı)
            demo_data = np.tile(demo_data, (seq_len, 1))
            sequences['demo'].append(demo_data)
        
        sequence_ids.append(seq_id)
    
    # Numpy dizilerine dönüştür
    for key in sequences:
        if sequences[key]:
            sequences[key] = np.array(sequences[key])
        else:
            sequences[key] = None  # Yoksa None olarak ayarla
    
    return sequences, sequence_ids

def pad_or_truncate(data, target_length):
    if len(data) < target_length:
        # Doldurma
        pad_length = target_length - len(data)
        padded_data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        return padded_data
    else:
        # Kırpma
        return data[:target_length]

# Gelişmiş Çok Modlu Model Mimarisi
def create_advanced_model(feature_config, num_classes):
    # Girdi katmanları
    inputs = {}
    
    # IMU dalı - zamansal işleme
    imu_input = layers.Input(shape=(Config.MAX_SEQ_LEN, len(feature_config['imu'])), name='imu_input')
    inputs['imu'] = imu_input
    
    # IMU işleme + dikkat mekanizması
    x_imu = layers.Conv1D(128, 5, activation='relu', padding='same')(imu_input)
    x_imu = layers.BatchNormalization()(x_imu)
    x_imu = layers.Conv1D(128, 5, activation='relu', padding='same')(x_imu)
    x_imu = layers.MaxPooling1D(2)(x_imu)
    x_imu = layers.Dropout(Config.DROPOUT_RATE)(x_imu)
    
    # Çok kafalı öz-dikkat
    query = layers.Dense(128)(x_imu)
    key = layers.Dense(128)(x_imu)
    value = layers.Dense(128)(x_imu)
    attention = layers.Attention(use_scale=True)([query, key, value])
    x_imu = layers.Concatenate()([x_imu, attention])
    
    x_imu = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x_imu)
    x_imu = layers.GlobalAveragePooling1D()(x_imu)
    
    # Thermopile dalı (mevcutsa)
    if feature_config['thm']:
        thm_input = layers.Input(shape=(Config.MAX_SEQ_LEN, len(feature_config['thm'])), name='thm_input')
        inputs['thm'] = thm_input
        
        x_thm = layers.Conv1D(64, 3, activation='relu', padding='same')(thm_input)
        x_thm = layers.BatchNormalization()(x_thm)
        x_thm = layers.LSTM(64, return_sequences=False)(x_thm)
    else:
        x_thm = None
    
    # ToF dalı (mevcutsa) - uzamsal işleme
    if feature_config['tof_sensors']:
        tof_input = layers.Input(
            shape=(Config.MAX_SEQ_LEN, 8, 8, len(feature_config['tof_sensors'])), 
            name='tof_input')
        inputs['tof'] = tof_input
        
        # CNN ile her zaman adımını işle
        x_tof = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(tof_input)
        x_tof = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x_tof)
        x_tof = layers.TimeDistributed(layers.Flatten())(x_tof)
        x_tof = layers.LSTM(64, return_sequences=False)(x_tof)
    else:
        x_tof = None
    
    # Demografik dal
    if feature_config['demo']:
        demo_input = layers.Input(shape=(Config.MAX_SEQ_LEN, len(feature_config['demo'])), name='demo_input')
        inputs['demo'] = demo_input
        x_demo = layers.Dense(32, activation='relu')(demo_input[:, 0, :])  # Sadece ilk zaman adımı
    else:
        # Demografik veri yoksa sıfır vektörü
        demo_input = layers.Input(shape=(Config.MAX_SEQ_LEN, 1), name='demo_input')
        inputs['demo'] = demo_input
        x_demo = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 1)))(demo_input)
    
    # Dalları birleştir
    combined = [x_imu, x_demo]
    if x_thm is not None:
        combined.append(x_thm)
    if x_tof is not None:
        combined.append(x_tof)
    
    x = layers.Concatenate()(combined)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(Config.DROPOUT_RATE/2)(x)
    
    # Çıktılar
    binary_output = layers.Dense(1, activation='sigmoid', name='binary_output')(x)
    multi_output = layers.Dense(num_classes, activation='softmax', name='multi_output')(x)
    
    return models.Model(inputs=inputs, outputs=[binary_output, multi_output])

# Ana Fonksiyon (DÜZELTİLDİ)
def main():
    # Veri yükleme
    train_df, test_df = load_data()
    
    # Özellik yapılandırması
    feature_config = get_feature_columns(train_df)
    print("Özellik yapılandırması oluşturuldu")
    
    # Veri ön işleme
    train_df = preprocess_data(train_df, feature_config)
    test_df = preprocess_data(test_df, feature_config)
    print("Veri ön işlendi")
    
    # Hedef değişkenler (sadece eğitim seti için)
    if 'sequence_type' in train_df.columns:
        train_df['binary_target'] = (train_df['sequence_type'] == 'Target').astype(int)
    else:
        # Test seti için geçici değerler
        train_df['binary_target'] = 0
    
    if 'gesture' in train_df.columns:
        le = LabelEncoder()
        train_df['gesture_encoded'] = le.fit_transform(train_df['gesture'])
        num_classes = len(le.classes_)
        print(f"Sınıf sayısı: {num_classes}")
        print(f"Sınıflar: {le.classes_}")
    else:
        # Test seti için geçici değerler
        train_df['gesture_encoded'] = 0
        num_classes = 1
        le = None
    
    # Sekansları oluştur
    print("Eğitim sekansları oluşturuluyor...")
    train_sequences, train_seq_ids = create_sequences(train_df, feature_config)
    print("Test sekansları oluşturuluyor...")
    test_sequences, test_seq_ids = create_sequences(test_df, feature_config)
    
    # Hedef değişkenleri hazırla (sadece eğitim seti için)
    if 'sequence_id' in train_df.columns and 'binary_target' in train_df.columns:
        train_meta = train_df.drop_duplicates('sequence_id').set_index('sequence_id')
        train_meta = train_meta.loc[train_seq_ids]
        y_binary = train_meta['binary_target'].values
        y_multi = train_meta['gesture_encoded'].values
    else:
        # Test seti için geçici değerler
        y_binary = np.zeros(len(train_seq_ids))
        y_multi = np.zeros(len(train_seq_ids))
    
    # Normalizasyon
    scalers = {}
    for key in ['imu', 'thm', 'demo']:
        if train_sequences[key] is not None and len(train_sequences[key]) > 0:
            # Ölçeklendirme için yeniden şekillendir
            orig_shape = train_sequences[key].shape
            flattened = train_sequences[key].reshape(-1, orig_shape[-1])
            
            scaler = StandardScaler()
            scaled = scaler.fit_transform(flattened)
            train_sequences[key] = scaled.reshape(orig_shape)
            
            if test_sequences[key] is not None and len(test_sequences[key]) > 0:
                orig_shape_test = test_sequences[key].shape
                flattened_test = test_sequences[key].reshape(-1, orig_shape_test[-1])
                scaled_test = scaler.transform(flattened_test)
                test_sequences[key] = scaled_test.reshape(orig_shape_test)
            
            scalers[key] = scaler
    
    # Çapraz doğrulama (sadece eğitim verisi varsa)
    if len(y_binary) > 0 and num_classes > 1:
        skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
        
        oof_binary = np.zeros(len(train_seq_ids))
        oof_multi = np.zeros((len(train_seq_ids), num_classes))
        
        test_pred_binary = np.zeros(len(test_seq_ids))
        test_pred_multi = np.zeros((len(test_seq_ids), num_classes))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_sequences['imu'], y_binary)):
            print(f"\n=== Fold {fold + 1} ===")
            
            # Girdileri hazırla
            train_input = {'imu': train_sequences['imu'][train_idx]}
            val_input = {'imu': train_sequences['imu'][val_idx]}
            test_input = {'imu': test_sequences['imu']}
            
            # Diğer modaliteleri ekle (mevcutsa)
            for modality in ['thm', 'tof', 'demo']:
                if modality in train_sequences and train_sequences[modality] is not None:
                    train_input[modality] = train_sequences[modality][train_idx]
                    val_input[modality] = train_sequences[modality][val_idx]
                    if modality in test_sequences and test_sequences[modality] is not None:
                        test_input[modality] = test_sequences[modality]
            
            # Model oluştur
            model = create_advanced_model(feature_config, num_classes)
            
            # Modeli derle
            model.compile(
                optimizer=optimizers.Adam(learning_rate=Config.LR),
                loss={
                    'binary_output': 'binary_crossentropy',
                    'multi_output': 'sparse_categorical_crossentropy'
                },
                loss_weights={'binary_output': 1.0, 'multi_output': 1.0},
                metrics={
                    'binary_output': ['accuracy'],
                    'multi_output': ['accuracy']
                }
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss', patience=Config.PATIENCE, 
                    restore_best_weights=True, verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=7, 
                    min_lr=1e-7, verbose=1
                )
            ]
            
            # Eğitim
            history = model.fit(
                train_input,
                {'binary_output': y_binary[train_idx], 'multi_output': y_multi[train_idx]},
                validation_data=(
                    val_input,
                    {'binary_output': y_binary[val_idx], 'multi_output': y_multi[val_idx]}
                ),
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                callbacks=callbacks_list,
                verbose=Config.VERBOSE
            )
            
            # Tahmin
            val_pred = model.predict(val_input, verbose=0)
            oof_binary[val_idx] = val_pred[0].flatten()
            oof_multi[val_idx] = val_pred[1]
            
            test_pred = model.predict(test_input, verbose=0)
            test_pred_binary += test_pred[0].flatten() / Config.N_SPLITS
            test_pred_multi += test_pred[1] / Config.N_SPLITS
            
            # Belleği temizle
            del model
            tf.keras.backend.clear_session()
        
        # Değerlendirme
        oof_binary_class = (oof_binary > 0.5).astype(int)
        oof_multi_class = np.argmax(oof_multi, axis=1)
        
        binary_f1 = f1_score(y_binary, oof_binary_class)
        multi_f1 = f1_score(y_multi, oof_multi_class, average='macro')
        final_score = (binary_f1 + multi_f1) / 2
        
        print(f"\n=== Sonuçlar ===")
        print(f"Binary F1: {binary_f1:.4f}")
        print(f"Multi F1: {multi_f1:.4f}")
        print(f"Final Skor: {final_score:.4f}")
    else:
        # Sadece test seti için tahmin yap
        test_pred_binary = np.zeros(len(test_seq_ids))
        test_pred_multi = np.zeros((len(test_seq_ids), num_classes))
    
    # Submission dosyası hazırlama
    if le is not None:
        test_binary_class = (test_pred_binary > 0.5).astype(int)
        test_multi_class = np.argmax(test_pred_multi, axis=1)
        
        submission_preds = []
        for i in range(len(test_binary_class)):
            if test_binary_class[i] == 0:
                submission_preds.append('non_target')
            else:
                submission_preds.append(le.inverse_transform([test_multi_class[i]])[0])
        
        submission = pd.DataFrame({
            'sequence_id': test_seq_ids,
            'gesture': submission_preds
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Submission dosyası kaydedildi!")
    else:
        print("LabelEncoder bulunamadı, submission oluşturulamadı")

if __name__ == "__main__":
    main()

