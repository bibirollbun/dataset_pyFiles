# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

# ==================== 配置部分 ====================
DATA_DIR = '/kaggle/input/birdclef-2025/'
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, 'train_audio')
TEST_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, 'test_soundscapes')
TRAIN_METADATA = os.path.join(DATA_DIR, 'train.csv')
TAXONOMY = os.path.join(DATA_DIR, 'taxonomy.csv')
SUBMISSION_PATH = '/kaggle/working/submission.csv'

# ==================== 特征提取部分 ====================
def extract_features(audio, sr=32000, n_mels=128, n_mfcc=20):
    """提取音频特征（Mel频谱图+MFCC+Chromagram）"""
    if len(audio) < sr * 5:
        audio = np.pad(
            audio,
            (0, max(0, sr * 5 - len(audio))),
            mode='constant'
        )
    else:
        audio = audio[:sr * 5]

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        fmin=20, fmax=16000,
        n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr,
        n_fft=2048, hop_length=512
    )

    features = np.concatenate([
        np.mean(mel_spec_db, axis=1),
        np.std(mel_spec_db, axis=1),
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1),
        np.mean(chroma, axis=1)
    ])
    return (features - features.min()) / (features.max() - features.min() + 1e-8)

# ==================== 模型定义部分 ====================
def create_model(input_shape, num_classes):
    """创建CNN+BiLSTM混合模型"""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3, 1), activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D((2, 1)),
        Dropout(0.3),

        Conv2D(64, (3, 1), activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D((2, 1)),
        Dropout(0.3),

        Conv2D(128, (3, 1), activation='relu', padding='valid'),
        BatchNormalization(),
        MaxPooling2D((2, 1)),
        Dropout(0.3),

        Reshape((-1, 128)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==================== 数据准备部分 ====================
def load_and_prepare_data():
    """加载并准备训练数据"""
    metadata = pd.read_csv(TRAIN_METADATA)
    taxonomy = pd.read_csv(TAXONOMY)
    data = metadata.merge(taxonomy, on='primary_label', how='left')

    data['filepath'] = data['filename'].apply(
        lambda f: os.path.join(TRAIN_AUDIO_DIR, f)
    )
    if 'rating' in data.columns:
        data = data[data['rating'] >= 3.0]
    return data

# ==================== 训练部分 ====================
def train_model(train_data, classes):
    """训练音频分类模型"""
    le = LabelEncoder()
    y = le.fit_transform(train_data['primary_label'])

    X = []
    print("正在提取特征...")
    for filepath in tqdm(train_data['filepath']):
        audio, sr = librosa.load(filepath, sr=32000)
        features = extract_features(audio, sr)
        X.append(features)

    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1, 1))

    model = create_model(input_shape=(X.shape[1], X.shape[2], X.shape[3]),
                         num_classes=len(classes))
    model.fit(
        X, y,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            # 修复了 ModelCheckpoint 路径名和变量拼写问题
ModelCheckpoint('best_model.weights.h5', save_best_only=True, save_weights_only=True)
# TEST_SOUNDSCAPESDIR 改为 TEST_SOUNDSCAPES_DIR

        ],
        verbose=1
    )
    return model, le

# ==================== 预测部分 ====================
def predict_audio_segment(model, audio_segment, sr=32000):
    """预测单个5秒音频片段"""
    features = extract_features(audio_segment, sr)
    x = features.reshape((1, features.shape[0], 1, 1))
    return model.predict(x, verbose=0)[0]


def process_test_file(file_path, model, classes, sr=32000):
    """处理1分钟测试音频文件"""
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        segments = [audio[i*sr*5:(i+1)*sr*5] for i in range(12)]
        return np.array([predict_audio_segment(model, seg, sr) for seg in segments])
    except Exception as e:
        print(f"处理文件 {file_path} 出错: {str(e)}")
        return None

# ==================== 主流程 ====================
def main():
    print("加载数据...")
    train_data = load_and_prepare_data()
    classes = train_data['primary_label'].unique().tolist()

    print("\n训练模型...")
    model, label_encoder = train_model(train_data, classes)

    print("\n生成提交文件...")
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    required_species = sample_sub.columns[1:].tolist()

    submission = pd.DataFrame(columns=['row_id'] + required_species)
    for file in tqdm(list(Path(TEST_SOUNDSCAPES_DIR).glob('*.ogg')), desc="处理测试音景"):
        soundscape_id = file.stem
        preds = process_test_file(str(file), model, classes)
        if preds is not None:
            for i, pred in enumerate(preds):
                row_data = {'row_id': f"{soundscape_id}_{(i+1)*5}"}
                for species, prob in zip(classes, pred):
                    if species in required_species:
                        row_data[species] = prob
                submission = submission.append(row_data, ignore_index=True)
    
    submission = submission.reindex(columns=['row_id'] + required_species, fill_value=0.0)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n提交文件已保存: {SUBMISSION_PATH}")
    print(submission.head())

if __name__ == "__main__":
    main()


