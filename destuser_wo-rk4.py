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





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import librosa
import os

# 假设数据路径（这里根据你上传文件路径调整，若实际不同需修改）
data_root = "/kaggle/input/birdclef-2025"

# 训练数据的标注文件路径，为你上传的 train.csv
train_metadata_path = os.path.join(data_root, "train.csv")

# 检查训练数据标注文件是否存在
if not os.path.exists(train_metadata_path):
    print(f"训练数据标注文件 {train_metadata_path} 不存在，请检查文件名是否正确。")
else:
    print(f"训练数据标注文件 {train_metadata_path} 存在。")
    # 加载训练数据的标注信息
    train_metadata = pd.read_csv(train_metadata_path)

    # 定义音频特征提取函数
    def extract_audio_features(audio_path, n_fft=2048, hop_length=512):
        audio, sr = librosa.load(audio_path)
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(spectrogram)
        mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(S=spectrogram, sr=sr)
        mel = librosa.feature.melspectrogram(audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=sr)
        features = np.concatenate((np.mean(spectrogram, axis=1), np.mean(mfccs, axis=1),
                                   np.mean(chroma, axis=1), np.mean(mel, axis=1),
                                   np.mean(contrast, axis=1)))
        return features

    # 构建完整的音频文件路径并提取特征，同时获取标签
    X = []
    y = []
    for index, row in train_metadata.iterrows():
        audio_name = row['filename']
        # 这里根据音频文件名的实际情况，可能需要进一步处理路径等信息，
        # 目前简单假设音频文件就在 train_audio 目录下，若实际不同需修改
        audio_path = os.path.join(data_root, "train_audio", audio_name)
        species_label = row['primary_label']
        try:
            features = extract_audio_features(audio_path)
            X.append(features)
            y.append(species_label)
        except:
            print(f"处理音频 {audio_path} 时出错，已跳过")

    X = np.array(X)
    y = np.array(y)

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 构建分类器管道（以线性SVM为例）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC())
    ])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 预测并评估
    y_pred_proba = pipeline.predict_proba(X_test)
    # 由于竞赛评估指标是跳过无真正阳性标签类别的宏平均ROC-AUC，这里简化计算，实际需按竞赛数据处理
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    print(f"Macro-averaged ROC-AUC score: {auc_score}")

    # 以下部分为对测试集进行预测（如果竞赛有测试集数据且需要预测结果的话）
    # 假设测试数据标注文件路径
    test_metadata_path = os.path.join(data_root, "test.csv")

    # 检查测试数据标注文件是否存在
    if not os.path.exists(test_metadata_path):
        print(f"测试数据标注文件 {test_metadata_path} 不存在，请检查文件名是否正确。")
    else:
        print(f"测试数据标注文件 {test_metadata_path} 存在。")
        test_metadata = pd.read_csv(test_metadata_path)

        test_X = []
        for index, row in test_metadata.iterrows():
            audio_name = row['filename']
            audio_path = os.path.join(data_root, "test_audio", audio_name)
            try:
                features = extract_audio_features(audio_path)
                test_X.append(features)
            except:
                print(f"处理测试音频 {audio_path} 时出错，已跳过")

        test_X = np.array(test_X)
        test_y_pred = pipeline.predict(test_X)
        test_y_pred_labels = label_encoder.inverse_transform(test_y_pred)

        # 这里可以将预测结果保存为合适的格式，以便提交竞赛
        # 例如保存为CSV文件
        submission = pd.DataFrame({
            'filename': test_metadata['filename'],
            'primary_label': test_y_pred_labels
        })
        submission.to_csv('submission.csv', index=False)

