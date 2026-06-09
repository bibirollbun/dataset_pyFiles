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


import pandas as pd
import re
from tqdm import tqdm
import sys
from collections import defaultdict
import spacy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import spacy

# 경로 추가 (zip 내부 폴더 기준)
sys.path.append("/kaggle/input/package1")

# eng_to_ipa 불러오기
import eng_to_ipa as ipa

# spaCy 모델 불러오기
nlp = spacy.load("en_core_web_sm")

# 학습 데이터
train_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

# 테스트 데이터
test_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")

# 제출 양식
submission_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")


def build_ipa_class_map():
    ipa_class_map = defaultdict(lambda: defaultdict(list))

    # === 자음 분류 ===
    ipa_class_map['manner']['plosive']     += ['p', 'b', 't', 'd', 'k', 'g']
    ipa_class_map['manner']['fricative']   += ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']
    ipa_class_map['manner']['affricate']   += ['tʃ', 'dʒ']
    ipa_class_map['manner']['nasal']       += ['m', 'n', 'ŋ']
    ipa_class_map['manner']['approximant'] += ['ɹ', 'j', 'w']
    ipa_class_map['manner']['lateral']     += ['l']

    ipa_class_map['place']['bilabial']      += ['p', 'b', 'm']
    ipa_class_map['place']['labiodental']   += ['f', 'v']
    ipa_class_map['place']['dental']        += ['θ', 'ð']
    ipa_class_map['place']['alveolar']      += ['t', 'd', 's', 'z', 'n', 'l']
    ipa_class_map['place']['post-alveolar'] += ['ʃ', 'ʒ', 'tʃ', 'dʒ']
    ipa_class_map['place']['velar']         += ['k', 'g', 'ŋ']
    ipa_class_map['place']['glottal']       += ['h', 'ʔ']
    ipa_class_map['place']['palatal']       += ['j']

    ipa_class_map['height']['close'] += ['i', 'ɪ', 'u', 'ʊ']
    ipa_class_map['height']['mid']   += ['e', 'ə', 'ɛ', 'ʌ', 'o', 'ɔ']
    ipa_class_map['height']['open']  += ['a', 'æ', 'ɑ']

    ipa_class_map['backness']['front']   += ['i', 'ɪ', 'e', 'ɛ', 'æ']
    ipa_class_map['backness']['central'] += ['ə', 'ʌ']
    ipa_class_map['backness']['back']    += ['u', 'ʊ', 'o', 'ɔ', 'ɑ']

    ipa_class_map['rounding']['rounded']   += ['u', 'ʊ', 'o', 'ɔ']
    ipa_class_map['rounding']['unrounded'] += ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ə', 'ʌ']

    ipa_class_map['length']['long']  += ['iː', 'uː', 'ɔː', 'ɑː', 'ɜː']
    ipa_class_map['length']['short'] += ['ɪ', 'ʊ', 'ə', 'ʌ', 'ɛ', 'æ']

    ipa_class_map['diphthong']['diphthong'] += [
        'aɪ', 'eɪ', 'ɔɪ', 'aʊ', 'oʊ', 'əʊ', 'ɪə', 'eə', 'ʊə'
    ]

    return ipa_class_map

def spacy_ratio_df(df, id_col='id', text_col='text'):
    tags = ['NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ', 'SCONJ', 'NUM', 'INTJ']
    result = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing POS ratios"):
        text = row[text_col]
        text_id = row[id_col]
        doc = nlp(text)
        total = len(doc)
        pos_counts = {tag: 0 for tag in tags}

        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1

        ratios = {f"{tag.lower()}_ratio": pos_counts[tag]/total if total > 0 else 0 for tag in tags}
        ratios['id'] = text_id
        result.append(ratios)

    df_ratio = pd.DataFrame(result)
    return df_ratio

def classify_ipa_df(df, ipa_class_map, id_col='id', text_col='text', label_col=None):
    results = []

    all_categories = list(ipa_class_map.keys())
    all_subgroups = {cat: list(ipa_class_map[cat].keys()) for cat in all_categories}
    all_symbols = sum([v for d in ipa_class_map.values() for v in d.values()], [])

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting IPA class ratios"):
        text = row[text_col]
        text_id = row[id_col]
        label = row[label_col] if label_col and label_col in row else None

        ipa_text = ipa.convert(text)

        tokens = []
        idx = 0
        while idx < len(ipa_text):
            two = ipa_text[idx:idx+2]
            one = ipa_text[idx]
            if two in all_symbols:
                tokens.append(two)
                idx += 2
            elif re.match(r'[a-zɪʊəɔæɑʌɛʃʒθðŋɹʔ]', one):
                tokens.append(one)
                idx += 1
            else:
                idx += 1

        total = len(tokens)
        counts = {f"{cat}_{sub}_ratio": 0 for cat in all_categories for sub in all_subgroups[cat]}

        for token in tokens:
            for cat in all_categories:
                for sub in all_subgroups[cat]:
                    if token in ipa_class_map[cat][sub]:
                        counts[f"{cat}_{sub}_ratio"] += 1

        for key in counts:
            counts[key] = counts[key] / total if total > 0 else 0

        counts['id'] = text_id
        if label is not None:
            counts['generated'] = label

        results.append(counts)

    return pd.DataFrame(results)

def preprocess_features(df, id_col='id', label_col='generated'):
    df = df.dropna().reset_index(drop=True)
    feature_cols = [col for col in df.columns if col not in [id_col, label_col]]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df[label_col].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val


ipa_class_map = build_ipa_class_map()
print("세팅 성공")


# Train 데이터
df_pos = spacy_ratio_df(train_df)
df_ipa = classify_ipa_df(train_df, ipa_class_map, label_col='generated')
df_full = pd.merge(df_pos, df_ipa, on='id')

# Test 데이터
df_pos_test = spacy_ratio_df(test_df)
df_ipa_test = classify_ipa_df(test_df, ipa_class_map)
df_test_full = pd.merge(df_pos_test, df_ipa_test, on='id')


X_train, X_val, y_train, y_val = preprocess_features(df_full)

# test set 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 학습 데이터의 피처 목록 추출
feature_cols = [col for col in df_full.columns if col not in ['id', 'generated']]
# 테스트셋에서 동일한 피처만 선택
X_test = df_test_full.drop(columns=['id']).values
X_test = df_test_full[feature_cols].values
X_test_scaled = scaler.transform(X_test)


# ✅ 텐서플로우 기반 MLP 모델 정의 + 학습
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# 클래스 가중치 자동 계산
weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print("Class Weights:", class_weights)

# 1. 모델 구조 정의 함수
def build_model(input_dim):
    model = models.Sequential()

    # 입력 레이어
    model.add(layers.Input(shape=(input_dim,)))

    # 첫 번째 Dense 블록
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # 두 번째 Dense 블록
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    # 세 번째 Dense
    model.add(layers.Dense(64, activation='relu'))

    # 출력층 (이진 분류이므로 sigmoid)
    model.add(layers.Dense(1, activation='sigmoid'))

    # 컴파일
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 2. EarlyStopping 콜백 설정
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# 3. 모델 생성 및 학습
model = build_model(X_train_scaled.shape[1])
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    class_weight=class_weights
)

# 4. 검증 성능 평가
y_pred_val = (model.predict(X_val_scaled) > 0.5).astype(int)
print(classification_report(y_val, y_pred_val))


# 테스트셋 예측
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
print(y_pred.shape)

# 제출 파일 생성
submission = submission_df.copy()
try:
    submission['generated'] = y_pred
    submission.to_csv("submission.csv", index=False)
    print("✅ 제출 파일 저장 완료: submission.csv")
except:
    submission.to_csv("submission.csv", index=False)
    print("✅ 제출 파일 저장 완료: (except)submission.csv")




