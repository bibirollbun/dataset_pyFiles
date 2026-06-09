# =================================================================
# ■■■ セル 1：セットアップ ■■■
# =================================================================
# まず、このセルだけを実行してください。
# 実行が完了したら、次のセルを実行してください。
# =================================================================
print("■ ステップ0：ライブラリの準備")

# TensorFlowと、それと互換性のあるNumpyのバージョンを正確に指定してインストール
!pip install -q tensorflow==2.15.0 keras==2.15.0 numpy==1.26.4

# DICOMファイルを扱うためのライブラリをインストール
!pip install -q pydicom

print("\n▶ ライブラリのインストールが完了しました。")
print("▶▶▶ 次のセルに進んでください。◀◀◀")


# =================================================================
# ■■■ セル 2：メインの実験コード ■■■
# =================================================================
# 上のセル1の実行が完了した後に、このセルを実行してください。
# =================================================================
print("■ ステップ1：ライブラリのインポートとデータ準備")

# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm.notebook import tqdm

# TensorFlowとKerasをインポート
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# グラフのスタイルを設定
sns.set_style("whitegrid")

# Kaggle環境でのファイルパスを定義
kaggle_input_path = '/kaggle/input/rsna-pneumonia-detection-challenge/'
labels_path = os.path.join(kaggle_input_path, 'stage_2_train_labels.csv')
detailed_info_path = os.path.join(kaggle_input_path, 'stage_2_detailed_class_info.csv')
dicom_image_dir = os.path.join(kaggle_input_path, 'stage_2_train_images')

# 作業用フォルダを作成
working_dir = '/kaggle/working/'
png_image_dir = os.path.join(working_dir, 'train_images_png')
os.makedirs(png_image_dir, exist_ok=True)

# --- 1. メタデータの作成 ---
df_labels = pd.read_csv(labels_path)
df_detailed = pd.read_csv(detailed_info_path)
df_combined = pd.merge(df_labels, df_detailed, on='patientId')

ages = []
sexes = []
view_positions = []

print("DICOMファイルからメタ情報を抽出中...")
for patient_id in tqdm(df_combined['patientId'].unique()):
    dcm_path = os.path.join(dicom_image_dir, f"{patient_id}.dcm")
    dcm_data = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    ages.append(dcm_data.PatientAge)
    sexes.append(dcm_data.PatientSex)
    view_positions.append(dcm_data.ViewPosition)

meta_dict = {
    'patientId': df_combined['patientId'].unique(),
    'Age': ages,
    'Sex': sexes,
    'ViewPosition': view_positions
}
df_meta = pd.DataFrame(meta_dict)

df_full = pd.merge(df_combined, df_meta, on='patientId')
df_full['Age'] = df_full['Age'].astype(int)

# --- 2. 画像分類タスク用にデータを整形 ---
df_class = df_full[['patientId', 'Target']].drop_duplicates().copy()
df_class['Target'] = df_class['Target'].astype(str)
df_class['filename'] = df_class['patientId'].apply(lambda x: f"{x}.png")

# --- 3. PNGへの変換 ---
print("DICOMをPNGに変換中...")
for patient_id in tqdm(df_class['patientId']):
    dcm_path = os.path.join(dicom_image_dir, f"{patient_id}.dcm")
    png_path = os.path.join(png_image_dir, f"{patient_id}.png")
    
    dcm_data = pydicom.dcmread(dcm_path)
    image = dcm_data.pixel_array
    cv2.imwrite(png_path, image)

# --- 4. 最終的な学習・検証データの作成 ---
df_train, df_val = train_test_split(df_class, test_size=0.2, random_state=42, stratify=df_class['Target'])

print(f"\nデータ総数: {len(df_class)}件")
print(f"学習用データ数: {len(df_train)}件")
print(f"検証用データ数: {len(df_val)}件")
print("▶ 全てのデータ準備完了")


# =================================================================
# ■■■ ステップ2：関数定義 ■■■
# =================================================================
print("\n■ ステップ2：学習・評価用の関数を定義")

def train_classification_model(train_df, val_df, epochs=5, image_size=224, batch_size=32):
    # --- 1. ImageDataGeneratorの準備 ---
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
        brightness_range=[0.9, 1.1]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, directory=png_image_dir, x_col='filename', y_col='Target',
        target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary'
    )
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, directory=png_image_dir, x_col='filename', y_col='Target',
        target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary', shuffle=False
    )
    
    # --- 2. クラスの重みを計算 ---
    weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
    )
    class_weights = {i : weights[i] for i in range(len(weights))}
    
    # --- 3. モデルの構築 ---
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # --- 4. 最初の学習 ---
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    print(f"--- {len(train_df)}件のデータで学習開始 (Phase 1) ---")
    model.fit(
        train_generator, epochs=epochs, validation_data=validation_generator,
        class_weight=class_weights, verbose=2
    )
    
    # --- 5. ファインチューニング ---
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    print(f"--- ファインチューニング開始 (Phase 2) ---")
    model.fit(
        train_generator, epochs=epochs, validation_data=validation_generator,
        class_weight=class_weights, verbose=2
    )

    # --- 6. 最終評価 ---
    loss, accuracy = model.evaluate(validation_generator)
    print(f"--- 評価完了 ---")
    return accuracy

print("▶ 関数定義完了")


# =================================================================
# ■■■ ステップ3：実験A 実行 ■■■
# =================================================================
print("\n■ ステップ3：実験Aの実行")

# 学習に使うデータの上限を設定
MAX_TRAIN_SAMPLES = 6000
df_train_lean = df_train.sample(n=MAX_TRAIN_SAMPLES, random_state=42)

# 検証するデータ量を定義
data_sizes = [500, 1000, 2000, 4000, 6000] 
results_A = []

# 各データサイズで学習と評価を繰り返す
for size in data_sizes:
    df_train_subset = df_train_lean.sample(n=size, random_state=42)
    print(f"\nデータ量 {size}件 での処理を開始...")
    accuracy = train_classification_model(df_train_subset, df_val, epochs=5, image_size=128, batch_size=32)
    print(f"★★ データ量 {size}件 の最終スコア(Accuracy): {accuracy:.4f} ★★")
    results_A.append({'size': size, 'accuracy': accuracy})

print("\n▶ 全ての実験が完了しました。")


# =================================================================
# ■■■ ステップ4：結果可視化 ■■■
# =================================================================
print("\n■ ステップ4：実験Aの結果を可視化")

df_results_A = pd.DataFrame(results_A)
df_results_A.to_csv('/kaggle/working/experiment_A_results.csv', index=False) # 結果をCSVで保存

plt.figure(figsize=(10, 6))
sns.lineplot(x='size', y='accuracy', data=df_results_A, marker='o', color='royalblue')
plt.title('Experiment A: Model Accuracy vs. Data Quantity', fontsize=16)
plt.xlabel('Number of Training Samples Used', fontsize=12)
plt.ylabel('Accuracy (Performance)', fontsize=12)
plt.xticks(data_sizes)
plt.ylim(0.6, 1.0) # 精度に合わせてY軸を調整
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('/kaggle/working/experiment_A_chart.png') # グラフを画像で保存
plt.show()

print("\n実験結果のCSVとグラフが /kaggle/working/ に保存されました。")


