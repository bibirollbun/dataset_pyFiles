# =================================================================
# ■ ステップ1：ライブラリと環境の準備
# =================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import pydicom # DICOMファイルの読み込みに必要

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow import keras

from tqdm.notebook import tqdm

# --- 環境判定とパス設定 ---
# Kaggle環境かColab環境かを自動で判定し、パスを切り替えます
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    print("Kaggle環境を検出しました。")
    # Kaggleのデータセットパス
    BASE_DIR = Path('/kaggle/input/rsna-pneumonia-detection-challenge')
    DICOM_DIR = BASE_DIR / 'stage_2_train_images'
    PNG_DIR = Path('/kaggle/input/rsna-pneu-train-png/orig')
    # モデルの保存/読み込みパス
    NOTEBOOK_SLUG = os.environ.get('KAGGLE_URL_BASE', 'default-notebook').split('/')[-1]
    MODEL_LOAD_PATH = Path(f'/kaggle/input/{NOTEBOOK_SLUG}/best_pneumonia_classifier.keras')
    MODEL_SAVE_PATH = Path('/kaggle/working/best_pneumonia_classifier.keras')
else:
    print("Colab環境またはローカル環境を検出しました。")
    from google.colab import drive
    drive.mount('/content/drive')
    # Google Driveのパス
    BASE_DIR = Path('/content/drive/MyDrive/RSNA_Pneumonia/')
    PNG_DIR = BASE_DIR / 'stage_2_train_images_png'
    # モデルの保存/読み込みパス
    MODEL_SAVE_PATH = BASE_DIR / 'best_pneumonia_classifier.keras'
    MODEL_LOAD_PATH = MODEL_SAVE_PATH

print(f"画像ディレクトリ: {PNG_DIR}")
print(f"モデル保存先: {MODEL_SAVE_PATH}")
print(f"モデル読み込み元 (次のセッション用): {MODEL_LOAD_PATH}")

print("\nTensorFlow Version:", tf.__version__)
print("▶ ライブラリの準備完了")

# グラフのスタイルを設定
sns.set_style("whitegrid")


# =================================================================
# ■ ステップ2：データの読み込みと準備
# =================================================================
# --- 1. ラベルデータの読み込み ---
df_labels = pd.read_csv(BASE_DIR / 'stage_2_train_labels.csv')

# --- 2. 分類タスク用のデータを作成 ---
df_class = df_labels.drop_duplicates('patientId')[['patientId', 'Target']].copy()
df_class['filename'] = df_class['patientId'].apply(lambda x: f"{x}.png")
df_class['Target'] = df_class['Target'].astype(str)

# 実際に存在する画像ファイルのみを対象にする
print("\n存在する画像ファイルを確認中...")
df_class['filepath'] = df_class['filename'].apply(lambda f: PNG_DIR / f)
df_class['file_exists'] = df_class['filepath'].apply(lambda p: p.exists())
missing_files_count = len(df_class) - df_class['file_exists'].sum()
if missing_files_count > 0:
    print(f"注意: {missing_files_count}件の画像ファイルが見つかりませんでした。これらはデータセットから除外されます。")
df_class = df_class[df_class['file_exists']].copy().reset_index(drop=True)

# 公平性分析のためにメタ情報を自動で追加
# この処理はKaggle環境でのみ実行されます（Colabでは前処理済みと想定）
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    ages = []
    sexes = []
    view_positions = []
    print("\nDICOMファイルからメタ情報を抽出中...")
    for patient_id in tqdm(df_class['patientId']):
        dcm_path = DICOM_DIR / f"{patient_id}.dcm"
        dcm_data = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        ages.append(dcm_data.PatientAge)
        sexes.append(dcm_data.PatientSex)
        view_positions.append(dcm_data.ViewPosition)
    df_class['Age'] = ages
    df_class['Sex'] = sexes
    df_class['ViewPosition'] = view_positions
    df_class['Age'] = df_class['Age'].astype(int)
    print("メタ情報の抽出完了。")

# --- 4. データの分割 ---
df_train, df_val = train_test_split(
    df_class,
    test_size=0.2,
    random_state=42,
    stratify=df_class['Target']
)

print(f"\n利用可能な画像総数: {len(df_class)}件")
print(f"学習用データ数: {len(df_train)}件")
print(f"検証用データ数: {len(df_val)}件")
print("\n▶ データの準備完了")


# =================================================================
# ■ ステップ3：学習・評価用の関数を定義
# =================================================================
def train_and_finetune_stable_model(train_df, val_df, image_dir, image_size=224, batch_size=32, initial_epochs=8, fine_tune_epochs=25):
    print(f"\n--- 画像サイズ: {image_size}x{image_size}, バッチサイズ: {batch_size} ---")
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, directory=image_dir, x_col='filename', y_col='Target',
        target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary',
        validate_filenames=False
    )
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, directory=image_dir, x_col='filename', y_col='Target',
        target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary', shuffle=False,
        validate_filenames=False
    )
    if len(train_generator.classes) == 0:
        print("学習データが見つかりません。")
        return None, None
    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"▶ 計算されたクラス重み: {class_weight_dict}")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    METRICS = [keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    base_model.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=METRICS)
    print("\n--- Phase 1: ヘッド層の学習開始 ---")
    model.fit(train_generator, epochs=initial_epochs, validation_data=validation_generator, class_weight=class_weight_dict, verbose=2)
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=METRICS)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
    ]
    print("\n--- Phase 2: ファインチューニング開始 ---")
    model.fit(
        train_generator, epochs=initial_epochs + fine_tune_epochs, initial_epoch=initial_epochs,
        validation_data=validation_generator, class_weight=class_weight_dict, callbacks=callbacks, verbose=2
    )
    print("\n--- 検証データで評価中 ---")
    y_true = validation_generator.classes
    y_pred_proba = model.predict(validation_generator)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
    return metrics, model

print("▶ 関数定義完了")


# =================================================================
# ■ ステップ4：【最終版】実験Aの実行（ベストモデルを自動保存）
# =================================================================
# 検証するデータ量を定義
data_sizes = [500, 1000, 2000, 4000, 8000]
results_A = []

# ★★★ ここから変更 ★★★
# ベストスコアとベストモデルを保持するための変数を初期化
best_f1_score = -1.0
best_model = None
best_model_size = 0

for size in data_sizes:
    print(f"\n{'='*60}\n▶▶▶ データ量 {size}件 での処理を開始... \n{'='*60}")

    # 層化サンプリングで学習用サブセットを作成
    _, df_train_subset = train_test_split(
        df_train,
        train_size=size,
        random_state=42,
        stratify=df_train['Target']
    )

    metrics, trained_model = train_and_finetune_stable_model(
        train_df=df_train_subset,
        val_df=df_val,
        image_dir=PNG_DIR,
        image_size=224,
        batch_size=16,
        initial_epochs=8,
        fine_tune_epochs=22
    )
    
    if metrics:
        metrics['size'] = size
        results_A.append(metrics)
        
        # ★★★ 新しいF1スコアが今までのベストスコアより高ければ、モデルを更新 ★★★
        if trained_model and metrics['f1_score'] > best_f1_score:
            print(f"\n✨ ベストスコア更新！ (F1: {best_f1_score:.4f} -> {metrics['f1_score']:.4f})")
            best_f1_score = metrics['f1_score']
            best_model = trained_model
            best_model_size = size

# ★★★ ループ終了後、保持していたベストモデルを保存 ★★★
if best_model:
    best_model.save(MODEL_SAVE_PATH)
    print(f"\n▶▶▶ 最強の分類モデル（データ量 {best_model_size}件、F1スコア {best_f1_score:.4f}）を '{MODEL_SAVE_PATH}' として保存しました。")
else:
    print("\n▶▶▶ 注意: 有効なモデルが学習されなかったため、モデルは保存されませんでした。")


print("\n▶ 実験Aが完了しました。")
df_results = pd.DataFrame(results_A)


# =================================================================
# ■ ステップ5：実験Aの結果を可視化
# =================================================================
# (このセクションは変更ありません)
print("\n--- Experiment A: Results Summary ---")
display(df_results)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Relationship between Data Size and Performance Metrics', fontsize=20, y=1.02)
axes = axes.flatten()
for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    sns.lineplot(x='size', y=metric, data=df_results, ax=axes[i], marker='o', color='royalblue')
    axes[i].set_title(title, fontsize=14)
    axes[i].set_xlabel('Training Data Size', fontsize=12)
    axes[i].set_ylabel('Score', fontsize=12)
    axes[i].set_ylim(0.0, 1.0)
    axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    for index, row in df_results.iterrows():
        axes[i].text(row['size'], row[metric], f"{row[metric]:.3f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# =================================================================
# ■ ステップ6：実験Bの準備と予測
# =================================================================
print("\n--- 実験B：公平性分析を開始します ---")
best_model = None
# まず現在のセッションで保存したモデルを探す
if MODEL_SAVE_PATH.exists():
    print(f"現在のセッションで保存したモデル '{MODEL_SAVE_PATH}' を読み込みます。")
    best_model = load_model(MODEL_SAVE_PATH)
# 見つからなければ、前のバージョンの出力（input）を探す
elif MODEL_LOAD_PATH.exists():
    print(f"以前のバージョンで保存されたモデル '{MODEL_LOAD_PATH}' を読み込みます。")
    best_model = load_model(MODEL_LOAD_PATH)
else:
    print(f"エラー: モデルファイルが見つかりません。")
    print(f"確認したパス: {MODEL_SAVE_PATH}, {MODEL_LOAD_PATH}")

if best_model:
    validation_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=df_val, directory=PNG_DIR, x_col='filename', y_col='Target',
        target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False,
        validate_filenames=False
    )
    print("\n▶ 検証データ全体の予測を実行中...")
    y_true = validation_generator.classes
    y_pred_proba = best_model.predict(validation_generator)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    df_val_results = df_val.copy()
    df_val_results['y_true'] = y_true
    df_val_results['y_pred'] = y_pred
    print("▶ 予測完了。")

# =================================================================
# ■ ステップ7：実験Bのグループ評価
# =================================================================
def evaluate_on_groups(df_results, group_col):
    results = []
    for name, group_df in df_results.groupby(group_col):
        if len(group_df) == 0: continue
        y_true_group, y_pred_group = group_df['y_true'], group_df['y_pred']
        metrics = {
            'group_name': name, 'group_col': group_col, 'count': len(group_df),
            'accuracy': accuracy_score(y_true_group, y_pred_group),
            'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
            'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
            'f1_score': f1_score(y_true_group, y_pred_group, zero_division=0)
        }
        results.append(metrics)
    return pd.DataFrame(results)

if 'df_val_results' in locals() and all(c in df_val_results for c in ['Age', 'Sex', 'ViewPosition']):
    df_val_results['AgeGroup'] = pd.cut(df_val_results['Age'], bins=[0, 59, 150], labels=['Under 60', '60 and Over'])
    results_sex = evaluate_on_groups(df_val_results, 'Sex')
    results_age = evaluate_on_groups(df_val_results, 'AgeGroup')
    results_view = evaluate_on_groups(df_val_results, 'ViewPosition')
    overall_metrics = {
        'group_name': 'Overall', 'group_col': 'Overall', 'count': len(df_val_results),
        'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0), 'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    df_overall = pd.DataFrame([overall_metrics])
    df_fairness_results = pd.concat([df_overall, results_sex, results_age, results_view], ignore_index=True)
    print("\n--- Fairness Analysis Results Summary ---")
    display(df_fairness_results)
else:
    print("\n--- 公平性分析に必要なメタ情報（Age, Sexなど）がないため、実験Bをスキップします ---")


# =================================================================
# ■ ステップ8：実験Bの結果を可視化
# =================================================================
if 'df_fairness_results' in locals():
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 20), sharex=False)
    fig.suptitle('Experiment B: Model Performance Comparison by Attribute Group', fontsize=20, y=1.0)
    for i, metric in enumerate(metrics_to_plot):
        data_to_plot = df_fairness_results[df_fairness_results['group_col'].isin(['Overall', 'Sex', 'AgeGroup', 'ViewPosition'])]
        sns.barplot(x='group_name', y=metric, data=data_to_plot, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Comparison of {titles[i]}', fontsize=14)
        axes[i].set_xlabel('Attribute Group', fontsize=12)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].set_ylim(0.0, 1.0)
        axes[i].tick_params(axis='x', rotation=10)
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.3f}",
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                             textcoords='offset points')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


