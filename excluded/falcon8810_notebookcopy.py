# # =================================================================
# # ■ ステップ1：ライブラリの準備
# # =================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import pydicom # DICOMファイルを読むために追加
# import japanize_matplotlib # ★★★ インポートして有効化 ★★★

# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.models import load_model

# from tensorflow import keras

# from tqdm.notebook import tqdm

# print("TensorFlow Version:", tf.__version__)
# print("▶ ライブラリの準備完了")

# # グラフのスタイルを設定
# sns.set_style("whitegrid")
# plt.rcParams['font.family'] = 'sans-serif'


# # =================================================================
# # ■ ステップ2：データの読み込みと準備
# # =================================================================
# # Kaggleのデータセットパスを定義
# BASE_DIR = '/kaggle/input/rsna-pneumonia-detection-challenge'
# PNG_DIR = '/kaggle/input/rsna-pneu-train-png/orig'
# DICOM_DIR = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'

# # --- 1. ラベルデータの読み込み ---
# df_labels = pd.read_csv(os.path.join(BASE_DIR, 'stage_2_train_labels.csv'))

# # --- 2. 分類タスク用のデータを作成 ---
# # patientIdで重複を削除し、必要なカラムのみに絞る
# df_class = df_labels.drop_duplicates('patientId')[['patientId', 'Target']].copy()

# # ファイル名カラムを追加
# df_class['filename'] = df_class['patientId'].apply(lambda x: f"{x}.png")
# df_class['Target'] = df_class['Target'].astype(str)

# # --- 3. DICOMからメタ情報(年齢、性別、撮影方向)を抽出 ---
# # この処理は少し時間がかかります
# ages = []
# sexes = []
# view_positions = []
# print("DICOMファイルからメタ情報を抽出中...")
# for patient_id in tqdm(df_class['patientId']):
#     dcm_path = os.path.join(DICOM_DIR, f"{patient_id}.dcm")
#     dcm_data = pydicom.dcmread(dcm_path, stop_before_pixels=True)
#     ages.append(dcm_data.PatientAge)
#     sexes.append(dcm_data.PatientSex)
#     view_positions.append(dcm_data.ViewPosition)

# # 抽出した情報をデータフレームに追加
# df_class['Age'] = ages
# df_class['Sex'] = sexes
# df_class['ViewPosition'] = view_positions
# df_class['Age'] = df_class['Age'].astype(int) # 年齢を数値型に変換

# # --- 4. データの分割 ---
# df_train, df_val = train_test_split(
#     df_class,
#     test_size=0.2,
#     random_state=42,
#     stratify=df_class['Target']
# )

# print(f"\nデータ総数: {len(df_class)}件")
# print(f"学習用データ数: {len(df_train)}件")
# print(f"検証用データ数: {len(df_val)}件")
# print("\n--- 検証データの属性情報サマリー ---")
# # この行でエラーが出ていましたが、修正により解決するはずです
# print(df_val[['Age', 'Sex', 'ViewPosition']].info())
# print("\n▶ データの準備完了")


# # =================================================================
# # ■ ステップ3：【安定化版】学習・評価用の関数を定義
# # =================================================================

# def train_and_finetune_stable_model(train_df, val_df, image_dir, image_size=224, batch_size=32, initial_epochs=8, fine_tune_epochs=25):
#     """
#     学習プロセスを安定させ、Recallを重視するよう修正した関数
#     """
#     print(f"\n--- 画像サイズ: {image_size}x{image_size}, バッチサイズ: {batch_size} ---")

#     # --- 1. ImageDataGeneratorの準備 ---
#     train_datagen = ImageDataGenerator(
#         rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
#         shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
#     )
#     val_datagen = ImageDataGenerator(rescale=1./255)
    
#     train_generator = train_datagen.flow_from_dataframe(
#         dataframe=train_df, directory=image_dir, x_col='filename', y_col='Target',
#         target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary'
#     )
#     validation_generator = val_datagen.flow_from_dataframe(
#         dataframe=val_df, directory=image_dir, x_col='filename', y_col='Target',
#         target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary', shuffle=False
#     )
    
#     # --- 2. クラス重みの計算 ---
#     if len(train_generator.classes) == 0:
#         print("学習データが見つかりません。処理を中断します。")
#         return None, None
#     class_weights = compute_class_weight(
#         'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
#     )
#     class_weight_dict = dict(enumerate(class_weights))
#     print(f"▶ 計算されたクラス重み: {class_weight_dict}")

#     # --- 3. モデルの構築 ---
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
    
#     # Kerasに標準で含まれていないRecallをメトリクスとして追加
#     METRICS = [
#         keras.metrics.TruePositives(name='tp'),
#         keras.metrics.FalsePositives(name='fp'),
#         keras.metrics.TrueNegatives(name='tn'),
#         keras.metrics.FalseNegatives(name='fn'), 
#         keras.metrics.BinaryAccuracy(name='accuracy'),
#         keras.metrics.Precision(name='precision'),
#         keras.metrics.Recall(name='recall'),
#     ]

#     # =================================================================
#     # ★★★ Phase 1: ヘッド層の学習 ★★★
#     # =================================================================
#     base_model.trainable = False
    
#     # ★ 変更点: 初期学習率を慎重な値に設定
#     model.compile(optimizer=Adam(learning_rate=1e-4),
#                   loss='binary_crossentropy',
#                   metrics=METRICS)
    
#     print("\n--- Phase 1: ヘッド層の学習開始 ---")
#     model.fit(
#         train_generator,
#         epochs=initial_epochs,
#         validation_data=validation_generator,
#         class_weight=class_weight_dict,
#         verbose=2
#     )

#     # =================================================================
#     # ★★★ Phase 2: ファインチューニング ★★★
#     # =================================================================
#     base_model.trainable = True
#     fine_tune_at = int(len(base_model.layers) * 0.8)
#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable = False

#     model.compile(optimizer=Adam(learning_rate=1e-5),
#                   loss='binary_crossentropy',
#                   metrics=METRICS)
                  
#     # ★ 変更点: Recallを監視し、patienceを増やす
#     callbacks = [
#         ReduceLROnPlateau(monitor='val_f1_score', mode='max', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
#         EarlyStopping(monitor='val_f1_score', mode='max', patience=8, restore_best_weights=True, verbose=1)
#     ]

#     print("\n--- Phase 2: ファインチューニング開始 ---")
#     total_epochs = initial_epochs + fine_tune_epochs
#     model.fit(
#         train_generator,
#         epochs=total_epochs,
#         initial_epoch=initial_epochs,
#         validation_data=validation_generator,
#         class_weight=class_weight_dict,
#         callbacks=callbacks,
#         verbose=2
#     )

#     # --- 6. 最終評価 ---
#     print("\n--- 検証データで評価中 ---")
#     y_true = validation_generator.classes
#     y_pred_proba = model.predict(validation_generator)
#     y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
#     metrics = {
#         'accuracy': accuracy_score(y_true, y_pred),
#         'precision': precision_score(y_true, y_pred, zero_division=0),
#         'recall': recall_score(y_true, y_pred, zero_division=0),
#         'f1_score': f1_score(y_true, y_pred, zero_division=0)
#     }
    
#     print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
    
#     return metrics, model

# print("▶ 関数定義完了")


# # =================================================================
# # ■ ステップ4：【安定化版】実験の実行
# # =================================================================
# # 検証するデータ量を定義
# data_sizes = [500,1000,2000,4000,8000]
# results_A = []
# final_model = None 

# for size in data_sizes:
#     # ★ 変更点: ループの最後はサンプリングしない
#     if size == len(df_train):
#         df_train_subset = df_train.copy()
#     else:
#         # クラス比率を保ったままサンプリング
#         n_class_0 = int(size * 0.775)
#         n_class_1 = size - n_class_0
#         df_train_0 = df_train[df_train['Target'] == '0'].sample(n=n_class_0, random_state=42)
#         df_train_1 = df_train[df_train['Target'] == '1'].sample(n=n_class_1, random_state=42)
#         df_train_subset = pd.concat([df_train_0, df_train_1])

#     print(f"\n{'='*60}\n▶▶▶ データ量 {size}件 での処理を開始... \n{'='*60}")

#     metrics, trained_model = train_and_finetune_stable_model(
#         train_df=df_train_subset,
#         val_df=df_val,
#         image_dir=PNG_DIR,
#         image_size=224,
#         batch_size=16, # バッチサイズを小さくして学習をより安定させる
#         initial_epochs=8,
#         fine_tune_epochs=22
#     )
    
#     if metrics:
#         metrics['size'] = size
#         results_A.append(metrics)
    
#     if size == 8000 and trained_model:
#         final_model = trained_model
        
# if final_model:
#     final_model.save('best_pneumonia_classifier.keras')
#     print("\n▶▶▶ 最強の分類モデルを 'best_pneumonia_classifier.keras' として保存しました。")

# print("\n▶ 実験Aが完了しました。")

# df_results = pd.DataFrame(results_A)


# # =================================================================
# # ■ ステップ5：実験結果の可視化
# # =================================================================
# print("\n--- 実験結果サマリー ---")
# display(df_results)

# # --- 描画 ---
# metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
# titles = ['正解率 (Accuracy)', '適合率 (Precision)', '再現率 (Recall)', 'F1スコア (F1-Score)']

# # 2x2のグリッドでグラフを作成
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('データ量と各評価指標の関係', fontsize=20, y=1.02)
# axes = axes.flatten() # 2x2のaxesを1次元配列に変換してループしやすくする

# for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
#     sns.lineplot(x='size', y=metric, data=df_results, ax=axes[i], marker='o', color='royalblue')
#     axes[i].set_title(title, fontsize=14)
#     axes[i].set_xlabel('学習データ数', fontsize=12)
#     axes[i].set_ylabel('スコア', fontsize=12)
#     axes[i].set_ylim(0.0, 1.0) # ★y軸の範囲を0.0〜1.0に調整
#     axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     # 各点に数値を表示
#     for index, row in df_results.iterrows():
#         axes[i].text(row['size'], row[metric], f"{row[metric]:.3f}", ha='center', va='bottom', fontsize=9)

# plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.show()


# # =================================================================
# # ■ ステップ6以降の修正版コード
# # =================================================================
# # このセルを実行する前に、ステップ1とステップ2が実行済みであることを確認してください。

# # --- モデルの読み込み ---
# MODEL_PATH = '/kaggle/working/best_pneumonia_classifier.keras'
# PNG_DIR = '/kaggle/input/rsna-pneu-train-png/orig' # 画像パスを再定義

# try:
#     best_model = load_model(MODEL_PATH)
#     print(f"▶ 保存されたモデル {MODEL_PATH} の読み込みに成功しました。")
# except Exception as e:
#     print(f"モデルの読み込みに失敗しました: {e}")
#     best_model = None

# # --- 検証データ全体の予測を実行 ---
# if best_model:
#     # 検証データ用のジェネレータを再作成
#     val_datagen = ImageDataGenerator(rescale=1./255)
#     validation_generator = val_datagen.flow_from_dataframe(
#         dataframe=df_val, 
#         directory=PNG_DIR, 
#         x_col='filename', 
#         y_col='Target',
#         target_size=(224, 224), 
#         batch_size=32, 
#         class_mode='binary', 
#         shuffle=False
#     )
    
#     print("\n▶ 検証データ全体の予測を実行中...")
#     y_true = validation_generator.classes
#     y_pred_proba = best_model.predict(validation_generator)
#     y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
#     # ジェネレータが実際に読み込んだファイル名を取得
#     valid_filenames = validation_generator.filenames
    
#     # 実際に読み込まれたファイルの結果だけで新しいDataFrameを作成
#     df_pred_results = pd.DataFrame({
#         'filename': valid_filenames,
#         'y_true': y_true,
#         'y_pred': y_pred
#     })
    
#     # 元の検証データフレーム(df_val)と、予測結果のフレームを'filename'をキーにして結合する
#     df_val_results = pd.merge(
#         df_val,
#         df_pred_results,
#         on='filename'
#     )
    
#     print(f"▶ 予測完了。検証に成功した {len(df_val_results)} 件のデータで分析を続行します。")

#     # --- グループ評価の実行 ---
#     df_val_results['AgeGroup'] = pd.cut(df_val_results['Age'], 
#                                         bins=[0, 59, 150], 
#                                         labels=['59歳以下', '60歳以上'])

#     results_sex = evaluate_on_groups(df_val_results, 'Sex')
#     results_age = evaluate_on_groups(df_val_results, 'AgeGroup')
#     results_view = evaluate_on_groups(df_val_results, 'ViewPosition')
    
#     overall_metrics = {
#         'group_name': '全体', 'group_col': 'Overall', 'count': len(df_val_results),
#         'accuracy': accuracy_score(df_val_results['y_true'], df_val_results['y_pred']), 
#         'precision': precision_score(df_val_results['y_true'], df_val_results['y_pred'], zero_division=0),
#         'recall': recall_score(df_val_results['y_true'], df_val_results['y_pred'], zero_division=0), 
#         'f1_score': f1_score(df_val_results['y_true'], df_val_results['y_pred'], zero_division=0)
#     }
#     df_overall = pd.DataFrame([overall_metrics])
    
#     df_fairness_results = pd.concat([df_overall, results_sex, results_age, results_view], ignore_index=True)
    
#     print("\n--- 公平性分析 結果サマリー ---")
#     display(df_fairness_results)

#     # --- 結果の可視化 ---
#     metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
#     metric_titles = ['正解率 (Accuracy)', '適合率 (Precision)', '再現率 (Recall)', 'F1スコア (F1-Score)']
    
#     fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 20), sharex=False)
#     fig.suptitle('実験B: 属性グループごとのモデル性能比較', fontsize=20, y=1.0)

#     for i, metric in enumerate(metrics_to_plot):
#         data_to_plot = df_fairness_results[df_fairness_results['group_col'].isin(['Overall', 'Sex', 'AgeGroup', 'ViewPosition'])]
#         sns.barplot(x='group_name', y=metric, data=data_to_plot, ax=axes[i], palette='viridis')
#         axes[i].set_title(f'{metric_titles[i]} の比較', fontsize=14)
#         axes[i].set_xlabel('属性グループ', fontsize=12)
#         axes[i].set_ylabel('スコア', fontsize=12)
#         axes[i].set_ylim(0.0, 1.0)
#         axes[i].tick_params(axis='x', rotation=10)
        
#         for p in axes[i].patches:
#             axes[i].annotate(f"{p.get_height():.3f}",
#                              (p.get_x() + p.get_width() / 2., p.get_height()),
#                              ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
#                              textcoords='offset points')

#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()


# =================================================================
# ■【最終実行用】学習・分析・保存 全機能版コード
# =================================================================
# このセルを一度だけ「Save Version」で実行してください。
# 2回目以降の実行では、保存された結果を読み込み、学習をスキップします。

# --- 1. 必要なライブラリの準備 ---
print("■ ステップ1：ライブラリの準備")
!pip install -q japanize-matplotlib
import japanize_matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from IPython.display import display
from tqdm.notebook import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
print("▶ ライブラリの準備完了")

# --- 2. データの準備 ---
print("\n■ ステップ2：データの準備")
BASE_DIR = '/kaggle/input/rsna-pneumonia-detection-challenge'
DICOM_DIR = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'
PNG_DIR = '/kaggle/input/rsna-pneu-train-png/orig'
df_labels = pd.read_csv(os.path.join(BASE_DIR, 'stage_2_train_labels.csv'))
df_class = df_labels.drop_duplicates('patientId')[['patientId', 'Target']].copy()
df_class['filename'] = df_class['patientId'].apply(lambda x: f"{x}.png")
df_class['Target'] = df_class['Target'].astype(str)

# メタ情報抽出
ages, sexes, view_positions = [], [], []
print("DICOMファイルからメタ情報を抽出中...")
for patient_id in tqdm(df_class['patientId']):
    dcm_path = os.path.join(DICOM_DIR, f"{patient_id}.dcm")
    dcm_data = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    ages.append(dcm_data.PatientAge)
    sexes.append(dcm_data.PatientSex)
    view_positions.append(dcm_data.ViewPosition)

df_class['Age'] = ages
df_class['Sex'] = sexes
df_class['ViewPosition'] = view_positions
df_class['Age'] = df_class['Age'].astype(int)

# データの分割
df_train, df_val = train_test_split(df_class, test_size=0.2, random_state=42, stratify=df_class['Target'])
print(f"▶ 検証用データ {len(df_val)} 件、学習用データ {len(df_train)} 件の準備完了")

# --- 3. グループ評価用の関数を定義 ---
def evaluate_on_groups(df_results, group_col):
    results = []
    for name, group_df in df_results.groupby(group_col):
        if len(group_df) == 0: continue
        y_true_group = group_df['y_true']
        y_pred_group = group_df['y_pred']
        metrics = {
            'group_name': name, 'group_col': group_col, 'count': len(group_df),
            'accuracy': accuracy_score(y_true_group, y_pred_group),
            'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
            'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
            'f1_score': f1_score(y_true_group, y_pred_group, zero_division=0)
        }
        results.append(metrics)
    return pd.DataFrame(results)
print("▶ 評価用関数の準備完了")

# --- 4. モデルの読み込み、または学習の実行 ---
print("\n■ ステップ4：モデルの読み込み、または学習の実行")
MODEL_PATH = '/kaggle/working/best_pneumonia_classifier.keras'
RESULTS_CSV_PATH = '/kaggle/working/experiment_A_results.csv'
best_model = None
df_results = None

if os.path.exists(MODEL_PATH) and os.path.exists(RESULTS_CSV_PATH):
    print(f"▶▶▶ 発見済みのモデル {MODEL_PATH} を読み込みます。")
    try:
        best_model = load_model(MODEL_PATH)
        df_results = pd.read_csv(RESULTS_CSV_PATH)
        print("▶ モデルと実験結果の読み込みに成功しました。学習はスキップされます。")
    except Exception as e:
        print(f"▶ 読み込みに失敗しました: {e}。学習を再実行します。")
        best_model = None

if best_model is None:
    print("\n▶▶▶ モデルが存在しないため、実験Aを最初から実行します...")
    
    # --- 学習用の関数定義 ---
    def train_and_finetune_stable_model(train_df, val_df, image_dir, image_size=224, batch_size=32, initial_epochs=8, fine_tune_epochs=25):
        print(f"\n--- 画像サイズ: {image_size}x{image_size}, バッチサイズ: {batch_size} ---")
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, directory=image_dir, x_col='filename', y_col='Target', target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary')
        validation_generator = val_datagen.flow_from_dataframe(dataframe=val_df, directory=image_dir, x_col='filename', y_col='Target', target_size=(image_size, image_size), batch_size=batch_size, class_mode='binary', shuffle=False)
        
        if len(train_generator.classes) == 0: return None, None
        class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
        class_weight_dict = dict(enumerate(class_weights))
        print(f"▶ 計算されたクラス重み: {class_weight_dict}")

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        METRICS = [keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        
        base_model.trainable = False
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=METRICS)
        print("\n--- Phase 1: ヘッド層の学習開始 ---")
        model.fit(train_generator, epochs=initial_epochs, validation_data=validation_generator, class_weight=class_weight_dict, verbose=2)

        base_model.trainable = True
        fine_tune_at = int(len(base_model.layers) * 0.8)
        for layer in base_model.layers[:fine_tune_at]: layer.trainable = False
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=METRICS)
        callbacks = [ReduceLROnPlateau(monitor='val_recall', mode='max', factor=0.2, patience=5, min_lr=1e-7, verbose=1), EarlyStopping(monitor='val_recall', mode='max', patience=8, restore_best_weights=True, verbose=1)]
        print("\n--- Phase 2: ファインチューニング開始 ---")
        model.fit(train_generator, epochs=initial_epochs + fine_tune_epochs, initial_epoch=initial_epochs, validation_data=validation_generator, class_weight=class_weight_dict, callbacks=callbacks, verbose=2)
        
        y_true = validation_generator.classes
        y_pred = (model.predict(validation_generator) > 0.5).astype(int).flatten()
        metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1_score': f1_score(y_true, y_pred, zero_division=0)}
        return metrics, model

    # --- 学習ループ ---
    data_sizes = [500, 1000, 2000, 4000, 8000]
    results_A = []
    
    for size in data_sizes:
        n_class_0 = int(size * 0.775)
        n_class_1 = size - n_class_0
        df_train_0 = df_train[df_train['Target'] == '0'].sample(n=n_class_0, random_state=42)
        df_train_1 = df_train[df_train['Target'] == '1'].sample(n=n_class_1, random_state=42)
        df_train_subset = pd.concat([df_train_0, df_train_1])
        print(f"\n{'='*60}\n▶▶▶ データ量 {size}件 での処理を開始... \n{'='*60}")
        metrics, trained_model = train_and_finetune_stable_model(train_df=df_train_subset, val_df=df_val, image_dir=PNG_DIR, image_size=224, batch_size=16, initial_epochs=8, fine_tune_epochs=22)
        if metrics:
            metrics['size'] = size
            results_A.append(metrics)
        if size == 8000 and trained_model:
            best_model = trained_model
            
    if best_model:
        best_model.save(MODEL_PATH)
        print(f"\n▶▶▶ 8000件で学習したモデルを '{MODEL_PATH}' として保存しました。")
    df_results = pd.DataFrame(results_A)
    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"▶ 実験Aの結果を '{RESULTS_CSV_PATH}' に保存しました。")

# --- 5. 実験Aの結果を可視化 ---
if df_results is not None:
    print("\n■ ステップ5：実験Aの結果を可視化")
    display(df_results)
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['正解率 (Accuracy)', '適合率 (Precision)', '再現率 (Recall)', 'F1スコア (F1-Score)']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('データ量と各評価指標の関係', fontsize=20, y=1.02)
    axes = axes.flatten()
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        sns.lineplot(x='size', y=metric, data=df_results, ax=axes[i], marker='o', color='royalblue')
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('学習データ数', fontsize=12)
        axes[i].set_ylabel('スコア', fontsize=12)
        axes[i].set_ylim(0.0, 1.0)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        for index, row in df_results.iterrows():
            axes[i].text(row['size'], max(row[metric], 0.01), f"{row[metric]:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# --- 6. 予測と公平性分析の実行 ---
if best_model:
    print("\n■ ステップ6：予測と公平性分析の実行")
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_dataframe(dataframe=df_val, directory=PNG_DIR, x_col='filename', y_col='Target', target_size=(224, 244), batch_size=32, class_mode='binary', shuffle=False)
    
    print("▶ 検証データ全体の予測を実行中...")
    y_pred_proba = best_model.predict(validation_generator)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    valid_filenames = validation_generator.filenames
    df_pred_results = pd.DataFrame({'filename': valid_filenames, 'y_true': validation_generator.classes, 'y_pred': y_pred})
    df_val_results = pd.merge(df_val, df_pred_results, on='filename')
    print(f"▶ 予測完了。検証に成功した {len(df_val_results)} 件のデータで分析を続行します。")

    df_val_results['AgeGroup'] = pd.cut(df_val_results['Age'], bins=[0, 59, 150], labels=['59歳以下', '60歳以上'])
    results_sex = evaluate_on_groups(df_val_results, 'Sex')
    results_age = evaluate_on_groups(df_val_results, 'AgeGroup')
    results_view = evaluate_on_groups(df_val_results, 'ViewPosition')
    
    overall_metrics = {'group_name': '全体', 'group_col': 'Overall', 'count': len(df_val_results), 'accuracy': accuracy_score(df_val_results['y_true'], df_val_results['y_pred']), 'precision': precision_score(df_val_results['y_true'], df_val_results['y_pred'], zero_division=0), 'recall': recall_score(df_val_results['y_true'], df_val_results['y_pred'], zero_division=0), 'f1_score': f1_score(df_val_results['y_true'], df_val_results['y_pred'], zero_division=0)}
    df_overall = pd.DataFrame([overall_metrics])
    df_fairness_results = pd.concat([df_overall, results_sex, results_age, results_view], ignore_index=True)
    
    print("\n--- 公平性分析 結果サマリー ---")
    display(df_fairness_results)

    # --- 7. 公平性分析の結果を可視化 ---
    print("\n■ ステップ7：公平性分析の結果を可視化")
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_titles = ['正解率 (Accuracy)', '適合率 (Precision)', '再現率 (Recall)', 'F1スコア (F1-Score)']
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 20), sharex=False)
    fig.suptitle('実験B: 属性グループごとのモデル性能比較', fontsize=20, y=1.0)
    for i, metric in enumerate(metrics_to_plot):
        data_to_plot = df_fairness_results[df_fairness_results['group_col'].isin(['Overall', 'Sex', 'AgeGroup', 'ViewPosition'])]
        sns.barplot(x='group_name', y=metric, data=data_to_plot, ax=axes[i], palette='viridis')
        axes[i].set_title(f'{metric_titles[i]} の比較', fontsize=14)
        axes[i].set_xlabel('属性グループ', fontsize=12)
        axes[i].set_ylabel('スコア', fontsize=12)
        axes[i].set_ylim(0.0, 1.0)
        axes[i].tick_params(axis='x', rotation=10)
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

else:
    print("エラー: モデルが利用できないため、分析を中断しました。")


