import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 設定パラメータ
N_SAMPLES = 75  # 各トリガーセグメントの長さ
N_MODELS = 45   # テスト対象のポイズンドモデル数
CHANNELS = ['channel_44', 'channel_45', 'channel_46']  # 使用するチャンネル


INPUT_DIR = '/kaggle/input/trojan-horse-hunt-in-space'
CLEAN_MODEL_PATH = os.path.join(INPUT_DIR, 'clean_model')
POISONED_MODELS_PATH = os.path.join(INPUT_DIR, 'poisoned_models')
SUBMISSION_PATH = 'submission.csv'

DEBUG = False
if DEBUG:
    INPUT_DIR = './data'
    CLEAN_MODEL_PATH = os.path.join(INPUT_DIR, 'clean_model')
    POISONED_MODELS_PATH = os.path.join(INPUT_DIR, 'poisoned_models')
    SUBMISSION_PATH = 'submission.csv'

def create_zero_trigger_submission():
    """
    ゼロで初期化されたトリガーマトリックスを作成し、
    正確な提出形式のCSVを生成する関数
    """
    # ゼロトリガーベクトルを作成
    zero_trigger = np.zeros(N_SAMPLES * len(CHANNELS))
    
    # 各モデルに対して同じゼロトリガーを複製
    data = np.tile(zero_trigger, (N_MODELS, 1))
    
    # データフレームに変換
    df = pd.DataFrame(data)
    
    # チャンネルごとの列名を生成
    channel_cols = [
        f"{ch}_{i+1}"
        for ch in CHANNELS
        for i in range(N_SAMPLES)
    ]
    
    # 列名を設定
    df.columns = channel_cols
    
    # model_idカラムを追加（1から始まる）
    df.insert(0, "model_id", range(1, N_MODELS + 1))
    
    # インデックスを1から始める
    df.index = df.index + 1
    
    # CSVファイルとして保存
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"提出ファイルが保存されました: {SUBMISSION_PATH}")
    
    return df

def main():
    """メイン実行関数"""
    print("Trojan Horse Hunt in Space - ベースライン提出コードを実行中...")
    
    # ゼロトリガーの提出ファイルを生成
    df = create_zero_trigger_submission()
    
    # 提出ファイルの形状を確認
    print(f"提出データフレームの形状: {df.shape}")
    print("完了しました！")

if __name__ == "__main__":
    main()


