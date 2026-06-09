import numpy as np
import pandas as pd

train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")


target_gestures = [
    'Above ear - pull hair', 'Cheek - pinch skin', 'Eyebrow - pull hair',
    'Eyelash - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
    'Neck - pinch skin', 'Neck - scratch',
]
non_target_gestures = [
    'Write name on leg', 'Wave hello', 'Glasses on/off', 'Text on phone',
    'Write name in air', 'Feel around in tray and pull out an object',
    'Scratch knee/leg skin', 'Pull air toward your face',
    'Drink from bottle/cup', 'Pinch knee/leg skin'
]

# 1. ジェスチャーと種類をマッピングする辞書を作成
gesture_type_map = {gesture: 'Target' for gesture in target_gestures}
gesture_type_map.update({gesture: 'Non-Target' for gesture in non_target_gestures})

# 2. 各ジェスチャーの出現回数をカウントし、多い順にソート
gesture_counts = train['gesture'].value_counts()

# 3. DataFrameに変換し、'gesture_type'列を追加
result_df = pd.DataFrame(gesture_counts).reset_index()
result_df.columns = ['gesture', 'count']
result_df['gesture_type'] = result_df['gesture'].map(gesture_type_map)

# 4. 出力
print("ジェスチャーごとの種類と出現回数 (数が多い順):\n")
print(result_df)

