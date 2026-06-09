import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


def load_all_annotations(base_folder):

    all_data = pd.DataFrame()

    for lab_id in os.listdir(base_folder):
        lab_folder = os.path.join(base_folder, lab_id)

   
        if os.path.isdir(lab_folder):

            for file in os.listdir(lab_folder):
                if file.endswith('.parquet'):
                    file_path = os.path.join(lab_folder, file)
                    temp_data = pd.read_parquet(file_path)
                    video_id = os.path.splitext(file)[0]  # 去掉文件扩展名
                    temp_data['video_id'] = video_id
                    temp_data['lab_id'] = lab_id  # 添加实验室 ID

                    # 合并到总 DataFrame
                    all_data = pd.concat([all_data, temp_data], ignore_index=True)

    return all_data

annotation_folder = '/kaggle/input/MABe-mouse-behavior-detection/train_annotation/'

annotation_data = load_all_annotations(annotation_folder)
annotation_data


single = annotation_data[annotation_data['agent_id'] == annotation_data['target_id']]  # agent_id 等於 target_id 的行
pair = annotation_data[annotation_data['agent_id'] != annotation_data['target_id']]  # agent_id 不等於 target_id 的行

# 儲存兩個 DataFrame
#single.to_csv('single.csv', index=False)  # 儲存為 CSV
#pair.to_csv('pair.csv', index=False)  # 儲存為 CSV

#print("Data has been split and saved successfully.")


import os
import pandas as pd

# 設定文件夾路徑
train_annotation_path = '/kaggle/input/MABe-mouse-behavior-detection/train_annotation/'
train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking/'

# 用於存儲數據的列表
tracking_data = []
annotation_data = []

# 遍歷 train_tracking 文件夾
for lab_id in os.listdir(train_tracking_path):
    lab_path = os.path.join(train_tracking_path, lab_id)
    if os.path.isdir(lab_path):  # 確保是文件夾
        for file in os.listdir(lab_path):
            if file.endswith('.parquet'):
                # 去掉 .parquet 擴展名
                tracking_data.append({'lab_id': lab_id, 'video_id': file[:-8]})

# 遍歷 train_annotation 文件夾
for lab_id in os.listdir(train_annotation_path):
    lab_path = os.path.join(train_annotation_path, lab_id)
    if os.path.isdir(lab_path):  # 確保是文件夾
        for file in os.listdir(lab_path):
            if file.endswith('.parquet'):
                # 去掉 .parquet 擴展名
                annotation_data.append({'lab_id': lab_id, 'video_id': file[:-8]})

# 創建 DataFrame
tracking_df = pd.DataFrame(tracking_data)
annotation_df = pd.DataFrame(annotation_data)
# 進行合併以獲取在兩個 DataFrame 中都存在的視頻 ID
common_videos_df = tracking_df.merge(annotation_df, on='video_id', suffixes=('_tracking', '_annotation'))

# 檢查結果
print("Common Videos DataFrame:")
print(common_videos_df)


import pandas as pd
import os
from tqdm import tqdm  # 引入 tqdm
import pandas as pd
import os
from tqdm import tqdm

# 假設 df 是你的 DataFrame，包含 video_id 列
# df = pd.read_csv('your_video_id_file.csv')  # 讀取你的 video_id 文件

# 將 df 中的 video_id 列轉換為集合
video_ids_set = set(common_videos_df['video_id'].unique())

# 初始化一個空的 DataFrame 用於保存結果
result_df = pd.DataFrame()

# 指定 train_tracking 文件夾的路徑
train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking/'

# 遍歷 train_tracking 文件夾中的每個實驗室文件夾
for lab_folder in os.listdir(train_tracking_path):
    lab_folder_path = os.path.join(train_tracking_path, lab_folder)

    # 確保是文件夾
    if os.path.isdir(lab_folder_path):
        # 獲取所有 parquet 文件
        parquet_files = [file for file in os.listdir(lab_folder_path) if file.endswith('.parquet')]
        
        # 使用 tqdm 來顯示進度條
        for file_name in tqdm(parquet_files, desc=f"Processing folder: {lab_folder}", unit="file"):
            # 提取 video_id，去掉文件擴展名
            video_id = file_name[:-8]  # 去掉 ".parquet"

            # 檢查 video_id 是否在集合中
            if video_id in video_ids_set:
                file_path = os.path.join(lab_folder_path, file_name)

                # 讀取 parquet 文件
                temp_df = pd.read_parquet(file_path)

                # 將讀取的數據添加到結果 DataFrame 中
                result_df = pd.concat([result_df, temp_df], ignore_index=True)

# 現在 result_df 包含了所有需要的數據
print(result_df)


import pandas as pd
import os
from tqdm import tqdm


video_ids_set = set(common_videos_df['video_id'].unique())

result_df = pd.DataFrame()

train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking/'

for lab_folder in os.listdir(train_tracking_path):
    lab_folder_path = os.path.join(train_tracking_path, lab_folder)

    if os.path.isdir(lab_folder_path):
        # 獲取所有 parquet 文件
        parquet_files = [file for file in os.listdir(lab_folder_path) if file.endswith('.parquet')]
        
        # 使用 tqdm 來顯示進度條
        for file_name in tqdm(parquet_files, desc=f"Processing folder: {lab_folder}", unit="file"):
            # 提取 video_id，去掉文件擴展名
            video_id = file_name[:-8]  # 去掉 ".parquet"

            # 檢查 video_id 是否在集合中
            if video_id in video_ids_set:
                file_path = os.path.join(lab_folder_path, file_name)

                # 讀取 parquet 文件
                temp_df = pd.read_parquet(file_path)

                # 添加 video_id 和 lab_id 列
                temp_df['video_id'] = video_id
                temp_df['lab_id'] = lab_folder  # lab_folder 作為 lab_id

                # 將讀取的數據添加到結果 DataFrame 中
                result_df = pd.concat([result_df, temp_df], ignore_index=True)

# 現在 result_df 包含了所有需要的數據
print(result_df)


import pandas as pd
import os
from tqdm import tqdm

video_ids_set = set(common_videos_df['video_id'].unique())

train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking/'

for lab_folder in os.listdir(train_tracking_path):
    lab_folder_path = os.path.join(train_tracking_path, lab_folder)

    if os.path.isdir(lab_folder_path):
        # 獲取所有 parquet 文件
        parquet_files = [file for file in os.listdir(lab_folder_path) if file.endswith('.parquet')]
        
        # 使用 tqdm 來顯示進度條
        for file_name in tqdm(parquet_files, desc=f"Processing folder: {lab_folder}", unit="file"):
            # 提取 video_id，去掉文件擴展名
            video_id = file_name[:-8]  # 去掉 ".parquet"

            # 檢查 video_id 是否在集合中
            if video_id in video_ids_set:
                file_path = os.path.join(lab_folder_path, file_name)

                # 讀取 parquet 文件
                temp_df = pd.read_parquet(file_path)

                # 添加 video_id 和 lab_id 列
                temp_df['video_id'] = video_id
                temp_df['lab_id'] = lab_folder  # lab_folder 作為 lab_id

                # 生成文件名
                output_file = f"{lab_folder}.csv"

                # 將數據附加到相應的 CSV 文件
                temp_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

                print(f"Appended data to: {output_file}")


import pandas as pd
import os
from tqdm import tqdm

video_ids_set = set(common_videos_df['video_id'].unique())

train_tracking_path = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking/'

for lab_folder in os.listdir(train_tracking_path):
    lab_folder_path = os.path.join(train_tracking_path, lab_folder)

    if os.path.isdir(lab_folder_path):
        # 獲取所有 parquet 文件
        parquet_files = [file for file in os.listdir(lab_folder_path) if file.endswith('.parquet')]
        
        # 使用 tqdm 來顯示進度條
        for file_name in tqdm(parquet_files, desc=f"Processing folder: {lab_folder}", unit="file"):
            # 提取 video_id，去掉文件擴展名
            video_id = file_name[:-8]  # 去掉 ".parquet"

            # 檢查 video_id 是否在集合中
            if video_id in video_ids_set:
                file_path = os.path.join(lab_folder_path, file_name)

                # 讀取 parquet 文件
                temp_df = pd.read_parquet(file_path)

                # 添加 video_id 和 lab_id 列
                temp_df['video_id'] = video_id
                temp_df['lab_id'] = lab_folder  # lab_folder 作為 lab_id

                # 生成文件名
                output_file = f"{lab_folder}.csv"

                # 將數據附加到相應的 CSV 文件
                temp_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

                print(f"Appended data to: {output_file}")


drop_body_parts = [
    'ear_left', 'ear_right',
    'headpiece_bottombackleft', 'headpiece_bottombackright', 
    'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 
    'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'tail_midpoint'
]

# 計算每個 (video_id, mouse_id) 的 bodypart 數量
bodypart_counts = result_df.groupby(['video_id', 'mouse_id', 'bodypart']).size().reset_index(name='count')

# 找出出現 5 個或以上 bodypart 的 (video_id, mouse_id)
to_drop = bodypart_counts.groupby(['video_id', 'mouse_id']).filter(lambda x: x['count'].count() >= 5)

# 獲取需要刪除的 video_id 和 mouse_id 的組合
drop_ids = to_drop[['video_id', 'mouse_id']].drop_duplicates()

# 筛选需要删除的行
df_dropped = result_df[~result_df[['video_id', 'mouse_id']].apply(tuple, axis=1).isin(drop_ids.apply(tuple, axis=1)) | df['bodypart'].isin(drop_body_parts)]

# 顯示結果
print(df_dropped)

