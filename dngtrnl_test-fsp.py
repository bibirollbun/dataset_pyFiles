import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import json
import matplotlib
import matplotlib.pyplot as plt
import random
import pyarrow.parquet as pq
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from matplotlib import animation, rc


file_path = '/kaggle/input/asl-fingerspelling/supplemental_landmarks/1032110484.parquet'  # ví dụ: 'data.txt'
size_bytes = os.path.getsize(file_path)
print(f"Dung lượng file: {size_bytes/(1024*1024*1024)} Gbytes")



table = pq.read_table(file_path)
# Convert to pandas DataFrame
print(table.shape)


face=0
right_hand=0
left_hand=0
pose=0
df = table.to_pandas()
for i in df.columns:
    if "face" in i:
        face+=1
        continue
    if "left_hand" in i:
        left_hand+=1
        continue
    if "right_hand" in i:
        right_hand+=1
        continue
    pose+=1
print(f"face: {face/3} ")
print(f"right_hand: {right_hand/3} ")
print(f"left_hand: {left_hand/3} ")
print(f"pose: {(pose-1)/3} ")#-1 for frame


LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE


X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]


FEATURE_COLUMNS = X + Y + Z
print(FEATURE_COLUMNS)


RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]


print(RHAND_IDX)
print(LHAND_IDX)
print(RPOSE_IDX)
print(LPOSE_IDX)


dataset_df = pd.read_csv('/kaggle/input/asl-fingerspelling/supplemental_metadata.csv')


# Đặt độ dài của mỗi chuỗi frame là 128
FRAME_LEN = 128

# Tạo thư mục để lưu dữ liệu đã xử lý
if not os.path.isdir("test"):
    os.mkdir("test")  # Nếu thư mục chưa tồn tại thì tạo mới
else:
    shutil.rmtree("test")  # Nếu đã tồn tại thì xóa thư mục cũ
    os.mkdir("test")       # Sau đó tạo lại thư mục mới

# Lặp qua từng file_id duy nhất trong dataset
for file_id in tqdm(dataset_df.file_id.unique()):
    # Đường dẫn đến file parquet chứa dữ liệu landmark
    pq_file = f"/kaggle/input/asl-fingerspelling/supplemental_landmarks/{file_id}.parquet"

    # Lọc ra các hàng trong dataset_df có cùng file_id
    file_df = dataset_df.loc[dataset_df["file_id"] == file_id]

    # Đọc file parquet, chỉ lấy cột sequence_id và các cột đặc trưng (FEATURE_COLUMNS)
    parquet_df = pq.read_table(
        f"/kaggle/input/asl-fingerspelling/supplemental_landmarks/{str(file_id)}.parquet",
        columns=['sequence_id'] + FEATURE_COLUMNS
    ).to_pandas()

    # Tên file TFRecord sẽ lưu dữ liệu đã xử lý
    tf_file = f"test/{file_id}.tfrecord"

    # Chuyển DataFrame thành numpy array để xử lý nhanh hơn
    parquet_numpy = parquet_df.to_numpy()

    # Ghi dữ liệu đã xử lý vào file TFRecord
    with tf.io.TFRecordWriter(tf_file) as file_writer:
        # Lặp qua từng chuỗi (sequence) và cụm từ (phrase) trong file
        for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
            # Lấy tất cả các frame tương ứng với sequence_id
            frames = parquet_numpy[parquet_df.index == seq_id]

            # Đếm số lượng frame không có giá trị NaN cho từng tay
            r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)  # tay phải
            l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)  # tay trái
            no_nan = max(r_nonan, l_nonan)  # Lấy số lượng frame tốt nhất giữa hai tay

            # Nếu số lượng frame tốt > 2 lần độ dài của cụm từ thì mới lấy
            if 2 * len(phrase) < no_nan:
                # Tạo dictionary các đặc trưng để ghi vào TFRecord
                features = {
                    FEATURE_COLUMNS[i]: tf.train.Feature(
                        float_list=tf.train.FloatList(value=frames[:, i])
                    ) for i in range(len(FEATURE_COLUMNS))
                }

                # Thêm nhãn (phrase) vào dữ liệu
                features["phrase"] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')])
                )

                # Tạo đối tượng Example và ghi vào file
                record_bytes = tf.train.Example(
                    features=tf.train.Features(feature=features)
                ).SerializeToString()

                # Ghi vào file TFRecord
                file_writer.write(record_bytes)


