import numpy as np

SEED = 42
np.random.seed(SEED)


byu_train_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'

working_yolo_path = '/kaggle/working/BYU_YOLO_dataset'


import os

working_images_train_path = os.path.join(working_yolo_path, 'images', 'train')
working_images_test_path = os.path.join(working_yolo_path, 'images', 'val')
working_labels_train_path = os.path.join(working_yolo_path, 'labels', 'train')
working_labels_test_path = os.path.join(working_yolo_path, 'labels', 'val')


for path in [working_images_train_path, 
             working_images_test_path, 
             working_labels_train_path, 
             working_labels_test_path]:
    
    os.makedirs(path, exist_ok=True)


def normalize_slice(slice_data):
    
    percentile_2th = np.percentile(slice_data, 2)
    percentile_98th = np.percentile(slice_data, 98)
    
    clipped_data = np.clip(slice_data, percentile_2th, percentile_98th)
    
    normalized = 255 * (clipped_data - percentile_2th) / (percentile_98th - percentile_2th)
    
    return np.uint8(normalized)


import pandas as pd

train_labels_df = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')


train_labels_df


train_labels_df['Number of motors'].value_counts()


len(train_labels_df.tomo_id.unique())


train_labels_df.tomo_id.value_counts()


train_labels_df[train_labels_df['Number of motors'] > 0]


train_labels_df[train_labels_df.tomo_id == 'tomo_00e047']


train_labels_df[train_labels_df.tomo_id == 'tomo_00e463']


for _, row in train_labels_df[train_labels_df.tomo_id == 'tomo_00e463'].iterrows():
    print(row)
    print(50 * '-')


tomo_id_unique_motors_exist = train_labels_df[train_labels_df['Number of motors'] > 0].tomo_id.unique()


train_split_len = len(tomo_id_unique_motors_exist) * 4 // 5
train_split_len


np.random.shuffle(tomo_id_unique_motors_exist)

train_tomo_ids = tomo_id_unique_motors_exist[:train_split_len]
test_tomo_ids = tomo_id_unique_motors_exist[train_split_len:]


from tqdm.notebook import tqdm
from PIL import Image

TRUST = 4
BOX_SIZE = 24

def process_tomogram_set(train_test_tomo_ids, train_test_images_path, train_test_labels_path):
    
    motor_axis_data = []
    
    for train_test_tomo_id in train_test_tomo_ids:
        train_test_tomo_id_df = train_labels_df[train_labels_df.tomo_id == train_test_tomo_id]
        
        for _, row in train_test_tomo_id_df.iterrows():
            
            if pd.isna(row['Motor axis 0']):
                continue
            motor_axis_data.append(
                (train_test_tomo_id, 
                 int(row['Motor axis 0']), 
                 int(row['Motor axis 1']), 
                 int(row['Motor axis 2']),
                 int(row['Array shape (axis 0)']))
            )
    
    for tomo_id, motor_axis_0, motor_axis_1, motor_axis_2, array_shape_axis_0 in tqdm(motor_axis_data):
        
        motor_axis_0_min = max(0, motor_axis_0 - TRUST)
        array_shape_axis_0 = min(array_shape_axis_0 - 1, motor_axis_0 + TRUST)
        
        for axis_0 in range(motor_axis_0_min, array_shape_axis_0 + 1):
            
            slice_filename = f'slice_{axis_0:04d}.jpg'
            
            current_path = os.path.join(byu_train_path, tomo_id, slice_filename)
            
            if not os.path.exists(current_path):
                print(f"Warning: {current_path} does not exist, skipping.")
                continue
                
            image = Image.open(current_path)
            img_array = np.array(image)
            
            normalized_img = normalize_slice(img_array)
            
            dest_filename = f'{tomo_id}_z{axis_0:04d}_y{motor_axis_1:04d}_x{motor_axis_2:04d}.jpg'
            dest_path = os.path.join(train_test_images_path, dest_filename)
            
            Image.fromarray(normalized_img).save(dest_path)
            
            img_width, img_height = image.size
            
            motor_axis_2_norm = motor_axis_2 / img_width
            motor_axis_1_norm = motor_axis_1 / img_height
            box_width_norm = BOX_SIZE / img_width
            box_height_norm = BOX_SIZE / img_height

            label_path = os.path.join(train_test_labels_path, dest_filename.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f"0 {motor_axis_2_norm} {motor_axis_1_norm} {box_width_norm} {box_height_norm}\n")


process_tomogram_set(train_tomo_ids, working_images_train_path, working_labels_train_path)
process_tomogram_set(test_tomo_ids, working_images_test_path, working_labels_test_path)


import yaml

yaml_content = {
    'path': working_yolo_path,
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'motor'}
}

with open(os.path.join(working_yolo_path, 'dataset.yaml'), 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)




