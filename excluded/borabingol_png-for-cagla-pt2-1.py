import pandas as pd
import numpy as np
import os
import pydicom
import yaml
import cv2
from sklearn.model_selection import train_test_split


#base_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
output_dir = "/kaggle/working/"


# Dosya yollarını hazırla
base_dir="/kaggle/working"
def create_dataset_structure(condition_groups, base_dir):
    for group_name, conditions in condition_groups.items():
        group_dir = os.path.join(base_dir, group_name.replace(' ', '_'))
        labels_dir = os.path.join(group_dir, 'labels')
        train_labels_dir = os.path.join(labels_dir, 'train')
        val_labels_dir = os.path.join(labels_dir, 'val')
        images_dir = os.path.join(group_dir, 'images')
        train_images_dir = os.path.join(images_dir, 'train')
        val_images_dir = os.path.join(images_dir, 'val')

        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        # Oluşturulan yolları yazdır
        print(f"Klasörler oluşturuldu:\n"
              f"Train Images: {train_images_dir}\n"
              f"Validation Images: {val_images_dir}\n"
              f"Train Labels: {train_labels_dir}\n"
              f"Validation Labels: {val_labels_dir}\n")
        print(train_images_dir, train_labels_dir, val_labels_dir, val_images_dir)


condition_groups = {
    'Spinal Canal Stenosis': ['Spinal Canal Stenosis'],
    'Neural Foraminal Narrowing': ['Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing'],
    'Subarticular Stenosis': ['Right Subarticular Stenosis', 'Left Subarticular Stenosis']
}

create_dataset_structure(condition_groups, base_dir)

###

#Veri setini severe'e indirgeme
df = pd.read_csv('/kaggle/input/png-for-cagla/clean_data.csv')

normal_mild = df[df['severity'] == 'normal_mild']
moderate = df[df['severity'] == 'moderate']
severe = df[df['severity'] == 'severe']

severe_count = len(severe)
#print(severe_count)

normal_mild_sample = normal_mild.sample(n=severe_count, random_state=42)
moderate_sample = moderate.sample(n=severe_count, random_state=42)

balanced_df = pd.concat([normal_mild_sample, moderate_sample, severe])

balanced_df.to_csv('/kaggle/working/balanced_veri_seti.csv', index=False)


# Gruplara ayır
for group_name, conditions in condition_groups.items():
    # Filter rows based on condition
    filtered_df = balanced_df[balanced_df['condition'].isin(conditions)]
    # Split the data into train and validation sets 
    train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    # Save to new CSV file
    group_name_save=group_name.replace(' ','_')
    train_file_path = os.path.join(output_dir, f'{group_name_save}_train.csv')
    val_file_path = os.path.join(output_dir, f'{group_name_save}_val.csv')
    train_df.to_csv(train_file_path, index=False)
    val_df.to_csv(val_file_path, index=False)
    
    #train_df.to_csv(f'{group_name_save}_train.csv', index=False) 
    #val_df.to_csv(f'{group_name_save}_val.csv', index=False)
    
    print(group_name_save)


# Görüntüleri istenen formata dönüştür, pathler image_path'de
def dicom_to_png(df, is_train, target_size=(224,224)):
    
    images_list = []
    height_list = []
    width_list = []

    for index, row in df.iterrows():
        
        image_path = row['image_path']
        condition = row['condition']

        group_name = None 
        for key, values in condition_groups.items(): 
            if condition in values: 
                group_name = key.replace(' ', '_') 
                break 
        if group_name is None: 
                print(f"Condition '{condition}' için group_name bulunamadı.") 
                continue
        
        group_dir = os.path.join(base_dir, group_name)
        added_dir = os.path.join(group_dir, 'images')

        if is_train:
            output_dir = os.path.join(added_dir, 'train')
        else:
            output_dir = os.path.join(added_dir, 'val')

        # Grup dizinleri oluşturma
        group_dir = os.path.join(base_dir, group_name)
        added_dir = os.path.join(group_dir, 'images')

        if is_train:
            output_dir = os.path.join(added_dir, 'train')
        else:
            output_dir = os.path.join(added_dir, 'val')

        # DICOM dosyasını yükle
        dicom_file = pydicom.dcmread(image_path)
        
        # Pixel array'i al
        image_array = dicom_file.pixel_array

        # RescaleIntercept ve RescaleSlope uygulama
        intercept = getattr(dicom_file, "RescaleIntercept", 0)
        slope = getattr(dicom_file, "RescaleSlope", 1)
        image_array = image_array * slope + intercept
        
        # Görüntü boyutlarını kaydet
        height, width = image_array.shape
        height_list.append(height)
        width_list.append(width)

        # Normalize et (0-255 aralığına)
        normalized_image = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Görüntüyü hedef boyutlara resize et
        resized_image = cv2.resize(normalized_image, target_size) 
        images_list.append(resized_image)
    
        # PNG olarak kaydet
        png_path = os.path.join(output_dir, f'image_{index}.png') 
        os.makedirs(output_dir, exist_ok=True)  # Çıkış klasörünü oluştur
        success = cv2.imwrite(png_path, resized_image)
        if success:
            print(f"Saved {image_path} to {png_path}")
        else:
            print(f"Failed to save {image_path} to {png_path}")
        
    print(f"All images in group '{group_name}' are saved.")

    return height_list, width_list


class_mapping = { 'normal_mild': 0, 'moderate': 1, 'severe': 2 }


def create_annotations(df, is_train, output_dir):
    # bu formatta olacak: <class_id> <x_center> <y_center> <width> <height>
    heights, widths = dicom_to_png(df, is_train)
    
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        severity = row['severity'] 
        class_id = class_mapping[severity]

        x_center = row['x'] / widths[index]
        y_center = row['y'] / heights[index]
        box_width = 50 / widths[index]
        box_height = 50 / heights[index]

        annotation_path = os.path.join(output_dir, f'image_{index}.txt')
        with open(annotation_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

        #print(f'Saved annotation for image_{index} to {annotation_path}')
    print("Annotation files are saved.")


data_path = "/kaggle/working"
def create_yaml(data_path, class_names, yaml_path):
    # veri setinin yolunu ve sınıf isimlerini belirten bir konfigürasyon dosyası (yaml formatında) oluştur

    yaml_content = {
        'path': os.path.abspath(data_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
        print(f'YAML dosyası {yaml_path} oluşturuldu.')


for group_name, conditions in condition_groups.items():
    train_labels_dir = os.path.join(base_dir, group_name.replace(' ', '_'), 'labels', 'train') 
    val_labels_dir = os.path.join(base_dir, group_name.replace(' ', '_'), 'labels', 'val') 

    group_name_save=group_name.replace(' ','_')
    tr_df = pd.read_csv(f'{group_name_save}_train.csv')
    val_df = pd.read_csv(f'{group_name_save}_val.csv')
    is_train = 1
    create_annotations(tr_df, is_train, train_labels_dir) 
    is_train = 0
    create_annotations(val_df,is_train, val_labels_dir)

for group_name in condition_groups.keys(): 
    group_dir = os.path.join(base_dir, group_name.replace(' ', '_')) 
    yaml_path = os.path.join(group_dir, 'dataset.yaml') 
    create_yaml(group_dir, ['normal_mild', 'moderate', 'severe'], yaml_path)

