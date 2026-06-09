import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil


output_dir = '/kaggle/working/'  
data_path = '/kaggle/input/global-wheat-detection/train'  

os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/valid'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/valid'), exist_ok=True)

def process_data(data, data_type='train'):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id']
        bounding_boxes = row['bboxes']
        yolo_data = []
        
        for bbox in bounding_boxes:
            x, y, w, h = bbox 
            x_center = (x + w / 2) / 1024.0  # Normalize x_center
            y_center = (y + h / 2) / 1024.0  # Normalize y_center
            w_norm = w / 1024.0  # Normalize width
            h_norm = h / 1024.0  # Normalize height
            yolo_data.append([0, x_center, y_center, w_norm, h_norm]) 
            
        yolo_data = np.array(yolo_data)  
    
        np.savetxt(
            os.path.join(output_dir, f'labels/{data_type}/{image_name}.txt'),
            yolo_data,
            fmt='%d %.6f %.6f %.6f %.6f',  # Format: class_id x_center y_center width height
            delimiter=' '
        )
        shutil.copyfile(
            os.path.join(data_path, f'{image_name}.jpg'),
            os.path.join(output_dir, f'images/{data_type}/{image_name}.jpg')
        )


df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")
df.head()


df_show = []
columns = df . columns 
for i in columns : 
    types = df[i] . dtypes
    unique_data = df[i] . nunique()
    NAN_value=df[i].isnull().sum()
    duplicated= df.duplicated().sum()  
    
    df_show . append ([i , types , unique_data , NAN_value,duplicated])
        
df_info = pd . DataFrame (df_show)
df_info . columns =['name of column' , 'types' ,'unique_data' , 'NAN value',"duplicated"]




df_info.style.highlight_max(color = 'pink', axis = 0)


def expand_bbox(x):
    r = np.array(re.findall("[0-9]+[.]?[0-9]*", x)).astype(np.float32)
    if len(r) != 4:
        print("WARNING! Bbox dimension not equal to 4")
    return r

df.bbox=df.bbox.apply(expand_bbox)


df.bbox


df = df.groupby('image_id')['bbox'].apply(list).reset_index(name='bboxes')


df


df_train, df_vaild = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)



df_train = df_train.reset_index(drop=True)
df_vaild = df_vaild.reset_index(drop=True)


process_data(df_train, data_type='train')
process_data(df_vaild, data_type='valid')


data_yaml_content=f"""
train: {os.path.join(output_dir,'images/train')}
val: {os.path.join(output_dir,'images/valid')}
nc: 1
names: ['wheat']
"""

data_yaml_path= os.path.join(output_dir, 'data.yaml')


with open(data_yaml_path, 'w') as file:
    file.write(data_yaml_content)
print(f"data.yaml file saved at: {data_yaml_path}")


with open(data_yaml_path,'r') as file:
    print(file.read())


!pip install ultralytics



from ultralytics import YOLO

model = YOLO('/kaggle/input/yolo11n/pytorch/default/1/yolo11n.pt')  
results = model.train(
    data=data_yaml_path,  
    epochs=2,                         
    imgsz=1024,                        
    batch=16,                          
    patience=10,                                                 
    cos_lr=True,                       
    warmup_epochs=3,                  
    augment=True,                     
    weight_decay=0.0005,               
    momentum=0.937,                    
    dropout=0.2                        
)


metrics = model.val(data='/kaggle/working/data.yaml', split='val')

mAP50 = metrics.box.map50  
mAP50_95 = metrics.box.map  
precision = metrics.box.p  
recall = metrics.box.r  

# Print metrics
print(f"mAP50: {mAP50:.4f}")
print(f"mAP50-95: {mAP50_95:.4f}")
print(f"Precision: {np.mean(precision):.4f}")  
print(f"Recall: {np.mean(recall):.4f}") 


sample_image_path = os.path.join(output_dir, 'images/valid', df_vaild.iloc[89]['image_id'] + '.jpg')
image = cv2.imread(sample_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

results = model.predict(source=sample_image_path, conf=0.5)  

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  

plt.figure(figsize=(6, 4))
plt.imshow(image)
plt.axis('off')
plt.show()


model.save('custom_yolo.pt')




