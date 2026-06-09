# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pydicom
ds = pydicom.dcmread("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/0005e8e3701dfb1dd93d53e2ff537b6e.dicom")


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
plt.imshow(ds.pixel_array, cmap = plt.cm.gray)
plt.show()


def dicom_conversion(dicom_path):
    # Load DICOM
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(float)
    
    # Check original image properties
    print(f"Original range: {img.min()} to {img.max()}")
    print(f"Original dtype: {img.dtype}")
    print(f"Image shape: {img.shape}")
    
    # Method 1: Simple normalization (often fails)
    simple_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    # Method 2: Percentile-based normalization (usually better)
    p1, p99 = np.percentile(img, (1, 99))  # Remove outliers
    percentile_norm = np.clip((img - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    
    # Method 3: Histogram equalization
    from skimage import exposure
    hist_eq = exposure.equalize_hist(img) * 255
    hist_eq = hist_eq.astype(np.uint8)
    
    # Compare all methods
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Original DICOM
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Original DICOM (Raw)')
    
    # Simple normalization
    axes[0,1].imshow(simple_norm, cmap='gray')
    axes[0,1].set_title('Simple Normalization')
    
    # Percentile normalization  
    axes[1,0].imshow(percentile_norm, cmap='gray')
    axes[1,0].set_title('Percentile Normalization (1-99%)')
    
    # Histogram equalized
    axes[1,1].imshow(hist_eq, cmap='gray')
    axes[1,1].set_title('Histogram Equalization')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return the best looking one
    return percentile_norm  # Usually this works best

# Test with your DICOM file
better_img = dicom_conversion("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/0005e8e3701dfb1dd93d53e2ff537b6e.dicom")


pip install ultralytics


from ultralytics import YOLO

model = YOLO("yolo11m.pt")


print(os.getcwd())


os.makedirs("val/images")
os.makedirs("val/labels")
os.makedirs("train/images")
os.makedirs("train/labels")


def dicom_to_jpeg(path,output):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(float)
    p1, p99 = np.percentile(img, (1, 99))  # Remove outliers
    percentile_norm = np.clip((img - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(percentile_norm).save(output,quality=95)

dicom_to_jpeg("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test/002a34c58c5b758217ed1f584ccbcfe9.dicom","test1.jpg")
img = Image.open("/kaggle/working/test1.jpg")
plt.imshow(img, cmap='gray')  # gray colormap for X-rays
plt.title("Your Converted X-ray Image")
plt.axis('off')  # Remove axes
plt.show()


n= 1
for dirpath, dirnames, filenames in os.walk("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train"):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        # os.remove(file_path)
        if(n<=3000):
            jpeg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpeg_path = os.path.join("/kaggle/working/train/images", jpeg_filename)
            dicom_to_jpeg(file_path,jpeg_path)
        else:
            break
        n+=1


n =1
for dirpath, dirnames, filenames in os.walk("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train"):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        if(n>4000 and n<=4700):
            jpeg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpeg_path = os.path.join("/kaggle/working/val/images", jpeg_filename)
            dicom_to_jpeg(file_path,jpeg_path)
        elif(n>4700):
            break
        n+=1
        


df = pd.read_csv("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")


for dirpath, dirnames, filenames in os.walk("/kaggle/working/train/images"):
    for filename in filenames:
        img = Image.open(f"/kaggle/working/train/images/{filename}")
        width, height = img.size
        jpeg_filename = os.path.splitext(filename)[0]
        rows = df[df["image_id"]==jpeg_filename]
        with open(f"/kaggle/working/train/labels/{jpeg_filename}.txt", "w") as f:
            f.write("")
        if rows["class_id"].iloc[0] != 14:
            rows = rows.sort_values(by = 'class_id',ascending = True)
            count = (df["image_id"]==jpeg_filename).sum()
            for i in range(0,count):
                text = str(rows["class_id"].iloc[i]) + " "
                x_cen = (rows["x_min"].iloc[i] + rows["x_max"].iloc[i])/2
                y_cen = (rows["y_min"].iloc[i] + rows["y_max"].iloc[i])/2
                wid = (rows["x_max"].iloc[i] - rows["x_min"].iloc[i])
                hei = (rows["y_max"].iloc[i] - rows["y_min"].iloc[i])
                norm_x = x_cen/width
                norm_y = y_cen/height
                norm_w = wid/width
                norm_h = hei/height
                text=text+str(norm_x)+" "+str(norm_y)+" "+str(norm_w)+" "+str(norm_h)+"\n"
                # print(text)
                with open(f"/kaggle/working/train/labels/{jpeg_filename}.txt", "a") as f:
                    f.write(text)


f = open("/kaggle/working/train/labels/00aca42a24e4ea6066cca2546150c36e.txt")
print(f.read())
f.close()


for dirpath, dirnames, filenames in os.walk("/kaggle/working/val/images"):
    for filename in filenames:
        img = Image.open(os.path.join(dirpath,filename))
        width, height = img.size
        jpeg_filename = os.path.splitext(filename)[0]
        rows = df[df["image_id"]==jpeg_filename]
        with open(f"/kaggle/working/val/labels/{jpeg_filename}.txt", "w") as f:
            f.write("")
        if rows["class_id"].iloc[0] != 14:
            rows = rows.sort_values(by = 'class_id',ascending = True)
            count = (df["image_id"]==jpeg_filename).sum()
            for i in range(0,count):
                text = str(rows["class_id"].iloc[i]) + " "
                x_cen = (rows["x_min"].iloc[i] + rows["x_max"].iloc[i])/2
                y_cen = (rows["y_min"].iloc[i] + rows["y_max"].iloc[i])/2
                wid = (rows["x_max"].iloc[i] - rows["x_min"].iloc[i])
                hei = (rows["y_max"].iloc[i] - rows["y_min"].iloc[i])
                norm_x = x_cen/width
                norm_y = y_cen/height
                norm_w = wid/width
                norm_h = hei/height
                text=text+str(norm_x)+" "+str(norm_y)+" "+str(norm_w)+" "+str(norm_h)+"\n"
                # print(text)
                with open(f"/kaggle/working/val/labels/{jpeg_filename}.txt", "a") as f:
                    f.write(text)


f = open("/kaggle/working/val/labels/02cd1d17763c869ff3d4af5e28539456.txt")
print(f.read())
f.close()


import yaml

# Your dataset configuration
config = {
    'train': '../train/images',
    'val': '../val/images',
    'nc': 14,
    'names': {0: 'Aortic enlargement', 1: 'Atelectasis', 2:'Calcification', 3:'Cardiomegaly', 4:'Consolidation',5:'ILD',6:'Infiltration',7:'Lung Opacity',8:'Nodule/Mass',9:'Other lesion',10:'Pleural effusion',11:'Pleural thickening', 12:'Pneumothorax', 13:'Pulmonary fibrosis'}
}

# Write to file
with open('data.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False,sort_keys=False)


f = open("/kaggle/working/data.yaml","r")
print(f.read())
f.close()


results = model.train(data="data.yaml", epochs=100, imgsz=640,batch=16,device="0,1")


tr_model = YOLO("/kaggle/working/runs/detect/train22/weights/last.pt")
tr_results = tr_model.train(resume = True)


img = Image.open("/kaggle/working/runs/detect/train22/results.png")
plt.figure(figsize = (10,10))
plt.imshow(img)
plt.axis("off")
plt.show()


x_ray_model = YOLO("/kaggle/working/runs/detect/train22/weights/best.pt")


def show_bbimg(path,filename,ax):
    img = Image.open(path)
    ax.imshow(img,cmap='gray')
    rows = df[df["image_id"]==filename]
    if rows["class_id"].iloc[0] != 14:
        dic={}
        rows = rows.sort_values(by = 'class_id',ascending = True)
        count = (df["image_id"]==filename).sum()
        for i in range(0,count):
            wid = (rows["x_max"].iloc[i] - rows["x_min"].iloc[i])
            hei = (rows["y_max"].iloc[i] - rows["y_min"].iloc[i])
            label = rows["class_name"].iloc[i]
            rect = patches.Rectangle((rows["x_min"].iloc[i], rows["y_min"].iloc[i]), wid, hei, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(rows["x_min"].iloc[i], rows["y_min"].iloc[i] - 5, str(label), color="white", fontsize=10,
                    bbox=dict(facecolor="red", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"))
            if label in dic:
                dic[label]+=1
            else:
                dic[label] = 1
    print("Ground Truth",dic)
    ax.axis("off")


dicom_to_jpeg("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/009d837e29ba400e03856cf8d6a5b545.dicom","/kaggle/working/test2.jpg")


results=x_ray_model("test2.jpg")


fig, ax = plt.subplots(1,2,figsize=(10,8))
show_bbimg("/kaggle/working/test2.jpg","009d837e29ba400e03856cf8d6a5b545",ax[0])
ax[0].set_title("Ground Truth")
ax[1].set_title("Predicted X-ray Image")
ax[1].imshow(results[0].plot())
ax[1].axis('off') 
plt.show()


img_id="ff924bcbd38f123aec723aa7040d7e43"
filepath=f"/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/{img_id}.dicom"
jpg_path = "/kaggle/working/test3.jpg"

dicom_to_jpeg(filepath,jpg_path)
res2=x_ray_model(jpg_path)
fig, ax = plt.subplots(1,2,figsize=(10,8))
show_bbimg(jpg_path,img_id,ax[0])
ax[0].set_title("Ground Truth")
ax[1].set_title("Predicted X-ray Image")
ax[1].imshow(res2[0].plot())
ax[1].axis('off') 
plt.show()


img_id="f9dda1a40ac162af4e9fbc6027ed5375"
filepath=f"/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/{img_id}.dicom"
jpg_path = "/kaggle/working/test4.jpg"

dicom_to_jpeg(filepath,jpg_path)
res=x_ray_model(jpg_path)
fig, ax = plt.subplots(1,2,figsize=(10,8))
show_bbimg(jpg_path,img_id,ax[0])
ax[0].set_title("Ground Truth")
ax[1].set_title("Predicted X-ray Image")
ax[1].imshow(res[0].plot())
ax[1].axis('off') 
plt.show()


metrics = x_ray_model.val(data="data.yaml", split="val")
print(metrics) 


print("mAP@0.5:", metrics.box.map50) 
print("mAP@0.5:0.95:", metrics.box.map) 


img = Image.open("runs/detect/val/confusion_matrix.png")
plt.figure(figsize=(14,14))
plt.imshow(img)
plt.axis("off")
plt.show()

