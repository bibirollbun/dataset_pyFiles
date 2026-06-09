! pip install pyyaml
! pip install ultralytics
! pip install opencv-python
! pip install matplotlib
! pip install numpy
! pip install requests
! pip install kagglehub
! pip install pandas


import os
import shutil
import requests
import zipfile
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from ultralytics import YOLO
import kagglehub
import pandas as pd


os.makedirs(os.path.abspath("./newDatasetFolder"), exist_ok=True)


finalDirs = os.listdir(os.path.abspath("./newDatasetFolder"))
print(finalDirs)

nameIndex = 0

if not os.path.exists(os.path.abspath("./newDatasetFolder/A_train_dataset")): 
    os.makedirs(os.path.abspath("./newDatasetFolder/A_train_dataset"), exist_ok=True)
    os.makedirs(os.path.abspath("./newDatasetFolder/A_train_dataset/images"), exist_ok=True)
    os.makedirs(os.path.abspath("./newDatasetFolder/A_train_dataset/labels"), exist_ok=True)


modelPath = kagglehub.dataset_download("niazmahmud0201/finetunedmodel-2")

print("modelPath: ", modelPath)


dest_image_path = os.path.abspath("./newDatasetFolder/A_train_dataset/images")
dest_label_path = os.path.abspath("./newDatasetFolder/A_train_dataset/labels")

not_founded_labels = []
print("yes")


listDSDir = ["finetunedmodel-2"]
print(listDSDir)

for x in listDSDir: 
    ppat = os.path.join(os.path.abspath("/kaggle/input"), x)

    if x == "finetunedmodel-2": 
        continue

    insidePath = os.path.join(ppat, os.listdir(ppat)[0])
    
    dirPath_image = os.path.join(insidePath, "images")
    dirPath_label = os.path.join(insidePath, "labels")
    # print(dirPath_image, dirPath_label)

    images = os.listdir(dirPath_image)
    labels = os.listdir(dirPath_label)

    for index, filename in enumerate(images):
        img_src_path = os.path.join(dirPath_image, filename)
        label_src_path = os.path.join(dirPath_label, f"{filename.split('.')[0]}.txt") 

        if os.path.exists(label_src_path) and os.path.exists(img_src_path):
            label_dest_path = os.path.join(dest_label_path,  f"{int(nameIndex):09}.txt")
            shutil.copy2(label_src_path, label_dest_path)

            img_dst_path = os.path.join(dest_image_path, f"{int(nameIndex):09}.png")
            shutil.copy2(img_src_path, img_dst_path)

            nameIndex += 1
            if index%100 == 0: 
                print(filename, f"{filename.split('.')[0]}.txt", "  ||index:", nameIndex)
        else: 
            not_founded_labels.append(f"{filename.split('.')[0]}.txt")
            print(f"label not for:{filename}")
            


print("Num of labelsnot found for desired iamge: ", len(not_founded_labels))
not_founded_labels


urls = ["https://storage.googleapis.com/duality-public-share/Hackathons/kaggle2/Kaggle2StartingDataset.zip",
        "https://storage.googleapis.com/duality-public-share/Hackathons/kaggle2/coolLighting.zip",
        "https://storage.googleapis.com/duality-public-share/Hackathons/kaggle2/cameraDistance.zip",
        "https://storage.googleapis.com/duality-public-share/Hackathons/kaggle2/furniture.zip", 
        "https://storage.googleapis.com/duality-public-share/Hackathons/kaggle2/plants.zip",
        ]

fileName = 1
for x in urls: 
    response = requests.get(x)
    
    zip_path = os.path.join(os.path.abspath("./newDatasetFolder"), f"{fileName}.zip")
    fileName += 1
    print(zip_path)
    
    with open(zip_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to: {zip_path}")

print("Downloaded Completed!!!")


downloaded_folder = os.listdir(os.path.abspath("./newDatasetFolder"))
print(downloaded_folder)

for x in downloaded_folder: 
    if x.split(".")[-1] == "zip": 
        dataPath = os.path.join(os.path.abspath("./newDatasetFolder"), x)
        with zipfile.ZipFile(dataPath, 'r') as zip_ref:
            zip_ref.extractall(os.path.abspath("./newDatasetFolder"))
        os.remove(dataPath)
    
print("extraction Done!!!")


def imgAdderLight(img): 
    normal_less_light_kernel1 = np.ones((3, 3), np.float32)/(5*5) 
    img_less_light1 = cv2.filter2D(img, -1, normal_less_light_kernel1)
    return img_less_light1


finalDirs = os.listdir(os.path.abspath("./newDatasetFolder"))
finalDirs.remove("A_train_dataset")
print(finalDirs)

not_founded_labels = []

for x in finalDirs: 
    dirPath = os.path.join(os.path.abspath("./newDatasetFolder"), x)
        
    if x == "Kaggle2StartingDataset": 
        dirPath_image = os.path.join(dirPath, "train/images")
        dirPath_label = os.path.join(dirPath, "train/labels")
        print(dirPath_image, dirPath_label)
    else: 
        dirPath_image = os.path.join(dirPath, "images")
        dirPath_label = os.path.join(dirPath, "labels")
        print(dirPath_image, dirPath_label)
        
    images = os.listdir(dirPath_image)
    labels = os.listdir(dirPath_label)
    
    # print(len(images), len(labels))
    for index, filename in enumerate(images):
        img_src_path = os.path.join(dirPath_image, filename)
        label_src_path = os.path.join(dirPath_label, f"{filename.split('.')[0]}.txt") 
        
        if os.path.exists(label_src_path) and os.path.exists(img_src_path):
             
            label_dest_path0 = os.path.join(dest_label_path,  f"{int(nameIndex):09}.txt")
            label_dest_path1 = os.path.join(dest_label_path,  f"{int(nameIndex+1):09}.txt")          
              
            shutil.copy2(label_src_path, label_dest_path0)
            shutil.copy2(label_src_path, label_dest_path1)
        
            cvtImg = cv2.imread(img_src_path, cv2.IMREAD_COLOR)
            cvtImg = cv2.cvtColor(cvtImg, cv2.COLOR_BGR2RGB)
            retImgst = imgAdderLight(cvtImg)
            
            img_dst_path = os.path.join(dest_image_path, f"{int(nameIndex):09}.png")
            shutil.copy2(img_src_path, img_dst_path)
            cv2.imwrite(os.path.join(dest_image_path, f"{int(nameIndex+1):09}.png"), retImgst)
            
            nameIndex += 2
            if index%100 == 0: 
                print(filename, f"{filename.split('.')[0]}.txt", "  ||index:", nameIndex)
        else: 
            not_founded_labels.append(f"{filename.split('.')[0]}.txt")
            print(f"label not for:{filename}")
            
    if x == "Kaggle2StartingDataset": 
        shutil.rmtree(os.path.join(dirPath, "train"))
        print("removed: ", os.path.join(dirPath, "train"))
    else:
        shutil.rmtree(dirPath)
        print("removed: ", dirPath)


print("Num of labelsnot found for desired iamge: ", len(not_founded_labels))
not_founded_labels


nameIndex = 0

if not os.path.exists(os.path.abspath("./newDatasetFolder/A_val_dataset")): 
    os.makedirs(os.path.abspath("./newDatasetFolder/A_val_dataset"), exist_ok=True)
    os.makedirs(os.path.abspath("./newDatasetFolder/A_val_dataset/images"), exist_ok=True)
    os.makedirs(os.path.abspath("./newDatasetFolder/A_val_dataset/labels"), exist_ok=True)


srcValPath_img = os.path.abspath("./newDatasetFolder/Kaggle2StartingDataset/val/images")
srcValPath_labels = os.path.abspath("./newDatasetFolder/Kaggle2StartingDataset/val/labels")

srcValPath_img_list = os.listdir(srcValPath_img)
srcValPath_labels_list = os.listdir(srcValPath_labels)

valNotFounded_labels = []
for index, filename in enumerate(srcValPath_img_list):
    img_src_path = os.path.join(srcValPath_img, filename)
    img_dst_path = os.path.join(os.path.abspath("./newDatasetFolder/A_val_dataset/images"), f"{nameIndex:09}.png")
    

    label_src_path = os.path.join(srcValPath_labels, f"{filename.split('.')[0]}.txt")
    label_dest_path = os.path.join(os.path.abspath("./newDatasetFolder/A_val_dataset/labels"),  f"{nameIndex:09}.txt")
    
    if os.path.exists(label_src_path) and os.path.exists(img_src_path):     
        shutil.copy2(img_src_path, img_dst_path)
        shutil.copy2(label_src_path, label_dest_path)

        nameIndex+=1
        if index%100 == 0: 
            print(filename, f"{filename.split('.')[0]}.txt")
    else: 
        valNotFounded_labels.append(f"{filename}")


def getRectDetailsForImg(h, w , labelPath): 
    rectDetails = []
    with open(labelPath, "r") as f: 
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # noow convert the value in pixel form
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            rectDetails.append([x1, y1, x2, y2])
    return rectDetails


imgPathAnalyze = os.path.abspath("./newDatasetFolder/A_train_dataset/images")
labelPathAnalyze = os.path.abspath("./newDatasetFolder/A_train_dataset/labels")

imgPathAnalyzeList = os.listdir(imgPathAnalyze)

fig = plt.figure(figsize=(15, 6)) 
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.15, hspace=0.2)

for index in range(10): 
    x = f"{random.randint(0, len(imgPathAnalyzeList) - 1):09}.png"
    
    labelNameAnz = f"{x.split('.')[0]}.txt"
    
    imgPath1Anz = os.path.join(imgPathAnalyze, x)
    labelPath1Anz = os.path.join(labelPathAnalyze, labelNameAnz)
    
    image = cv2.imread(imgPath1Anz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width , channel = image.shape
    
    rectDetails = getRectDetailsForImg(height, width, labelPath1Anz)
    for xp in rectDetails: 
        cv2.rectangle(image, (xp[0], xp[1]), (xp[2], xp[3]), (0, 0,0), 2)
    
    
    image = cv2.resize(image, (int(width*0.5), int(height*0.5)))
    plt.subplot(2, 5, index+1)
    plt.imshow(image)
    plt.title(x, fontsize=8)
    plt.xticks([])
    plt.yticks([])
    

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


model = YOLO(os.path.join(modelPath, os.listdir(modelPath)[0]))

file_list = os.listdir(os.path.abspath('./newDatasetFolder/A_train_dataset/images'))
image_path = os.path.join(os.path.abspath('./newDatasetFolder/A_train_dataset/images'),  f"{random.randint(1, len(file_list) - 1):09}.png")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predRes = model(image)[0].plot()
predRes = cv2.resize(predRes, (int(image.shape[1]*0.5), int(image.shape[0]*0.5)))
plt.imshow(predRes)
cv2.waitKey(0)
cv2.destroyAllWindows()


yamlContent = {
    "path": os.path.abspath("./newDatasetFolder"), 
    "train": "A_train_dataset",
    "val": "A_val_dataset",
    "test": "Kaggle2StartingDataset/testImages",
    "nc": 1, 
    "names": ["soup"]
}

with open('z_custom_data.yaml', 'w') as file:
    yaml.dump(yamlContent, file, default_flow_style=False)


with open('z_custom_data.yaml', 'r', encoding="utf-8") as file:
    print(file.read())


import torch

print("Number of gpu's: ", torch.cuda.device_count())
print("Number of cpu's:", os.cpu_count())


results = model.train(data="z_custom_data.yaml", 
                    epochs=30,
                    batch=20,
                    lr0=0.0005, 
                    device= [0,1],
                    optimizer= "auto", 
                    cos_lr=True, 
                    augment=True, 
                    workers= os.cpu_count(),  
                    flipud=0.232, 
                    fliplr=0.232, 
                    translate=0.077, 
                    scale=0.121, 
                    shear=0.001 
                    ) 


trainInfoPath = os.path.abspath("./runs/detect/train")

fig = plt.figure(figsize=(15, 10)) 

trainF1 = cv2.imread(os.path.join(trainInfoPath, "F1_curve.png"))
plt.subplot(2, 2, 1)
plt.axis('off')
plt.imshow(trainF1)

trainPR = cv2.imread(os.path.join(trainInfoPath, "PR_curve.png"))
plt.subplot(2, 2, 2)
plt.axis('off')
plt.imshow(trainPR)

trainP = cv2.imread(os.path.join(trainInfoPath, "P_curve.png"))
plt.subplot(2, 2, 3)
plt.axis('off')
plt.imshow(trainP)

trainR = cv2.imread(os.path.join(trainInfoPath, "R_curve.png"))
plt.subplot(2, 2, 4)
plt.axis('off')
plt.imshow(trainR)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


trainCF = cv2.imread(os.path.abspath("./runs/detect/train/confusion_matrix.png"))
trainCF = cv2.cvtColor(trainCF, cv2.COLOR_BGR2RGB)

plt.imshow(trainCF)
cv2.waitKey(0)
cv2.destroyAllWindows()


trainSumfig = plt.figure(figsize=(15, 10))

trainResult = cv2.imread(os.path.abspath("./runs/detect/train/results.png"))
trainResult = cv2.cvtColor(trainResult, cv2.COLOR_BGR2RGB)
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(trainResult)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


testModelPath = os.path.join(os.path.abspath("./runs/detect/train/weights"), "last.pt")
TestModel = YOLO(testModelPath)


# testimgPath = os.path.abspath("./newDatasetFolder/Kaggle2StartingDataset/testImages/images")
testimgPath = os.path.abspath("/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images")

imgPathAnalyzeList = os.listdir(testimgPath)

fig = plt.figure(figsize=(15, 6)) 
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.15, hspace=0.2)

preds = []
for index in range(10): 
    while True: 
        x= imgPathAnalyzeList[random.randint(0, len(imgPathAnalyzeList) - 1)]
        if x not in preds: 
            preds.append(x)
            break 
        print("img Found Same in randint!!!")
    
    imgPath1Anz = os.path.join(testimgPath, x)
    
    image = cv2.imread(imgPath1Anz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predTestImg = TestModel.predict(image, conf=0.85, verbose=False)[0].plot()
    # predTestImg = cv2.resize(predTestImg, (int(width*0.5), int(height*0.5)))
    
    plt.subplot(2, 5, index+1)
    plt.axis('off')
    plt.imshow(predTestImg)
    

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



valResult  = TestModel.val(data="z_custom_data.yaml", split='test')

print("\n\n")
print("Precision:", valResult.box.p)
print("Recall:", valResult.box.r)
print("mAP@0.5(mean average precision): ", valResult.box.map50)
print("mAP: ", valResult.box.map)


import pandas as pd

testimgPath = os.path.abspath("/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images")
listTestImgs = os.listdir(testimgPath)

csvContent = []

for x in listTestImgs: 
    testImgss = cv2.imread(os.path.join(testimgPath, x))
    testImgss = cv2.cvtColor(testImgss, cv2.COLOR_BGR2RGB)
    height, width, color_channel = testImgss.shape 
    
    pred = TestModel.predict(testImgss, verbose=False, conf=0.8)[0]
    
    output_line = ""
    for box in pred.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xCenter, yCenter, imgW, imgH = box.xywh[0]
        
        xCenter /= width
        yCenter /= height
        imgW /= width
        imgH /= height
        output_line += f"{cls_id} {conf} {xCenter} {yCenter} {imgW} {imgH} "
        
        
    csvContent.append({
        "image_id": x.split(".")[0], 
        "prediction_string": output_line.strip() if len(output_line.strip())>1 else "no boxes"
    })


df = pd.DataFrame(csvContent)
subFilePath = os.path.join(os.path.abspath("."), "submission.csv")
if os.path.exists(subFilePath): 
    os.remove(subFilePath)
    
df.to_csv(subFilePath, index=False)
print("submission.csv is created")

