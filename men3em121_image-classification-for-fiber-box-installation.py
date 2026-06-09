pip install ultralytics==8.0.55


import os
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


initial_train = os.listdir("/kaggle/input/ada-image-recognition-fiber/dataset/images/train")
train_imgs, val_imgs = train_test_split(initial_train, train_size=0.9, random_state=1)


train_txt = "train.txt"
val_txt = "val.txt"
train_dir = "/kaggle/input/ada-image-recognition-fiber/dataset/images/train"
for imgs, txt in zip([train_imgs, val_imgs], [train_txt, val_txt]):
    with open(txt, 'w') as file:
        for img in imgs:
            file.write(os.path.join(train_dir, img)+"\n")


yaml_path = "/kaggle/working/custom.yaml"
yaml_content = f""" 
train: {train_txt}
val: {val_txt}
test: /kaggle/input/ada-image-recognition-fiber/dataset/images/test

# class names
names: 
  0: Screw,
  1: Foam
  2: Plastic cover
  3: Tie-wrap, 
  4: Rubbers
"""

with open(yaml_path, 'w') as file:
    file.write(yaml_content)


model = YOLO("yolov8n.pt")


model.train(
    data=yaml_path,
    epochs=100,
    device="cuda:0",
    project="dtp-f")


trained_model = YOLO("/kaggle/working/dtp-f/train/weights/best.pt")
results = trained_model.predict(
    source="/kaggle/input/ada-image-recognition-fiber/dataset/images/test",
    device="0",
    save=True, 
    save_txt=True,
    project="dtp-f")  # save predictions as labels

