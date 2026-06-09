%%capture --no-display
!pip install ultralytics
!pip install split-folders
!pip install -U ipywidgets


from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import splitfolders
from IPython.display import display, Image
import pandas as pd


model = YOLO('yolov8n-cls.pt')


splitfolders.ratio("/kaggle/input/state-farm-distracted-driver-detection/imgs/train", output="output", seed=1337, ratio=(0.7, 0.15, 0.15))


results = model.train(data = '/kaggle/working/output', epochs = 5)


model.val()


df = pd.read_csv("/kaggle/working/runs/classify/train/results.csv")
df.head()


Image("/kaggle/working/runs/classify/train/results.png")


Image("/kaggle/working/runs/classify/train/confusion_matrix_normalized.png")


path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test/"
model_weights = "/kaggle/working/runs/classify/train/weights/best.pt"
pred = [(path+i,model.predict(path+i, model = model_weights)[0].probs.top1) for i in os.listdir(path)[:45]]


labels = {
    0: 'normal driving',
1: 'texting - right',
2: 'talking on the phone - right',
3: 'texting - left',
4: 'talking on the phone - left',
5: 'operating the radio',
6: 'drinking',
7: 'reaching behind',
8: 'hair and makeup',
9: 'talking to passenger'}


rows = 9
cols = 5
fig, ax = plt.subplots(rows, cols, figsize=(20, 20))
for i, (img, label) in enumerate(pred):
    row = i // cols
    col = i % cols
    ax[row, col].imshow(plt.imread(img))
    ax[row, col].set_title(labels.get(label))
    ax[row, col].axis('off')

plt.suptitle("Predicted Images")
plt.tight_layout()
plt.show()


!zip -r output.zip output

from IPython.display import FileLink
FileLink(r'output.zip')


