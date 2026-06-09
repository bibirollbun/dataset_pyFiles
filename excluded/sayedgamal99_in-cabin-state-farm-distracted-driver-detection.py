%%capture --no-display
!pip install ultralytics
!pip install roboflow
!pip install ultralytics
!pip install split-folders


import ultralytics
from ultralytics import YOLO
from IPython.display import Image
from roboflow import Roboflow
from kaggle_secrets import UserSecretsClient
import os
import splitfolders

ultralytics.checks()


splitfolders.ratio("/kaggle/input/state-farm-distracted-driver-detection/imgs/train", output="dataset", seed=32, ratio=(0.7, 0.15, 0.15))



model = YOLO('yolo11s-cls.pt')


results = model.train(data = '/kaggle/working/dataset', epochs = 150, batch=32, imgsz=640,degrees=10, patience=8,seed=42)


Image("/kaggle/working/runs/classify/train/train_batch0.jpg", width=600)


metrics = model.val()
print("Validation Metrics:")
print(metrics)



test_img_path = '/kaggle/working/dataset/test/c0/img_100074.jpg'  # adjust path as needed
results = model.predict(test_img_path, imgsz=640)
result = results[0]


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
img = Image.open(test_img_path)
plt.imshow(img)
plt.title(f'Predicted: {result.names[result.probs.top1]}')
plt.axis('off')

# Show probability distribution
plt.subplot(1, 2, 2)
probs = result.probs.data.cpu().numpy()
class_names = list(result.names.values())
y_pos = np.arange(len(class_names))
plt.barh(y_pos, probs)
plt.yticks(y_pos, class_names)
plt.xlabel('Probability')
plt.title('Class Probabilities')

plt.tight_layout()
plt.show()




