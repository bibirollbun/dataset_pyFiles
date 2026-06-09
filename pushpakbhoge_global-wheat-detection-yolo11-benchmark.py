!pip install /kaggle/input/sam3-project-offline-packages/wheels/ultralytics-8.3.233-py3-none-any.whl --no-deps 


from ultralytics import YOLO

### build pretrained YOLO v11 model
WEIGHT_FILE = "/kaggle/input/global-wheat-head-detection/pytorch/yolo-v11-large/1/best.pt"
model = YOLO(WEIGHT_FILE)


# visualize a random sample
import matplotlib.pyplot as plt
import random
import os

# randomely select one image
image_base_path = "/kaggle/input/global-wheat-detection/test"
image_names = os.listdir(image_base_path)
image_name = random.choice(image_names)
image_path = os.path.join(image_base_path, image_name)

# run the model on one image
yolo_predictions = model.predict(image_path, verbose=False)[0]

# draw the bounding boxes on image and showcase
image_drawn = yolo_predictions.plot(labels=False)
plt.figure(figsize=(16, 16))
plt.imshow(image_drawn)


import pandas as pd
from PIL import Image


submission_data = []
for image_name in image_names:
    # collect the image
    image_path = os.path.join(image_base_path, image_name)
    image_pil = Image.open(image_path)
    image_width, image_height = image_pil.size
    
    # run the model on one image
    yolo_predictions = model.predict(image_path, verbose=False)[0]

    # collect the predictions
    predictions = []
    for bbox in yolo_predictions.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = bbox
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(x2, 1024)), int(min(y2, 1024))
        width, height = x2-x1, y2- y1
        predictions += [round(score, 4), x1, y1, width, height]

    # append the predictions to the submission data
    submission_data.append({
        "image_id": image_name.split('.')[0],
        "PredictionString": " ".join(list(map(str, predictions)))
    })


submission_df = pd.DataFrame(submission_data)
submission_df.to_csv("submission.csv", index=False)
submission_df




