!pip install /kaggle/input/sam3-project-offline-packages/wheels/portalocker-3.2.0-py3-none-any.whl
!pip install /kaggle/input/sam3-project-offline-packages/wheels/iopath-0.1.10.tar.gz
!pip install /kaggle/input/sam3-project-offline-packages/wheels/ftfy-6.1.1-py3-none-any.whl
!pip install /kaggle/input/sam3-project-offline-packages/wheels/decord-0.6.0-py3-none-manylinux2010_x86_64.whl 
!pip install /kaggle/input/sam3-project-offline-packages/wheels/ultralytics-8.3.233-py3-none-any.whl --no-deps 
!cp /kaggle/input/sam3-project-offline-packages/sam3_repo -r ./sam3_repo


# add repository in the system path
import sys
sys.path.insert(0, "./sam3_repo")

SAM3_WEIGHT_FILE = "/kaggle/input/sam3-project-offline-packages/weights/sam3.pt"
PROMPT = "wheat, flower, seed head, green seed head, spike, green spike"


from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

### build SAM3 model
model = build_sam3_image_model(checkpoint_path=SAM3_WEIGHT_FILE, load_from_HF=False)
processor = Sam3Processor(model, confidence_threshold=0.3)


from sam3.model.sam3_image_processor import Sam3Processor
from typing import Union, Optional, Dict
from PIL import Image, ImageDraw
import torch


def run_sam3(processor: Sam3Processor, image: Union[str, Image], prompt: str):
    if isinstance(image, str):
        image = Image.open(image)

    # run the model
    inference_state  = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return output

def visualize_predictions(image_path: str, outputs: Dict):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for bbox in outputs['boxes']:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
    return image

def nms(boxes, scores, iou_threshold=0.5):
    # boxes: (N, 4) in xyxy
    # scores: (N,)
    idxs = scores.argsort(descending=True)
    keep = []

    while idxs.numel() > 0:
        current = idxs[0].item()
        keep.append(current)

        if idxs.numel() == 1:
            break

        rest = idxs[1:]

        x1 = torch.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = torch.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = torch.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = torch.minimum(boxes[current, 3], boxes[rest, 3])

        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter = inter_w * inter_h

        area_current = (boxes[current, 2] - boxes[current, 0]) * \
                       (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * \
                    (boxes[rest, 3] - boxes[rest, 1])

        union = area_current + area_rest - inter
        iou = inter / union

        idxs = rest[iou < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

def perform_non_max_suppression(outputs, iou_threshold: float = 0.5):
    keep_idx = nms(outputs['boxes'], outputs['scores'], iou_threshold=iou_threshold)
    outputs['scores'] = outputs['scores'][keep_idx]
    outputs['boxes'] = outputs['boxes'][keep_idx, :]
    outputs['masks'] = outputs['masks'][keep_idx, :, :]
    return outputs


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
sam3_predictions = run_sam3(processor, image_path, prompt=PROMPT)
sam3_predictions = perform_non_max_suppression(sam3_predictions, iou_threshold=0.5)

# draw the bounding boxes on image and showcase
image_drawn = visualize_predictions(image_path, sam3_predictions)
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
    sam3_predictions = run_sam3(processor, image_pil, prompt=PROMPT)
    sam3_predictions = perform_non_max_suppression(sam3_predictions, iou_threshold=0.5)

    # collect the predictions
    predictions = []
    for score, bbox in zip(sam3_predictions['scores'].tolist(), sam3_predictions['boxes'].to(int).tolist()):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2, 1024), min(y2, 1024)
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




