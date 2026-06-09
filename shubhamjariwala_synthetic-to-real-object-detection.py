yaml_text = """
path: /kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2
train: train/images
val: val/images
test: testImages/images
lr0: 0.01
lrf: 0.01
nc: 1
names: ['soup']
"""

with open('hyp_full.yaml', 'w') as f:
    f.write(yaml_text)



# yaml_text = """
# path: /kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2
# train: train/images
# val: val/images
# test: testImages/images
# lr0: 0.002
# lrf: 0.01
# momentum: 0.937
# weight_decay: 0.0005
# warmup_epochs: 3.0
# warmup_momentum: 0.8
# warmup_bias_lr: 0.1
# box: 0.05
# cls: 0.5
# cls_pw: 1.0
# obj: 1.0
# obj_pw: 1.0
# iou_t: 0.35
# anchor_t: 4.0
# fl_gamma: 0.0
# hsv_h: 0.015
# hsv_s: 0.7
# hsv_v: 0.4
# flipud: 0.0
# fliplr: 0.5
# mosaic: 1.0
# mixup: 0.2
# copy_paste: 0.0
# shear: 0.01
# translate: 0.1
# scale: 0.5
# perspective: 0.0005
# nc: 1
# names: ['soup']
# """

# with open('hyp_raw.yaml', 'w') as f:
#     f.write(yaml_text)



# yaml_text = """
# path: /kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2
# train: train/images
# val: val/images
# test: testImages/images
# lr0: 0.0001
# lrf: 0.01
# momentum: 0.937
# weight_decay: 0.0005
# warmup_epochs: 3.0
# warmup_momentum: 0.8
# warmup_bias_lr: 0.1
# box: 0.05
# cls: 0.5
# cls_pw: 1.0
# obj: 1.0
# obj_pw: 1.0
# iou_t: 0.35
# anchor_t: 4.0
# fl_gamma: 0.0
# hsv_h: 0.005
# hsv_s: 0.1
# hsv_v: 0.1
# flipud: 0.0
# fliplr: 0.3
# mosaic: 0.0
# mixup: 0.0
# copy_paste: 0.0
# shear: 0.0
# translate: 0.05
# scale: 0.1
# perspective: 0.0
# nc: 1
# names: ['soup']
# """

# with open('hyp_basic.yaml', 'w') as f:
#     f.write(yaml_text)



# Install YOLOv8 via Ultralytics
!pip install ultralytics --upgrade -q


# Iteration 1
from ultralytics import YOLO

# Load YOLOv8 model (v8n is light, v8m or v8l are stronger but slower)
model = YOLO('yolo11m.pt')  # or 'yolov8m.pt'

# Train the model
model.train(
    data='hyp_full.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    name='soup_detector',
    project='runs/train',
    exist_ok=True
)



from ultralytics import YOLO
# # Iteration 2
# # Load YOLOv8 model (v8n is light, v8m or v8l are stronger but slower)
# model = YOLO('yolov8n.pt')  # or 'yolov8m.pt'

# # Train the model
# model.train(
#     data='soup_data.yaml',
#     epochs=100,               # ğŸ”¼ More epochs to learn better
#     imgsz=960,                # ğŸ”¼ Larger image size for better spatial features
#     batch=8,                  # ğŸ”½ Reduce if you increase image size (to avoid OOM)
#     lr0=0.002,                # ğŸ”§ Start with lower initial learning rate
#     warmup_epochs=3,         # ğŸ”§ Gradual learning warmup
#     weight_decay=0.01,       # âœ… Regularization
#     hsv_h=0.015,             # ğŸ”� Color augmentation (hue)
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=0.3,             # ğŸ”� Small rotations
#     translate=0.1,           # ğŸ”� Slight translations
#     scale=0.5,               # ğŸ”� Random zoom-in/out
#     shear=0.01,
#     perspective=0.0005,
#     flipud=0.0,              # ğŸ”„ Disable vertical flip (unrealistic for soup cans)
#     fliplr=0.5,              # ğŸ”„ Left-right flip
#     mosaic=1.0,              # ğŸ§© Keep full mosaic augmentation
#     mixup=0.2,               # ğŸ�›ï¸� Mixup aug helps with generalization
#     patience=20,             # â�³ Stop early if no improvement
#     name='soup_detector_opt',
#     project='runs/train',
#     exist_ok=True
# )



# #Iteration 3

# model = YOLO('yolov8n.pt')  # or 'yolov8m.pt'

# model.train(
#     data='soup_data.yaml',
#     epochs=100,         # Longer training for better learning
#     imgsz=640,          # Higher image resolution for more detail
#     batch=8,            # Reduce if GPU RAM is limited
#     name='soup_detector_v2',
#     project='runs/train',
#     exist_ok=True,
#     patience=20,        # Early stopping
#     lr0=0.001,          # Lower learning rate
#     warmup_epochs=3,     # Gradual ramp-up for stability
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     degrees=0.0,
#     translate=0.1,
#     scale=0.5,
#     shear=0.0,
#     perspective=0.0,
#     flipud=0.0,
#     fliplr=0.5,
#     mosaic=1.0,
#     mixup=0.0

# )



# # Load the trained model
# model = YOLO('/kaggle/input/model_synthetic2realobject/pytorch/default/2/best.pt')
# datasets = ['/kaggle/working/hyp_basic.yaml', '/kaggle/working/hyp_full.yaml', '/kaggle/working/hyp_raw.yaml']

# for dataset in datasets:
#     model.train(
#         data=dataset,
#         epochs=30,  # fine-tuning, not from scratch
#         imgsz=640,
#         batch=16,
#         name=f'hyp_{dataset.split(".")[0]}',
#         project='runs/train',
#         exist_ok=True,
#         device='cuda'
#     )
# # Run inference on test images
# # results = model.predict(source='/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images', save=False, conf=0.25)




# # Load the trained model
# model = YOLO('/kaggle/input/model_synthetic2realobject/pytorch/default/2/best.pt')

# model.train(
#     data='hyp_raw.yaml',
#     epochs=50,  # fine-tuning, not from scratch
#     imgsz=640,
#     batch=16,
#     name='soup_finetuned_v2',
#     project='runs/train',
#     patience=20,
#     exist_ok=True
# )
# # Run inference on test images
# # results = model.predict(source='/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images', save=False, conf=0.25)




# results = model.val()
# print(results.box.map)   # mAP@0.5
# print(results.box.map50) # mAP@0.5
# print(results.box.map75) # mAP@0.75




# paths = {
#     "Raw": "/kaggle/working/runs/train/hyp_/kaggle/working/hyp_raw/weights/best.pt",
#     "Basic": "/kaggle/working/runs/train/hyp_/kaggle/working/hyp_basic/weights/best.pt",
#     "Full": "/kaggle/working/runs/train/hyp_/kaggle/working/hyp_full/weights/best.pt"
# }

# results = {}

# for name, path in paths.items():
#     model = YOLO(path)
#     val_result = model.val(
#         data='/kaggle/working/soup_data.yaml',  # Keep dataset consistent
#         imgsz=640,
#         split='val',
#         conf=0.25
#     )
#     results[name] = {
#         "mAP50": val_result.box.map50,
#         "mAP50-95": val_result.box.map
#     }



# results = model.predict(
#     source='/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images',
#     conf=0.05,         # Lower confidence to catch more objects
#     iou=0.5,           # IOU for NMS
#     imgsz=960,         # Match training resolution
#     max_det=100,       # Allow up to 100 detections per image
#     augment=True,      # Enable test-time augmentation
#     half=True,         # Use FP16 precision (faster on GPU)
#     device='cuda'      # Force GPU if available
# )



from ultralytics import YOLO

model = YOLO('/kaggle/working/runs/train/soup_detector/weights/best.pt')  # Adjust path if needed


results = model.predict(
    source='/kaggle/input/synthetic-2-real-object-detection-challenge-2/Synthetic to Real Object Detection Challenge 2/testImages/images',
    conf=0.05,
    iou=0.5,
    imgsz=960,
    max_det=100,
    augment=True,
    half=True,
    device='cuda',
    save=True,         # save prediction images
    save_txt=True,     # save predictions in YOLO text format
    project='runs/predict',
    name='hyp_raw_final',
    exist_ok=True
)


# import pandas as pd
# import os
# import cv2
# import os
# from ultralytics.utils.plotting import Annotator


# save_dir = 'runs/predict/hyp_raw_final_filtered'
# os.makedirs(save_dir, exist_ok=True)

# for result in results:
#     img_path = result.path
#     img = cv2.imread(img_path)
#     annotator = Annotator(img)
    
#     for box in result.boxes:
#         conf = box.conf.item()
#         cls = int(box.cls.item())
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         width = x2 - x1
#         height = y2 - y1
#         aspect_ratio = width / height if height > 0 else 0
#         label = model.names[cls]

#         # âœ… FILTERING LOGIC
#         if label == 'soup' and aspect_ratio > 2:
#             continue  # discard: too wide to be a real soup can
#         if conf < 0.5 or width < 20 or height < 20:
#             continue  # discard: low confidence or tiny region

#         # Draw and save filtered prediction
#         annotator.box_label([x1, y1, x2, y2], f'{label} {conf:.2f}')

#     # Save filtered image
#     filename = os.path.basename(img_path)
#     save_path = os.path.join(save_dir, filename)
#     cv2.imwrite(save_path, annotator.result())



# import os
# from PIL import Image
# import matplotlib.pyplot as plt

# # Folder with predicted images
# image_dir = 'runs/predict/hyp_raw_final'
# image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# # Display first 10 predictions (change the range as needed)
# for img_file in image_files[:10]:
#     img_path = os.path.join(image_dir, img_file)
#     img = Image.open(img_path)

#     plt.figure(figsize=(8, 6))
#     plt.imshow(img)
#     plt.title(img_file)
#     plt.axis('off')
#     plt.show()



# import pandas as pd
# import os

# pred_rows = []
# for r in results:
#     image_id = os.path.basename(r.path).split('.')[0]
#     prediction_string = ''
#     for box in r.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])
#         x_center, y_center, width, height = box.xywh[0]
#         img_w, img_h = r.orig_shape[1], r.orig_shape[0]
        
#         # Normalize bbox values
#         x_c = x_center / img_w
#         y_c = y_center / img_h
#         w = width / img_w
#         h = height / img_h

#         prediction_string += f'{cls} {conf:.6f} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} '

#     pred_rows.append({
#         'image_id': image_id,
#         'prediction_string': prediction_string.strip()
#     })

# no_box = "0 0.0 0.5 0.5 0.01 0.01"
# df = pd.DataFrame(pred_rows)
# # Replace null or empty prediction_strings
# df["prediction_string"] = df["prediction_string"].fillna(no_box)
# df.loc[df["prediction_string"].str.strip() == "", "prediction_string"] = no_box

# # Drop any rows with null image_id
# df = df.dropna(subset=['image_id'])

# df.to_csv('submission.csv', index=False)



import pandas as pd
import os

pred_rows = []
for r in results:
    image_id = os.path.basename(r.path).split('.')[0]
    prediction_string = ''
    img_w, img_h = r.orig_shape[1], r.orig_shape[0]

    for box in r.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        x_center, y_center, width, height = box.xywh[0]

        # Filtering logic
        aspect_ratio = float(width) / float(height) if height > 0 else 0
        if label == 'soup' and aspect_ratio > 2:
            continue  # Discard: too wide to be a soup can
        if conf < 0.5 or width < 20 or height < 20:
            continue  # Discard: low confidence or tiny object

        # Normalize bbox values
        x_c = float(x_center) / img_w
        y_c = float(y_center) / img_h
        w = float(width) / img_w
        h = float(height) / img_h

        prediction_string += f'{cls} {conf:.6f} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} '

    pred_rows.append({
        'image_id': image_id,
        'prediction_string': prediction_string.strip()
    })

# Handle empty predictions
no_box = "0 0.0 0.5 0.5 0.01 0.01"
df = pd.DataFrame(pred_rows)
df["prediction_string"] = df["prediction_string"].fillna(no_box)
df.loc[df["prediction_string"].str.strip() == "", "prediction_string"] = no_box
df = df.dropna(subset=['image_id'])
df.to_csv('submission.csv', index=False)





