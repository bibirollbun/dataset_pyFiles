# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install ultralytics tqdm pillow


import os
import cv2
import csv
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# 1. ë””ë°”ì�´ìŠ¤ ì„¤ì •
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"âœ… Using device: {device}")

# 2. ConvNeXt ëª¨ë�¸ (ImageNet pretrained + 2-class head)
#model = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
#checkpoint = torch.load(args.finetune, map_location='cpu')
#state_dict = model.state_dict()
#model.head = nn.Sequential(
#    nn.AdaptiveAvgPool2d(1),  # [B, 768, H, W] â†’ [B, 768, 1, 1]
#    nn.Flatten(1),            # [B, 768]
#    nn.Linear(768, 2)         # [B, 2]
#)
model = timm.create_model("efficientnet_b0", num_classes=2, checkpoint_path='/kaggle/input/classifier3/pytorch/default/1/checkpoint-1.pth')
model.to(device)
model.eval()

# 3. YOLOv8 ì–¼êµ´ íƒ�ì§€ê¸° ë¡œë“œ
face_detector = YOLO("/kaggle/input/yolo-face11n/pytorch/default/1/yolov11n-face.pt", task='detect')
face_detector.to(device)

# 4. ì�´ë¯¸ì§€ ì „ì²˜ë¦¬
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# 5. ê²½ë¡œ ì„¤ì •
video_dir = "/kaggle/input/Deepfake_Detection_and_Generation_Challenge_Blue_Team/test_sample_videos"
output_csv = "submission.csv"



# 6. ë¹„ë””ì˜¤ ë‹¨ìœ„ ì˜ˆì¸¡ í•¨ìˆ˜
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    is_fake = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ì–¼êµ´ ê²€ì¶œ
        results = face_detector(frame, verbose=False)
        boxes = results[0].boxes.xyxy if results[0].boxes else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ì–¼êµ´ ì¤‘ì‹¬ ê¸°ì¤€ìœ¼ë¡œ 320x320 crop ì˜�ì—­ ì„¤ì •
            half = 160
            top, bottom = cy - half, cy + half
            left, right = cx - half, cx + half

            pad_top = max(0, -top)
            pad_bottom = max(0, bottom - h)
            pad_left = max(0, -left)
            pad_right = max(0, right - w)

            # crop with padding
            crop = frame[max(0, top):min(h, bottom), max(0, left):min(w, right)]
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # ì�´ë¯¸ì§€ ë³€í™˜ í›„ ëª¨ë�¸ ì¶”ë¡ 
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prob = F.softmax(output, dim=1)

                if prob[0, 0] >= 0.6:
#                    print("ì²« ë²ˆì§¸ ë…¸ë“œì�˜ í™•ë¥ ì�´ 0.6 ì�´ìƒ�ì�…ë‹ˆë‹¤.")
                    pred = 0
                else:
#                    print("ì²« ë²ˆì§¸ ë…¸ë“œì�˜ í™•ë¥ ì�´ 0.6 ë¯¸ë§Œì�…ë‹ˆë‹¤.")
                    pred = 1
                    
                #pred = torch.argmax(output, dim=1).item()

            if pred == 0:
                is_fake = True
                break  # í•˜ë‚˜ë�¼ë�„ fakeë©´ ì¢…ë£Œ

        if is_fake:
            break

    cap.release()
    return 1 if is_fake else 0


# 7. ì „ì²´ ë¹„ë””ì˜¤ ì²˜ë¦¬ ë°� ê²°ê³¼ ìˆ˜ì§‘
results = []
for video_name in tqdm(os.listdir(video_dir), desc="ğŸ”� Predicting"):
    if not video_name.endswith(".mp4"):
        continue
    video_path = os.path.join(video_dir, video_name)
    label = predict_video(video_path)
    results.append((video_name, label))


#for i in range(2151):
#    results.append(("dummy.mp3", 0))

# 8. CSV ì €ì�¥
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "label"])
    writer.writerows(results)

print(f"\nğŸ�¯ Inference ì™„ë£Œ! ê²°ê³¼ CSV: {output_csv}")

