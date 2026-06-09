!pip install ultralytics transformers sentencepiece
!pip install ultralytics
!pip install cvzone


from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import cvzone
import math
import time
import os

model = YOLO("/kaggle/input/yolov8l/pytorch/default/1/yolov8l.pt")

cap = cv2.VideoCapture("/kaggle/input/testing-video-1/video_20250205_180325.mp4")

output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


while True:
    success, img = cap.read()
    if not success:
        break
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    output_path = os.path.join(output_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}.jpg")
    cv2.imwrite(output_path, img)
cap.release()


output_video_path = "output_video_4.mp4"
img_array = []
for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Video created successfully.")

