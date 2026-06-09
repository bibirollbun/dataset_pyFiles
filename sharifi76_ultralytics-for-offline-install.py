!pip download -d ./packages ultralytics
!tar cfvz archive.tar.gz ./packages


!tar xfvz archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages 


import os
from ultralytics import YOLO

# Create a directory to store the weights
os.makedirs("yolov8-weights", exist_ok=True)

# This will download the YOLOv8 nano weights
model = YOLO("yolov8s.pt")

# Save the downloaded weights to a local file
model.save("yolov8-weights/yolov8s.pt")

print(f"YOLOv8 weights saved to: {os.path.abspath('yolov8-weights/yolov8s.pt')}")


model = YOLO("yolov8n.pt")
model.save("yolov8-weights/yolov8n.pt")
model = YOLO("yolov8m.pt")
model.save("yolov8-weights/yolov8m.pt")

