!pip download -d ./packages ultralytics
!tar cfvz archive.tar.gz ./packages


!tar xfvz archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages 


import os
from ultralytics import YOLO

os.makedirs("yolov12-weights", exist_ok=True)


model = YOLO("yolo12n.pt")
model.save("yolov12-weights/yolo12n.pt")
model = YOLO("yolo12s.pt")
model.save("yolov12-weights/yolo12s.pt")
model = YOLO("yolo12m.pt")
model.save("yolov12-weights/yolo12m.pt")
model = YOLO("yolo12l.pt")
model.save("yolov12-weights/yolo12l.pt")
model = YOLO("yolo12x.pt")
model.save("yolov12-weights/yolo12x.pt")

