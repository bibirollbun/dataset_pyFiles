!pip install -q roboflow
!pip install ultralytics
!pip uninstall opencv-python -y
!pip install opencv-python-headless  # First uninstall headless if present
!pip install opencv-python           # This version includes GUI support
from roboflow import Roboflow
import os
from IPython.display import clear_output
from ultralytics import YOLO
import cv2

clear_output()


rf = Roboflow(api_key="3kCCYyTsGpQ7FN4YaH5B")
project = rf.workspace("sperm-motility-analysis-from-microscopic-videos").project("sperm-head-detection")
version = project.version(10)
dataset = version.download("yolov8")


!pip uninstall opencv-python -y
!pip install opencv-python-headless  # First uninstall headless if present
!pip install opencv-python           # This version includes GUI support


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

class SpermDetectionDataset:
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Load labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO format to pixel coordinates
                    x_center *= original_w
                    y_center *= original_h
                    width *= original_w
                    height *= original_h
                    
                    x_min = x_center - width/2
                    y_min = y_center - height/2
                    x_max = x_center + width/2
                    y_max = y_center + height/2
                    
                    # Scale coordinates to new image size
                    x_min = x_min * (self.img_size / original_w)
                    y_min = y_min * (self.img_size / original_h)
                    x_max = x_max * (self.img_size / original_w)
                    y_max = y_max * (self.img_size / original_h)
                    
                    boxes.append([class_id, x_min, y_min, x_max, y_max])
        
        return img, np.array(boxes)
    
    def visualize(self, idx):
        img, boxes = self.__getitem__(idx)
        
        # Draw bounding boxes
        for box in boxes:
            class_id, x_min, y_min, x_max, y_max = box
            cv2.rectangle(img, 
                          (int(x_min), int(y_min)), 
                          (int(x_max), int(y_max)), 
                          (255, 0, 0), 2)
            
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = SpermDetectionDataset(image_dir="/kaggle/working/Sperm-Head-Detection-10/train/images", 
                                   label_dir='/kaggle/working/Sperm-Head-Detection-10/train/labels')
    
    # Visualize a sample
    dataset.visualize(0)
    
    # You can then use this dataset with a deep learning framework
    # like PyTorch or TensorFlow for training a sperm detection model


# Install required packages
# pip install ultralytics

from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolov8s.pt')  # or yolov8s.pt

# Modify your training command to use CPU
results = model.train(
    data='/kaggle/working/Sperm-Head-Detection-10/data.yaml',
    epochs=5,
    imgsz=640,
    batch=16,
    device='cpu'  # Changed from '0' to 'cpu'
)

# Evaluate the model
metrics = model.val()


import cv2
from matplotlib import pyplot as plt

def test_single_image(image_path, model_path='/kaggle/working/runs/detect/train2/weights/best.pt'):
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    
    # Show results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.show()
        
        # Print detection info
        print("Detected sperms:", len(r.boxes))
        for box in r.boxes:
            print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xywh}")

# Example usage
test_single_image('/kaggle/working/Sperm-Head-Detection-10/test/images/EVA_T0_SM-187001_tile_2_jpg.rf.985375f77c6518c3363ce3d24c77a608.jpg')


import cv2
from ultralytics import YOLO

def process_video_no_display(
    video_path, 
    model_path='/kaggle/working/runs/detect/train2/weights/best.pt',
    output_path='/kaggle/working/output.mp4',
    conf_threshold=0.25
):
    # Load your trained model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame, conf=conf_threshold)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Write frame to output file
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Results saved to {output_path}")
    
    # Return the output path for download
    return output_path

# Example usage
output_video = process_video_no_display(
    video_path='/kaggle/input/evisan-multi-sperm-detection-and-tracking/video/test_200x.wmv',
    conf_threshold=0.3
)

# In Kaggle, you can then download the output file
from IPython.display import FileLink
FileLink(output_video)




