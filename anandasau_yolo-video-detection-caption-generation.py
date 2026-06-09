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

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Video capture
cap = cv2.VideoCapture("/kaggle/input/d/anandasau/object-detection-video-test/Huskey.mp4")

# Output folder for saving frames
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
        break  # Break out of the loop if there are no more frames to read

    # Perform object detection with YOLO
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Save frame with YOLO detections applied
    output_path = os.path.join(output_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}.jpg")
    cv2.imwrite(output_path, img)

# Release video capture
cap.release()


import os
import cv2

output_folder = "output_frames"  # Change this to your folder path
output_video_path = "output_video.mp4"

img_array = []
size = None  # Initialize as None instead of 0

for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable images

        height, width, layers = img.shape
        if size is None:  # Assign size once from the first valid image
            size = (width, height)

        img_array.append(img)

if size is not None and img_array:  # Ensure we have valid images
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for img in img_array:
        out.write(img)
    out.release()
    print("Video created successfully.")
else:
    print("No valid images found.")



!pip install --upgrade ipywidgets jupyterlab-widgets



from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import shutil

# Load the BLIP (Bootstrapping Language-Image Pre-training) model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Path to the Open Images dataset
image_folder = "/kaggle/input/open-images-2019-object-detection/test"

# Output folder to save images with predictions and bounding boxes
output_folder = "predicted_images_with_captions"
os.makedirs(output_folder, exist_ok=True)

# Function to perform object detection, draw bounding boxes, and generate image captions
def detect_draw_and_caption(image_path, model, output_folder, processor, blip_model):
    # Read the image
    img = cv2.imread(image_path)
    
    # Perform object detection with YOLO
    results = model(img)
    
    # Loop through each result (there may be multiple objects detected)
    for result in results:
        boxes = result.boxes  # Get the bounding boxes
        
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # These are the bounding box corner points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class id
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get class name
            class_name = model.names[cls]
            
            # Display class and confidence
            label = f'{class_name} {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Generate caption using the BLIP model
    pil_img = Image.open(image_path).convert("RGB")
    inputs = processor(pil_img, return_tensors="pt")
    caption = blip_model.generate(**inputs)
    description = processor.decode(caption[0], skip_special_tokens=True)
    
    # Save the image with bounding boxes
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    # Display the image and its caption
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Caption: {description}")
    plt.show()

# Example usage for a few images in the folder
for i, image_name in enumerate(os.listdir(image_folder)):
    if image_name.endswith(".jpg") and i < 5:  # Change the number '5' to process more images
        image_path = os.path.join(image_folder, image_name)
        detect_draw_and_caption(image_path, model, output_folder, processor, blip_model)

# Compress the output folder into a ZIP file for easy downloading
shutil.make_archive("predicted_images_with_captions", 'zip', output_folder)

print("Object detection and captioning completed, images saved and compressed for download.")



image_path = '/kaggle/input/video-detect/PetalsWebDesignerCushionChair_Beige_packof2.jpg'
detect_draw_and_caption(image_path, model, output_folder, processor, blip_model)



image_path = '/kaggle/input/video-detect/107032274-1647540069295-gettyimages-1084167640-2018_10_13-n1_office_0312.jpeg'
detect_draw_and_caption(image_path, model, output_folder, processor, blip_model)


image_path = '/kaggle/input/video-detect/img.jpeg'
detect_draw_and_caption(image_path, model, output_folder, processor, blip_model)


image_path = '/kaggle/input/video-detect/images (2).jpeg'
detect_draw_and_caption(image_path, model, output_folder, processor, blip_model)




