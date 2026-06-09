!pip install ultralytics comet_ml


with open("yolo_params.yaml", "w") as f:
    f.write("""
# Dataset paths
train: /kaggle/input/falcon-object-detection/ObjectDetectionDataset/train/images  # Path to training images
val: /kaggle/input/falcon-object-detection/ObjectDetectionDataset/val/images      # Path to validation images
test: /kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test    # Path to test images

# Class information
nc: 1                     # Number of classes
names: ['cheerios']       # Class names
""")


import comet_ml
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
COMET_API_KEY = user_secrets.get_secret("COMET_API_KEY")

comet_ml.login(project_name="synthetic-to-real", api_key=COMET_API_KEY)


from ultralytics import YOLO
# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

IMG_SIZE = 640
# Train the model
results = model.train(data="yolo_params.yaml", 
                      epochs=100,
                      imgsz=IMG_SIZE, 
                      patience=20,
                      cos_lr=True,
                      dropout=0.4, 
                      mosaic=0.2, 
                      lr0=0.0001, 
                      optimizer="SGD", 
                      momentum=0.975,
                      weight_decay=0.0001,
                      single_cls=True, 
                      plots=True,
                      cache=True,
                      flipud=0.25,
                      scale=1.0
                     )


results = model.predict("/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images",
                       #iou=0.2,
                       imgsz=IMG_SIZE,
                       save=True,
                       #show_labels=False,
                       conf=0.001,
                       #max_det=1,
                       #visualize=True,
                       #max_det=2,
                       verbose=False
                       )


from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml


# Function to predict and save images
def predict_and_save(model, image_path, output_path_txt):
    """
    Predict bounding boxes for an image and save them in YOLO format.
    
    Args:
        model: YOLOv8 model.
        image_path: Path to the input image.
        output_path_txt: Path to save the predictions.
    """

    # Perform prediction
    results = model.predict(image_path,
                            imgsz=IMG_SIZE,
                            conf=0.001)
    result = results[0]
    img_height, img_width = result.orig_shape

    # Save bounding boxes in YOLO format
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()  
            # print(box.xywh[0][0].item(), box.xywh[0][1].item())
            x_center = box.xywh[0][0].item() / img_width
            y_center = box.xywh[0][1].item() / img_height
            # print(x_center, y_center)
            width = box.xywh[0][2].item() / img_width
            height = box.xywh[0][3].item() / img_height
            f.write(f"{cls_id} {conf} {x_center} {y_center} {width} {height}\n")

def main():
    # Set working directory
    __file__ = "/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/predict.py"
    this_dir = Path(__file__).parent
    working_dir = Path("/kaggle/working/")
    os.chdir(this_dir)

    # Load test path from YAML
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' not in data or not data['test']:
            print("Add 'test: path/to/test/images' to yolo_params.yaml")
            exit()
        images_dir = Path(data['test'])
    
    # Validate test directory
    if not images_dir.exists():
        print(f"Test directory {images_dir} does not exist")
        exit()
    if not any(images_dir.glob('*')):
        print(f"Test directory {images_dir} is empty")
        exit()

    # Load the latest trained YOLO model
    detect_path = working_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")
    idx = 0
    """if len(train_folders) > 1:
        choice = -1
        choices = list(range(len(train_folders)))
        while choice not in choices:
            print("Select the training folder:")
            for i, folder in enumerate(train_folders):
                print(f"{i}: {folder}")
            choice = input()
            if not choice.isdigit():
                choice = -1
            else:
                choice = int(choice)
        idx = choice"""

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    model = YOLO(model_path)

    # Directory with images to generate predictions
    output_dir = working_dir / "predictions" # Replace with the directory where you want to save predictions
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create labels subdirectories
    labels_output_dir = output_dir / 'labels'
    
    # images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg','.jpeg']:
            continue
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(model, img_path, output_path_txt)

    print(f"Bounding box labels saved in {labels_output_dir}")
    data = this_dir / 'yolo_params.yaml'
    print(f"Model parameters saved in {data}")

if __name__ == '__main__':
    main()



from pathlib import Path
import pandas as pd
import csv
import sys

def predictions_to_csv(
    preds_folder: str = "predictions/labels", 
    output_csv: str = "submission.csv", 
    test_images_folder: str = "data/test/images",
    allowed_extensions: tuple = (".jpg", ".png", ".jpeg")
):
    """
    Convert YOLO prediction files to Kaggle submission CSV format
    with strict validation.
    """
    # Validate inno boxputs
    preds_path = Path(preds_folder)
    if not preds_path.exists():
        print(f"ERROR: Prediction folder '{preds_folder}' does not exist")
        sys.exit(1)

    # Get test image IDs (without extensions)
    test_images_path = Path(test_images_folder)
    if not test_images_path.exists():
        print(f"ERROR: Test images folder '{test_images_folder}' not found")
        sys.exit(1)
        
    test_images = {
        p.stem: True 
        for p in test_images_path.glob("*") 
        if p.suffix.lower() in allowed_extensions
    }
    print(f"Found {len(test_images)} test images")

    # Collect predictions with validation
    predictions = []
    error_count = 0
    bboxes = 0
    
    for txt_file in preds_path.glob("*.txt"):
        image_id = txt_file.stem
        
        # Validate image_id
        if image_id not in test_images:
            print(f"Skipping non-test image prediction: {txt_file.name}")
            continue
            
        with open(txt_file, "r") as f:
            valid_lines = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                    
                parts = line.split()
                # Validate YOLO format: 6 values per line
                if len(parts) != 6:
                    print(f"Invalid prediction in {txt_file.name} line {line_num}: {line}")
                    error_count += 1
                    continue
                    
                try:
                    # Validate numerical values
                    [float(x) for x in parts]
                    valid_lines.append(line)
                    bboxes += 1
                except ValueError:
                    print(f"Non-numeric values in {txt_file.name} line {line_num}: {line}")
                    error_count += 1
                    continue

        pred_str = " ".join(valid_lines) if valid_lines else "no box"
        predictions.append({"image_id": image_id, "prediction_string": pred_str})

    # Create submission dataframe
    submission_df = pd.DataFrame({"image_id": list(test_images.keys())})
    
    if predictions:
        preds_df = pd.DataFrame(predictions)
        final_df = submission_df.merge(preds_df, on="image_id", how="left").fillna("no boxes")
    else:
        final_df = submission_df
        final_df["prediction_string"] = "no boxes"

    # Save with CSV quoting rules
    final_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    print(f"\n Success! Submission saved to {output_csv}")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Total bounding boxes: {bboxes}")
    print(f"   Validation errors: {error_count}")

if __name__ == "__main__":

    predictions_to_csv(
        preds_folder="/kaggle/working/predictions/labels",
        output_csv="/kaggle/working/submission.csv",
        test_images_folder="/kaggle/input/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/test/images"
    )



!zip -r /kaggle/working/train.zip /kaggle/working/runs/detect/train


!zip -r /kaggle/working/predictions.zip /kaggle/working/runs/detect/train2

