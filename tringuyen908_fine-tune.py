import pandas as pd
import shutil
import os
from PIL import Image
from pycocotools.coco import COCO


!pip install ultralytics


import pandas as pd
import shutil
import os
from PIL import Image
from pycocotools.coco import COCO
import random


output_dir = '/kaggle/input/tp-finetunedatanew/finetune_ds'



from ultralytics import YOLO

# Load pre-trained model (cÃ³ thá»ƒ dÃ¹ng yolo11n.pt, yolo11s.pt, yolo11m.pt)






model = YOLO("yolo11l.pt")  # DÃ¹ng model lá»›n hÆ¡n Ä‘á»ƒ cÃ³ Ä‘á»™ chÃ­nh xÃ¡c cao hÆ¡n
# Fine-tune vá»›i cÃ¡c tham sá»‘ tá»‘i Æ°u
results = model.train(
    data=os.path.join(output_dir, 'data.yaml'),
    epochs=50,  # TÄƒng epochs
    imgsz=640,
    batch=16,
    patience=10,
    save=True,
    device=0,  # GPU
    workers=4,
    lr0=0.001,  # Learning rate tháº¥p hÆ¡n cho fine-tuning
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    box=0.05,  # Box loss gain
    cls=0.5,   # Class loss gain
    dfl=1.5,   # DFL loss gain
    hsv_h=0.015,  # Image HSV-Hue augmentation
    hsv_s=0.7,    # Image HSV-Saturation augmentation
    hsv_v=0.4,    # Image HSV-Value augmentation
    degrees=0.0,  # Image rotation
    translate=0.1,  # Image translation
    scale=0.5,    # Image scale
    shear=0.0,    # Image shear
    perspective=0.0,  # Image perspective
    flipud=0.0,   # Image flip up-down
    fliplr=0.5,   # Image flip left-right
    mosaic=1.0,   # Mosaic augmentation
    mixup=0.5,    # Mixup augmentation
    copy_paste=0.1,  # Copy-paste augmentation
)


# Validate model
metrics = model.val()
print(f"\nmAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

print("\n=== LÆ°u mÃ´ hÃ¬nh vÃ  káº¿t quáº£ ===")

# Táº¡o thÆ° má»¥c lÆ°u káº¿t quáº£
results_dir = '/kaggle/working/model_results'
os.makedirs(results_dir, exist_ok=True)

# 1. LÆ°u model weights (file .pt)
best_model_path = '/kaggle/working/best_model.pt'
model.save(best_model_path)
print(f"Ä�Ã£ lÆ°u best model táº¡i: {best_model_path}")

# 2. LÆ°u model á»Ÿ Ä‘á»‹nh dáº¡ng khÃ¡c nhau
try:
    # Export ONNX
    onnx_path = '/kaggle/working/model.onnx'
    model.export(format='onnx', dynamic=True)
    print(f"Ä�Ã£ export ONNX model")
    
    # Export TorchScript
    torchscript_path = '/kaggle/working/model.torchscript'
    model.export(format='torchscript')
    print(f"Ä�Ã£ export TorchScript model")
    
except Exception as e:
    print(f"Lá»—i khi export model: {e}")

# 3. LÆ°u káº¿t quáº£ training
training_results = {
    'mAP50': float(metrics.box.map50) if metrics.box.map50 is not None else 0,
    'mAP50-95': float(metrics.box.map) if metrics.box.map is not None else 0,
    'epochs_trained': 50,
    'classes': ['person', 'phone', 'reflex_camera', 'polaroid_camera'],
    'class_counts': class_counts
}

# LÆ°u káº¿t quáº£ dÆ°á»›i dáº¡ng JSON
import json
with open('/kaggle/working/training_results.json', 'w') as f:
    json.dump(training_results, f, indent=2)
print("Ä�Ã£ lÆ°u káº¿t quáº£ training táº¡i: /kaggle/working/training_results.json")

# 4. LÆ°u confusion matrix vÃ  cÃ¡c metrics chi tiáº¿t
try:
    import matplotlib.pyplot as plt
    
    # LÆ°u confusion matrix
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(metrics.confusion_matrix.matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.savefig('/kaggle/working/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Ä�Ã£ lÆ°u confusion matrix")
    
except Exception as e:
    print(f"Lá»—i khi lÆ°u confusion matrix: {e}")

# 5. Copy training logs vÃ  charts tá»« runs/detect/train
import glob
try:
    # TÃ¬m thÆ° má»¥c runs má»›i nháº¥t
    train_dirs = glob.glob('/kaggle/working/runs/detect/train*')
    if train_dirs:
        latest_train_dir = max(train_dirs, key=os.path.getctime)
        
        # Copy cÃ¡c file quan trá»�ng
        important_files = ['results.png', 'confusion_matrix.png', 'results.csv', 'weights/best.pt', 'weights/last.pt']
        
        for file_pattern in important_files:
            source_files = glob.glob(os.path.join(latest_train_dir, file_pattern))
            for source_file in source_files:
                if os.path.exists(source_file):
                    filename = os.path.basename(source_file)
                    if filename.endswith('.pt'):
                        # Ä�á»•i tÃªn weights Ä‘á»ƒ phÃ¢n biá»‡t
                        if 'best' in filename:
                            filename = 'yolo_best_weights.pt'
                        elif 'last' in filename:
                            filename = 'yolo_last_weights.pt'
                    
                    dest_file = os.path.join('/kaggle/working', filename)
                    shutil.copy(source_file, dest_file)
                    print(f"Ä�Ã£ copy {filename}")
        
        print(f"Training results available in: {latest_train_dir}")
    
except Exception as e:
    print(f"Lá»—i khi copy training files: {e}")

# 6. Táº¡o file README vá»›i thÃ´ng tin mÃ´ hÃ¬nh
readme_content = f"""# YOLO Fine-tuned Model

## Model Information
- Base Model: YOLOv11s
- Classes: person, phone, reflex_camera, polaroid_camera
- Training Epochs: 50
- Image Size: 640x640

## Performance Metrics
- mAP50: {training_results['mAP50']:.4f}
- mAP50-95: {training_results['mAP50-95']:.4f}

## Class Distribution
"""

for i, (class_id, count) in enumerate(class_counts.items()):
    class_names = ['person', 'phone', 'reflex_camera', 'polaroid_camera']
    readme_content += f"- {class_names[i]}: {count} instances\n"

readme_content += f"""
## Files Included
- best_model.pt: Best model weights
- model.onnx: ONNX format model
- model.torchscript: TorchScript format model
- training_results.json: Detailed training metrics
- results.png: Training charts
- confusion_matrix.png: Confusion matrix visualization

## Usage
```python
from ultralytics import YOLO
model = YOLO('best_model.pt')
results = model('image.jpg')
```
"""

with open('/kaggle/working/README.md', 'w') as f:
    f.write(readme_content)
print("Ä�Ã£ táº¡o README.md")

# 7. Hiá»ƒn thá»‹ danh sÃ¡ch táº¥t cáº£ files Ä‘Ã£ táº¡o
print("\n=== Files Ä‘Ã£ táº¡o trong /kaggle/working/ ===")
working_files = os.listdir('/kaggle/working/')
for file in sorted(working_files):
    file_path = os.path.join('/kaggle/working/', file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
        print(f"{file} ({size:.2f} MB)")

print(f"\nTá»•ng cá»™ng {len(working_files)} files/folders trong /kaggle/working/")
print("CÃ¡c file nÃ y sáº½ cÃ³ thá»ƒ download Ä‘Æ°á»£c thÃ´ng qua Kaggle Kernels Output!")

# Test model vá»›i 1 áº£nh máº«u (náº¿u cÃ³)
try:
    if len(all_images) > 0:
        test_image_path = os.path.join(output_dir, 'images', all_images[0])
        if os.path.exists(test_image_path):
            print(f"\n=== Test model vá»›i áº£nh máº«u ===")
            results = model(test_image_path)
            print(f"Detected {len(results[0].boxes)} objects in test image")
            
            # LÆ°u káº¿t quáº£ detection
            results[0].save('/kaggle/working/test_detection.jpg')
            print("Ä�Ã£ lÆ°u káº¿t quáº£ detection táº¡i: /kaggle/working/test_detection.jpg")
            
except Exception as e:
    print(f"Lá»—i khi test model: {e}")

print("\nğŸ�‰ HoÃ n thÃ nh! Model vÃ  táº¥t cáº£ files Ä‘Ã£ Ä‘Æ°á»£c lÆ°u vÃ o /kaggle/working/")
print("Báº¡n cÃ³ thá»ƒ download chÃºng báº±ng lá»‡nh: kaggle kernels output username/kernel-name -p ./")


import os
import shutil
import zipfile
import json

print("ğŸš€ GIáº¢I PHÃ�P DOWNLOAD MODEL Tá»ª KAGGLE\n")

# === GIáº¢I PHÃ�P 1: Copy vÃ  nÃ©n file nhá»� hÆ¡n ===
def solution_1_compress():
    print("ğŸ“¦ GIáº¢I PHÃ�P 1: NÃ©n file model")
    print("-" * 50)
    
    # TÃ¬m táº¥t cáº£ file .pt trong runs
    model_files = []
    for root, dirs, files in os.walk('/kaggle/working'):
        for file in files:
            if file.endswith('.pt'):
                full_path = os.path.join(root, file)
                size_mb = os.path.getsize(full_path) / (1024**2)
                model_files.append((full_path, size_mb))
                print(f"TÃ¬m tháº¥y: {file} ({size_mb:.1f} MB)")
    
    if not model_files:
        print("â�Œ KhÃ´ng tÃ¬m tháº¥y file .pt nÃ o!")
        return
    
    # Copy file best.pt ra working directory
    best_pt = None
    for path, size in model_files:
        if 'best.pt' in path:
            best_pt = path
            break
    
    if best_pt:
        # Copy vÃ  nÃ©n
        output_path = '/kaggle/working/best_model.pt'
        shutil.copy(best_pt, output_path)
        
        # Táº¡o zip file
        zip_path = '/kaggle/working/yolo_model.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_path, 'best_model.pt')
        
        zip_size = os.path.getsize(zip_path) / (1024**2)
        print(f"\nâœ… Ä�Ã£ táº¡o: yolo_model.zip ({zip_size:.1f} MB)")
        print(f"ğŸ“¥ Download file nÃ y tá»« Output tab")

solution_1_compress()


# CÃ i Ä‘áº·t thÆ° viá»‡n Ultralytics
!pip install -q ultralytics

import os
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# --- Load YOLOv8 model ---
# DÃ¹ng mÃ´ hÃ¬nh YOLOv8 Ä‘Ã£ Ä‘Æ°á»£c huáº¥n luyá»‡n trÆ°á»›c vá»›i dá»¯ liá»‡u COCO
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')  # Sá»­ dá»¥ng model YOLOv8 medium pre-trained

# --- Load áº£nh ---
# Chá»�n má»™t áº£nh ngáº«u nhiÃªn tá»« bá»™ validation cá»§a COCO
img_path = '/kaggle/input/testimage/WIN_20250607_19_34_01_Pro.jpg'  # Cáº­p nháº­t Ä‘Ãºng Ä‘Æ°á»�ng dáº«n áº£nh náº¿u cáº§n
img = Image.open(img_path).convert('RGB')

# --- Cháº¡y inference vá»›i YOLOv8 ---
results = model(img_path)[0]  # Sá»­ dá»¥ng mÃ´ hÃ¬nh Ä‘Ã£ táº£i Ä‘á»ƒ cháº¡y inference
boxes = results.boxes.xyxy.cpu().numpy()  # Láº¥y toáº¡ Ä‘á»™ bounding boxes (x1, y1, x2, y2)
classes = results.boxes.cls.cpu().numpy().astype(int)  # Láº¥y lá»›p cá»§a cÃ¡c object
confidences = results.boxes.conf.cpu().numpy()  # Láº¥y Ä‘á»™ tá»± tin cá»§a cÃ¡c prediction

# --- Váº½ káº¿t quáº£ vÃ  Ä‘Ã¡nh giÃ¡ hÃ nh Ä‘á»™ng ---
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

# Táº­p há»£p cÃ¡c box theo class
by_cls = {}
for box, cls, conf in zip(boxes, classes, confidences):
    by_cls.setdefault(cls, []).append((box, conf))

# Kiá»ƒm tra hÃ nh Ä‘á»™ng "chá»¥p áº£nh mÃ n hÃ¬nh"
persons = by_cls.get(0, [])  # Lá»›p person (ID=0)
phones = by_cls.get(1, [])  # Lá»›p cell phone (ID=67)

detected = False
for p_box, _ in persons:
    for ph_box, _ in phones:
        # Kiá»ƒm tra xem Ä‘iá»‡n thoáº¡i cÃ³ gáº§n mÃ n hÃ¬nh (cÃ³ thá»ƒ báº¡n cáº§n thÃªm "screen" náº¿u muá»‘n)
        cx, cy = (ph_box[0] + ph_box[2]) / 2, (ph_box[1] + ph_box[3]) / 2  # Tá»�a Ä‘á»™ trung tÃ¢m cá»§a Ä‘iá»‡n thoáº¡i
        px, py = (p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2  # Tá»�a Ä‘á»™ trung tÃ¢m cá»§a ngÆ°á»�i

        # Kiá»ƒm tra xem Ä‘iá»‡n thoáº¡i cÃ³ náº±m trong vÃ¹ng táº§m tay cá»§a ngÆ°á»�i hay khÃ´ng
        if p_box[0] < cx < p_box[2] and p_box[1] < cy < p_box[3]:
            # Váº½ cÃ¡c bounding boxes
            draw.rectangle(p_box.tolist(), outline='blue', width=2)
            draw.rectangle(ph_box.tolist(), outline='green', width=2)
            draw.text((ph_box[0], ph_box[1]-10), "likely taking photo", fill='green', font=font)
            detected = True
            break
    if detected: break

if detected:
    print("Detected taking-photo action based on rule-based heuristic.")
else:
    print("No taking-photo action detected.")

# --- Hiá»ƒn thá»‹ káº¿t quáº£ ---
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.show()


"""
Advanced Screen Capture Detection Algorithm
A Computer Vision Approach for Detecting Screen Photography Behavior

This algorithm implements a multi-stage detection pipeline combining:
1. Object detection (YOLO)
2. Spatial relationship analysis
3. Pose estimation features
4. Temporal consistency (for video)
"""

import numpy as np
import cv2
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import norm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque
import json

@dataclass
class Detection:
    """Data structure for object detection results"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]
    area: float
    aspect_ratio: float
    
    @classmethod
    def from_yolo_box(cls, box, class_names):
        """Create Detection from YOLO box object"""
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls)
        
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        return cls(
            bbox=[x1, y1, x2, y2],
            confidence=float(box.conf),
            class_id=class_id,
            class_name=class_names[class_id],
            center=center,
            area=area,
            aspect_ratio=aspect_ratio
        )

class ScreenCaptureDetector:
    """
    Advanced detector for screen capture behavior analysis
    
    Key innovations:
    1. Multi-modal feature extraction
    2. Probabilistic confidence scoring
    3. Temporal consistency for video streams
    4. Explainable AI outputs
    """
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """Initialize detector with YOLO model and configuration"""
        from ultralytics import YOLO
        
        self.model = YOLO(model_path)
        self.class_names = ['person', 'phone', 'reflex_camera', 'polaroid_camera']
        
        # Default configuration (can be tuned)
        self.config = config or {
            'spatial': {
                'max_distance_ratio': 0.8,  # Max distance as ratio of person height
                'optimal_distance_ratio': 0.4,  # Optimal holding distance
                'min_overlap_ratio': 0.1,  # Minimum spatial overlap
                'camera_height_range': (0.3, 0.8),  # Camera position relative to person
            },
            'geometric': {
                'phone_tilt_range': (-30, 45),  # Degrees, negative = toward camera
                'camera_angle_range': (15, 75),  # Viewing angle to phone
                'min_phone_visibility': 0.3,  # Minimum visible phone area
            },
            'temporal': {
                'window_size': 10,  # Frames for temporal analysis
                'min_consistency': 0.7,  # Minimum temporal consistency
            },
            'weights': {
                'spatial': 0.3,
                'geometric': 0.25,
                'pose': 0.25,
                'context': 0.2
            }
        }
        
        # Temporal buffer for video analysis
        self.temporal_buffer = deque(maxlen=self.config['temporal']['window_size'])
        
    def detect_screen_capture(self, image_path: str, 
                            visualize: bool = True,
                            return_details: bool = True) -> Dict:
        """
        Main detection pipeline
        
        Returns:
            Dict containing:
            - is_capturing: bool
            - confidence: float (0-1)
            - evidence: detailed analysis results
            - visualization: annotated image (if requested)
        """
        # Run YOLO detection
        results = self.model(image_path, conf=0.2, verbose=False)
        
        if len(results[0].boxes) == 0:
            return {
                'is_capturing': False,
                'confidence': 0.0,
                'evidence': {'reason': 'No objects detected'},
                'visualization': None
            }
        
        # Parse detections
        detections = [Detection.from_yolo_box(box, self.class_names) 
                     for box in results[0].boxes]
        
        # Group by class
        detections_by_class = self._group_detections_by_class(detections)
        
        # Analyze screen capture behavior
        analysis = self._analyze_capture_behavior(detections_by_class, image_path)
        
        # Visualize if requested
        visualization = None
        if visualize:
            visualization = self._visualize_analysis(
                image_path, detections, analysis
            )
        
        # Compile results
        result = {
            'is_capturing': analysis['final_score'] > 0.5,
            'confidence': analysis['final_score'],
            'evidence': analysis if return_details else None,
            'visualization': visualization
        }
        
        # Update temporal buffer for video analysis
        self.temporal_buffer.append(result)
        
        return result
    
    def _group_detections_by_class(self, detections: List[Detection]) -> Dict:
        """Group detections by class for easier analysis"""
        groups = {name: [] for name in self.class_names}
        for det in detections:
            groups[det.class_name].append(det)
        return groups
    
    def _analyze_capture_behavior(self, detections_by_class: Dict, 
                                 image_path: str) -> Dict:
        """
        Core analysis algorithm combining multiple evidence sources
        """
        analysis = {
            'spatial_score': 0.0,
            'geometric_score': 0.0,
            'pose_score': 0.0,
            'context_score': 0.0,
            'final_score': 0.0,
            'details': {}
        }
        
        # Check prerequisites
        persons = detections_by_class['person']
        phones = detections_by_class['phone']
        cameras = (detections_by_class['reflex_camera'] + 
                  detections_by_class['polaroid_camera'])
        
        if not persons or not phones:
            analysis['details']['missing'] = 'Required objects not detected'
            return analysis
        
        # Find best person-phone-camera combination
        best_combo = None
        best_score = 0.0
        
        for person in persons:
            for phone in phones:
                # Analyze with camera
                if cameras:
                    for camera in cameras:
                        score, details = self._analyze_combination(
                            person, phone, camera, with_camera=True
                        )
                        if score > best_score:
                            best_score = score
                            best_combo = (person, phone, camera)
                            analysis['details'] = details
                else:
                    # Analyze without camera (phone self-camera)
                    score, details = self._analyze_combination(
                        person, phone, None, with_camera=False
                    )
                    if score > best_score:
                        best_score = score
                        best_combo = (person, phone, None)
                        analysis['details'] = details
        
        # Calculate component scores
        if best_combo:
            analysis.update(self._calculate_component_scores(best_combo, analysis['details']))
        
        # Weighted final score
        weights = self.config['weights']
        analysis['final_score'] = (
            weights['spatial'] * analysis['spatial_score'] +
            weights['geometric'] * analysis['geometric_score'] +
            weights['pose'] * analysis['pose_score'] +
            weights['context'] * analysis['context_score']
        )
        
        # Apply temporal consistency if available
        if len(self.temporal_buffer) > 0:
            analysis['final_score'] = self._apply_temporal_smoothing(
                analysis['final_score']
            )
        
        return analysis
    
    def _analyze_combination(self, person: Detection, phone: Detection, 
                           camera: Optional[Detection], with_camera: bool) -> Tuple[float, Dict]:
        """Analyze specific person-phone-camera combination"""
        details = {
            'with_camera': with_camera,
            'spatial_relations': {},
            'geometric_features': {},
            'pose_indicators': {}
        }
        
        # 1. Spatial relationship analysis
        spatial_score = self._analyze_spatial_relations(
            person, phone, camera, details['spatial_relations']
        )
        
        # 2. Geometric configuration analysis
        geometric_score = self._analyze_geometric_config(
            person, phone, camera, details['geometric_features']
        )
        
        # 3. Pose-based indicators
        pose_score = self._analyze_pose_indicators(
            person, phone, camera, details['pose_indicators']
        )
        
        # Combined score
        total_score = (spatial_score + geometric_score + pose_score) / 3
        
        return total_score, details
    
    def _analyze_spatial_relations(self, person: Detection, phone: Detection,
                                 camera: Optional[Detection], details: Dict) -> float:
        """Analyze spatial relationships between objects"""
        score = 0.0
        
        # Person-phone distance
        person_height = person.bbox[3] - person.bbox[1]
        phone_distance = euclidean(person.center, phone.center)
        normalized_distance = phone_distance / person_height
        
        details['phone_distance_normalized'] = normalized_distance
        
        # Optimal distance scoring (Gaussian distribution)
        optimal = self.config['spatial']['optimal_distance_ratio']
        distance_score = norm.pdf(normalized_distance, optimal, 0.2) / norm.pdf(optimal, optimal, 0.2)
        score += distance_score * 0.4
        
        # Phone position relative to person
        phone_in_front = (
            person.bbox[0] < phone.center[0] < person.bbox[2] and
            phone.center[1] < person.center[1]  # Phone above center
        )
        details['phone_in_front'] = phone_in_front
        if phone_in_front:
            score += 0.3
        
        # Camera analysis if present
        if camera:
            # Camera-phone alignment
            camera_to_phone = np.array(phone.center) - np.array(camera.center)
            camera_to_person = np.array(person.center) - np.array(camera.center)
            
            # Check if camera points toward phone
            cos_angle = np.dot(camera_to_phone, camera_to_person) / (
                np.linalg.norm(camera_to_phone) * np.linalg.norm(camera_to_person)
            )
            alignment_score = max(0, cos_angle)
            details['camera_alignment'] = alignment_score
            score += alignment_score * 0.3
        
        return min(1.0, score)
    
    def _analyze_geometric_config(self, person: Detection, phone: Detection,
                                 camera: Optional[Detection], details: Dict) -> float:
        """Analyze geometric configuration and angles"""
        score = 0.0
        
        # Phone orientation (aspect ratio indicates orientation)
        phone_vertical = phone.aspect_ratio < 0.7
        details['phone_vertical'] = phone_vertical
        if not phone_vertical:  # Horizontal phone more likely for viewing
            score += 0.3
        
        # Estimate viewing angle
        if camera:
            # Vector from camera to phone
            view_vector = np.array(phone.center) - np.array(camera.center)
            view_angle = np.degrees(np.arctan2(view_vector[1], view_vector[0]))
            
            # Check if angle is suitable for screen capture
            angle_range = self.config['geometric']['camera_angle_range']
            if angle_range[0] <= abs(view_angle) <= angle_range[1]:
                score += 0.4
            details['viewing_angle'] = view_angle
        
        # Phone size relative to person (indicates distance)
        person_area = (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1])
        phone_relative_size = phone.area / person_area
        details['phone_relative_size'] = phone_relative_size
        
        # Optimal relative size (not too close, not too far)
        if 0.01 < phone_relative_size < 0.1:
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_pose_indicators(self, person: Detection, phone: Detection,
                                camera: Optional[Detection], details: Dict) -> float:
        """Analyze pose-based behavioral indicators"""
        score = 0.0
        
        # Hand position estimation (simplified without keypoints)
        # Assume hands are near phone if phone is in person's bounding box
        phone_center_y = phone.center[1]
        person_upper_body = person.bbox[1] + (person.bbox[3] - person.bbox[1]) * 0.4
        
        hands_raised = phone_center_y < person_upper_body
        details['hands_raised'] = hands_raised
        if hands_raised:
            score += 0.4
        
        # Body orientation (simplified)
        if camera:
            # Check if person faces camera while phone faces away
            person_to_camera = np.array(camera.center) - np.array(person.center)
            person_to_phone = np.array(phone.center) - np.array(person.center)
            
            # Angle between vectors
            cos_angle = np.dot(person_to_camera, person_to_phone) / (
                np.linalg.norm(person_to_camera) * np.linalg.norm(person_to_phone) + 1e-6
            )
            
            # Negative correlation expected (opposite directions)
            if cos_angle < 0:
                score += 0.3
            details['body_phone_camera_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Stability indicator (low confidence might indicate motion blur)
        avg_confidence = (person.confidence + phone.confidence) / 2
        if camera:
            avg_confidence = (avg_confidence + camera.confidence) / 1.5
        
        details['detection_stability'] = avg_confidence
        if avg_confidence > 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_component_scores(self, combo: Tuple, details: Dict) -> Dict:
        """Calculate individual component scores from analysis details"""
        scores = {
            'spatial_score': 0.0,
            'geometric_score': 0.0,
            'pose_score': 0.0,
            'context_score': 0.0
        }
        
        # Spatial score
        if details.get('spatial_relations', {}).get('phone_in_front'):
            scores['spatial_score'] += 0.5
        if details.get('spatial_relations', {}).get('camera_alignment', 0) > 0.7:
            scores['spatial_score'] += 0.5
        
        # Geometric score
        if not details.get('geometric_features', {}).get('phone_vertical', True):
            scores['geometric_score'] += 0.4
        if 0.01 < details.get('geometric_features', {}).get('phone_relative_size', 0) < 0.1:
            scores['geometric_score'] += 0.6
        
        # Pose score
        if details.get('pose_indicators', {}).get('hands_raised'):
            scores['pose_score'] += 0.5
        if details.get('pose_indicators', {}).get('detection_stability', 0) > 0.7:
            scores['pose_score'] += 0.5
        
        # Context score (additional evidence)
        if combo[2] is not None:  # Has camera
            scores['context_score'] += 0.6
        if len(self.temporal_buffer) > 5:  # Consistent behavior over time
            scores['context_score'] += 0.4
        
        return scores
    
    def _apply_temporal_smoothing(self, current_score: float) -> float:
        """Apply temporal smoothing for video sequences"""
        if len(self.temporal_buffer) < 2:
            return current_score
        
        # Get recent scores
        recent_scores = [b['confidence'] for b in list(self.temporal_buffer)[-5:]]
        recent_scores.append(current_score)
        
        # Weighted average with emphasis on recent frames
        weights = np.exp(np.linspace(0, 1, len(recent_scores)))
        weights /= weights.sum()
        
        smoothed = np.average(recent_scores, weights=weights)
        return float(smoothed)
    
    def _visualize_analysis(self, image_path: str, detections: List[Detection],
                          analysis: Dict) -> np.ndarray:
        """Create visualization with analysis results"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Color scheme
        colors = {
            'person': (255, 0, 0),      # Red
            'phone': (0, 255, 0),       # Green
            'reflex_camera': (0, 0, 255), # Blue
            'polaroid_camera': (255, 255, 0) # Yellow
        }
        
        # Draw detections
        for det in detections:
            color = colors.get(det.class_name, (128, 128, 128))
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Bounding box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Label with confidence
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_rgb, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(img_rgb, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
        
        # Add analysis overlay
        overlay = img_rgb.copy()
        
        # Draw connections and indicators
        if analysis['details']:
            # Add visual indicators based on analysis
            # This is simplified - in practice, would add more sophisticated visualization
            pass
        
        # Add analysis results panel
        panel_height = 150
        panel = np.ones((panel_height, img_rgb.shape[1], 3), dtype=np.uint8) * 255
        
        # Add text to panel
        y_offset = 30
        cv2.putText(panel, f"Screen Capture Detection Results", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        y_offset += 30
        status = "DETECTED" if analysis['final_score'] > 0.5 else "NOT DETECTED"
        color = (0, 128, 0) if analysis['final_score'] > 0.5 else (128, 0, 0)
        cv2.putText(panel, f"Status: {status}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 25
        cv2.putText(panel, f"Confidence: {analysis['final_score']:.1%}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Component scores
        y_offset += 25
        cv2.putText(panel, f"Spatial: {analysis['spatial_score']:.1%} | "
                          f"Geometric: {analysis['geometric_score']:.1%} | "
                          f"Pose: {analysis['pose_score']:.1%} | "
                          f"Context: {analysis['context_score']:.1%}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Combine image and panel
        result = np.vstack([img_rgb, panel])
        
        return result
    
    def evaluate_on_dataset(self, dataset_path: str, 
                          ground_truth_path: str) -> Dict:
        """
        Evaluate algorithm performance on annotated dataset
        For research paper evaluation
        """
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        import json
        
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        predictions = []
        true_labels = []
        confidence_scores = []
        
        # Process each image
        for img_name, label in ground_truth.items():
            img_path = f"{dataset_path}/{img_name}"
            result = self.detect_screen_capture(img_path, visualize=False)
            
            predictions.append(result['is_capturing'])
            true_labels.append(label)
            confidence_scores.append(result['confidence'])
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        auc = roc_auc_score(true_labels, confidence_scores)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'accuracy': np.mean(np.array(predictions) == np.array(true_labels))
        }


# === USAGE EXAMPLE ===
def main():
    """Example usage for research paper"""
    
    # Initialize detector
    detector = ScreenCaptureDetector(
        model_path='/kaggle/input/testimage/best_model.pt'
    )
    
    # Single image analysis
    result = detector.detect_screen_capture(
        '/kaggle/input/testimage/WIN_20250607_19_34_01_Pro.jpg',
        visualize=True,
        return_details=True
    )
    
    # Display results
    print("=" * 60)
    print("SCREEN CAPTURE DETECTION RESULTS")
    print("=" * 60)
    print(f"Detection: {'YES' if result['is_capturing'] else 'NO'}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if result['evidence']:
        print("\nDetailed Analysis:")
        print(f"- Spatial Score: {result['evidence']['spatial_score']:.1%}")
        print(f"- Geometric Score: {result['evidence']['geometric_score']:.1%}")
        print(f"- Pose Score: {result['evidence']['pose_score']:.1%}")
        print(f"- Context Score: {result['evidence']['context_score']:.1%}")
        
        print("\nKey Evidence:")
        details = result['evidence'].get('details', {})
        for category, features in details.items():
            if isinstance(features, dict):
                print(f"\n{category.upper()}:")
                for key, value in features.items():
                    print(f"  - {key}: {value}")
    
    # Visualize
    if result['visualization'] is not None:
        plt.figure(figsize=(15, 10))
        plt.imshow(result['visualization'])
        plt.axis('off')
        plt.title('Screen Capture Detection Analysis')
        plt.tight_layout()
        plt.show()
    
    # For research evaluation
    # metrics = detector.evaluate_on_dataset(
    #     dataset_path='/path/to/test/images',
    #     ground_truth_path='/path/to/annotations.json'
    # )
    # print(f"\nEvaluation Metrics:")
    # print(f"- Precision: {metrics['precision']:.3f}")
    # print(f"- Recall: {metrics['recall']:.3f}")
    # print(f"- F1-Score: {metrics['f1_score']:.3f}")
    # print(f"- AUC: {metrics['auc']:.3f}")


if __name__ == "__main__":
    main()


"""
Simplified Screen Capture Detection
Quick implementation for practical use
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class SimpleScreenCaptureDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_map = {
            0: 'person',
            1: 'phone', 
            2: 'reflex_camera',
            3: 'polaroid_camera'
        }
    
    def detect(self, image_path):
        # Run detection
        results = self.model(image_path, conf=0.25)
        
        if len(results[0].boxes) == 0:
            return False, 0.0, "No objects detected"
        
        # Parse detections
        detections = {'person': [], 'phone': [], 'camera': []}
        
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = self.class_map[cls_id]
            
            if cls_name == 'person':
                detections['person'].append(box)
            elif cls_name == 'phone':
                detections['phone'].append(box)
            elif cls_name in ['reflex_camera', 'polaroid_camera']:
                detections['camera'].append(box)
        
        # Check conditions
        if not detections['person'] or not detections['phone']:
            return False, 0.0, "Missing person or phone"
        
        # Simple rule-based detection
        confidence = 0.0
        reason = []
        
        # Check each person-phone pair
        for person in detections['person']:
            p_box = person.xyxy[0].cpu().numpy()
            p_center = [(p_box[0] + p_box[2])/2, (p_box[1] + p_box[3])/2]
            p_height = p_box[3] - p_box[1]
            
            for phone in detections['phone']:
                ph_box = phone.xyxy[0].cpu().numpy()
                ph_center = [(ph_box[0] + ph_box[2])/2, (ph_box[1] + ph_box[3])/2]
                
                # Distance check
                distance = np.linalg.norm(np.array(p_center) - np.array(ph_center))
                normalized_dist = distance / p_height
                
                # Phone in front and raised
                phone_raised = ph_center[1] < p_center[1]
                phone_in_range = 0.2 < normalized_dist < 0.8
                
                if phone_raised and phone_in_range:
                    confidence = max(confidence, 0.6)
                    reason.append("Phone held up")
                    
                    # Bonus if camera present
                    if detections['camera']:
                        confidence = min(confidence + 0.3, 0.9)
                        reason.append("Camera detected")
        
        is_capturing = confidence > 0.5
        reason_str = ", ".join(reason) if reason else "No capture behavior detected"
        
        return is_capturing, confidence, reason_str
    
    def visualize(self, image_path):
        """Quick visualization"""
        results = self.model(image_path)
        
        # Get annotated image
        annotated = results[0].plot()
        
        # Detect
        is_capturing, confidence, reason = self.detect(image_path)
        
        # Add text overlay
        status = "SCREEN CAPTURE DETECTED" if is_capturing else "NO SCREEN CAPTURE"
        color = (0, 255, 0) if is_capturing else (0, 0, 255)
        
        cv2.putText(annotated, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated, f"Confidence: {confidence:.0%}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, reason, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated

# Usage
detector = SimpleScreenCaptureDetector('/kaggle/input/testimage/best_model.pt')

# Detect
image_path = '/kaggle/input/testimage/5e4c00bead0bcb001cd9294b.jpg'
is_capturing, confidence, reason = detector.detect(image_path)

print(f"Screen capture: {is_capturing}")
print(f"Confidence: {confidence:.0%}")
print(f"Reason: {reason}")

# Visualize
annotated = detector.visualize(image_path)
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# Ä�Æ°á»�ng dáº«n dataset
imagenet_dir = '/kaggle/input/imagenet-object-localization-challenge'
coco_dir = '/kaggle/input/coco-2017-dataset/coco2017'
output_dir = '/kaggle/working/dataset'


# Táº¡o thÆ° má»¥c output
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)


# Xá»­ lÃ½ ImageNet
annotations_file = os.path.join(imagenet_dir, 'LOC_train_solution.csv')
annotations = pd.read_csv(annotations_file)
desired_classes = ['n02992529', 'n04069434', 'n03976467']
class_map = {'n02992529': 1, 'n04069434': 2, 'n03976467': 3}  # phone, reflex_camera, polaroid_camera



filtered_annotations = annotations[annotations['PredictionString'].str.contains('|'.join(desired_classes))]
for idx, row in filtered_annotations.iterrows():
    image_id = row['ImageId']
    prediction = row['PredictionString'].split()
    class_id = prediction[0]
    if class_id not in desired_classes:
        continue
    xmin, ymin, xmax, ymax = map(float, prediction[1:5])
    
    # Ä�Æ°á»�ng dáº«n hÃ¬nh áº£nh
    image_path = os.path.join(imagenet_dir, 'ILSVRC/Data/CLS-LOC/train', class_id, f'{image_id}.JPEG')
    if not os.path.exists(image_path):
        continue
    
    # Sao chÃ©p hÃ¬nh áº£nh
    output_image_path = os.path.join(output_dir, 'images', f'imagenet_{image_id}.jpg')
    shutil.copy(image_path, output_image_path)
    
    # Láº¥y kÃ­ch thÆ°á»›c hÃ¬nh áº£nh
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    
    # Táº¡o label YOLO
    x_center = (xmin + xmax) / 2 / image_width
    y_center = (ymin + ymax) / 2 / image_height
    width_norm = (xmax - xmin) / image_width
    height_norm = (ymax - ymin) / image_height
    our_class_id = class_map[class_id]
    
    label = f"{our_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
    
    # LÆ°u label
    with open(os.path.join(output_dir, 'labels', f'imagenet_{image_id}.txt'), 'w') as f:
        f.write(label)



# Xá»­ lÃ½ COCO
coco = COCO(os.path.join(coco_dir, 'annotations/instances_train2017.json'))
desired_category_ids = [1, 68]  # person, cell phone
category_map = {1: 0, 68: 1}  # person: 0, cell phone: 1 (phone)
img_ids = coco.getImgIds(catIds=desired_category_ids)

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=desired_category_ids)
    anns = coco.loadAnns(ann_ids)
    
    # Sao chÃ©p hÃ¬nh áº£nh
    image_path = os.path.join(coco_dir, 'train2017', img_info['file_name'])
    output_image_path = os.path.join(output_dir, 'images', f'coco_{img_info["file_name"]}')
    shutil.copy(image_path, output_image_path)
    
    # Táº¡o label YOLO
    label_path = os.path.join(output_dir, 'labels', os.path.splitext(f'coco_{img_info["file_name"]}')[0] + '.txt')
    with open(label_path, 'w') as f:
        for ann in anns:
            category_id = ann['category_id']
            if category_id in category_map:
                our_class_id = category_map[category_id]
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                img_width = img_info['width']
                img_height = img_info['height']
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                f.write(f"{our_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")



# Táº¡o train.txt
images = [os.path.join(output_dir, 'images', f) for f in os.listdir(os.path.join(output_dir, 'images')) if f.endswith('.jpg')]
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for image in images:
        f.write(image + '\n')



!pip install ultralytics


# Táº¡o data.yaml
yaml_content = f"""
train: {os.path.join(output_dir, 'train.txt')}
val: {os.path.join(output_dir, 'train.txt')}
nc: 4
names: ['person', 'phone', 'reflex_camera', 'polaroid_camera']
"""
with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

# Fine-tune YOLO11
from ultralytics import YOLO

# Load mÃ´ hÃ¬nh pre-trained
model = YOLO("yolo11n.pt")

# Huáº¥n luyá»‡n mÃ´ hÃ¬nh
model.train(data=os.path.join(output_dir, 'data.yaml'), epochs=10, imgsz=640)




