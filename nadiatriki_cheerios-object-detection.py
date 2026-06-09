# Competition Submission: Cheerios Object Detection (0.99553 Score)

## ğŸš€ Approach Overview  
This solution uses YOLOv8 to find and locate Cheerios boxes in images. By carefully adjusting the training process we achieved a **0.99553 mAP** score in the Synthetic-to-Real Object Detection Challenge. Hereâ€™s how:.  

---

### 1. Data Strategy ğŸ—ƒï¸�  
- **Dataset Structure**:  Used the full dataset (training + validation) data available on falcon, and testing the test data provided in the competition. 
- **Class Handling**: Single-class setup (`cheerios`) with YOLO-style annotations.  
- **Augmentations**:
Added small changes to the images (like flipping, rotating, and adjusting colors) to help the model learn better:
  - HSV color space manipulation (`HueÂ±0.015`, `SaturationÂ±0.7`, `ValueÂ±0.4`).  
  - Spatial transforms: Flip (`50%`), Translation (`10%`), Scaling (`50%`), Shear (`Â±0.01`).  
  - Aspect ratio preservation during resizing.  

---

### 2. Model Architecture ğŸ§   
- **Base Model**: YOLOv8 Large (`yolov8l.pt`).  
- **Optimization**:  
  - SGD optimizer with momentum (`0.937`).  
  - Cosine learning rate scheduler (initial `lr=0.0005`).  
  - Weight decay (`0.0001`) for regularization.  
  - 20-epoch training.  

---

### 3. Inference Enhancements ğŸ”�  
- **Test-Time Augmentation (TTA)**: During testing, we made the model look at each image multiple times with small changes (like zooming or flipping) to get more accurate results. 
  - Enabled multi-scale inference with flipped versions.  
  - Confidence threshold: `0.05` (optimized for recall). 
- **Post-processing**:  
  - Strict validation of prediction format.  
  - Automatic `no boxes` handling for missing predictions.  

---

### 4. Validation Strategy âœ…  
- **Holdout Validation**: Dedicated validation set from competition data.  
- **Metric Focus**: Optimized for `mAP@0.5IOU`.  
- **Reproducibility**:  
  - Fixed random seeds in augmentation pipeline.  
  - Hardware: NVIDIA Tesla T4 GPU.

