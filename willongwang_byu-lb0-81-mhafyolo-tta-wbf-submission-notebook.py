# !tar xfz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
# !pip install --no-index --find-links=./packages -q ultralytics
# !rm -rf ./packages
# print("package installed ...........................")

!cp -r /kaggle/input/mhafyolo/pytorch/default/1/MHAF-YOLO-main /kaggle/working/
print('PIP INSTALL OK!!!')


import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2
from tqdm.notebook import tqdm
from pathlib import Path
current_dir = Path.cwd()
print("this_dir:", current_dir)

target_dir = Path("/kaggle/working/MHAF-YOLO-main") 
os.chdir(target_dir)

from ultralytics import YOLOv10
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from ultralytics.utils.ops import non_max_suppression


# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv"

# Model path - adjust if your best model is saved in a different location
model_path = "/kaggle/input/mahf-yolo-train/mayolov2f.pt"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.8  # Lower threshold to catch more potential motors
MAX_DETECTIONS_PER_TOMO = 1  # Keep track of top N detections per tomogram
NMS_IOU_THRESHOLD = 0.2  # Non-maximum suppression threshold for 3D clustering
CONCENTRATION = 1 # ONLY PROCESS 1/20 slices for fast submission
SIZE = 1024


# GPU profiling context manager
class GPUProfiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")

# Check GPU availability and set up optimizations
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = ['cuda:0', 'cuda:1']
BATCH_SIZE = 8  # Default batch size, will be adjusted dynamically if GPU available

if device.startswith('cuda'):
    # Set CUDA optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
    print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    
    # Get available GPU memory and set batch size accordingly
    free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
    BATCH_SIZE = max(8, min(32, int(free_mem * 4)))  # 4 images per GB as rough estimate
    print(f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
else:
    print("GPU not available, using CPU")
    BATCH_SIZE = 8  # Reduce batch size for CPU


def weighted_box_fusion(boxes, iou_threshold=0.4):
    """Applies Weighted Box Fusion to combine overlapping bounding boxes."""
    fused_boxes = []
    # print(type(boxes))
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        
        similar_boxes = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            if iou(boxes[i], boxes[j]) > iou_threshold:
                similar_boxes.append(boxes[j])
                used[j] = True

        # Compute weighted average for the final box
        # similar_boxes = np.array(similar_boxes)
        # similar_boxes = similar_boxes.cpu().numpy()
        similar_boxes = np.array([t.cpu() for t in similar_boxes])
        # print("similar_boxes: ",similar_boxes)
        confidences = similar_boxes[:,:, 4]
        weights = confidences / confidences.sum()

        fused_x1 = np.sum(similar_boxes[:, :, 0] * weights)
        fused_y1 = np.sum(similar_boxes[:, :, 1] * weights)
        fused_x2 = np.sum(similar_boxes[:, :, 2] * weights)
        fused_y2 = np.sum(similar_boxes[:, :, 3] * weights)
        fused_confidence = np.mean(confidences) #np.max(confidences)  # Take max confidence

        fused_boxes.append((fused_x1, fused_y1, fused_x2, fused_y2, fused_confidence))

    return fused_boxes

def predict_ensemble_tta(single_model, image_np, device, img_size):
    """
    For a 640x640 numpy image:
    - Multiple models (list of models)
    - Multiple TTA (original, hflip, vflip, rot90)
    Do the NMS for the last time
    Return: [K,6] => x1,y1,x2,y2,conf,cls
    """
    all_boxes = []
    all_confs = []
    all_clss = []

    def do_infer(img_tta, invert_func):
        #for m in models:
            res = single_model(img_tta, 
                            imgsz=img_size, 
                            # conf=conf_thres,
                            device=device, 
                            verbose=False)
            # res = single_model(img_tta,verbose=False)
            for r in res:
                # print('res box:', r.boxes)
                boxes = r.boxes
                if boxes is None or len(boxes)==0:
                    # print('predict box is none!')
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                # Inverse transformation
                xyxy_orig = invert_func(xyxy)
                all_boxes.append(xyxy_orig)
                all_confs.append(confs)
                all_clss.append(clss)

    # Master drawing
    do_infer(image_np, invert_func=lambda x: x)

    # Horizontal flip
    img_hflip = cv2.flip(image_np, 1)

    org_img_h, org_img_w = image_np.shape[:2]
    # then in invert_hflip:
    def invert_hflip(xyxy):
        new_ = xyxy.copy()
        x1 = org_img_w - xyxy[:,2]
        x2 = org_img_w - xyxy[:,0]
        new_[:,0] = x1
        new_[:,2] = x2
        return new_
    do_infer(img_hflip, invert_func=invert_hflip)

    # Vertical flip
    # img_vflip = cv2.flip(image_np, 0)
    # def invert_vflip(xyxy):
    #     new_ = xyxy.copy()
    #     y1 = img_size - xyxy[:,3]
    #     y2 = img_size - xyxy[:,1]
    #     new_[:,1] = y1
    #     new_[:,3] = y2
    #     return new_
    # do_infer(img_vflip, invert_func=invert_vflip)

    # Rotate 90 degrees (clockwise)
    # img_rot90 = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
    # def invert_rot90(xyxy):
    #     new_ = xyxy.copy()
    #     # (x,y)->(y,640-x)
    #     # Inverse transform (x',y')->(640-y',x')
    #     x1_old,y1_old = xyxy[:,0], xyxy[:,1]
    #     x2_old,y2_old = xyxy[:,2], xyxy[:,3]
    #     X1 = img_size - y2_old
    #     Y1 = x1_old
    #     X2 = img_size - y1_old
    #     Y2 = x2_old
    #     new_[:,0] = X1
    #     new_[:,1] = Y1
    #     new_[:,2] = X2
    #     new_[:,3] = Y2
    #     return new_
    # do_infer(img_rot90, invert_func=invert_rot90)

    if len(all_boxes)==0:
        # print('all boxes is None!')
        return None

    boxes_cat = np.concatenate(all_boxes, axis=0)
    confs_cat = np.concatenate(all_confs, axis=0)
    clss_cat  = np.concatenate(all_clss, axis=0)
    cat_data = np.column_stack([boxes_cat, confs_cat, clss_cat])  # shape [N,6]
    # print('final:',cat_data)

    # Need to add batch dimension => [1,N,6]
    cat_tensor = torch.from_numpy(cat_data).float().unsqueeze(0).to(device)
    # NMS 
    # nms_out = non_max_suppression(cat_tensor, iou_thres=0.5, max_det=300)
    nms_out = weighted_box_fusion(cat_tensor, iou_threshold=0.5)
    # nms_out => list length =1(a graph), take nms_out[0]
    if len(nms_out)==0 or nms_out[0] is None or len(nms_out[0])==0:
        return None
    # final_nms = nms_out[0].cpu().numpy()  # shape [K,6]
    final_nms = list(nms_out[0]) # shape [K,6]
    # print("final_nms : ", final_nms)
    return final_nms


def make_predict(sub_path, model, device, img_size):
    # print("img path:", sub_path)
    res_list = []
    for img in sub_path:
        # print('images:', img)
        img_np = cv2.imread(img)
        if img_np is None:
            res_list.append((0, 0, 0, 0, 0.0))
            continue
        res_nms = predict_ensemble_tta(model, img_np, device, img_size = img_size)
        if res_nms is None:
            # print('nms is none!')
            res_list.append((0, 0, 0, 0, 0.0))
        else:
            res_list.append(res_nms)

    return res_list
    


def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles for better contrast
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

def preload_image_batch(file_paths):
    """Preload a batch of images to CPU memory"""
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            # Try with PIL as fallback
            img = np.array(Image.open(path))
        images.append(img)
    return images

def process_tomogram(tomo_id, model, index=0, total=1,SIZE=SIZE):
    """
    Process a single tomogram and return the most confident motor detection
    """
    print(f"Processing tomogram {tomo_id} ({index}/{total})")
    
    # Get all slice files for this tomogram
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    # Apply CONCENTRATION to reduce the number of slices processed
    # This will process approximately CONCENTRATION fraction of all slices
    selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]
    
    print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices based on CONCENTRATION={CONCENTRATION}")
    
    # Create a list to store all detections
    all_detections = []
    
    # Create CUDA streams for parallel processing if using GPU
    if device.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(8, BATCH_SIZE))]
    else:
        streams = [None]
    
    # Variables for preloading
    next_batch_thread = None
    next_batch_images = None
    
    # Process slices in batches
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        # Wait for previous preload thread if it exists
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None
            
        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]
        
        # Start preloading next batch
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None
        
        # Split batch across streams for parallel processing
        sub_batches = np.array_split(batch_files, len(streams))
        sub_batch_results = []
        
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
                
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                # Process sub-batch
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                
                # Run inference with profiling
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    # sub_results = model(sub_batch_paths, verbose=False)
                    sub_results = make_predict(sub_batch_paths, model, device, SIZE)
                    # print(sub_results)
                
                # Process each result in this sub-batch
                # for result in sub_results:
                    # print('nms_res1:', result)
                for j,res in enumerate(sub_results):
                    # print('nms_res2', res)
                    # x1,y1,x2,y2, confidence, cls_ = res
                    x1,y1,x2,y2, confidence = res
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Calculate center coordinates
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                                
                        # Store detection with 3D coordinates
                        all_detections.append({
                                'z': round(sub_batch_slice_nums[j]),
                                'y': round(y_center),
                                'x': round(x_center),
                                'confidence': float(confidence)
                            })
                    # print("all_detections:", all_detections)

                    # if len(result.boxes) > 0:
                    #     boxes = result.boxes
                    #     for box_idx, confidence in enumerate(boxes.conf):
                    #         if confidence >= CONFIDENCE_THRESHOLD:
                    #             # Get bounding box coordinates
                    #             x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                                
                    #             # Calculate center coordinates
                    #             x_center = (x1 + x2) / 2
                    #             y_center = (y1 + y2) / 2
                                
                    #             # Store detection with 3D coordinates
                    #             all_detections.append({
                    #                 'z': round(sub_batch_slice_nums[j]),
                    #                 'y': round(y_center),
                    #                 'x': round(x_center),
                    #                 'confidence': float(confidence)
                    #             })
        
        # Synchronize streams
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    
    # Clean up thread if still running
    if next_batch_thread is not None:
        next_batch_thread.join()
    
    # 3D Non-Maximum Suppression to merge nearby detections across slices
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    
    # Sort detections by confidence (highest first)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # If there are no detections, return NA values
    if not final_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }
    
    # Take the detection with highest confidence
    best_detection = final_detections[0]
    
    # Return result with integer coordinates
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }

def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # List to store final detections after NMS
    final_detections = []
    
    # Define 3D distance function
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + 
                       (d1['y'] - d2['y'])**2 + 
                       (d1['x'] - d2['x'])**2)
    
    # Maximum distance threshold (based on box size and slice gap)
    box_size = 24  # Same as annotation box size
    distance_threshold = box_size * iou_threshold
    
    # Process each detection
    while detections:
        # Take the detection with highest confidence
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        # Filter out detections that are too close to the best detection
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections

def generate_submission():
    """
    Main function to generate the submission file
    """
    # Get list of test tomograms
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    
    print(f"Found {total_tomos} tomograms in test directory")
    
    # Debug image loading for the first tomogram
    # if test_tomos:
    #     debug_image_loading(test_tomos[0])
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize model once outside the processing loop
    print(f"Loading YOLO model from {model_path}")
    model = YOLOv10(model_path)
    model.to(device)
    
    # Additional optimizations for inference
    if device.startswith('cuda'):
        # Fuse conv and bn layers for faster inference
        model.fuse()
        
        # Enable model half precision (FP16) if on compatible GPU
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or newer
            model.model.half()
            print("Using half precision (FP16) for inference")
    
    # Process tomograms with parallelization
    results = []
    motors_found = 0
    
    # Using ThreadPoolExecutor with max_workers=1 since each worker uses the GPU already
    # and we're parallelizing within each tomogram processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_tomo = {}
        
        # Submit all tomograms for processing
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id
        
        # Process completed futures as they complete
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            result = future.result()
            results.append(result)
                
                # Update motors found count
            has_motor = not pd.isna(result['Motor axis 0'])
            if has_motor:
                motors_found += 1
                print(f"Motor found in {tomo_id} at position: "
                      f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
            else:
                print(f"No motor detected in {tomo_id}")
                    
            print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            
            # try:
            #     # Clear CUDA cache between tomograms
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
                    
            #     result = future.result()
            #     results.append(result)
                
            #     # Update motors found count
            #     has_motor = not pd.isna(result['Motor axis 0'])
            #     if has_motor:
            #         motors_found += 1
            #         print(f"Motor found in {tomo_id} at position: "
            #               f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
            #     else:
            #         print(f"No motor detected in {tomo_id}")
                    
            #     print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            
            # except Exception as e:
            #     print(f"Error processing {tomo_id}: {e}")
            #     # Create a default entry for failed tomograms
            #     results.append({
            #         'tomo_id': tomo_id,
            #         'Motor axis 0': -1,
            #         'Motor axis 1': -1,
            #         'Motor axis 2': -1
            #     })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    
    # Ensure proper column order
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    
    # Save the submission file
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission complete!")
    print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {submission_path}")
    
    # Display first few rows of submission
    print("\nSubmission preview:")
    print(submission_df.head())
    
    return submission_df


# Run the submission pipeline
if __name__ == "__main__":
    # Time entire process
    start_time = time.time()
    
    # Generate submission
    submission = generate_submission()
    
    # Print total execution time
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


# tomo_00e047	169	546	603
# tomo_01a877	147	638	286

