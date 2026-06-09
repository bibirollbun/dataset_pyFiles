!pip install --no-index --find-links=/kaggle/input/insightface-pkg/pytorch/default/1/insightface_pkg insightface


!pip install --no-index --find-links=/kaggle/input/onnxruntime-gpu/pytorch/default/1/onnxruntime_pkg onnxruntime-gpu


import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List
from insightface.app import FaceAnalysis
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import os
import shutil
import gc


metadata_metaforensics1 = {f'/kaggle/input/finals/MetaForensics/1/m_{filename}': {'label': 'FAKE', 'split': 'test', 'original': f'/kaggle/input/finals/MetaForensics/0/{filename}'} for filename in sorted(os.listdir('/kaggle/input/finals/MetaForensics/0'))}
metadata_metaforensics0 = {f'/kaggle/input/finals/MetaForensics/0/{filename}': {'label': 'FAKE', 'split': 'test', 'original': f'/kaggle/input/finals/MetaForensics/0/{filename}'} for filename in sorted(os.listdir('/kaggle/input/finals/MetaForensics/0'))}
metadata = {**metadata_metaforensics0, **metadata_metaforensics1}


src_model_dir = '/kaggle/input/deepfake1/asset/models/antelopev2'

dst_root = '/kaggle/working/antelope_root'
dst_model_dir = os.path.join(dst_root, 'models', 'antelopev2')

if not os.path.exists(dst_model_dir):
    shutil.copytree(src_model_dir, dst_model_dir)


from insightface.app import FaceAnalysis

app = FaceAnalysis(name='antelopev2', root=dst_root, providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)


# Set paths
base_path = '/kaggle/input/'
videos_dir = os.path.join(base_path, 'Deepfake_Detection_and_Generation_Challenge_Blue_Team/test_sample_videos')
best_model_path = os.path.join(base_path, 'vivit-model/vivit_best_f1_0.6347.pth')  # Update to latest ViViT model path

# Initialize VideoMAE processor
processor = VideoMAEImageProcessor.from_pretrained("/kaggle/input/videomae_processor/pytorch/default/1/videomae_processor")

# with open('/kaggle/input/Deepfake_Detection_and_Generation_Challenge_Blue_Team/metadata.json', 'r') as f:
#     metadata = json.load(f)


def get_video_paths_from_metadata(metadata, videos_dir, split=None):
    """
    Get video path list from metadata
    Args:
        metadata: Dictionary containing video information
        videos_dir: Video directory path
        split: Optional, specify dataset split ('train', 'test', 'val')
    Returns:
        video_paths: List of video paths
    """
    video_paths = []
    for video_name, video_info in metadata.items():
        # If split is specified, only return videos from that split
        if split is not None and video_info.get('split') != split:
            continue
        
        video_path = os.path.join(videos_dir, video_name)
        if os.path.exists(video_path):
            video_paths.append(video_path)
        else:
            print(f"Warning: Video file not found: {video_path}")
    
    return video_paths

def extract_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract all frames from video and return image list
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        List[np.ndarray]: List containing all video frames, each element is a numpy array representing an image
    """
    frames = []
    reader = cv2.VideoCapture(video_path)
    
    while reader.isOpened():
        success, frame = reader.read()
        if not success:
            break
        frames.append(frame)
    
    reader.release()
    return frames

def expand_bbox(bbox, img_shape, ratio=0.2):
    """
    Expand face bbox and prevent out of bounds
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    expand_w = int(w * ratio)
    expand_h = int(h * ratio)

    new_x1 = max(0, x1 - expand_w)
    new_y1 = max(0, y1 - expand_h)
    new_x2 = min(img_shape[1], x2 + expand_w)
    new_y2 = min(img_shape[0], y2 + expand_h)

    return [new_x1, new_y1, new_x2, new_y2]

def extract_faces_from_one_video(video_path, ratio=0.2, max_frames=32):
    
    frames = extract_frames(video_path)
    
    face_cropped_frames = []
    for frame in frames:
        faces = app.get(frame)
        if len(faces) == 0:
            continue
        else:
            face = faces[0]
            bbox = face.bbox.astype(int)
            expanded_bbox = expand_bbox(bbox, frame.shape, ratio=ratio)
            x1, y1, x2, y2 = expanded_bbox
            face_crop = frame[y1:y2, x1:x2]
            face_cropped_frames.append(face_crop)
        if len(face_cropped_frames) >= max_frames:
            break
    
    # Release original frame memory
    del frames
    gc.collect()
    
    return face_cropped_frames

def process_single_video(video_path, metadata, model, device, max_frames=16, min_sequence_length=8):
    """
    Process a single video and return prediction results
    """
    video_name = os.path.basename(video_path)
    
    # Extract face frames
    face_frames = extract_faces_from_one_video(video_path, max_frames=max_frames * 4)
    
    # Check if there are enough face frames
    if len(face_frames) < min_sequence_length:
        print(f"Warning: {video_name} has only {len(face_frames)} face frames, minimum required is {min_sequence_length}")
        return None
    
    # Create multiple sequence segments
    sequences = []
    for i in range(0, len(face_frames) - max_frames + 1, max_frames):
        frame_chunk = face_frames[i:i+max_frames]
        sequences.append({
            'face_frames': frame_chunk
        })
    
    # Release face frame memory
    del face_frames
    gc.collect()
    
    # Perform inference on each sequence
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for sequence in sequences:
            # Process images
            imgs = []
            for face_frame in sequence['face_frames']:
                if isinstance(face_frame, np.ndarray):
                    face_frame = Image.fromarray(face_frame)
                face_frame = face_frame.resize((224, 224))  # ViViT input size
                imgs.append(face_frame)
            
            # Use VideoMAE processor to process images
            pixel_values = processor(images=imgs, return_tensors="pt")["pixel_values"]  # [1, C, T, H, W]
            pixel_values = pixel_values.to(device)
            
            # Model inference
            outputs = model(pixel_values)
            
            # Get prediction probability and predicted class
            prob = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1)
            
            # Convert all fake classes (1-5) to 1, youtube class (0) remains 0
            binary_pred_class = 1 if pred_class.item() != 0 else 0
            
            predictions.append(binary_pred_class)
            probabilities.append(prob[0][pred_class].item())
            
            # Release current sequence memory
            del imgs, pixel_values, outputs, prob, pred_class
            gc.collect()
    
    # Vote to decide final prediction
    if predictions:
        # Count prediction results
        pred_counts = {}
        total_prob = 0
        
        for pred, prob in zip(predictions, probabilities):
            if pred not in pred_counts:
                pred_counts[pred] = 0
            pred_counts[pred] += 1
            total_prob += prob
        
        # Vote to decide final prediction (majority voting)
        final_prediction = max(pred_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average probability and confidence
        avg_probability = total_prob / len(predictions)
        confidence = pred_counts[final_prediction] / len(predictions)
        
        result = {
            'ID': video_name,
            'label': final_prediction
        }
    else:
        result = None
    
    # Release sequence memory
    del sequences, predictions, probabilities
    gc.collect()
    
    return result

# ViViT Model
class ViViTForDeepfake(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits

def load_model(model_path, device):
    """Load trained ViViT model"""
    print(f"Loading ViViT model from: {model_path}")
    
    # Initialize model
    model = ViViTForDeepfake(num_classes=6)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"ViViT model loaded successfully!")
    
    return model

def inference_on_videos_one_by_one(video_paths, metadata, model, device):
    """Perform inference on videos one by one to avoid memory explosion"""
    print(f"Starting inference on {len(video_paths)} videos (one by one)...")
    
    results = []
    skipped_videos = []
    
    for i, video_path in enumerate(tqdm(video_paths, desc="Processing videos")):
        try:
            # Process single video
            result = process_single_video(
                video_path=video_path,
                metadata=metadata,
                model=model,
                device=device,
                max_frames=16,
                min_sequence_length=8
            )
            
            if result is not None:
                results.append(result)
            else:
                skipped_videos.append(os.path.basename(video_path))
            
            # Force garbage collection every 10 videos
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                print(f"Processed {i + 1}/{len(video_paths)} videos, memory cleaned")
                
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            skipped_videos.append(os.path.basename(video_path))
            continue
    
    print(f"Successfully processed {len(results)} videos")
    print(f"Skipped {len(skipped_videos)} videos")
    
    return results

def save_results(results, output_path):
    """Save inference results"""
    if not results:
        print("No results to save!")
        return None
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    return df

# Main inference process
def main_inference():
    """Main inference function"""
    print("Starting main inference process...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(best_model_path, device)
    
    # Get all video paths (no split restriction, process all videos)
    all_video_paths = get_video_paths_from_metadata(metadata, videos_dir)
    print(f"Found {len(all_video_paths)} videos for inference")
    
    # Perform inference (process videos one by one)
    results = inference_on_videos_one_by_one(all_video_paths, metadata, model, device)
    
    # Save results
    output_path = os.path.join('/kaggle/working', 'submission_1.csv')
    df = save_results(results, output_path)
    
    # Display first few results
    print("\nFirst 10 inference results:")
    print(df.head(10))
    
    return results

# Usage example
if __name__ == "__main__":
    # Run main inference process
    main_inference()


df = pd.read_csv('/kaggle/working/inference_results.csv')


df.to_csv('/kaggle/working/submission_1.csv')


import onnxruntime as ort

print("Available Execution Providers:")
print(ort.get_available_providers())

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("✅ CUDAExecutionProvider is available! You can use GPU.")
else:
    print("❌ CUDAExecutionProvider is NOT available. ONNX Runtime will run on CPU.")





