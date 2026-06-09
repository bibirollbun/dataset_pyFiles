%pip install mtcnn


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os, glob
import cv2


TEST_VIDEO_DIR = r'/kaggle/input/deepfake-detection-challenge/test_videos'
TRAIN_VIDEO_DIR = r'/kaggle/input/deepfake-detection-challenge/train_sample_videos'


test_paths = sorted(glob.glob(os.path.join(TEST_VIDEO_DIR, '*.mp4')))
train_paths = sorted(glob.glob(os.path.join(TRAIN_VIDEO_DIR, '*.mp4')))

metadata_path = os.path.join(TRAIN_VIDEO_DIR, 'metadata.json')
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

pd.DataFrame(metadata)


def extract_iframes(video_path, output_dir, num_frames=20):
    """
    Extract equidistant frames from a video using cv2.
    
    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory where frames will be saved.
        num_frames (int): Number of frames to extract.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If requested frames > available, limit it
    num_frames = min(num_frames, total_frames)

    # Compute step size
    step = total_frames // num_frames

    frame_ids = [i * step for i in range(num_frames)]

    for idx, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(frame_filename, frame)
        else:
            print(f"Warning: could not read frame {frame_id} in {video_path}")

    cap.release()
    print(f"Extracted {len(frame_ids)} frames from {video_path}")
    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    return frames


from mtcnn import MTCNN
def crop_faces_mtcnn(image_paths, output_dir, min_confidence=0.95, detector=None):
    """
    Crops the highest-confidence face from each image using MTCNN.
    Returns a list of saved cropped face image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    if detector is None:
        detector = MTCNN()
    
    cropped_files = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = detector.detect_faces(img)
        if not results:
            continue

        best_face = max(results, key=lambda x: x['confidence'])
        conf = best_face['confidence']

        if conf < min_confidence:
            continue

        x, y, w, h = best_face['box']
        x, y = max(0, x), max(0, y)
        face_crop = img[y:y+h, x:x+w]

        out_file = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_file, face_crop)
        cropped_files.append(out_file)

    return cropped_files


def extract_faces_from_video(video_path, work_dir, min_confidence=0.95, max_frames=None, quiet=True):
    """
    Complete pipeline: extracts I-frames from a video, then crops faces with MTCNN.
    Returns list of cropped face image paths.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(work_dir, f"{base_name}_frames")
    crops_dir = os.path.join(work_dir, f"{base_name}_crops")

    # Step 1: Extract I-frames
    frames = extract_iframes(video_path, frames_dir)

    # Optional downsampling (if too many frames)
    if max_frames and len(frames) > max_frames:
        import random
        frames = random.sample(frames, max_frames)

    # Step 2: Crop faces
    detector = MTCNN()
    cropped_faces = crop_faces_mtcnn(frames, crops_dir, min_confidence=min_confidence, detector=detector)

    return cropped_faces


import gc

if __name__ == "__main__":
    video_paths = test_paths
    for video_path in video_paths:
        output_root = "/kaggle/working/processed"

        cropped_faces = extract_faces_from_video(
            video_path,
            work_dir=output_root,
            min_confidence=0.95,
        )

        print(f"Extracted and cropped {len(cropped_faces)} faces from {video_path}")
        gc.collect()


!rm -rf /kaggle/working/processed/*_frames

