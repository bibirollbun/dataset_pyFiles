!pip install -q "torch" "torchvision" "transformers" "Pillow" "opencv-python" "accelerate" "bitsandbytes" "ultralytics" "tqdm"
print("Installation complete.")


# ==============================================================================
# 1. SETUP AND INSTALLATION
# ==============================================================================
import os
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import shutil
import re
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import List, Dict, Tuple
import warnings
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
VIDEO_DIRECTORY = "/kaggle/input/2coool-unified-test-videos/test_videos/"
OUTPUT_FILE = "/kaggle/working/submission.csv"
MODEL_CACHE_DIR = "/kaggle/working/models"
NUM_FRAMES_TO_ANALYZE = 5 # Using 5 frames to get a clear Before, Peak, and After

if os.path.exists(MODEL_CACHE_DIR):
    shutil.rmtree(MODEL_CACHE_DIR)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ==============================================================================
# 3. ADVANCED PROMPT ENGINEERING TEMPLATES
# ==============================================================================

# Advanced prompt for the "Caption Before Incident" to maximize NLG scores
CAPTION_PROMPT = """
You are an AI assistant for a vehicle safety system. Your task is to provide a detailed scene description from the dashcam's point of view. 
In your description, you MUST include:
1. The type of road (e.g., multi-lane highway, residential street, intersection).
2. The current weather and lighting conditions (e.g., sunny day, overcast, nighttime).
3. The density of traffic (e.g., light, moderate, heavy).
4. Any significant objects or road users visible (e.g., other cars, trucks, cyclists, pedestrians).

Compose this information into a single, well-written paragraph.
"""

# Advanced Chain-of-Thought prompt for the "Reason of Incident" to maximize NLG scores
REASON_PROMPT = """
You are an expert accident analyst. Your goal is to explain the root cause of the incident shown in the image sequence (Before, Peak, After). 
Follow these steps in your reasoning:
1. Identify the primary 'actors' involved (e.g., 'the white sedan', 'the cyclist', 'the ego-vehicle').
2. Describe the state and action of each actor in the 'Before' frame.
3. Describe the critical event that occurs in the 'Peak' frame, specifying the precise interaction between the actors.
4. Explain the immediate outcome shown in the 'After' frame.
5. Synthesize these points into a concise, professional paragraph explaining the direct cause of the incident.
"""

# ==============================================================================
# 4. COMPETITION LABELS
# ==============================================================================
COMPETITION_LABELS = {
    "hazard_labels": [
        "animal on the road", "scooter on the road", "bicycle on road",
        "pedestrian is crossing the street", "pedestrian on the road",
        "vehicle overtakes", "flying object near the car",
        "obstacle on the road", "vehicle moving erratically",
    ],
    "collision_labels": [
        "ego-car hits barrier", "flying object hit the car", "ego-car hit an animal",
        "many cars/pedestrians/cyclists collided", "car hits barrier", "ego-car hits a pedestrian",
        "car flipped over", "ego-car hits a crossing cyclist", "vehicle drives into another vehicle",
        "ego-car loses control", "vehicle hits ego-car", "ego-car hits a vehicle"
    ],
    "crash_severity_labels": [
        "0. No Crash", "1. Ego-car kept moving", "2. Ego-car collided and could not continue moving",
        "3. Ego-car collided with at least one person or cyclist",
        "4. Other cars collided with person/car/object but ego-car is ok",
        "5. Multiple vehicles collided with ego-car"
    ]
}

# ==============================================================================
# 5. CORE FUNCTIONS (MODELS, FRAME EXTRACTION, IMAGE STITCHING)
# ==============================================================================

def setup_models():
    """Loads and initializes the VLM."""
    print("Setting up and loading pre-trained models... ðŸ¤–")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Primary device: {device}. Found {torch.cuda.device_count()} GPUs.")
    print("Loading VLM (llava-hf/bakLlava-v1-hf)...")
    vlm_model_id = "llava-hf/bakLlava-v1-hf"
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id, cache_dir=MODEL_CACHE_DIR)
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        vlm_model_id, torch_dtype=torch.float16, cache_dir=MODEL_CACHE_DIR, device_map="auto"
    )
    print("All models loaded successfully. âœ…")
    return {"device": device, "vlm": (vlm_processor, vlm_model)}

def find_peak_motion_frame(video_path: str) -> int:
    """Finds the frame with the most motion, which is a key strength of our pipeline."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0
        max_diff, peak_frame_idx, frame_idx = -1, 0, 0
        ret, prev_frame = cap.read()
        if not ret: cap.release(); return 0
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        while True:
            ret, current_frame = cap.read()
            if not ret: break
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, current_gray)
            current_diff = np.mean(diff)
            if current_diff > max_diff:
                max_diff, peak_frame_idx = current_diff, frame_idx + 1
            prev_gray, frame_idx = current_gray, frame_idx + 1
        cap.release()
        return peak_frame_idx
    except Exception:
        return 0

def extract_targeted_frames(video_path: str, num_frames: int, peak_frame_idx: int) -> Tuple[List[Image.Image], int]:
    """Extracts frames centered around the peak motion frame."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or total_frames < num_frames: return [], 0
        half_num_frames = num_frames // 2
        start_frame = max(0, peak_frame_idx - half_num_frames)
        end_frame = min(total_frames - 1, peak_frame_idx + half_num_frames)
        if end_frame - start_frame + 1 < num_frames:
            if start_frame == 0: end_frame = min(total_frames - 1, num_frames - 1)
            else: start_frame = max(0, total_frames - num_frames)
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames, total_frames
    except Exception as e: return [], 0

def stitch_images(images: List[Image.Image], labels: List[str]) -> Image.Image:
    """Creates the 'Storyboard' image by stitching frames horizontally with labels."""
    if not images: return None
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    stitched_image = Image.new('RGB', (total_width, max_height + 40))
    draw = ImageDraw.Draw(stitched_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    except IOError:
        font = ImageFont.load_default()

    x_offset = 0
    for i, img in enumerate(images):
        stitched_image.paste(img, (x_offset, 40))
        label_pos = (x_offset + img.width // 2 - 30, 5)
        draw.text(label_pos, labels[i], fill="white", font=font)
        x_offset += img.width
        
    return stitched_image

def query_vlm(frame: Image.Image, models, prompt_text: str, max_tokens: int = 150) -> str:
    """Generic function to query the VLM."""
    vlm_processor, vlm_model = models["vlm"]
    if not frame: return "Response unavailable."
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
    inputs = vlm_processor(text=prompt, images=frame, return_tensors="pt").to(vlm_model.device, torch.float16)
    with torch.no_grad():
        # Removed 'temperature' flag to prevent harmless warnings
        generate_ids = vlm_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        response = vlm_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response.split("ASSISTANT:")[-1].strip()

def vlm_count_involved_entities(storyboard_image: Image.Image, models: dict) -> Dict[str, int]:
    """Uses the VLM to count entities specifically involved in an incident."""
    counts = {}
    entity_map = {"vehicles": "vehicles (excluding the ego-car)", "pedestrians": "pedestrians", "animals": "animals", "bicyclists": "bicyclists or scooters"}
    for key, name in entity_map.items():
        prompt = (f"Analyze the sequence of events in the image. Count how many distinct {name} are *directly involved* "
                  "in the incident. Provide a single number as your answer.")
        response = query_vlm(storyboard_image, models, prompt, max_tokens=5)
        digits = re.findall(r'\d+', response)
        counts[key] = int(digits[0]) if digits else 0
    return counts

# ==============================================================================
# 6. MAIN VIDEO PROCESSING PIPELINE
# ==============================================================================
def process_video(video_info, models, labels):
    """Processes a single video using the full storyboard pipeline."""
    video_id, frames, total_frames, peak_frame_idx = video_info
    
    def create_no_incident_row():
        return {'video': int(video_id), 'Incident window start frame': -1, 'Incident Detection': -1, 'Crash Severity': "0. No Crash", 'Ego-car involved': 0, 'Label': "no incident", 'Number of Bicyclists/Scooters': 0, 'Number of animals involved': 0, 'Number of pedestrians involved': 0, 'Number of vehicles involved (excluding ego-car)': 0, 'Caption Before Incident': "no incident", 'Reason of Incident': "no incident"}

    if not frames or len(frames) < 3:
        return create_no_incident_row()

    frame_before = frames[0]
    frame_peak = frames[len(frames) // 2]
    frame_after = frames[-1]
    
    storyboard_image = stitch_images([frame_before, frame_peak, frame_after], ["Before", "Peak", "After"])
    if not storyboard_image: return create_no_incident_row()

    triage_prompt = ("Analyze the sequence of three images (Before, Peak, After). Does this sequence show a traffic accident, a potential hazard, or normal driving? "
                     "Your answer must be one of these three options only: 'Accident', 'Hazard', or 'No Incident'.")
    triage_result = query_vlm(storyboard_image, models, triage_prompt, max_tokens=10).lower()

    if "no incident" in triage_result or total_frames < 50:
        return create_no_incident_row()

    is_accident = "accident" in triage_result
    incident_type = 1 if is_accident else 0
    
    specific_label_list = labels["collision_labels"] if is_accident else labels["hazard_labels"]
    label_prompt = (f"From the list below, which label best describes the sequence of events shown in the three images?\n"
                    f"Options: {', '.join(specific_label_list)}\nProvide only the text of the best matching label.")
    incident_label = query_vlm(storyboard_image, models, label_prompt, max_tokens=50)
    if incident_label not in specific_label_list: incident_label = specific_label_list[0]

    row = {'video': int(video_id), 'Incident window start frame': peak_frame_idx, 'Incident Detection': incident_type, 'Label': incident_label}
    
    if is_accident:
        severity_prompt = (f"Based on the event sequence, especially the 'After' frame, determine the crash severity from these options:\n"
                           f"Options: {', '.join(labels['crash_severity_labels'])}\nChoose the most appropriate option and provide only its text.")
        crash_severity = query_vlm(storyboard_image, models, severity_prompt, max_tokens=50)
        matched_severity = [s for s in labels['crash_severity_labels'] if s.lower() in crash_severity.lower()]
        row['Crash Severity'] = matched_severity[0] if matched_severity else "1. Ego-car kept moving"
    else:
        row['Crash Severity'] = "0. No Crash"

    row["Caption Before Incident"] = query_vlm(frame_before, models, CAPTION_PROMPT)
    row["Reason of Incident"] = query_vlm(storyboard_image, models, REASON_PROMPT)
    
    entity_counts = vlm_count_involved_entities(storyboard_image, models)
    row.update({"Number of vehicles involved (excluding ego-car)": entity_counts["vehicles"], "Number of pedestrians involved": entity_counts["pedestrians"], "Number of animals involved": entity_counts["animals"], "Number of Bicyclists/Scooters": entity_counts["bicyclists"], "Ego-car involved": 1 if "ego-car" in row["Label"].lower() else 0})

    return row

def create_submission_file(results: List[dict]):
    """Generates the final submission CSV file."""
    print(f"\nGenerating submission file: {OUTPUT_FILE} ðŸ“„")
    df = pd.DataFrame(results)
    column_order = ["video", "Incident window start frame", "Incident Detection", "Crash Severity", "Ego-car involved", "Label", "Number of Bicyclists/Scooters", "Number of animals involved", "Number of pedestrians involved", "Number of vehicles involved (excluding ego-car)", "Caption Before Incident", "Reason of Incident"]
    for col in column_order:
        if col not in df.columns: df[col] = -1
    df = df[column_order]
    df.fillna({'Incident Detection': -1, 'Incident window start frame': -1, 'Ego-car involved': 0}, inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Submission file created successfully! ðŸŽ‰\nTop 5 rows:")
    print(df.head())

# ==============================================================================
# 7. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # Check for font file needed for image stitching
    if not os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        print("Font file not found, attempting to install...")
        os.system("sudo apt-get -y install fonts-dejavu-core")

    models = setup_models()
    video_files = sorted([f for f in os.listdir(VIDEO_DIRECTORY) if f.endswith('.mp4')], 
                         key=lambda x: int(os.path.splitext(x)[0]))
    
    print(f"Found {len(video_files)} video files to process.")

    all_results = []
    start_time = time.time()

    with tqdm(total=len(video_files), desc="ðŸŽ¥ Processing videos") as pbar:
        for video_file in video_files:
            video_id = os.path.splitext(video_file)[0]
            try:
                video_path = os.path.join(VIDEO_DIRECTORY, video_file)
                
                peak_frame_idx = find_peak_motion_frame(video_path)
                frames, total_frames = extract_targeted_frames(video_path, NUM_FRAMES_TO_ANALYZE, peak_frame_idx)
                
                video_info = (video_id, frames, total_frames, peak_frame_idx)
                result = process_video(video_info, models, COMPETITION_LABELS)
                all_results.append(result)

            except Exception as e:
                # This robust error handling ensures a row is created for every video
                print(f"\n---!!! An error occurred while processing video: {video_file} !!!---")
                print(f"Error details: {e}")
                print("--- Creating a default 'no incident' row and continuing. ---")
                
                default_row = {
                    'video': int(video_id), 'Incident window start frame': -1, 'Incident Detection': -1,
                    'Crash Severity': "0. No Crash", 'Ego-car involved': 0, 'Label': "no incident",
                    'Number of Bicyclists/Scooters': 0, 'Number of animals involved': 0,
                    'Number of pedestrians involved': 0, 'Number of vehicles involved (excluding ego-car)': 0,
                    'Caption Before Incident': "no incident", 'Reason of Incident': "error processing video"
                }
                all_results.append(default_row)
            
            finally:
                # Ensures the progress bar always updates
                pbar.update(1)

    print(f"\nâœ… Analysis complete. Processed {len(all_results)} videos.")
    incident_count = sum(1 for r in all_results if r.get('Incident Detection', -1) != -1)
    print(f"Detected incidents in {incident_count}/{len(video_files)} videos.")
    
    create_submission_file(all_results)
    
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

