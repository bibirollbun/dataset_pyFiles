# Install required packages 
!pip install unsloth

# Install latest transformers for Gemma 3n
!pip install --no-deps --upgrade transformers==4.54.1
!pip install --no-deps --upgrade timm==1.0.19


import torch

from unsloth import FastModel

# Set TorchDynamo cache size limit before using TorchDynamo features
torch._dynamo.config.cache_size_limit = 64

# Set up the tokenizer and model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False
)


from base64 import b64encode
from IPython.display import HTML

# Define the URL of the video we want to analyze
VIDEO_URL="/kaggle/input/emergency-dataset/01_kids_near_pool.mp4"

# Display the image to verify it's loaded correctly
video = open(VIDEO_URL,'rb').read()
src = 'data:video/mp4;base64,' + b64encode(video).decode()
html = '<video width=480 controls><source src="%s" type="video/mp4"></video>' % src 
HTML(html)


# This prompt template forces the model into the desired persona and output structure for an image.
system_prompt_text = """You are an intelligent security agent responsible for protecting people, homes, and shops from threats. 
You will analyze visual input and describe whether a dangerous or suspicious situation is present. 
Your analysis should be logical, cautious, and actionable.
You will receive concise descriptions of what is seen in images (e.g., "a bear near the backyard", "a person entering the shop after hours", or "a dog walking past the gate"). Based on this, assess the risk level using systematic threat evaluation.

ANALYSIS PROCESS:
1. Identify all entities and their behaviors in the scene
2. Assess potential threat level based on context, time, location, and behavior patterns
3. Consider environmental factors and normal vs. abnormal activity
4. Determine appropriate response level
            
EMERGENCY LEVELS:
- **HIGH**: Immediate physical danger (wild animals near entry points, armed intruders, violent behavior, fire, medical emergencies)
- **MEDIUM**: Suspicious activity requiring attention (unknown persons after hours, unusual behavior patterns, potential security breaches)
- **LOW**: Normal, safe situations requiring no action

RESPONSE FORMAT(JSON)
- emergency_level: low|medium|high
- reason: Brief explanation why threat exists
- advice: Specific actionable recommendation
- notification_types: one or more of google_home_alarm, google_nest, push_notifications, emergency_calls (leave empty if emergency_level is safe)

EXAMPLE OUTPUT:
```json
{
    "emergency_level" : "high",
    "reason" : "Large bear attempting to enter through front door",
    "advice" : "Immediately secure all entrances and contact wildlife control",
    "notification_types" : "google_home_alarm, push_notifications, emergency_calls"
}
```

RULES:
You must provide only the output in strict JSON format, with no additional text, explanations, or formatting.
If you are unsure, lean toward safety, but never hallucinate.
Be concise but smart."""


from transformers import TextStreamer

# Inference method to generate predictions or responses
def inference(messages, use_streamer=False) -> str:
    # Prepare the inputs for the model
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

    # Generate the response with default recommended settings 
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(**inputs, streamer = text_streamer if use_streamer else None, max_new_tokens=128, temperature=1.0, top_p=0.95, top_k=64)
    
    # Decode and print the final result
    input_len = inputs["input_ids"].shape[-1]
    results = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return results


def get_analysis(system_prompt: str, frames: [], prompt: str=None, use_streamer=False) -> str:
    # Prepare the content for user message
    content = []
    for i, frame in enumerate(frames, start=1):
        # Convert frame data to image 
        image = convert_to_image(frame)
        # Add each image to the content list
        content.append({"type": "image", "image": image})

    # Add a descriptive text to the content
    if prompt is not None:
        content.append({"type": "text", "text": prompt})

    # Prepare messages for the inference function
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content}
    ]

    # Call the inference function and return its result
    return inference(messages, use_streamer) 


import cv2

from PIL import Image

def convert_to_image(frame, resize=None) -> Image:
    # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)

    return pil_image


import cv2

# Open the video file
cap = cv2.VideoCapture(VIDEO_URL)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video info: {total_frames} total frames, {fps:.2f} FPS")

frame_count = 0
sec = 0
interval_seconds = 3

try:
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # End of video
    
        # Save frame at every full second
        if int(frame_count % (fps * interval_seconds)) == 0:

            # Analyze frame
            print(f"Time {sec // 60:02}:{sec % 60:02}")
            results = get_analysis(system_prompt_text, [frame], "This is a live feed image from the pool camera.", False)
            print(results[0])
            
            # Second increment
            sec += interval_seconds
    
        # Frame increment 
        frame_count += 1
finally:
    # Release the video capture object
    cap.release()


# This prompt template forces the model into the desired persona and output structure for an image.
emergency_level_system_prompt_text = """You are an intelligent security agent responsible for protecting people, homes, and shops from threats. 
You will analyze visual input and describe whether a dangerous or suspicious situation is present. 
Your analysis should be logical, cautious, and actionable.
You will receive concise descriptions of what is seen in images (e.g., "a bear near the backyard", "a person entering the shop after hours", or "a dog walking past the gate"). Based on this, assess the risk level using systematic threat evaluation.

ANALYSIS PROCESS:
1. Identify all entities and their behaviors in the scene
2. Assess potential threat level based on context, time, location, and behavior patterns
3. Consider environmental factors and normal vs. abnormal activity
4. Determine appropriate response level
            
EMERGENCY LEVELS:
- **HIGH**: Immediate physical danger (wild animals near entry points, armed intruders, violent behavior, fire, medical emergencies)
- **MEDIUM**: Suspicious activity requiring attention (unknown persons after hours, unusual behavior patterns, potential security breaches)
- **LOW**: Normal, safe situations requiring no action

RESPONSE FORMAT
You must ONLY provide single digit emergency level which is '2' FOR HIGH, '1' FOR MEDIUM AND '0' FOR LOW

EXAMPLE OUTPUT:
2
1
0

RULES:
If you are unsure, lean toward safety, but never hallucinate.
Be concise but smart."""


from pathlib import Path

class Camera:
    def __init__(self, index, name, video_path, emergency_level=0, notification_types=None):
        self.index = index
        self.name = name
        self.video_path = Path(video_path)
        self.emergency_level = emergency_level
        self.notification_types = notification_types or []
        
    def __repr__(self):
        return f"Camera(index={self.index}, name='{self.name}', emergency_level={self.emergency_level})"

# Simulate cameras with mp4 files
cameras = [
    Camera(0, "Living Room", "/kaggle/input/emergency-dataset/03_bear_break_in.mp4", 0, []),
    Camera(1, "Workshop", "/kaggle/input/emergency-dataset/05_gunslinger.mp4", 0, []),
    Camera(2, "Backyard", "/kaggle/input/emergency-dataset/01_kids_near_pool.mp4", 0, []),
    Camera(3, "Front Door", "/kaggle/input/emergency-dataset/02_crocodile_doorbell.mp4", 0, [])
]


import cv2
import json
import re

# Extract a frame at a specific time from a video
def extract_frame_at_time(video_path, time_seconds):
    # Start capturing frames
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if time_seconds > duration:
        cap.release()
        return None
    
    # Calculate frame number
    frame_number = int(time_seconds * fps)
    
    # Set the position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    else:
        print(f"Error: Could not read frame at {time_seconds}s from {video_path.name}")
        return None

def get_structured_log(camera_name, time, log_text):
    try:
        # Extract JSON content between ```json and ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, log_text, re.DOTALL)
        
        if match:
            json_string = match.group(1).strip()

            # Parse the JSON string to a Python dictionary
            parsed_json = json.loads(json_string)

            # Add camera information to the parsed JSON
            parsed_json['camera_name'] = camera_name
            parsed_json['time'] = time

            return parsed_json
        else:
            print("No JSON block found in the log text")
            return None
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


import cv2
import time

from queue import PriorityQueue
from pathlib import Path
from datetime import datetime

check_counter = 0
priority_queue = PriorityQueue()

def add_to_priority_queue(camera, frame_time, emergency_level, frame):
    minutes, seconds = divmod(frame_time, 60)
    
    record = {
        'camera_name': camera.name,
        'camera_index': camera.index,
        'time': f"{minutes:02}:{seconds:02}",
        'emergency_level': emergency_level,
        'frame': frame
    }
    
    priority_queue.put((-emergency_level, time.time(), record))

def process_priority_queue():
    """
    Process records from priority queue based on emergency levels
    Every 3 checks: process both level 2 and level 1 records
    Other checks: process only level 2 records
    """
    global check_counter
    logs = []
    
    # Process level 2 records immediately
    while not priority_queue.empty():
        priority, time, record = priority_queue.get()
        log = None
        if priority == -2:  # Level 2 emergency
            # Here you would handle the emergency (e.g., notify authorities)
            analysis_result = get_analysis(system_prompt_text, [record['frame']], f"This is a live feed image from the {record['camera_name']}.", False)
            log = get_structured_log(record['camera_name'], record['time'], analysis_result[0])
            print(f"Analyzing {record['camera_name']} Level 2 Emergency at {record['time']}s: {log['reason']}")
            logs.append(log)
            
            # Remove the record from the queue
            priority_queue.task_done()
        elif priority == -1 and check_counter % 3 == 0:  # Level 1 emergency every 3 checks
            # Handle level 1 emergency (e.g., log, notify user)
            analysis_result = get_analysis(system_prompt_text, [record['frame']], f"This is a live feed image from the {record['camera_name']}.", False)
            log = get_structured_log(record['camera_name'], record['time'], analysis_result[0])
            print(f"Analyzing {record['camera_name']} Level 1 Emergency at {record['time']}s: {log['reason']}")
            logs.append(log)
            
            # Remove the record from the queue
            priority_queue.task_done()
            break


    check_counter += 1
    return logs

def simulate_camera_feed(num_captures=20, interval_seconds=3):
    print(f"Starting multi-camera simulation...")

    # print camera details
    print("Cameras in simulation:")
    for camera in cameras:
        print(f"Camera {camera.index}: {camera.name} (Path: {camera.video_path})")

    print(f"Extracting frames every {interval_seconds} seconds")
    
    capture_count = 0
    simulation_log = []

    while capture_count < num_captures:
        for camera in cameras:
            frame = extract_frame_at_time(camera.video_path, interval_seconds*capture_count)
            if frame is None:
                continue
            analysis_result = get_analysis(emergency_level_system_prompt_text, [frame], f"This is a live feed image from the {camera.name}.", False)
            emergency_level = int(analysis_result[0])
            if emergency_level in [1, 2]:  # Assuming 1=warning, 2=emergency
                minutes, seconds = minutes, seconds = divmod(interval_seconds * capture_count, 60)
                print(f"{minutes:02}:{seconds:02} Camera: {camera.name}, Emergency Level: {emergency_level}")
                add_to_priority_queue(camera, interval_seconds*capture_count, emergency_level, frame)

        logs = process_priority_queue()
        if logs:
            simulation_log.extend(logs)
        capture_count += 1
    
    # Save simulation log to file
    log_file = Path("simulation_log.json")
    with log_file.open("w") as f:
        json.dump(simulation_log, f, indent=4)

    print(f"Simulation log saved to {log_file}")
    print(f"\nSimulation completed.")

simulate_camera_feed()

