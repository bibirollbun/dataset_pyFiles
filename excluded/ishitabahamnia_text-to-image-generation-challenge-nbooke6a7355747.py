import pandas as pd
import os

# Define the path to the prompt file
PROMPT_FILE = "/content/DreamLayer-Prompt-Kaggle (1).txt"

# Check if the file exists before attempting to read
if os.path.exists(PROMPT_FILE):
    # Load and clean prompts
    with open(PROMPT_FILE, "r") as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Limit to 49 prompts if there are more
    if len(prompts) > 49:
        prompts = prompts[:49]

    print(f"âœ… Successfully loaded {len(prompts)} prompts.")
else:
    print(f"â�Œ Error: Prompt file not found at {PROMPT_FILE}")
    prompts = []  # Initialize an empty list if the file is not found



# =====================================================
# ğŸ–¼ï¸� COMPLETE DREAMLAYER IMAGE GENERATION PIPELINE
# =====================================================

# Install required packages
!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet
StableDiffusionPipeline.from_pretrained("/kaggle/input/text-to-image-challenge/DreamLayer-Prompt-Kaggle.txt", ...)

import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
import base64
from IPython.display import display, HTML, Image
import json

# =====================================================
# 1ï¸�âƒ£ DEFINE ALL REQUIRED FUNCTIONS
# =====================================================

def create_dreamlayer_config():
    """Create the DreamLayer configuration file"""
    config = {
        "model": {
            "name": "runwayml/stable-diffusion-v1-5",
            "revision": "main",
            "torch_dtype": "float16"
        },
        "scheduler": {
            "name": "PNDMScheduler",
            "num_inference_steps": 25
        },
        "generation": {
            "width": 512,
            "height": 512,
            "guidance_scale": 7.5,
            "seed": 42
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    return config

def save_config(config, output_dir):
    """Save configuration to file"""
    config_path = os.path.join(output_dir, "config-dreamlayer.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

class DreamLayerGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.setup_model()
    
    def setup_model(self):
        """Initialize the diffusion model"""
        print("ğŸš€ Loading Stable Diffusion model...")
        
        try:
            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.float16 if self.config["device"] == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Optimizations
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_memory_efficient_attention()
            
            print("âœ… Model loaded successfully!")
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")
            raise
    
    def generate_image(self, prompt, prompt_id, output_dir):
        """Generate a single image from prompt"""
        # Create output filename
        filename = f"{prompt_id:04d}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"â�­ï¸� Skipping {filename} (already exists)")
            return output_path
        
        # Generate image
        try:
            with torch.autocast(self.device):
                image = self.pipe(
                    prompt,
                    width=self.config["generation"]["width"],
                    height=self.config["generation"]["height"],
                    guidance_scale=self.config["generation"]["guidance_scale"],
                    num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.config["generation"]["seed"] + prompt_id
                    )
                ).images[0]
            
            # Save image
            image.save(output_path)
            return output_path
        except Exception as e:
            print(f"â�Œ Error generating image {prompt_id}: {e}")
            return None

def main():
    # Create output directory
    output_dir = "dreamlayer_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define prompts (from your file)
    prompts = [
        "A man baking and preparing donuts to sell at shop.",
        "A zebra chews a flower in a fenced in field.",
        "A person skiing down a mountain kicking his leg up.",
        "An airplane in route with a cloudy sky behind it.",
        "A white and blue truck parked in the middle of a dirt road.",
        "PIcked peach flowers sit in a vase with water.",
        "Sheep are on a grassy field and one of them is a white and black baby.",
        "A blue rusted train engine sitting on top of rail tracks.",
        "An open laptop computer sitting on top of a wooden table.",
        "A woman twirling an umbrella with flowers on it.",
        "A woman sitting in a restaurant with Mexican food on her plate",
        "A woman holding a purse and a cellphone.",
        "A cat reaching for a knife that has it's blade out.",
        "The man smiles with a slice of pizza while next to a friend.",
        "Night picture of a car parked and some parking lights in the distance.",
        "A professional photograph of a motorcycle rider in the air.",
        "A person putting doughnuts into a bag in a shop.",
        "A very ornately decorated and brightly colored clock.",
        "A young man that is having some wine and something to eat.",
        "A picture of pink bathroom sink and a mirror.",
        "people standing in line beside a food truck",
        "Single sheep in a field looking back at camera.",
        "A few pizza slices next to a couple of bread slices.",
        "Two giraffe standing in a field next to trees.",
        "A person in rubber boots and a rain coat seated on a bench.",
        "A giraffe standing next to a bamboo building.",
        "A large combination pizza with two pieces gone.",
        "A bearded man in a suit eating pizza.",
        "An old yellow train is waiting at the station.",
        "People flying kites in a park next to a lake.",
        "A young man is body surfing and paddling in the water.",
        "A very large stuffed giraffe posed looking out a window.",
        "A dim room with toilet bowls lined along the wall",
        "Two trains parked at a train station as passenger wait to board them.",
        "A slightly made/messy bed against the corner in a white room.",
        "A bench sit in front of a blue and yellow train.",
        "A man is skate boarding down a path and a dog is running by his side.",
        "A man on a long exposure picture riding an electric skateboard.",
        "A person on skis and with poles in the snow and facing the blue sky.",
        "a cake with a section missing sitting next to a burning candle",
        "a woman selling jewelry laid on a blanket on the sidewalk",
        "A group of people on a tennis court.",
        "Two large elephants walking behind a wire fence on green grass.",
        "a woman on a tennis court getting ready to serve the ball",
        "Tabby cat with green eyes wearing a hat",
        "A small, white formica kitchen with a refrigerator, sink and small electrical appliances",
        "A tall clock tower with a large clock on it's face.",
        "A bathroom with a toilet, tub, mirror, window and a shower pole.",
        "A group of people standing on a snow covered hill."
    ]
    
    print(f"ğŸ�¨ Starting generation of {len(prompts)} images...")
    
    # Create configuration
    config = create_dreamlayer_config()
    save_config(config, output_dir)
    
    # Initialize generator
    generator = DreamLayerGenerator(config)
    
    # Generate images
    results = []
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"ğŸ”„ [{i}/{len(prompts)}] Generating: {prompt[:50]}...")
        
        try:
            image_path = generator.generate_image(prompt, i, output_dir)
            if image_path and os.path.exists(image_path):
                results.append({
                    "prompt_id": i,
                    "predicted": prompt,
                    "image_path": image_path,
                    "status": "success"
                })
                successful += 1
                print(f"âœ… Generated: {os.path.basename(image_path)}")
            else:
                results.append({
                    "prompt_id": i,
                    "predicted": prompt,
                    "image_path": f"{i:04d}.png",
                    "status": "failed"
                })
                failed += 1
                print(f"â�Œ Failed to generate image {i}")
        except Exception as e:
            print(f"â�Œ Error generating image {i}: {str(e)}")
            results.append({
                "prompt_id": i,
                "predicted": prompt,
                "image_path": f"{i:04d}.png",
                "status": "error"
            })
            failed += 1
    
    # Create results.csv
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\nğŸ�‰ Generation completed!")
    print(f"ğŸ“Š Summary: {successful} successful, {failed} failed")
    print(f"ğŸ“� Images saved to: {output_dir}")
    print(f"ğŸ“„ Results CSV: {results_csv_path}")
    
    return output_dir, results

# =====================================================
# 2ï¸�âƒ£ DISPLAY FUNCTION
# =====================================================

def display_all_images(output_dir, results):
    """Display all generated images in a beautiful grid"""
    
    # Get list of all generated images
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    image_files.sort()

    if not image_files:
        print("â�Œ No images found in the output directory.")
        return
    
    print(f"ğŸ“¸ Displaying {len(image_files)} generated images:")
    print(f"ğŸ“� Directory: {output_dir}")

    # Create a responsive grid layout
    cols = 4
    html_content = f"""
    <div style='
        display: grid; 
        grid-template-columns: repeat({cols}, 1fr); 
        gap: 15px; 
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 20px 0;
    '>
    """

    for img_name in image_files:
        img_path = os.path.join(output_dir, img_name)
        prompt_id = os.path.splitext(img_name)[0]
        
        # Get prompt text from results
        prompt_text = "Prompt not available"
        for result in results:
            if str(result["prompt_id"]) == prompt_id:
                prompt_text = result["predicted"]
                break
        
        # Truncate long prompts for display
        if len(prompt_text) > 60:
            display_text = prompt_text[:60] + "..."
        else:
            display_text = prompt_text

        if os.path.exists(img_path):
            try:
                # Get file size for display
                file_size = os.path.getsize(img_path) // 1024  # Size in KB
                
                # Embed image using base64
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                
                html_content += f"""
                <div style='
                    border: 2px solid #e0e0e0; 
                    padding: 10px; 
                    text-align: center; 
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                ' onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                    <img src='data:image/png;base64,{encoded_string}' 
                         style='
                             width: 100%; 
                             height: 200px; 
                             object-fit: cover;
                             border-radius: 4px;
                         '
                         alt='{prompt_id}'>
                    <div style='margin-top: 8px;'>
                        <div style='font-size: 12px; font-weight: bold; color: #2E86AB;'>ğŸ†” {prompt_id}</div>
                        <div style='font-size: 10px; color: #666; margin: 5px 0; line-height: 1.2;'>
                            {display_text}
                        </div>
                        <div style='font-size: 9px; color: #888;'>ğŸ“� {file_size} KB</div>
                    </div>
                </div>
                """
            except Exception as e:
                html_content += f"""
                <div style='
                    border: 2px solid #ffcccc; 
                    padding: 10px; 
                    text-align: center; 
                    background: #fff5f5;
                    border-radius: 8px;
                '>
                    <div style='color: #d63031; font-size: 24px;'>â�Œ</div>
                    <div style='font-size: 12px; font-weight: bold;'>Error Loading</div>
                    <div style='font-size: 10px; color: #666;'>ID: {prompt_id}</div>
                </div>
                """
        else:
            html_content += f"""
            <div style='
                border: 2px dashed #ccc; 
                padding: 10px; 
                text-align: center; 
                background: #f9f9f9;
                border-radius: 8px;
            '>
                <div style='color: #999; font-size: 24px;'>ğŸ“­</div>
                <div style='font-size: 12px; font-weight: bold;'>Missing Image</div>
                <div style='font-size: 10px; color: #666;'>ID: {prompt_id}</div>
            </div>
            """

    html_content += "</div>"
    
    # Add summary statistics
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in image_files if os.path.exists(os.path.join(output_dir, f))) // 1024
    
    summary_html = f"""
    <div style='
        background: #2E86AB; 
        color: white; 
        padding: 15px; 
        border-radius: 8px; 
        margin: 20px 0;
        text-align: center;
    '>
        <h3 style='margin: 0;'>ğŸ“Š Image Generation Summary</h3>
        <div style='display: flex; justify-content: center; gap: 30px; margin-top: 10px;'>
            <div>
                <div style='font-size: 24px; font-weight: bold;'>{len(image_files)}</div>
                <div style='font-size: 12px;'>Total Images</div>
            </div>
            <div>
                <div style='font-size: 24px; font-weight: bold;'>{total_size} KB</div>
                <div style='font-size: 12px;'>Total Size</div>
            </div>
            <div>
                <div style='font-size: 24px; font-weight: bold;'>{len([f for f in image_files if os.path.exists(os.path.join(output_dir, f))])}</div>
                <div style='font-size: 12px;'>Files Found</div>
            </div>
        </div>
    </div>
    """
    
    display(HTML(summary_html))
    display(HTML(html_content))

# =====================================================
# 3ï¸�âƒ£ RUN THE COMPLETE PIPELINE
# =====================================================

print("ğŸš€ Starting DreamLayer Image Generation Pipeline...")
print("=" * 60)

# Run the main function to generate images
output_directory, results = main()

# Display all generated images
print("\n" + "=" * 60)
print("ğŸ–¼ï¸� Displaying Generated Images...")
print("=" * 60)

display_all_images(output_directory, results)

print("\nğŸ�‰ Pipeline completed successfully!")
print("ğŸ“‹ Next steps:")
print("   1. Check the 'dreamlayer_output' folder for all images")
print("   2. Upload results.csv to Kaggle")
print("   3. Share your notebook with organizers")


# =====================================================
# ğŸ–¼ï¸� DREAMLAYER IMAGE GENERATION (ALL-IN-ONE)
# =====================================================

!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet

import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime

# -------------------------------
# 1ï¸�âƒ£ Functions & Classes
# -------------------------------

def create_dreamlayer_config():
    return {
        "model": {
            "name": "runwayml/stable-diffusion-v1-5",
            "revision": "main",
            "torch_dtype": "float16"
        },
        "scheduler": {
            "num_inference_steps": 25
        },
        "generation": {
            "width": 512,
            "height": 512,
            "guidance_scale": 7.5,
            "seed": 42
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

def save_config(config, output_dir):
    import json
    config_path = os.path.join(output_dir, "config-dreamlayer.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

class DreamLayerGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.setup_model()
    
    def setup_model(self):
        print("ğŸš€ Loading Stable Diffusion model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config["model"]["name"],
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        if self.device=="cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_memory_efficient_attention()
        print("âœ… Model loaded successfully!")
    
    def generate_image(self, prompt, prompt_id, output_dir):
        from PIL import Image
        output_path = os.path.join(output_dir, f"{prompt_id:04d}.png")
        if os.path.exists(output_path):
            return output_path
        with torch.autocast(self.device):
            img = self.pipe(
                prompt,
                width=self.config["generation"]["width"],
                height=self.config["generation"]["height"],
                guidance_scale=self.config["generation"]["guidance_scale"],
                num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                generator=torch.Generator(device=self.device).manual_seed(
                    self.config["generation"]["seed"] + prompt_id
                )
            ).images[0]
            img.save(output_path)
        return output_path

# -------------------------------
# 2ï¸�âƒ£ Main execution
# -------------------------------

def main():
    output_dir = "dreamlayer_output"
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = [
        # 49 prompts copied from your list
        "A man baking and preparing donuts to sell at shop.",
        "A zebra chews a flower in a fenced in field.",
        "A person skiing down a mountain kicking his leg up.",
        "An airplane in route with a cloudy sky behind it.",
        "A white and blue truck parked in the middle of a dirt road.",
        "PIcked peach flowers sit in a vase with water.",
        "Sheep are on a grassy field and one of them is a white and black baby.",
        "A blue rusted train engine sitting on top of rail tracks.",
        "An open laptop computer sitting on top of a wooden table.",
        "A woman twirling an umbrella with flowers on it.",
        "A woman sitting in a restaurant with Mexican food on her plate",
        "A woman holding a purse and a cellphone.",
        "A cat reaching for a knife that has it's blade out.",
        "The man smiles with a slice of pizza while next to a friend.",
        "Night picture of a car parked and some parking lights in the distance.",
        "A professional photograph of a motorcycle rider in the air.",
        "A person putting doughnuts into a bag in a shop.",
        "A very ornately decorated and brightly colored clock.",
        "A young man that is having some wine and something to eat.",
        "A picture of pink bathroom sink and a mirror.",
        "people standing in line beside a food truck",
        "Single sheep in a field looking back at camera.",
        "A few pizza slices next to a couple of bread slices.",
        "Two giraffe standing in a field next to trees.",
        "A person in rubber boots and a rain coat seated on a bench.",
        "A giraffe standing next to a bamboo building.",
        "A large combination pizza with two pieces gone.",
        "A bearded man in a suit eating pizza.",
        "An old yellow train is waiting at the station.",
        "People flying kites in a park next to a lake.",
        "A young man is body surfing and paddling in the water.",
        "A very large stuffed giraffe posed looking out a window.",
        "A dim room with toilet bowls lined along the wall",
        "Two trains parked at a train station as passenger wait to board them.",
        "A slightly made/messy bed against the corner in a white room.",
        "A bench sit in front of a blue and yellow train.",
        "A man is skate boarding down a path and a dog is running by his side.",
        "A man on a long exposure picture riding an electric skateboard.",
        "A person on skis and with poles in the snow and facing the blue sky.",
        "a cake with a section missing sitting next to a burning candle",
        "a woman selling jewelry laid on a blanket on the sidewalk",
        "A group of people on a tennis court.",
        "Two large elephants walking behind a wire fence on green grass.",
        "a woman on a tennis court getting ready to serve the ball",
        "Tabby cat with green eyes wearing a hat",
        "A small, white formica kitchen with a refrigerator, sink and small electrical appliances",
        "A tall clock tower with a large clock on it's face.",
        "A bathroom with a toilet, tub, mirror, window and a shower pole.",
        "A group of people standing on a snow covered hill."
    ]
    
    print(f"ğŸ�¨ Starting generation of {len(prompts)} images...")
    
    config = create_dreamlayer_config()
    save_config(config, output_dir)
    
    generator = DreamLayerGenerator(config)
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"ğŸ”„ [{i}/{len(prompts)}] {prompt[:50]}...")
        try:
            path = generator.generate_image(prompt, i, output_dir)
            results.append({"prompt_id": i, "prompt": prompt, "image_path": path})
        except Exception as e:
            print(f"â�Œ Failed {i}: {str(e)}")
            results.append({"prompt_id": i, "prompt": prompt, "image_path": None})
    
    # Save results CSV for Kaggle submission
    pd.DataFrame(results)[["prompt_id","prompt"]].to_csv(os.path.join(output_dir,"submission.csv"), index=False)
    print(f"\nğŸ�‰ Completed! Images and submission.csv saved in {output_dir}")
    return output_dir

# Run the pipeline
output_directory = main()



# =====================================================
# ğŸ”� COMPLETE EVALUATION PIPELINE WITH F1 SCORING
# Author: Ishita
# Features: Object detection, F1 scoring, Confusion matrices
# =====================================================

!pip install ultralytics scikit-learn pandas numpy matplotlib seaborn plotly --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report, precision_recall_fscore_support
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO
import os
from tqdm import tqdm

# =====================================================
# 1ï¸�âƒ£ SETUP EVALUATION SYSTEM
# =====================================================

class DreamLayerEvaluator:
    def __init__(self):
        # Load YOLO model for object detection
        print("ğŸš€ Loading YOLO model for object detection...")
        self.model = YOLO('yolov8n.pt')
        
        # Object mapping between YOLO classes and our expected objects
        self.object_mapping = {
            'person': ['man', 'woman', 'person', 'people'],
            'cat': ['cat'],
            'dog': ['dog'],
            'sheep': ['sheep', 'lamb', 'baby sheep'],
            'zebra': ['zebra'],
            'giraffe': ['giraffe'],
            'elephant': ['elephant'],
            'car': ['car'],
            'truck': ['truck'],
            'train': ['train'],
            'airplane': ['airplane', 'plane'],
            'motorcycle': ['motorcycle', 'motorbike'],
            'bus': ['bus'],
            'umbrella': ['umbrella'],
            'handbag': ['purse', 'handbag'],
            'cell phone': ['cellphone', 'phone', 'mobile phone'],
            'knife': ['knife'],
            'pizza': ['pizza'],
            'donut': ['donut', 'doughnut'],
            'cake': ['cake'],
            'sink': ['sink'],
            'refrigerator': ['refrigerator', 'fridge'],
            'toilet': ['toilet'],
            'clock': ['clock'],
            'vase': ['vase'],
            'bench': ['bench'],
            'skis': ['skis'],
            'snowboard': ['snowboard'],
            'sports ball': ['ball', 'tennis ball'],
            'kite': ['kite'],
            'teddy bear': ['stuffed giraffe', 'teddy bear', 'stuffed animal'],
            'wine glass': ['wine glass', 'wine'],
            'laptop': ['laptop', 'computer'],
            'book': ['book'],
            'chair': ['chair'],
            'dining table': ['table', 'dining table'],
            'oven': ['oven'],
            'bed': ['bed'],
            'tv': ['tv', 'television'],
            'cup': ['cup', 'glass'],
            'fork': ['fork'],
            'spoon': ['spoon'],
            'bowl': ['bowl'],
            'banana': ['banana'],
            'apple': ['apple'],
            'sandwich': ['sandwich'],
            'orange': ['orange'],
            'broccoli': ['broccoli'],
            'carrot': ['carrot'],
            'hot dog': ['hot dog'],
            'backpack': ['backpack'],
            'suitcase': ['suitcase'],
            'frisbee': ['frisbee'],
            'skateboard': ['skateboard'],
            'surfboard': ['surfboard'],
            'tennis racket': ['tennis racket'],
            'bottle': ['bottle', 'water bottle'],
            'flower': ['flower', 'flowers']
        }
        
        print("âœ… YOLO model loaded successfully!")
    
    def extract_expected_objects(self, prompt):
        """Extract expected objects from prompt text"""
        prompt_lower = prompt.lower()
        expected_objects = []
        
        for yolo_class, our_objects in self.object_mapping.items():
            for obj in our_objects:
                if obj in prompt_lower:
                    expected_objects.append(obj)
                    break  # Only add once per object group
        
        # Handle special cases
        if 'baby' in prompt_lower and 'sheep' in prompt_lower:
            expected_objects.append('baby sheep')
        if 'stuffed' in prompt_lower and 'giraffe' in prompt_lower:
            expected_objects.append('stuffed giraffe')
        if 'parking' in prompt_lower and 'light' in prompt_lower:
            expected_objects.append('parking light')
        if 'tennis court' in prompt_lower:
            expected_objects.extend(['tennis court', 'court'])
        
        return list(set(expected_objects))  # Remove duplicates
    
    def map_detected_to_expected(self, detected_objects):
        """Map YOLO detected objects to our expected object names"""
        mapped_objects = []
        for detected in detected_objects:
            for yolo_obj, our_objects in self.object_mapping.items():
                if detected == yolo_obj:
                    mapped_objects.extend(our_objects)
        return list(set(mapped_objects))
    
    def detect_objects_in_image(self, image_path):
        """Detect objects in an image using YOLO"""
        if not os.path.exists(image_path):
            return []
        
        try:
            results = self.model(image_path)
            detected_objects = []
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    object_name = result.names[class_id]
                    detected_objects.append(object_name)
            
            return detected_objects
        except Exception as e:
            print(f"â�Œ Error detecting objects in {image_path}: {e}")
            return []
    
    def calculate_f1_per_prompt(self, expected_objects, image_path):
        """Calculate F1 score for a single prompt-image pair"""
        if not os.path.exists(image_path):
            return 0.0, expected_objects, []
        
        try:
            # Detect objects in image
            detected_objects = self.detect_objects_in_image(image_path)
            
            # Map detected objects to our expected names
            mapped_detected = self.map_detected_to_expected(detected_objects)
            
            # Calculate F1 score
            if len(expected_objects) > 0:
                y_true = [1] * len(expected_objects)  # All expected objects should be present
                y_pred = [1 if obj in mapped_detected else 0 for obj in expected_objects]
                
                # Calculate precision, recall, f1
                tp = sum(y_pred)
                fp = len([obj for obj in mapped_detected if obj not in expected_objects])
                fn = len(expected_objects) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1 = 0.0
                precision = 0.0
                recall = 0.0
                tp = 0
                fp = 0
                fn = 0
            
            return f1, expected_objects, mapped_detected, precision, recall, tp, fp, fn
            
        except Exception as e:
            print(f"â�Œ Error evaluating {image_path}: {e}")
            return 0.0, expected_objects, [], 0.0, 0.0, 0, 0, 0
    
    def evaluate_all_prompts(self, prompt_mapping, output_dir):
        """Evaluate F1 scores for all prompts"""
        print("\nğŸ§® Starting evaluation of all prompts...")
        
        evaluation_results = []
        
        for prompt_id, prompt_data in tqdm(prompt_mapping.items(), desc="Evaluating prompts"):
            image_path = os.path.join(output_dir, prompt_data['image_name'])
            
            # Extract expected objects from prompt
            expected_objects = self.extract_expected_objects(prompt_data['prompt'])
            
            if os.path.exists(image_path):
                # Calculate F1 score
                f1, expected, detected, precision, recall, tp, fp, fn = self.calculate_f1_per_prompt(
                    expected_objects, image_path
                )
                
                evaluation_results.append({
                    'prompt_id': prompt_id,
                    'prompt': prompt_data['prompt'],
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'expected_objects': expected,
                    'detected_objects': detected,
                    'image_path': image_path,
                    'status': 'evaluated'
                })
            else:
                evaluation_results.append({
                    'prompt_id': prompt_id,
                    'prompt': prompt_data['prompt'],
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': len(expected_objects),
                    'expected_objects': expected_objects,
                    'detected_objects': [],
                    'image_path': image_path,
                    'status': 'missing_image'
                })
        
        print(f"âœ… Evaluation completed! Processed {len(evaluation_results)} prompts")
        return evaluation_results

# =====================================================
# 2ï¸�âƒ£ RUN THE EVALUATION
# =====================================================

# Initialize evaluator
evaluator = DreamLayerEvaluator()

# Run evaluation on your generated images
print("ğŸ”� Starting comprehensive evaluation...")
evaluation_results = evaluator.evaluate_all_prompts(prompt_mapping, OUTPUT_DIR)

# Display evaluation summary
successful_evals = len([r for r in evaluation_results if r['status'] == 'evaluated'])
missing_images = len([r for r in evaluation_results if r['status'] == 'missing_image'])
avg_f1 = np.mean([r['f1_score'] for r in evaluation_results if r['status'] == 'evaluated'])

print(f"\nğŸ“Š EVALUATION SUMMARY:")
print(f"   âœ… Successful evaluations: {successful_evals}")
print(f"   â�Œ Missing images: {missing_images}")
print(f"   ğŸ“ˆ Average F1 Score: {avg_f1:.4f}")

# =====================================================
# 3ï¸�âƒ£ CONFUSION MATRIX ANALYSIS (Now with defined evaluation_results)
# =====================================================

def prepare_confusion_matrix_data(evaluation_results):
    """Prepare data for multi-label confusion matrix analysis"""
    
    # Collect all unique objects across all evaluations
    all_objects = set()
    for res in evaluation_results:
        all_objects.update(res['expected_objects'])
        all_objects.update(res['detected_objects'])
    
    all_objects = sorted(list(all_objects))
    print(f"ğŸ“Š Found {len(all_objects)} unique objects across all prompts")
    
    # Prepare ground truth and predictions for each prompt
    y_true_list = []
    y_pred_list = []
    prompt_ids = []
    
    for res in evaluation_results:
        expected = res['expected_objects']
        detected = res['detected_objects']
        
        # Create binary vectors for expected and detected objects
        y_true_prompt = [1 if obj in expected else 0 for obj in all_objects]
        y_pred_prompt = [1 if obj in detected else 0 for obj in all_objects]
        
        y_true_list.append(y_true_prompt)
        y_pred_list.append(y_pred_prompt)
        prompt_ids.append(res['prompt_id'])
    
    # Convert to numpy arrays
    y_true_np = np.array(y_true_list)
    y_pred_np = np.array(y_pred_list)
    
    return y_true_np, y_pred_np, all_objects, prompt_ids

# Prepare the data for confusion matrix
y_true, y_pred, all_objects, prompt_ids = prepare_confusion_matrix_data(evaluation_results)

def calculate_detailed_metrics(y_true, y_pred, all_objects):
    """Calculate detailed confusion matrices and metrics for each object"""
    
    # Calculate multi-label confusion matrices
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    
    # Calculate metrics for each object
    object_metrics = []
    
    for i, obj in enumerate(all_objects):
        tn, fp, fn, tp = conf_matrices[i].ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Calculate support (how many times object was expected)
        support = np.sum(y_true[:, i])
        
        object_metrics.append({
            'object': obj,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': support,
            'detection_rate': tp / support if support > 0 else 0
        })
    
    return object_metrics, conf_matrices

# Calculate detailed metrics
object_metrics, conf_matrices = calculate_detailed_metrics(y_true, y_pred, all_objects)

# =====================================================
# 4ï¸�âƒ£ DISPLAY RESULTS
# =====================================================

def display_evaluation_results(evaluation_results, object_metrics):
    """Display comprehensive evaluation results"""
    
    # Convert to DataFrame for easier analysis
    eval_df = pd.DataFrame(evaluation_results)
    
    print("ğŸ“ˆ OVERALL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Prompts Evaluated: {len(eval_df)}")
    print(f"Average F1 Score: {eval_df['f1_score'].mean():.4f}")
    print(f"Average Precision: {eval_df['precision'].mean():.4f}")
    print(f"Average Recall: {eval_df['recall'].mean():.4f}")
    print(f"Total True Positives: {eval_df['true_positives'].sum()}")
    print(f"Total False Positives: {eval_df['false_positives'].sum()}")
    print(f"Total False Negatives: {eval_df['false_negatives'].sum()}")
    
    # Show top and bottom performing prompts
    print(f"\nğŸ�† TOP 5 BEST PERFORMING PROMPTS (by F1 Score):")
    top_prompts = eval_df.nlargest(5, 'f1_score')[['prompt_id', 'prompt', 'f1_score', 'precision', 'recall']]
    for _, row in top_prompts.iterrows():
        print(f"  {row['prompt_id']}: F1={row['f1_score']:.3f} | {row['prompt'][:60]}...")
    
    print(f"\nğŸ“‰ TOP 5 WORST PERFORMING PROMPTS (by F1 Score):")
    worst_prompts = eval_df.nsmallest(5, 'f1_score')[['prompt_id', 'prompt', 'f1_score', 'precision', 'recall']]
    for _, row in worst_prompts.iterrows():
        print(f"  {row['prompt_id']}: F1={row['f1_score']:.3f} | {row['prompt'][:60]}...")
    
    # Show object-level performance
    objects_df = pd.DataFrame(object_metrics)
    objects_with_support = objects_df[objects_df['support'] > 0]
    
    print(f"\nğŸ�¯ OBJECT-LEVEL PERFORMANCE (Objects with support > 0):")
    print(f"Total objects with support: {len(objects_with_support)}")
    print(f"Average F1 Score: {objects_with_support['f1_score'].mean():.4f}")
    
    print(f"\nğŸ”� BEST DETECTED OBJECTS:")
    best_objects = objects_with_support.nlargest(5, 'f1_score')[['object', 'f1_score', 'precision', 'recall', 'support']]
    print(best_objects.round(3))
    
    print(f"\nğŸ”» WORST DETECTED OBJECTS:")
    worst_objects = objects_with_support.nsmallest(5, 'f1_score')[['object', 'f1_score', 'precision', 'recall', 'support']]
    print(worst_objects.round(3))

# Display results
display_evaluation_results(evaluation_results, object_metrics)

# =====================================================
# 5ï¸�âƒ£ VISUALIZATION
# =====================================================

def plot_evaluation_visualizations(evaluation_results, object_metrics):
    """Create visualizations for evaluation results"""
    
    eval_df = pd.DataFrame(evaluation_results)
    objects_df = pd.DataFrame(object_metrics)
    objects_with_support = objects_df[objects_df['support'] > 0]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DreamLayer Evaluation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: F1 Score Distribution
    axes[0, 0].hist(eval_df['f1_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(eval_df['f1_score'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {eval_df["f1_score"].mean():.3f}')
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of F1 Scores across Prompts')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision vs Recall scatter
    scatter = axes[0, 1].scatter(eval_df['recall'], eval_df['precision'], 
                                c=eval_df['f1_score'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall (Color = F1 Score)')
    plt.colorbar(scatter, ax=axes[0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Object F1 Scores (top 15 by support)
    top_objects = objects_with_support.nlargest(15, 'support')
    axes[1, 0].barh(top_objects['object'], top_objects['f1_score'], color='lightgreen')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_title('F1 Scores for Top 15 Objects by Support')
    axes[1, 0].set_xlim(0, 1)
    
    # Plot 4: Detection performance
    performance_counts = [
        len(eval_df[eval_df['f1_score'] > 0.7]),
        len(eval_df[(eval_df['f1_score'] > 0.4) & (eval_df['f1_score'] <= 0.7)]),
        len(eval_df[eval_df['f1_score'] <= 0.4])
    ]
    labels = ['Good (F1 > 0.7)', 'Medium (0.4 < F1 â‰¤ 0.7)', 'Poor (F1 â‰¤ 0.4)']
    colors = ['lightgreen', 'gold', 'lightcoral']
    axes[1, 1].pie(performance_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Prompt Performance Distribution')
    
    plt.tight_layout()
    plt.show()

# Create visualizations
plot_evaluation_visualizations(evaluation_results, object_metrics)

# =====================================================
# 6ï¸�âƒ£ SAVE EVALUATION RESULTS
# =====================================================

def save_evaluation_results(evaluation_results, object_metrics, output_dir):
    """Save all evaluation results to files"""
    
    # Save prompt-level results
    prompt_results_df = pd.DataFrame(evaluation_results)
    prompt_results_file = os.path.join(output_dir, 'prompt_evaluation_results.csv')
    prompt_results_df.to_csv(prompt_results_file, index=False)
    
    # Save object-level results
    object_results_df = pd.DataFrame(object_metrics)
    object_results_file = os.path.join(output_dir, 'object_evaluation_results.csv')
    object_results_df.to_csv(object_results_file, index=False)
    
    # Create summary
    summary = {
        'total_prompts': len(evaluation_results),
        'average_f1_score': prompt_results_df['f1_score'].mean(),
        'average_precision': prompt_results_df['precision'].mean(),
        'average_recall': prompt_results_df['recall'].mean(),
        'total_objects_tracked': len(object_metrics),
        'objects_with_support': len(object_results_df[object_results_df['support'] > 0]),
        'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nğŸ’¾ Evaluation results saved:")
    print(f"   ğŸ“„ Prompt-level results: {prompt_results_file}")
    print(f"   ğŸ“„ Object-level results: {object_results_file}")
    print(f"   ğŸ“„ Evaluation summary: {summary_file}")
    
    return prompt_results_file, object_results_file, summary_file

# Save results
prompt_file, object_file, summary_file = save_evaluation_results(evaluation_results, object_metrics, OUTPUT_DIR)

print("\nâœ… EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
print("ğŸ“Š You now have comprehensive F1 scores and confusion matrix analysis for your DreamLayer submission!")


display(df_results)


import os
from IPython.display import display, HTML, Image

OUTPUT_DIR = "/kaggle/working/dreamlayer_output"

# Get list of all generated images (excluding the summary image if it exists)
image_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') and f != 'SUMMARY.png']
image_files.sort() # Sort to maintain order

if not image_files:
    print("No images found in the output directory.")
else:
    print(f"Displaying {len(image_files)} generated images in a grid:")

    # Define number of columns for the grid
    cols = 4
    html_content = "<div style='display: grid; grid-template-columns: repeat(" + str(cols) + ", 1fr); gap: 10px;'>"

    for img_name in image_files:
        img_path = os.path.join(OUTPUT_DIR, img_name)
        prompt_id = os.path.splitext(img_name)[0] # Extract prompt_id from filename

        # Optional: Get prompt text if available (requires prompts or results df)
        # Assuming prompts list is available from previous cells
        prompt_text = "N/A"
        if 'prompts' in globals() and len(prompts) >= int(prompt_id):
             # Adjust index as prompt_id is 1-based, list is 0-based
             prompt_text = prompts[int(prompt_id) - 1]
             # Truncate long prompts for display
             if len(prompt_text) > 50:
                 prompt_text = prompt_text[:50] + "..."
        elif 'df_results' in globals():
             # Look up in df_results if available
             prompt_row = df_results[df_results['prompt_id'] == prompt_id]
             if not prompt_row.empty:
                  prompt_text = prompt_row.iloc[0]['prompt']
                  if len(prompt_text) > 50:
                      prompt_text = prompt_text[:50] + "..."


        if os.path.exists(img_path):
            # Embed image using base64 for direct display in HTML
            try:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                img_tag = f"<img src='data:image/png;base64,{encoded_string}' style='width: 100%; height: auto;'>"
                html_content += f"""
                <div style='border: 1px solid #ccc; padding: 5px; text-align: center;'>
                    {img_tag}
                    <p style='font-size: 10px; margin-top: 5px;'><strong>ID:</strong> {prompt_id}</p>
                    <p style='font-size: 10px;'><strong>Prompt:</strong> {prompt_text}</p>
                </div>
                """
            except Exception as e:
                 print(f"Could not embed image {img_name}: {e}")
                 # Add a placeholder div if embedding fails
                 html_content += f"""
                 <div style='border: 1px solid #ccc; padding: 5px; text-align: center;'>
                     <p>Error loading image {img_name}</p>
                     <p style='font-size: 10px; margin-top: 5px;'><strong>ID:</strong> {prompt_id}</p>
                     <p style='font-size: 10px;'><strong>Prompt:</strong> {prompt_text}</p>
                 </div>
                 """


    html_content += "</div>"
    display(HTML(html_content))


from IPython.display import Image, display
import os

OUTPUT_DIR = "/kaggle/working/dreamlayer_output" # Make sure this matches the output directory in the generation cell

print("\nğŸ“¸ Sample Generated Images:\n")
# Display up to 3 images, checking if they exist
for i in range(min(3, len(prompts))): # Use the length of prompts to determine how many images to try and display
     img_name = f"{i+1:04d}.png"
     img_path = os.path.join(OUTPUT_DIR, img_name)
     if os.path.exists(img_path):
         display(Image(filename=img_path))
     else:
         print(f"Image {img_name} not found.")

print(f"\nğŸ“‚ All files saved under: {OUTPUT_DIR}")


!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet

import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime
from IPython.display import Image, display # Import display and Image

# =====================================================
# Configuration
# =====================================================
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")
MODEL_NAME = "stabilityai/sd-turbo"
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 5
GUIDANCE_SCALE = 0.0


os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Load Model
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
pipe = pipe.to(device)

# =====================================================
# Generate Images
# =====================================================
results = []

# Check for existing images to resume
existing_images = set([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])

for i, prompt in enumerate(tqdm(prompts, desc="Generating Images")):
    img_name = f"{i+1:04d}.png"
    img_path = os.path.join(OUTPUT_DIR, img_name)
    status = "success"

    result_entry = {
        "prompt_id": f"{i+1:04d}",
        "prompt": prompt,
        "image_path": img_name,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    results.append(result_entry)

    if img_name in existing_images:
        print(f"Skipping {img_name}, already exists.")
        results[-1]['status'] = 'skipped'
        continue

    try:
        image = pipe(prompt, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1],
                     num_inference_steps=NUM_INFERENCE_STEPS,
                     guidance_scale=GUIDANCE_SCALE).images[0]

        image.save(img_path)
    except Exception as e:
        print(f"Error generating image for prompt ID {i+1:04d}: {e}")
        results[-1]['status'] = 'failed'
        results[-1]['error'] = str(e)

# =====================================================
# Save Results CSV
# =====================================================
df_results = pd.DataFrame(results)

successful = (df_results["status"] == "success").sum()
failed = (df_results["status"] == "failed").sum()
skipped = (df_results["status"] == "skipped").sum()

summary_row = pd.DataFrame([{
    "prompt_id": "SUMMARY",
    "prompt": f"âœ… {successful} success | â�Œ {failed} failed | â�­ï¸� {skipped} skipped",
    "image_path": "",
    "status": "summary",
    "timestamp": datetime.now().isoformat()
}])

df_results = pd.concat([df_results, summary_row], ignore_index=True)

df_results.to_csv(RESULTS_FILE, index=False)
print("âœ… results.csv created!")

# =====================================================
# Show Sample Outputs
# =====================================================
print("\nğŸ“¸ Sample Generated Images:\n")
for i in range(min(3, len(results))):
     img_name = f"{i+1:04d}.png"
     img_path = os.path.join(OUTPUT_DIR, img_name)
     if os.path.exists(img_path):
         display(Image(filename=img_path))
     else:
         print(f"Image {img_name} not found.")

print(f"\nğŸ“‚ All files saved under: {OUTPUT_DIR}")


# =====================================================
# ğŸ–¼ï¸� DREAMLAYER COMPLETE PIPELINE: Display & Evaluate
# Author: Ishita
# Features: Sequential display, F1 evaluation
# =====================================================

# First, install all required packages
!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet
!pip install kaggle scikit-learn matplotlib seaborn --quiet
!pip install ultralytics opencv-python --quiet

!pip install ultralytics scikit-learn pandas numpy matplotlib seaborn plotly --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, classification_report, precision_recall_fscore_support
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO
import os
from tqdm import tqdm

# =====================================================
# 1ï¸�âƒ£ SETUP EVALUATION SYSTEM
# =====================================================

class DreamLayerEvaluator:
    def __init__(self):
        # Load YOLO model for object detection
        print("ğŸš€ Loading YOLO model for object detection...")
        self.model = YOLO('yolov8n.pt')

        # Object mapping between YOLO classes and our expected objects
        self.object_mapping = {
            'person': ['man', 'woman', 'person', 'people'],
            'cat': ['cat'],
            'dog': ['dog'],
            'sheep': ['sheep', 'lamb', 'baby sheep'],
            'zebra': ['zebra'],
            'giraffe': ['giraffe'],
            'elephant': ['elephant'],
            'car': ['car'],
            'truck': ['truck'],
            'train': ['train'],
            'airplane': ['airplane', 'plane'],
            'motorcycle': ['motorcycle', 'motorbike'],
            'bus': ['bus'],
            'umbrella': ['umbrella'],
            'handbag': ['purse', 'handbag'],
            'cell phone': ['cellphone', 'phone', 'mobile phone'],
            'knife': ['knife'],
            'pizza': ['pizza'],
            'donut': ['donut', 'doughnut'],
            'cake': ['cake'],
            'sink': ['sink'],
            'refrigerator': ['refrigerator', 'fridge'],
            'toilet': ['toilet'],
            'clock': ['clock'],
            'vase': ['vase'],
            'bench': ['bench'],
            'skis': ['skis'],
            'snowboard': ['snowboard'],
            'sports ball': ['ball', 'tennis ball'],
            'kite': ['kite'],
            'teddy bear': ['stuffed giraffe', 'teddy bear', 'stuffed animal'],
            'wine glass': ['wine glass', 'wine'],
            'laptop': ['laptop', 'computer'],
            'book': ['book'],
            'chair': ['chair'],
            'dining table': ['table', 'dining table'],
            'oven': ['oven'],
            'bed': ['bed'],
            'tv': ['tv', 'television'],
            'cup': ['cup', 'glass'],
            'fork': ['fork'],
            'spoon': ['spoon'],
            'bowl': ['bowl'],
            'banana': ['banana'],
            'apple': ['apple'],
            'sandwich': ['sandwich'],
            'orange': ['orange'],
            'broccoli': ['broccoli'],
            'carrot': ['carrot'],
            'hot dog': ['hot dog'],
            'backpack': ['backpack'],
            'suitcase': ['suitcase'],
            'frisbee': ['frisbee'],
            'skateboard': ['skateboard'],
            'surfboard': ['surfboard'],
            'tennis racket': ['tennis racket'],
            'bottle': ['bottle', 'water bottle'],
            'flower': ['flower', 'flowers']
        }

        print("âœ… YOLO model loaded successfully!")

    def extract_expected_objects(self, prompt):
        """Extract expected objects from prompt text"""
        prompt_lower = prompt.lower()
        expected_objects = []

        for yolo_class, our_objects in self.object_mapping.items():
            for obj in our_objects:
                if obj in prompt_lower:
                    expected_objects.append(obj)
                    break  # Only add once per object group

        # Handle special cases
        if 'baby' in prompt_lower and 'sheep' in prompt_lower:
            expected_objects.append('baby sheep')
        if 'stuffed' in prompt_lower and 'giraffe' in prompt_lower:
            expected_objects.append('stuffed giraffe')
        if 'parking' in prompt_lower and 'light' in prompt_lower:
            expected_objects.append('parking light')
        if 'tennis court' in prompt_lower:
            expected_objects.extend(['tennis court', 'court'])

        return list(set(expected_objects))  # Remove duplicates

    def map_detected_to_expected(self, detected_objects):
        """Map YOLO detected objects to our expected object names"""
        mapped_objects = []
        for detected in detected_objects:
            for yolo_obj, our_objects in self.object_mapping.items():
                if detected == yolo_obj:
                    mapped_objects.extend(our_objects)
        return list(set(mapped_objects))

    def detect_objects_in_image(self, image_path):
        """Detect objects in an image using YOLO"""
        if not os.path.exists(image_path):
            return []

        try:
            results = self.model(image_path)
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    object_name = result.names[class_id]
                    detected_objects.append(object_name)

            return detected_objects
        except Exception as e:
            print(f"â�Œ Error detecting objects in {image_path}: {e}")
            return []

    def calculate_f1_per_prompt(self, expected_objects, image_path):
        """Calculate F1 score for a single prompt-image pair"""
        if not os.path.exists(image_path):
            return 0.0, expected_objects, []

        try:
            # Detect objects in image
            detected_objects = self.detect_objects_in_image(image_path)

            # Map detected objects to our expected names
            mapped_detected = self.map_detected_to_expected(detected_objects)

            # Calculate F1 score
            if len(expected_objects) > 0:
                y_true = [1] * len(expected_objects)  # All expected objects should be present
                y_pred = [1 if obj in mapped_detected else 0 for obj in expected_objects]

                # Calculate precision, recall, f1
                tp = sum(y_pred)
                fp = len([obj for obj in mapped_detected if obj not in expected_objects])
                fn = len(expected_objects) - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1 = 0.0
                precision = 0.0
                recall = 0.0
                tp = 0
                fp = 0
                fn = 0

            return f1, expected_objects, mapped_detected, precision, recall, tp, fp, fn

        except Exception as e:
            print(f"â�Œ Error evaluating {image_path}: {e}")
            return 0.0, expected_objects, [], 0.0, 0.0, 0, 0, 0

    def evaluate_all_prompts(self, prompt_mapping, output_dir):
        """Evaluate F1 scores for all prompts"""
        print("\nğŸ§® Starting evaluation of all prompts...")

        evaluation_results = []

        for prompt_id, prompt_data in tqdm(prompt_mapping.items(), desc="Evaluating prompts"):
            image_path = os.path.join(output_dir, prompt_data['image_name'])

            # Extract expected objects from prompt
            expected_objects = self.extract_expected_objects(prompt_data['prompt'])

            if os.path.exists(image_path):
                # Calculate F1 score
                f1, expected, detected, precision, recall, tp, fp, fn = self.calculate_f1_per_prompt(
                    expected_objects, image_path
                )

                evaluation_results.append({
                    'prompt_id': prompt_id,
                    'prompt': prompt_data['prompt'],
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'expected_objects': expected,
                    'detected_objects': detected,
                    'image_path': image_path,
                    'status': 'evaluated'
                })
            else:
                evaluation_results.append({
                    'prompt_id': prompt_id,
                    'prompt': prompt_data['prompt'],
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': len(expected_objects),
                    'expected_objects': expected_objects,
                    'detected_objects': [],
                    'image_path': image_path,
                    'status': 'missing_image'
                })

        print(f"âœ… Evaluation completed! Processed {len(evaluation_results)} prompts")
        return evaluation_results

# =====================================================
# 2ï¸�âƒ£ RUN THE EVALUATION
# =====================================================

# Initialize evaluator
evaluator = DreamLayerEvaluator()

# Run evaluation on your generated images
print("ğŸ”� Starting comprehensive evaluation...")
evaluation_results = evaluator.evaluate_all_prompts(prompt_mapping, OUTPUT_DIR)

# Display evaluation summary
successful_evals = len([r for r in evaluation_results if r['status'] == 'evaluated'])
missing_images = len([r for r in evaluation_results if r['status'] == 'missing_image'])
avg_f1 = np.mean([r['f1_score'] for r in evaluation_results if r['status'] == 'evaluated'])

print(f"\nğŸ“Š EVALUATION SUMMARY:")
print(f"   âœ… Successful evaluations: {successful_evals}")
print(f"   â�Œ Missing images: {missing_images}")
print(f"   ğŸ“ˆ Average F1 Score: {avg_f1:.4f}")

# =====================================================
# 3ï¸�âƒ£ CONFUSION MATRIX ANALYSIS (Now with defined evaluation_results)
# =====================================================

def prepare_confusion_matrix_data(evaluation_results):
    """Prepare data for multi-label confusion matrix analysis"""

    # Collect all unique objects across all evaluations
    all_objects = set()
    for res in evaluation_results:
        all_objects.update(res['expected_objects'])
        all_objects.update(res['detected_objects'])

    all_objects = sorted(list(all_objects))
    print(f"ğŸ“Š Found {len(all_objects)} unique objects across all prompts")

    # Prepare ground truth and predictions for each prompt
    y_true_list = []
    y_pred_list = []
    prompt_ids = []

    for res in evaluation_results:
        expected = res['expected_objects']
        detected = res['detected_objects']

        # Create binary vectors for expected and detected objects
        y_true_prompt = [1 if obj in expected else 0 for obj in all_objects]
        y_pred_prompt = [1 if obj in detected else 0 for obj in all_objects]

        y_true_list.append(y_true_prompt)
        y_pred_list.append(y_pred_prompt)
        prompt_ids.append(res['prompt_id'])

    # Convert to numpy arrays
    y_true_np = np.array(y_true_list)
    y_pred_np = np.array(y_pred_list)

    return y_true_np, y_pred_np, all_objects, prompt_ids

# Prepare the data for confusion matrix
y_true, y_pred, all_objects, prompt_ids = prepare_confusion_matrix_data(evaluation_results)

def calculate_detailed_metrics(y_true, y_pred, all_objects):
    """Calculate detailed confusion matrices and metrics for each object"""

    # Calculate multi-label confusion matrices
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

    # Calculate metrics for each object
    object_metrics = []

    for i, obj in enumerate(all_objects):
        tn, fp, fn, tp = conf_matrices[i].ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # Calculate support (how many times object was expected)
        support = np.sum(y_true[:, i])

        object_metrics.append({
            'object': obj,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': support,
            'detection_rate': tp / support if support > 0 else 0
        })

    return object_metrics, conf_matrices

# Calculate detailed metrics
object_metrics, conf_matrices = calculate_detailed_metrics(y_true, y_pred, all_objects)

# =====================================================
# 4ï¸�âƒ£ DISPLAY RESULTS
# =====================================================

def display_evaluation_results(evaluation_results, object_metrics):
    """Display comprehensive evaluation results"""

    # Convert to DataFrame for easier analysis
    eval_df = pd.DataFrame(evaluation_results)

    print("ğŸ“ˆ OVERALL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Prompts Evaluated: {len(eval_df)}")
    print(f"Average F1 Score: {eval_df['f1_score'].mean():.4f}")
    print(f"Average Precision: {eval_df['precision'].mean():.4f}")
    print(f"Average Recall: {eval_df['recall'].mean():.4f}")
    print(f"Total True Positives: {eval_df['true_positives'].sum()}")
    print(f"Total False Positives: {eval_df['false_positives'].sum()}")
    print(f"Total False Negatives: {eval_df['false_negatives'].sum()}")

    # Show top and bottom performing prompts
    print(f"\nğŸ�† TOP 5 BEST PERFORMING PROMPTS (by F1 Score):")
    top_prompts = eval_df.nlargest(5, 'f1_score')[['prompt_id', 'prompt', 'f1_score', 'precision', 'recall']]
    for _, row in top_prompts.iterrows():
        print(f"  {row['prompt_id']}: F1={row['f1_score']:.3f} | {row['prompt'][:60]}...")

    print(f"\nğŸ“‰ TOP 5 WORST PERFORMING PROMPTS (by F1 Score):")
    worst_prompts = eval_df.nsmallest(5, 'f1_score')[['prompt_id', 'prompt', 'f1_score', 'precision', 'recall']]
    for _, row in worst_prompts.iterrows():
        print(f"  {row['prompt_id']}: F1={row['f1_score']:.3f} | {row['prompt'][:60]}...")

    # Show object-level performance
    objects_df = pd.DataFrame(object_metrics)
    objects_with_support = objects_df[objects_df['support'] > 0]

    print(f"\nğŸ�¯ OBJECT-LEVEL PERFORMANCE (Objects with support > 0):")
    print(f"Total objects with support: {len(objects_with_support)}")
    print(f"Average F1 Score: {objects_with_support['f1_score'].mean():.4f}")

    print(f"\nğŸ”� BEST DETECTED OBJECTS:")
    best_objects = objects_with_support.nlargest(5, 'f1_score')[['object', 'f1_score', 'precision', 'recall', 'support']]
    print(best_objects.round(3))

    print(f"\nğŸ”» WORST DETECTED OBJECTS:")
    worst_objects = objects_with_support.nsmallest(5, 'f1_score')[['object', 'f1_score', 'precision', 'recall', 'support']]
    print(worst_objects.round(3))

# Display results
display_evaluation_results(evaluation_results, object_metrics)

# =====================================================
# 5ï¸�âƒ£ VISUALIZATION
# =====================================================

def plot_evaluation_visualizations(evaluation_results, object_metrics):
    """Create visualizations for evaluation results"""

    eval_df = pd.DataFrame(evaluation_results)
    objects_df = pd.DataFrame(object_metrics)
    objects_with_support = objects_df[objects_df['support'] > 0]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DreamLayer Evaluation Results', fontsize=16, fontweight='bold')

    # Plot 1: F1 Score Distribution
    axes[0, 0].hist(eval_df['f1_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(eval_df['f1_score'].mean(), color='red', linestyle='--',
                      label=f'Mean: {eval_df["f1_score"].mean():.3f}')
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of F1 Scores across Prompts')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Precision vs Recall scatter
    scatter = axes[0, 1].scatter(eval_df['recall'], eval_df['precision'],
                                c=eval_df['f1_score'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall (Color = F1 Score)')
    plt.colorbar(scatter, ax=axes[0, 1])
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Object F1 Scores (top 15 by support)
    top_objects = objects_with_support.nlargest(15, 'support')
    axes[1, 0].barh(top_objects['object'], top_objects['f1_score'], color='lightgreen')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_title('F1 Scores for Top 15 Objects by Support')
    axes[1, 0].set_xlim(0, 1)

    # Plot 4: Detection performance
    performance_counts = [
        len(eval_df[eval_df['f1_score'] > 0.7]),
        len(eval_df[(eval_df['f1_score'] > 0.4) & (eval_df['f1_score'] <= 0.7)]),
        len(eval_df[eval_df['f1_score'] <= 0.4])
    ]
    labels = ['Good (F1 > 0.7)', 'Medium (0.4 < F1 â‰¤ 0.7)', 'Poor (F1 â‰¤ 0.4)']
    colors = ['lightgreen', 'gold', 'lightcoral']
    axes[1, 1].pie(performance_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Prompt Performance Distribution')

    plt.tight_layout()
    plt.show()

# Create visualizations
plot_evaluation_visualizations(evaluation_results, object_metrics)

# =====================================================
# 6ï¸�âƒ£ SAVE EVALUATION RESULTS
# =====================================================

def save_evaluation_results(evaluation_results, object_metrics, output_dir):
    """Save all evaluation results to files"""

    # Save prompt-level results
    prompt_results_df = pd.DataFrame(evaluation_results)
    prompt_results_file = os.path.join(output_dir, 'prompt_evaluation_results.csv')
    prompt_results_df.to_csv(prompt_results_file, index=False)

    # Save object-level results
    object_results_df = pd.DataFrame(object_metrics)
    object_results_file = os.path.join(output_dir, 'object_evaluation_results.csv')
    object_results_df.to_csv(object_results_file, index=False)

    # Create summary
    summary = {
        'total_prompts': len(evaluation_results),
        'average_f1_score': prompt_results_df['f1_score'].mean(),
        'average_precision': prompt_results_df['precision'].mean(),
        'average_recall': prompt_results_df['recall'].mean(),
        'total_objects_tracked': len(object_metrics),
        'objects_with_support': len(object_results_df[object_results_df['support'] > 0]),
        'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"\nğŸ’¾ Evaluation results saved:")
    print(f"   ğŸ“„ Prompt-level results: {prompt_results_file}")
    print(f"   ğŸ“„ Object-level results: {object_results_file}")
    print(f"   ğŸ“„ Evaluation summary: {summary_file}")

    return prompt_results_file, object_results_file, summary_file

# Save results
prompt_file, object_file, summary_file = save_evaluation_results(evaluation_results, object_metrics, OUTPUT_DIR)

print("\nâœ… EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
print("ğŸ“Š You now have comprehensive F1 scores and confusion matrix analysis for your DreamLayer submission!")


# Install required packages
!pip install dreamlayer diffusers transformers accelerate torchvision pillow
!pip install opencv-python ultralytics

import os
import json
import pandas as pd
import torch
from pathlib import Path
import shutil
from PIL import Image
import cv2


def create_dreamlayer_config():
    """Create the DreamLayer configuration file"""

    config = {
        "model": {
            "name": "runwayml/stable-diffusion-v1-5",
            "revision": "main",
            "torch_dtype": "float16"
        },
        "scheduler": {
            "name": "DPMSolverMultistepScheduler",
            "num_inference_steps": 20
        },
        "generation": {
            "width": 512,
            "height": 512,
            "guidance_scale": 7.5,
            "seed": 42
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    return config

def save_config(config, output_dir):
    """Save configuration to file"""
    config_path = os.path.join(output_dir, "config-dreamlayer.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path


class DreamLayerGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.setup_model()

    def setup_model(self):
        """Initialize the diffusion model"""
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        import torch

        print("Loading model...")

        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config["model"]["name"],
            torch_dtype=torch.float16 if self.config["device"] == "cuda" else torch.float32,
            revision=self.config["model"]["revision"]
        )

        # Set scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_memory_efficient_attention()

        print("Model loaded successfully!")

    def generate_image(self, prompt, prompt_id, output_dir):
        """Generate a single image from prompt"""
        from PIL import Image

        # Create output filename
        filename = f"{prompt_id:04d}.png"
        output_path = os.path.join(output_dir, filename)

        # Generate image
        with torch.autocast(self.device):
            image = self.pipe(
                prompt,
                width=self.config["generation"]["width"],
                height=self.config["generation"]["height"],
                guidance_scale=self.config["generation"]["guidance_scale"],
                num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                generator=torch.Generator(device=self.device).manual_seed(
                    self.config["generation"]["seed"] + prompt_id
                )
            ).images[0]

        # Save image
        image.save(output_path)
        return output_path


def main():
    # Create output directory
    output_dir = "dreamlayer_output"
    os.makedirs(output_dir, exist_ok=True)

    # Define prompts (from your file)
    prompts = [
        "A man baking and preparing donuts to sell at shop.",
        "A zebra chews a flower in a fenced in field.",
        "A person skiing down a mountain kicking his leg up.",
        "An airplane in route with a cloudy sky behind it.",
        "A white and blue truck parked in the middle of a dirt road.",
        "PIcked peach flowers sit in a vase with water.",
        "Sheep are on a grassy field and one of them is a white and black baby.",
        "A blue rusted train engine sitting on top of rail tracks.",
        "An open laptop computer sitting on top of a wooden table.",
        "A woman twirling an umbrella with flowers on it.",
        "A woman sitting in a restaurant with Mexican food on her plate",
        "A woman holding a purse and a cellphone.",
        "A cat reaching for a knife that has it's blade out.",
        "The man smiles with a slice of pizza while next to a friend.",
        "Night picture of a car parked and some parking lights in the distance.",
        "A professional photograph of a motorcycle rider in the air.",
        "A person putting doughnuts into a bag in a shop.",
        "A very ornately decorated and brightly colored clock.",
        "A young man that is having some wine and something to eat.",
        "A picture of pink bathroom sink and a mirror.",
        "people standing in line beside a food truck",
        "Single sheep in a field looking back at camera.",
        "A few pizza slices next to a couple of bread slices.",
        "Two giraffe standing in a field next to trees.",
        "A person in rubber boots and a rain coat seated on a bench.",
        "A giraffe standing next to a bamboo building.",
        "A large combination pizza with two pieces gone.",
        "A bearded man in a suit eating pizza.",
        "An old yellow train is waiting at the station.",
        "People flying kites in a park next to a lake.",
        "A young man is body surfing and paddling in the water.",
        "A very large stuffed giraffe posed looking out a window.",
        "A dim room with toilet bowls lined along the wall",
        "Two trains parked at a train station as passenger wait to board them.",
        "A slightly made/messy bed against the corner in a white room.",
        "A bench sit in front of a blue and yellow train.",
        "A man is skate boarding down a path and a dog is running by his side.",
        "A man on a long exposure picture riding an electric skateboard.",
        "A person on skis and with poles in the snow and facing the blue sky.",
        "a cake with a section missing sitting next to a burning candle",
        "a woman selling jewelry laid on a blanket on the sidewalk",
        "A group of people on a tennis court.",
        "Two large elephants walking behind a wire fence on green grass.",
        "a woman on a tennis court getting ready to serve the ball",
        "Tabby cat with green eyes wearing a hat",
        "A small, white formica kitchen with a refrigerator, sink and small electrical appliances",
        "A tall clock tower with a large clock on it's face.",
        "A bathroom with a toilet, tub, mirror, window and a shower pole.",
        "A group of people standing on a snow covered hill."
    ]

    print(f"Starting generation of {len(prompts)} images...")

    # Create configuration
    config = create_dreamlayer_config()
    save_config(config, output_dir)

    # Initialize generator
    generator = DreamLayerGenerator(config)

    # Generate images
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"Generating image {i}/{len(prompts)}: {prompt[:50]}...")

        try:
            image_path = generator.generate_image(prompt, i, output_dir)
            results.append({
                "prompt_id": i,
                "predicted": prompt,
                "image_path": image_path
            })
            print(f"âœ“ Generated: {image_path}")
        except Exception as e:
            print(f"âœ— Failed to generate image {i}: {str(e)}")
            # Add fallback entry
            results.append({
                "prompt_id": i,
                "predicted": prompt,
                "image_path": f"{i:04d}.png"
            })

    # Create results.csv
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "results.csv")
    results_df[["prompt_id", "predicted"]].to_csv(results_csv_path, index=False)

    print(f"\nGeneration completed!")
    print(f"Images saved to: {output_dir}")
    print(f"Results CSV: {results_csv_path}")

    return output_dir

# Run the main function
if __name__ == "__main__":
    output_directory = main()


def simple_generation_script():
    """Simplified version for quick testing"""
    from diffusers import StableDiffusionPipeline
    import torch
    import pandas as pd
    from PIL import Image
    import os

    # Setup
    output_dir = "simple_output"
    os.makedirs(output_dir, exist_ok=True)

    # Your prompts
    prompts = [
        "A man baking and preparing donuts to sell at shop.",
        "A zebra chews a flower in a fenced in field.",
        # ... include all your prompts here
    ]

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    # Generate images
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"Generating {i}/{len(prompts)}...")

        image = pipe(
            prompt,
            width=512,
            height=512,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=torch.Generator(device=device).manual_seed(42 + i)
        ).images[0]

        filename = f"{i:04d}.png"
        image.save(os.path.join(output_dir, filename))

        results.append({"prompt_id": i, "predicted": prompt})

    # Save results
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print("Done!")

# Uncomment to run the simple version
# simple_generation_script()


def prepare_kaggle_submission(output_dir):
    """Prepare files for Kaggle submission"""

    # Create submission.csv (this is what you upload to Kaggle)
    results_df = pd.read_csv(os.path.join(output_dir, "results.csv"))

    # For Kaggle submission, we just need prompt_id and predicted
    submission_df = results_df[['prompt_id', 'predicted']].copy()
    submission_df.to_csv('submission.csv', index=False)

    print("Submission file created: submission.csv")
    print(f"Number of entries: {len(submission_df)}")

    return submission_df

# After running main(), prepare submission
# submission_df = prepare_kaggle_submission(output_directory)


def get_optimized_config():
    """Get optimized configuration for better performance"""

    config_options = {
        "fast": {
            "model": "runwayml/stable-diffusion-v1-5",
            "steps": 15,
            "guidance": 7.5
        },
        "quality": {
            "model": "stabilityai/stable-diffusion-2-1",
            "steps": 25,
            "guidance": 8.0
        },
        "detailed": {
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "steps": 30,
            "guidance": 8.5
        }
    }

    return config_options

# Example of using different models
def switch_model(model_name):
    """Switch to a different model"""
    models = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"
    }
    return models.get(model_name, "runwayml/stable-diffusion-v1-5")


output_dir = main()


submission_df = prepare_kaggle_submission(output_dir)


# =====================================================
# ğŸ–¼ï¸� DREAMLAYER TEXT-TO-IMAGE CHALLENGE - OPTIMIZED NOTEBOOK
# Author: Ishita
# Optimized for fast image display and performance
# =====================================================

!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet
!pip install kaggle --quiet

import os
import pandas as pd
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime
from IPython.display import Image, display, HTML
import zipfile
import base64
from io import BytesIO
import concurrent.futures

# =====================================================
# 1ï¸�âƒ£ CONFIGURATION - OPTIMIZED
# =====================================================
PROMPT_FILE = "/content/DreamLayer-Prompt-Kaggle.txt"  # TODO: Replace actual path
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")
MODEL_NAME = "stabilityai/sd-turbo"  # Fast model for quick generation
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 4  # Reduced for speed
GUIDANCE_SCALE = 0.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 2ï¸�âƒ£ LOAD PROMPTS - OPTIMIZED
# =====================================================
def load_prompts(file_path):
    """Fast prompt loading with error handling"""
    try:
        with open(file_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"âœ… Loaded {len(prompts)} prompts.")
        return prompts
    except FileNotFoundError:
        print(f"â�Œ Prompt file not found at {file_path}")
        # Fallback to hardcoded prompts
        return [
            "A man baking and preparing donuts to sell at shop.",
            "A zebra chews a flower in a fenced in field.",
            "A person skiing down a mountain kicking his leg up."
        ]

prompts = load_prompts(PROMPT_FILE)

# =====================================================
# 3ï¸�âƒ£ LOAD MODEL - OPTIMIZED
# =====================================================
def setup_pipeline():
    """Optimized model loading with performance settings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ğŸš€ Using device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable for speed
        requires_safety_checker=False
    ).to(device)

    # Performance optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_memory_efficient_attention()
        torch.backends.cudnn.benchmark = True

    return pipe

pipe = setup_pipeline()

# =====================================================
# 4ï¸�âƒ£ FAST IMAGE GENERATION
# =====================================================
def generate_single_image(args):
    """Generate single image - optimized for parallel processing"""
    i, prompt, output_dir = args
    img_name = f"{i+1:04d}.png"
    img_path = os.path.join(output_dir, img_name)

    if os.path.exists(img_path):
        return {"prompt_id": f"{i+1:04d}", "status": "skipped", "image_path": img_name}

    try:
        image = pipe(
            prompt,
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1],
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device=pipe.device).manual_seed(i + 42)  # Consistent seeds
        ).images[0]

        image.save(img_path, optimize=True, quality=85)  # Optimized saving
        return {"prompt_id": f"{i+1:04d}", "status": "success", "image_path": img_name}

    except Exception as e:
        print(f"â�Œ Error generating image {img_name}: {e}")
        return {"prompt_id": f"{i+1:04d}", "status": "failed", "image_path": ""}

# Generate images in batches for better performance
print("ğŸš€ Generating images...")
results = []

# Use ThreadPoolExecutor for parallel generation (if supported by model)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    args_list = [(i, prompt, OUTPUT_DIR) for i, prompt in enumerate(prompts)]
    future_to_prompt = {executor.submit(generate_single_image, args): args for args in args_list}

    for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(prompts), desc="Generating"):
        result = future.result()
        results.append({
            **result,
            "prompt": prompts[int(result["prompt_id"]) - 1],
            "timestamp": datetime.now().isoformat()
        })

# =====================================================
# 5ï¸�âƒ£ FAST IMAGE DISPLAY - OPTIMIZED
# =====================================================
def display_images_fast(image_paths, max_display=6, columns=3):
    """Ultra-fast image display with grid layout"""
    if not image_paths:
        print("âš ï¸� No images to display")
        return

    # Limit number of images to display
    image_paths = image_paths[:max_display]

    # Create HTML grid for fastest display
    html_content = "<div style='display: grid; grid-template-columns: repeat(" + str(columns) + ", 1fr); gap: 10px;'>"

    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                # Convert image to base64 for instant display
                with open(img_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()

                # Get file size for info
                file_size = os.path.getsize(img_path) // 1024

                html_content += f"""
                <div style='text-align: center; border: 1px solid #ddd; padding: 5px; border-radius: 5px;'>
                    <img src='data:image/png;base64,{img_data}' style='max-width: 100%; height: auto; border-radius: 3px;'
                         onerror="this.style.display='none'">
                    <div style='font-size: 10px; color: #666; margin-top: 5px;'>
                        {os.path.basename(img_path)} ({file_size}KB)
                    </div>
                </div>
                """
            except Exception as e:
                html_content += f"<div>Error loading {img_path}</div>"
        else:
            html_content += f"<div>Missing: {os.path.basename(img_path)}</div>"

    html_content += "</div>"
    display(HTML(html_content))

def quick_preview(output_dir, sample_size=6):
    """Quick preview of generated images"""
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    image_files.sort()

    if not image_files:
        print("â�Œ No images generated yet")
        return

    print(f"ğŸ“¸ Quick Preview ({len(image_files)} total images):")

    # Display first few images in fast grid
    preview_images = [os.path.join(output_dir, f) for f in image_files[:sample_size]]
    display_images_fast(preview_images)

    # Show summary
    success_count = len([r for r in results if r.get('status') == 'success'])
    failed_count = len([r for r in results if r.get('status') == 'failed'])

    print(f"\nğŸ“Š Generation Summary:")
    print(f"   âœ… Success: {success_count}")
    print(f"   â�Œ Failed: {failed_count}")
    print(f"   â�­ï¸� Skipped: {len(image_files) - success_count}")

# =====================================================
# 6ï¸�âƒ£ SAVE RESULTS - OPTIMIZED
# =====================================================
def save_results_fast(results, output_file):
    """Fast results saving with summary"""
    df = pd.DataFrame(results)

    # Add summary row
    success_count = (df['status'] == 'success').sum()
    failed_count = (df['status'] == 'failed').sum()
    skipped_count = (df['status'] == 'skipped').sum()

    summary = f"âœ… {success_count} success | â�Œ {failed_count} failed | â�­ï¸� {skipped_count} skipped"
    print(f"\nğŸ“Š {summary}")

    # Save without summary row to keep CSV clean for submission
    df.to_csv(output_file, index=False)
    print(f"ğŸ’¾ Results saved to: {output_file}")

    return success_count, failed_count, skipped_count

success, failed, skipped = save_results_fast(results, RESULTS_FILE)

# =====================================================
# 7ï¸�âƒ£ INSTANT PREVIEW
# =====================================================
print("\n" + "="*50)
print("ğŸš€ INSTANT PREVIEW")
print("="*50)

# Immediate display without waiting
quick_preview(OUTPUT_DIR)

# =====================================================
# 8ï¸�âƒ£ SUBMISSION PREPARATION
# =====================================================
def create_submission_zip(output_dir, zip_name="submission.zip"):
    """Create submission zip efficiently"""
    zip_path = os.path.join(output_dir, zip_name)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all PNG files
        png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        for file in png_files:
            zipf.write(os.path.join(output_dir, file), file)

        # Add results.csv
        if os.path.exists(RESULTS_FILE):
            zipf.write(RESULTS_FILE, "results.csv")

    file_size = os.path.getsize(zip_path) // 1024
    print(f"ğŸ“¦ Submission package created: {zip_path} ({file_size} KB)")
    return zip_path

submission_zip = create_submission_zip(OUTPUT_DIR)

# =====================================================
# 9ï¸�âƒ£ KAGGLE SUBMISSION (OPTIONAL)
# =====================================================
def submit_to_kaggle(zip_file_path, competition_name):
    """Submit to Kaggle competition"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.competition_submit(
            file_name=zip_file_path,
            message=f"Auto-submission: {success} success, {failed} failed",
            competition=competition_name
        )
        print("âœ… Submission successful!")
        return True
    except Exception as e:
        print(f"âš ï¸� Kaggle submission skipped: {e}")
        print("ğŸ’¡ To enable auto-submission:")
        print("   1. Upload kaggle.json to /root/.kaggle/")
        print("   2. Ensure you've accepted competition rules")
        return False

# Uncomment to enable auto-submission
# submit_to_kaggle(submission_zip, "dreamlayer-text-to-image-challenge")

# =====================================================
# ğŸ”Ÿ FINAL SUMMARY
# =====================================================
print("\n" + "="*50)
print("ğŸ�‰ GENERATION COMPLETE!")
print("="*50)
print(f"ğŸ“� Output directory: {OUTPUT_DIR}")
print(f"ğŸ“Š Results: {success} successful, {failed} failed")
print(f"ğŸ“¦ Submission ready: {submission_zip}")
print("\nğŸ“‹ Next steps:")
print("   1. Download submission.zip")
print("   2. Upload to Kaggle competition")
print("   3. Share your notebook with organizers")
print("="*50)

# Force display cleanup and memory optimization
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =====================================================
# ğŸ–¼ï¸� DREAMLAYER COMPLETE PIPELINE: Display, Train & Evaluate
# Author: Ishita
# Features: Sequential display, Model training, F1 evaluation
# =====================================================

!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet
!pip install kaggle ultralytics sklearn matplotlib seaborn --quiet

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from datetime import datetime
from IPython.display import Image, display, HTML, clear_output
import zipfile
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from ultralytics import YOLO
import cv2

# =====================================================
# 1ï¸�âƒ£ CONFIGURATION
# =====================================================
PROMPT_FILE = "/content/DreamLayer-Prompt-Kaggle.txt"
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
BATCH_SIZE = 2
TRAIN_EPOCHS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 2ï¸�âƒ£ LOAD AND VALIDATE PROMPTS
# =====================================================
def load_and_validate_prompts(file_path):
    """Load prompts and create mapping with validation"""
    try:
        with open(file_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Create prompt mapping with validation
        prompt_mapping = {}
        for i, prompt in enumerate(prompts):
            prompt_id = f"{i+1:04d}"
            prompt_mapping[prompt_id] = {
                'prompt': prompt,
                'expected_objects': extract_expected_objects(prompt),
                'image_path': os.path.join(OUTPUT_DIR, f"{prompt_id}.png")
            }

        print(f"âœ… Loaded and validated {len(prompt_mapping)} prompts")
        return prompt_mapping
    except FileNotFoundError:
        print("â�Œ Prompt file not found, using sample prompts")
        return create_sample_prompts()

def extract_expected_objects(prompt):
    """Extract expected objects from prompt using simple NLP"""
    # Common objects from the competition prompts
    common_objects = [
        'man', 'donuts', 'shop', 'zebra', 'flower', 'fence', 'field',
        'person', 'mountain', 'airplane', 'sky', 'truck', 'road',
        'flowers', 'vase', 'water', 'sheep', 'grass', 'train', 'rails',
        'laptop', 'table', 'woman', 'umbrella', 'restaurant', 'food',
        'purse', 'cellphone', 'cat', 'knife', 'pizza', 'friend', 'car',
        'lights', 'motorcycle', 'clock', 'wine', 'sink', 'mirror',
        'people', 'bread', 'giraffe', 'trees', 'boots', 'coat', 'bench',
        'bamboo', 'building', 'elephants', 'tennis', 'court', 'hat',
        'kitchen', 'refrigerator', 'toilet', 'tub', 'shower', 'snow', 'hill'
    ]

    prompt_lower = prompt.lower()
    detected_objects = [obj for obj in common_objects if obj in prompt_lower]

    # Add custom object detection for specific cases
    if 'baby' in prompt_lower and 'sheep' in prompt_lower:
        detected_objects.append('baby sheep')
    if 'stuffed' in prompt_lower and 'giraffe' in prompt_lower:
        detected_objects.append('stuffed giraffe')

    return list(set(detected_objects))  # Remove duplicates

def create_sample_prompts():
    """Create sample prompts for testing"""
    sample_prompts = [
        "A man baking and preparing donuts to sell at shop.",
        "A zebra chews a flower in a fenced in field.",
        "A person skiing down a mountain kicking his leg up."
    ]

    mapping = {}
    for i, prompt in enumerate(sample_prompts):
        prompt_id = f"{i+1:04d}"
        mapping[prompt_id] = {
            'prompt': prompt,
            'expected_objects': extract_expected_objects(prompt),
            'image_path': os.path.join(OUTPUT_DIR, f"{prompt_id}.png")
        }
    return mapping

# Load prompts with mapping
prompt_mapping = load_and_validate_prompts(PROMPT_FILE)

# =====================================================
# 3ï¸�âƒ£ SEQUENTIAL IMAGE GENERATION WITH PROGRESS DISPLAY
# =====================================================
class SequentialImageGenerator:
    def __init__(self, model_name, device):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        if device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_memory_efficient_attention()

    def generate_sequential_with_display(self, prompt_mapping, output_dir):
        """Generate images sequentially with live display"""
        results = []
        total_prompts = len(prompt_mapping)

        for i, (prompt_id, prompt_data) in enumerate(prompt_mapping.items()):
            clear_output(wait=True)
            print(f"ğŸ”„ Generating {i+1}/{total_prompts}: {prompt_data['prompt'][:50]}...")

            # Display progress
            self.display_progress(i, total_prompts, prompt_data['prompt'])

            img_path = prompt_data['image_path']
            status = "success"

            if not os.path.exists(img_path):
                try:
                    image = self.pipe(
                        prompt_data['prompt'],
                        height=IMAGE_SIZE[0],
                        width=IMAGE_SIZE[1],
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=torch.Generator(device=self.device).manual_seed(i + 42)
                    ).images[0]
                    image.save(img_path)

                    # Display generated image immediately
                    self.display_single_image(prompt_id, prompt_data['prompt'], img_path)

                except Exception as e:
                    print(f"â�Œ Error: {e}")
                    status = "failed"
            else:
                status = "skipped"
                self.display_single_image(prompt_id, prompt_data['prompt'], img_path)

            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_data['prompt'],
                "image_path": os.path.basename(img_path),
                "status": status,
                "expected_objects": prompt_data['expected_objects'],
                "timestamp": datetime.now().isoformat()
            })

            # Small delay to see each image
            if i < total_prompts - 1:  # Don't delay after last image
                print("â�³ Preparing next image...")

        return results

    def display_progress(self, current, total, prompt):
        """Display progress bar"""
        progress = (current + 1) / total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        print(f"\nProgress: [{bar}] {progress*100:.1f}% ({current+1}/{total})")
        print(f"Current: {prompt[:60]}...")
        print("-" * 50)

    def display_single_image(self, prompt_id, prompt, image_path):
        """Display single image with prompt information"""
        if os.path.exists(image_path):
            try:
                # Display image
                display(Image(filename=image_path, width=300))

                # Display prompt info
                html_content = f"""
                <div style='background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    <strong>Prompt ID:</strong> {prompt_id}<br>
                    <strong>Prompt:</strong> {prompt}<br>
                    <strong>Expected Objects:</strong> {', '.join(extract_expected_objects(prompt))}
                </div>
                """
                display(HTML(html_content))

            except Exception as e:
                print(f"âš ï¸� Could not display image {image_path}: {e}")
        else:
            print(f"â�Œ Image not found: {image_path}")

# Initialize generator
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = SequentialImageGenerator(MODEL_NAME, device)

# Generate images with sequential display
print("ğŸ�¨ Starting sequential image generation with live display...")
results = generator.generate_sequential_with_display(prompt_mapping, OUTPUT_DIR)

# =====================================================
# 4ï¸�âƒ£ MODEL FINE-TUNING
# =====================================================
class PromptDataset(Dataset):
    def __init__(self, prompt_mapping, tokenizer, size=512):
        self.prompt_mapping = prompt_mapping
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.prompt_mapping)

    def __getitem__(self, idx):
        prompt_id = list(self.prompt_mapping.keys())[idx]
        prompt_data = self.prompt_mapping[prompt_id]

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt_data['prompt'],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "prompt": prompt_data['prompt'],
            "prompt_id": prompt_id
        }

def fine_tune_model(prompt_mapping, model_name, epochs=3):
    """Fine-tune the diffusion model on our prompts"""
    print("ğŸ”§ Starting model fine-tuning...")

    # Load model components
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

    # Create dataset
    dataset = PromptDataset(prompt_mapping, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    # Training loop
    unet.train()
    text_encoder.eval()  # Freeze text encoder

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # Forward pass would go here - simplified for example
            # In practice, you'd need to implement the full diffusion training

            loss = torch.tensor(0.1, requires_grad=True)  # Placeholder
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.4f}")

    print("âœ… Model fine-tuning completed")
    return unet, text_encoder

# Fine-tune model (commented for speed, uncomment to train)
# tuned_unet, tuned_text_encoder = fine_tune_model(prompt_mapping, MODEL_NAME, TRAIN_EPOCHS)

# =====================================================
# 5ï¸�âƒ£ F1 SCORE EVALUATION WITH YOLO
# =====================================================
class F1Evaluator:
    def __init__(self):
        # Load YOLO model for object detection
        self.model = YOLO('yolov8n.pt')
        self.object_mapping = {
            'person': ['man', 'woman', 'person', 'people'],
            'cat': ['cat'],
            'dog': ['dog'],
            'sheep': ['sheep', 'lamb'],
            'zebra': ['zebra'],
            'giraffe': ['giraffe'],
            'elephant': ['elephant'],
            'car': ['car', 'truck'],
            'train': ['train'],
            'airplane': ['airplane'],
            'motorcycle': ['motorcycle'],
            'bus': ['bus'],
            'truck': ['truck'],
            'umbrella': ['umbrella'],
            'handbag': ['purse', 'handbag'],
            'cell phone': ['cellphone', 'phone'],
            'knife': ['knife'],
            'pizza': ['pizza'],
            'donut': ['donut', 'doughnut'],
            'cake': ['cake'],
            'sink': ['sink'],
            'refrigerator': ['refrigerator'],
            'toilet': ['toilet'],
            'clock': ['clock'],
            'vase': ['vase'],
            'bench': ['bench'],
            'skis': ['skis'],
            'snowboard': ['snowboard'],
            'sports ball': ['tennis ball'],
            'kite': ['kite'],
            'teddy bear': ['stuffed giraffe', 'teddy bear'],
            'wine glass': ['wine glass']
        }

    def map_detected_to_expected(self, detected_objects):
        """Map YOLO detected objects to our expected object names"""
        mapped_objects = []
        for detected in detected_objects:
            for yolo_obj, our_objects in self.object_mapping.items():
                if detected == yolo_obj:
                    mapped_objects.extend(our_objects)
        return list(set(mapped_objects))

    def calculate_f1_per_prompt(self, expected_objects, image_path):
        """Calculate F1 score for a single prompt-image pair"""
        if not os.path.exists(image_path):
            return 0.0, [], []

        try:
            # Run object detection
            results = self.model(image_path)
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    object_name = result.names[class_id]
                    detected_objects.append(object_name)

            # Map detected objects to our expected names
            mapped_detected = self.map_detected_to_expected(detected_objects)

            # Calculate F1 score
            y_true = [1 if obj in expected_objects else 0 for obj in expected_objects]
            y_pred = [1 if obj in mapped_detected else 0 for obj in expected_objects]

            if len(y_true) > 0 and sum(y_true) > 0:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                f1 = 0.0

            return f1, expected_objects, mapped_detected

        except Exception as e:
            print(f"â�Œ Error evaluating {image_path}: {e}")
            return 0.0, expected_objects, []

    def evaluate_all_prompts(self, results):
        """Evaluate F1 scores for all prompts"""
        print("\nğŸ§® Calculating F1 scores...")

        evaluation_results = []
        all_expected = []
        all_detected = []

        for result in tqdm(results, desc="Evaluating"):
            if result['status'] == 'success' and os.path.exists(result['image_path']):
                f1, expected, detected = self.calculate_f1_per_prompt(
                    result['expected_objects'],
                    os.path.join(OUTPUT_DIR, result['image_path'])
                )

                evaluation_results.append({
                    'prompt_id': result['prompt_id'],
                    'prompt': result['prompt'],
                    'f1_score': f1,
                    'expected_objects': expected,
                    'detected_objects': detected,
                    'image_path': result['image_path']
                })

                all_expected.extend(expected)
                all_detected.extend(detected)

        return evaluation_results

    def plot_f1_scores(self, evaluation_results):
        """Plot F1 score distribution"""
        f1_scores = [result['f1_score'] for result in evaluation_results]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of F1 Scores')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        successful = [score for score in f1_scores if score > 0.5]
        plt.pie([len(successful), len(f1_scores) - len(successful)],
                labels=['F1 > 0.5', 'F1 â‰¤ 0.5'],
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Success Rate (F1 > 0.5)')

        plt.tight_layout()
        plt.show()

        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        print(f"\nğŸ“Š Average F1 Score: {avg_f1:.4f}")
        print(f"ğŸ�† Best F1 Score: {max(f1_scores) if f1_scores else 0:.4f}")
        print(f"ğŸ“‰ Worst F1 Score: {min(f1_scores) if f1_scores else 0:.4f}")

# Initialize evaluator and run evaluation
evaluator = F1Evaluator()
evaluation_results = evaluator.evaluate_all_prompts(results)

# =====================================================
# 6ï¸�âƒ£ COMPREHENSIVE RESULTS DISPLAY
# =====================================================
def display_comprehensive_results(evaluation_results, top_n=5):
    """Display comprehensive results with best/worst performers"""

    # Sort by F1 score
    sorted_results = sorted(evaluation_results, key=lambda x: x['f1_score'], reverse=True)

    print("\n" + "="*80)
    print("ğŸ�† TOP PERFORMING IMAGES (Highest F1 Scores)")
    print("="*80)

    # Display top performers
    for i, result in enumerate(sorted_results[:top_n]):
        print(f"\n#{i+1} - F1 Score: {result['f1_score']:.4f}")
        print(f"Prompt ID: {result['prompt_id']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Expected: {result['expected_objects']}")
        print(f"Detected: {result['detected_objects']}")

        # Display image
        img_path = os.path.join(OUTPUT_DIR, result['image_path'])
        if os.path.exists(img_path):
            display(Image(filename=img_path, width=300))

    # Display worst performers
    print("\n" + "="*80)
    print("ğŸ“‰ LOWEST PERFORMING IMAGES")
    print("="*80)

    for i, result in enumerate(sorted_results[-top_n:]):
        print(f"\n#{i+1} - F1 Score: {result['f1_score']:.4f}")
        print(f"Prompt: {result['prompt'][:80]}...")
        print(f"Expected: {result['expected_objects']}")
        print(f"Detected: {result['detected_objects']}")

    # Plot results
    evaluator.plot_f1_scores(evaluation_results)

# Display comprehensive results
display_comprehensive_results(evaluation_results)

# =====================================================
# 7ï¸�âƒ£ SAVE FINAL RESULTS
# =====================================================
def save_final_results(results, evaluation_results, output_file):
    """Save final results with F1 scores"""
    final_df = pd.DataFrame(results)

    # Add F1 scores
    f1_scores = {result['prompt_id']: result['f1_score'] for result in evaluation_results}
    final_df['f1_score'] = final_df['prompt_id'].map(f1_scores)

    # Calculate overall metrics
    successful_generations = len([r for r in results if r['status'] == 'success'])
    avg_f1 = final_df['f1_score'].mean()

    print(f"\nâœ… FINAL SUMMARY:")
    print(f"   ğŸ“� Total Prompts: {len(results)}")
    print(f"   ğŸ�¨ Successful Generations: {successful_generations}")
    print(f"   ğŸ“Š Average F1 Score: {avg_f1:.4f}")
    print(f"   ğŸ’¾ Results saved to: {output_file}")

    final_df.to_csv(output_file, index=False)
    return final_df

# Save final results
final_results_df = save_final_results(results, evaluation_results, RESULTS_FILE)

# =====================================================
# 8ï¸�âƒ£ CREATE SUBMISSION PACKAGE
# =====================================================
def create_kaggle_submission(output_dir, results_file):
    """Create Kaggle submission package"""
    zip_path = os.path.join(output_dir, "submission.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add all generated images
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.png'):
                zipf.write(os.path.join(output_dir, file_name), file_name)

        # Add results file
        if os.path.exists(results_file):
            zipf.write(results_file, "results.csv")

    # Create submission.csv for Kaggle
    submission_df = final_results_df[['prompt_id', 'prompt']].copy()
    submission_csv = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_csv, index=False)
    zipf.write(submission_csv, "submission.csv")

    file_size = os.path.getsize(zip_path) // 1024
    print(f"ğŸ“¦ Kaggle submission package created: {zip_path} ({file_size} KB)")
    return zip_path

submission_package = create_kaggle_submission(OUTPUT_DIR, RESULTS_FILE)

print("\nğŸ�‰ PIPELINE COMPLETED SUCCESSFULLY!")
print("ğŸ“‹ Next steps:")
print("   1. Download submission.zip")
print("   2. Upload to Kaggle competition")
print("   3. Share your notebook with organizers")


from IPython.display import display, Image

# Display a single image
display(Image(filename=os.path.join(OUTPUT_DIR, "0001.png")))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display a single image
img = mpimg.imread(os.path.join(OUTPUT_DIR, "0001.png"))
plt.imshow(img)
plt.axis('off')
plt.show()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the number of rows and columns for the grid
rows = 5
cols = 10

# Create a figure
fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the images and display them
for i, ax in enumerate(axes):
    if i < len(results):
        img_path = results[i]["image_path"]
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"ID: {results[i]['prompt_id']}", fontsize=8)
    else:
        ax.axis('off')  # Hide empty subplots

plt.tight_layout()
plt.show()
from IPython.display import display, HTML
import base64

# Function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Display images in a grid
cols = 4
html = f"<div style='display:grid;grid-template-columns:repeat({cols},1fr);gap:15px;'>"

for result in results:
    image_path = result["image_path"]
    if os.path.exists(image_path):
        b64_image = image_to_base64(image_path)
        html += f"""
        <div style='border:1px solid #ccc;padding:5px;text-align:center;'>
            <img src='data:image/png;base64,{b64_image}' style='width:100%;height:200px;object-fit:cover;'>
            <br>
            <span style='font-size:10px;'>{result['prompt_id']}: {result['prompt'][:50]}...</span>
        </div>
        """
html += "</div>"
display(HTML(html))
# Save images to the working directory
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display images in a grid
fig, axes = plt.subplots(10, 5, figsize=(20, 20))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(results):
        img_path = results[i]["image_path"]
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"ID: {results[i]['prompt_id']}", fontsize=8)

plt.tight_layout()
plt.show()



# =====================================================
# ğŸ–¼ï¸� DREAMLAYER COMPLETE PIPELINE: Display & Evaluate
# Author: Ishita
# Features: Sequential display, F1 evaluation
# =====================================================

# First, install all required packages
!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet
!pip install kaggle scikit-learn matplotlib seaborn --quiet
!pip install ultralytics opencv-python --quiet

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime
from IPython.display import Image, display, HTML, clear_output
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from ultralytics import YOLO
import cv2

# =====================================================
# 1ï¸�âƒ£ CONFIGURATION
# =====================================================
PROMPT_FILE = "/content/DreamLayer-Prompt-Kaggle (1).txt"
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = (512, 512)
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 2ï¸�âƒ£ LOAD AND VALIDATE PROMPTS
# =====================================================
def load_and_validate_prompts(file_path):
    """Load prompts and create mapping with validation"""
    try:
        with open(file_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Create prompt mapping with validation
        prompt_mapping = {}
        for i, prompt in enumerate(prompts):
            prompt_id = f"{i+1:04d}"
            prompt_mapping[prompt_id] = {
                'prompt': prompt,
                'expected_objects': extract_expected_objects(prompt),
                'image_name': f"{prompt_id}.png",
                'image_path': os.path.join(OUTPUT_DIR, f"{prompt_id}.png")
            }

        print(f"âœ… Loaded and validated {len(prompt_mapping)} prompts")
        return prompt_mapping
    except FileNotFoundError:
        print("â�Œ Prompt file not found, using sample prompts")
        return create_sample_prompts()

def extract_expected_objects(prompt):
    """Extract expected objects from prompt using simple NLP"""
    # Enhanced object list with more comprehensive coverage
    common_objects = [
        'man', 'donuts', 'shop', 'zebra', 'flower', 'fence', 'field',
        'person', 'mountain', 'airplane', 'sky', 'truck', 'road',
        'flowers', 'vase', 'water', 'sheep', 'grass', 'train', 'rails',
        'laptop', 'table', 'woman', 'umbrella', 'restaurant', 'food',
        'purse', 'cellphone', 'cat', 'knife', 'pizza', 'friend', 'car',
        'lights', 'motorcycle', 'clock', 'wine', 'sink', 'mirror',
        'people', 'bread', 'giraffe', 'trees', 'boots', 'coat', 'bench',
        'bamboo', 'building', 'elephants', 'tennis', 'court', 'hat',
        'kitchen', 'refrigerator', 'toilet', 'tub', 'shower', 'snow', 'hill',
        'donut', 'doughnut', 'engine', 'computer', 'Mexican food', 'plate',
        'parking', 'photo', 'rider', 'bag', 'clock', 'young man', 'eat',
        'bathroom', 'line', 'food truck', 'pizza slices', 'bread slices',
        'rubber boots', 'rain coat', 'combination pizza', 'bearded man', 'suit',
        'old yellow train', 'station', 'kites', 'park', 'lake', 'body surfing',
        'paddling', 'stuffed giraffe', 'window', 'dim room', 'toilet bowls',
        'wall', 'passenger', 'messy bed', 'corner', 'white room', 'skate boarding',
        'path', 'dog', 'long exposure', 'electric skateboard', 'skis', 'poles',
        'cake', 'candle', 'jewelry', 'blanket', 'sidewalk', 'group', 'wire fence',
        'green grass', 'tabby cat', 'green eyes', 'formica', 'small electrical appliances',
        'tall clock', 'tower', 'tub', 'shower pole', 'snow covered hill'
    ]

    prompt_lower = prompt.lower()
    detected_objects = [obj for obj in common_objects if obj in prompt_lower]

    # Add custom object detection for specific cases
    if 'baby' in prompt_lower and 'sheep' in prompt_lower:
        detected_objects.append('baby sheep')
    if 'stuffed' in prompt_lower and 'giraffe' in prompt_lower:
        detected_objects.append('stuffed giraffe')
    if 'parking' in prompt_lower and 'light' in prompt_lower:
        detected_objects.append('parking light')

    return list(set(detected_objects))  # Remove duplicates

def create_sample_prompts():
    """Create sample prompts for testing"""
    sample_prompts = [
        "A man baking and preparing donuts to sell at shop.",
        "A zebra chews a flower in a fenced in field.",
        "A person skiing down a mountain kicking his leg up."
    ]

    mapping = {}
    for i, prompt in enumerate(sample_prompts):
        prompt_id = f"{i+1:04d}"
        mapping[prompt_id] = {
            'prompt': prompt,
            'expected_objects': extract_expected_objects(prompt),
            'image_name': f"{prompt_id}.png",
            'image_path': os.path.join(OUTPUT_DIR, f"{prompt_id}.png")
        }
    return mapping

# Load prompts with mapping
print("ğŸ“� Loading prompts...")
prompt_mapping = load_and_validate_prompts(PROMPT_FILE)

# =====================================================
# 3ï¸�âƒ£ SEQUENTIAL IMAGE GENERATION WITH PROGRESS DISPLAY
# =====================================================
class SequentialImageGenerator:
    def __init__(self, model_name, device):
        self.device = device
        print(f"ğŸš€ Loading model: {model_name}")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)

            if device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_memory_efficient_attention()
            print("âœ… Model loaded successfully!")
        except Exception as e:
            print(f"â�Œ Error loading model: {e}")
            raise

    def generate_sequential_with_display(self, prompt_mapping, output_dir):
        """Generate images sequentially with live display"""
        results = []
        total_prompts = len(prompt_mapping)

        for i, (prompt_id, prompt_data) in enumerate(prompt_mapping.items()):
            clear_output(wait=True)
            print(f"ğŸ”„ Generating {i+1}/{total_prompts}: {prompt_data['prompt'][:50]}...")

            # Display progress
            self.display_progress(i, total_prompts, prompt_data['prompt'])

            img_path = prompt_data['image_path']
            status = "success"

            if not os.path.exists(img_path):
                try:
                    image = self.pipe(
                        prompt_data['prompt'],
                        height=IMAGE_SIZE[0],
                        width=IMAGE_SIZE[1],
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=torch.Generator(device=self.device).manual_seed(i + 42)
                    ).images[0]
                    image.save(img_path)

                    # Display generated image immediately
                    self.display_single_image(prompt_id, prompt_data['prompt'], img_path)

                except Exception as e:
                    print(f"â�Œ Error: {e}")
                    status = "failed"
            else:
                status = "skipped"
                self.display_single_image(prompt_id, prompt_data['prompt'], img_path)

            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_data['prompt'],
                "image_name": prompt_data['image_name'],
                "image_path": prompt_data['image_path'],
                "status": status,
                "expected_objects": prompt_data['expected_objects'],
                "timestamp": datetime.now().isoformat()
            })

        return results

    def display_progress(self, current, total, prompt):
        """Display progress bar"""
        progress = (current + 1) / total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = 'â–ˆ' * filled_length + 'â–‘' * (bar_length - filled_length)

        print(f"\nProgress: [{bar}] {progress*100:.1f}% ({current+1}/{total})")
        print(f"Current: {prompt[:60]}...")
        print("-" * 50)

    def display_single_image(self, prompt_id, prompt, image_path):
        """Display single image with prompt information"""
        if os.path.exists(image_path):
            try:
                # Display image
                display(Image(filename=image_path, width=300))

                # Display prompt info
                html_content = f"""
                <div style='background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    <strong>Prompt ID:</strong> {prompt_id}<br>
                    <strong>Prompt:</strong> {prompt}<br>
                    <strong>Expected Objects:</strong> {', '.join(extract_expected_objects(prompt))}
                </div>
                """
                display(HTML(html_content))

            except Exception as e:
                print(f"âš ï¸� Could not display image {image_path}: {e}")
        else:
            print(f"â�Œ Image not found: {image_path}")

# Initialize generator
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ğŸ”§ Using device: {device}")
generator = SequentialImageGenerator(MODEL_NAME, device)

# Generate images with sequential display
print("ğŸ�¨ Starting sequential image generation with live display...")
results = generator.generate_sequential_with_display(prompt_mapping, OUTPUT_DIR)

# =====================================================
# 4ï¸�âƒ£ F1 SCORE EVALUATION WITH YOLO
# =====================================================
class F1Evaluator:
    def __init__(self):
        # Load YOLO model for object detection
        print("ğŸš€ Loading YOLO model for object detection...")
        try:
            self.model = YOLO('yolov8n.pt')
            self.object_mapping = {
                'person': ['man', 'woman', 'person', 'people'],
                'cat': ['cat'],
                'dog': ['dog'],
                'sheep': ['sheep', 'lamb', 'baby sheep'],
                'zebra': ['zebra'],
                'giraffe': ['giraffe'],
                'elephant': ['elephant'],
                'car': ['car', 'truck'],
                'train': ['train'],
                'airplane': ['airplane'],
                'motorcycle': ['motorcycle'],
                'bus': ['bus'],
                'truck': ['truck'],
                'umbrella': ['umbrella'],
                'handbag': ['purse', 'handbag'],
                'cell phone': ['cellphone', 'phone'],
                'knife': ['knife'],
                'pizza': ['pizza'],
                'donut': ['donut', 'doughnut'],
                'cake': ['cake'],
                'sink': ['sink'],
                'refrigerator': ['refrigerator'],
                'toilet': ['toilet'],
                'clock': ['clock'],
                'vase': ['vase'],
                'bench': ['bench'],
                'skis': ['skis'],
                'snowboard': ['snowboard'],
                'sports ball': ['tennis ball'],
                'kite': ['kite'],
                'teddy bear': ['stuffed giraffe', 'teddy bear'],
                'wine glass': ['wine glass'],
                'laptop': ['laptop', 'computer'],
                'book': ['book'],
                'chair': ['chair'],
                'dining table': ['table', 'dining table'],
                'bed': ['bed'],
                'tv': ['tv', 'television'],
                'bottle': ['bottle', 'water bottle'],
                'flower': ['flower', 'flowers']
            }
            print("âœ… YOLO model loaded successfully!")
        except Exception as e:
            print(f"â�Œ Error loading YOLO model: {e}")
            raise

    def map_detected_to_expected(self, detected_objects):
        """Map YOLO detected objects to our expected object names"""
        mapped_objects = []
        for detected in detected_objects:
            for yolo_obj, our_objects in self.object_mapping.items():
                if detected == yolo_obj:
                    mapped_objects.extend(our_objects)
        return list(set(mapped_objects))

    def calculate_f1_per_prompt(self, expected_objects, image_path):
        """Calculate F1 score for a single prompt-image pair"""
        if not os.path.exists(image_path):
            return 0.0, [], []

        try:
            # Run object detection
            results = self.model(image_path)
            detected_objects = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    object_name = result.names[class_id]
                    detected_objects.append(object_name)

            # Map detected objects to our expected names
            mapped_detected = self.map_detected_to_expected(detected_objects)

            # Calculate F1 score
            y_true = [1] * len(expected_objects)  # All expected objects should be present
            y_pred = [1 if obj in mapped_detected else 0 for obj in expected_objects]

            if len(y_true) > 0 and sum(y_true) > 0:
                f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                f1 = 0.0

            return f1, expected_objects, mapped_detected

        except Exception as e:
            print(f"â�Œ Error evaluating {image_path}: {e}")
            return 0.0, expected_objects, []

    def evaluate_all_prompts(self, results):
        """Evaluate F1 scores for all prompts"""
        print("\nğŸ§® Calculating F1 scores...")

        evaluation_results = []

        for result in tqdm(results, desc="Evaluating"):
            if result['status'] == 'success' and os.path.exists(result['image_path']):
                f1, expected, detected = self.calculate_f1_per_prompt(
                    result['expected_objects'],
                    result['image_path']
                )

                evaluation_results.append({
                    'prompt_id': result['prompt_id'],
                    'prompt': result['prompt'],
                    'f1_score': f1,
                    'expected_objects': expected,
                    'detected_objects': detected,
                    'image_path': result['image_path']
                })
            else:
                evaluation_results.append({
                    'prompt_id': result['prompt_id'],
                    'prompt': result['prompt'],
                    'f1_score': 0.0,
                    'expected_objects': result['expected_objects'],
                    'detected_objects': [],
                    'image_path': result['image_path']
                })

        return evaluation_results

    def plot_f1_scores(self, evaluation_results):
        """Plot F1 score distribution"""
        f1_scores = [result['f1_score'] for result in evaluation_results]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of F1 Scores')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        successful = [score for score in f1_scores if score > 0.5]
        plt.pie([len(successful), len(f1_scores) - len(successful)],
                labels=['F1 > 0.5', 'F1 â‰¤ 0.5'],
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Success Rate (F1 > 0.5)')

        plt.tight_layout()
        plt.show()

        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        print(f"\nğŸ“Š Average F1 Score: {avg_f1:.4f}")
        print(f"ğŸ�† Best F1 Score: {max(f1_scores) if f1_scores else 0:.4f}")
        print(f"ğŸ“‰ Worst F1 Score: {min(f1_scores) if f1_scores else 0:.4f}")

# Initialize evaluator and run evaluation
print("ğŸ”� Starting F1 evaluation...")
evaluator = F1Evaluator()
evaluation_results = evaluator.evaluate_all_prompts(results)

# Plot F1 scores
evaluator.plot_f1_scores(evaluation_results)

# =====================================================
# 5ï¸�âƒ£ CONFUSION MATRIX ANALYSIS
# =====================================================
print("\n" + "="*80)
print("ğŸ“Š OBJECT-LEVEL CONFUSION MATRIX ANALYSIS")
print("="*80)

def analyze_confusion_matrix(evaluation_results):
    """Perform comprehensive confusion matrix analysis"""
    
    # 1. Collect all unique expected and detected objects
    all_objects = set()
    for res in evaluation_results:
        all_objects.update(res['expected_objects'])
        all_objects.update(res['detected_objects'])

    all_objects = sorted(list(all_objects))
    print(f"ğŸ“‹ Found {len(all_objects)} unique objects across all prompts")

    # 2. Prepare ground truth and predictions
    y_true_list = []
    y_pred_list = []

    for res in evaluation_results:
        expected = res['expected_objects']
        detected = res['detected_objects']

        # Create binary vectors for expected and detected objects
        y_true_prompt = [1 if obj in expected else 0 for obj in all_objects]
        y_pred_prompt = [1 if obj in detected else 0 for obj in all_objects]

        y_true_list.append(y_true_prompt)
        y_pred_list.append(y_pred_prompt)

    # Convert to numpy arrays
    y_true_np = np.array(y_true_list)
    y_pred_np = np.array(y_pred_list)

    # 3. Calculate confusion matrices
    if len(all_objects) > 0:
        conf_matrices = multilabel_confusion_matrix(y_true_np, y_pred_np)
        
        # 4. Display per object matrices (top 20 by frequency)
        print("\nğŸ”� OBJECT-LEVEL CONFUSION MATRICES (Top 20):")
        print("-" * 60)
        
        # Calculate object frequencies
        object_frequencies = {}
        for res in evaluation_results:
            for obj in res['expected_objects']:
                object_frequencies[obj] = object_frequencies.get(obj, 0) + 1
        
        # Sort by frequency
        sorted_objects = sorted(object_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        
        for obj, freq in sorted_objects:
            if obj in all_objects:
                idx = all_objects.index(obj)
                if conf_matrices[idx].shape == (2, 2):
                    tn, fp, fn, tp = conf_matrices[idx].ravel()
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    print(f"\nğŸ�¯ '{obj}' (Frequency: {freq})")
                    print(f"   âœ… True Positives: {tp:2d} | Precision: {precision:.3f}")
                    print(f"   â�Œ False Positives: {fp:2d} | Recall: {recall:.3f}")
                    print(f"   â�Œ False Negatives: {fn:2d} | F1: {2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0:.3f}")

        # 5. Summary statistics
        total_tp = np.sum(conf_matrices[:, 1, 1])
        total_fp = np.sum(conf_matrices[:, 0, 1])
        total_fn = np.sum(conf_matrices[:, 1, 0])

        print(f"\nğŸ“ˆ SUMMARY STATISTICS:")
        print(f"   âœ… Total True Positives: {total_tp}")
        print(f"   â�Œ Total False Positives: {total_fp}")
        print(f"   â�Œ Total False Negatives: {total_fn}")
        print(f"   ğŸ“Š Overall Precision: {total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0:.3f}")
        print(f"   ğŸ“Š Overall Recall: {total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0:.3f}")

    return all_objects, y_true_np, y_pred_np

# Run confusion matrix analysis
all_objects, y_true, y_pred = analyze_confusion_matrix(evaluation_results)

# =====================================================
# 6ï¸�âƒ£ COMPREHENSIVE RESULTS DISPLAY
# =====================================================
def display_comprehensive_results(evaluation_results, top_n=5):
    """Display comprehensive results with best/worst performers"""

    # Sort by F1 score
    sorted_results = sorted(evaluation_results, key=lambda x: x['f1_score'], reverse=True)

    print("\n" + "="*80)
    print("ğŸ�† TOP PERFORMING IMAGES (Highest F1 Scores)")
    print("="*80)

    # Display top performers
    for i, result in enumerate(sorted_results[:top_n]):
        print(f"\n#{i+1} - F1 Score: {result['f1_score']:.4f}")
        print(f"Prompt ID: {result['prompt_id']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Expected: {result['expected_objects']}")
        print(f"Detected: {result['detected_objects']}")

        # Display image
        if os.path.exists(result['image_path']):
            display(Image(filename=result['image_path'], width=300))

    # Display worst performers
    print("\n" + "="*80)
    print("ğŸ“‰ LOWEST PERFORMING IMAGES")
    print("="*80)

    num_worst = min(top_n, len(sorted_results))
    for i, result in enumerate(sorted_results[-num_worst:]):
        print(f"\n#{i+1} - F1 Score: {result['f1_score']:.4f}")
        print(f"Prompt ID: {result['prompt_id']}")
        print(f"Prompt: {result['prompt'][:80]}...")
        print(f"Expected: {result['expected_objects']}")
        print(f"Detected: {result['detected_objects']}")

# Display comprehensive results
display_comprehensive_results(evaluation_results)

# =====================================================
# 7ï¸�âƒ£ SAVE FINAL RESULTS
# =====================================================
def save_final_results(results, evaluation_results, output_file):
    """Save final results with F1 scores"""
    final_df = pd.DataFrame(results)

    # Add F1 scores and detection info
    f1_scores = {result['prompt_id']: result['f1_score'] for result in evaluation_results}
    detected_objects = {result['prompt_id']: result['detected_objects'] for result in evaluation_results}
    
    final_df['f1_score'] = final_df['prompt_id'].map(f1_scores)
    final_df['detected_objects'] = final_df['prompt_id'].map(detected_objects)

    # Calculate overall metrics
    successful_generations = len([r for r in results if r['status'] == 'success'])
    avg_f1 = final_df['f1_score'].mean()

    print(f"\nâœ… FINAL SUMMARY:")
    print(f"   ğŸ“� Total Prompts: {len(results)}")
    print(f"   ğŸ�¨ Successful Generations: {successful_generations}")
    print(f"   ğŸ“Š Average F1 Score: {avg_f1:.4f}")
    print(f"   ğŸ’¾ Results saved to: {output_file}")

    final_df.to_csv(output_file, index=False)
    return final_df

# Save final results
final_results_df = save_final_results(results, evaluation_results, RESULTS_FILE)

# =====================================================
# 8ï¸�âƒ£ CREATE SUBMISSION PACKAGE
# =====================================================
def create_kaggle_submission(output_dir, results_file):
    """Create Kaggle submission package"""
    zip_path = os.path.join(output_dir, "submission.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add all generated images
        for file_name in os.listdir(output_dir):
            if file_name.endswith('.png'):
                zipf.write(os.path.join(output_dir, file_name), file_name)

        # Add results file
        if os.path.exists(results_file):
            zipf.write(results_file, "results.csv")

    file_size = os.path.getsize(zip_path) // 1024
    print(f"ğŸ“¦ Kaggle submission package created: {zip_path} ({file_size} KB)")
    
    # Show package contents
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_list = zipf.namelist()
    
    png_files = [f for f in file_list if f.endswith('.png')]
    print(f"   ğŸ“� Contains {len(png_files)} images and {len(file_list) - len(png_files)} data files")
    
    return zip_path

# Create submission package
submission_package = create_kaggle_submission(OUTPUT_DIR, RESULTS_FILE)

print("\nğŸ�‰ PIPELINE COMPLETED SUCCESSFULLY!")
print("ğŸ“‹ Next steps:")
print("   1. Download submission.zip from the file browser")
print("   2. Upload to Kaggle competition")
print("   3. Share your notebook with organizers if required")


# =====================================================
# ğŸ–¼ï¸� DREAMLAYER IMAGE GENERATION + KAGGLE SUBMISSION (ROBUST)
# =====================================================

!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm --quiet

import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display, HTML
import base64

# -------------------------------
# 1ï¸�âƒ£ Setup paths
# -------------------------------
BASE_DIR = "/kaggle/working/text-to-image-challenge"
OUTPUT_DIR = os.path.join(BASE_DIR, "dreamlayer_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: local model path (upload HF model as Kaggle dataset)
LOCAL_MODEL_PATH = "/kaggle/input/stable-diffusion-v1-5"  # change if you uploaded

# -------------------------------
# 2ï¸�âƒ£ Functions & Classes
# -------------------------------
def create_dreamlayer_config():
    return {
        "model": {"name":"runwayml/stable-diffusion-v1-5", "revision":"main", "torch_dtype":"float16"},
        "scheduler":{"num_inference_steps":25},
        "generation":{"width":512,"height":512,"guidance_scale":7.5,"seed":42},
        "device":"cuda" if torch.cuda.is_available() else "cpu"
    }

def save_config(config, output_dir):
    import json
    config_path = os.path.join(output_dir, "config-dreamlayer.json")
    with open(config_path,'w') as f:
        json.dump(config,f,indent=2)
    return config_path

class DreamLayerGenerator:
    def __init__(self, config, local_model_path=None):
        self.config = config
        self.device = config["device"]
        self.local_model_path = local_model_path
        self.setup_model()
    
    def setup_model(self):
        print("ğŸš€ Loading Stable Diffusion model...")
        model_name = self.local_model_path if self.local_model_path and os.path.exists(self.local_model_path) else self.config["model"]["name"]
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            if self.device=="cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_memory_efficient_attention()
            print("âœ… Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate_image(self, prompt, prompt_id, output_dir):
        from PIL import Image
        output_path = os.path.join(output_dir,f"{prompt_id:04d}.png")
        if os.path.exists(output_path):
            return output_path
        with torch.autocast(self.device):
            img = self.pipe(
                prompt,
                width=self.config["generation"]["width"],
                height=self.config["generation"]["height"],
                guidance_scale=self.config["generation"]["guidance_scale"],
                num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                generator=torch.Generator(device=self.device).manual_seed(
                    self.config["generation"]["seed"] + prompt_id
                )
            ).images[0]
            img.save(output_path)
        return output_path

# -------------------------------
# 3ï¸�âƒ£ Prompts
# -------------------------------
prompts = [
    "A man baking and preparing donuts to sell at shop.",
    "A zebra chews a flower in a fenced in field.",
    "A person skiing down a mountain kicking his leg up.",
    "An airplane in route with a cloudy sky behind it.",
    "A white and blue truck parked in the middle of a dirt road.",
    "PIcked peach flowers sit in a vase with water.",
    "Sheep are on a grassy field and one of them is a white and black baby.",
    "A blue rusted train engine sitting on top of rail tracks.",
    "An open laptop computer sitting on top of a wooden table.",
    "A woman twirling an umbrella with flowers on it.",
    "A woman sitting in a restaurant with Mexican food on her plate",
    "A woman holding a purse and a cellphone.",
    "A cat reaching for a knife that has it's blade out.",
    "The man smiles with a slice of pizza while next to a friend.",
    "Night picture of a car parked and some parking lights in the distance.",
    "A professional photograph of a motorcycle rider in the air.",
    "A person putting doughnuts into a bag in a shop.",
    "A very ornately decorated and brightly colored clock.",
    "A young man that is having some wine and something to eat.",
    "A picture of pink bathroom sink and a mirror.",
    "people standing in line beside a food truck",
    "Single sheep in a field looking back at camera.",
    "A few pizza slices next to a couple of bread slices.",
    "Two giraffe standing in a field next to trees.",
    "A person in rubber boots and a rain coat seated on a bench.",
    "A giraffe standing next to a bamboo building.",
    "A large combination pizza with two pieces gone.",
    "A bearded man in a suit eating pizza.",
    "An old yellow train is waiting at the station.",
    "People flying kites in a park next to a lake.",
    "A young man is body surfing and paddling in the water.",
    "A very large stuffed giraffe posed looking out a window.",
    "A dim room with toilet bowls lined along the wall",
    "Two trains parked at a train station as passenger wait to board them.",
    "A slightly made/messy bed against the corner in a white room.",
    "A bench sit in front of a blue and yellow train.",
    "A man is skate boarding down a path and a dog is running by his side.",
    "A man on a long exposure picture riding an electric skateboard.",
    "A person on skis and with poles in the snow and facing the blue sky.",
    "a cake with a section missing sitting next to a burning candle",
    "a woman selling jewelry laid on a blanket on the sidewalk",
    "A group of people on a tennis court.",
    "Two large elephants walking behind a wire fence on green grass.",
    "a woman on a tennis court getting ready to serve the ball",
    "Tabby cat with green eyes wearing a hat",
    "A small, white formica kitchen with a refrigerator, sink and small electrical appliances",
    "A tall clock tower with a large clock on it's face.",
    "A bathroom with a toilet, tub, mirror, window and a shower pole.",
    "A group of people standing on a snow covered hill."
]

# -------------------------------
# 4ï¸�âƒ£ Generate images + submission.csv
# -------------------------------
print(f"ğŸ�¨ Generating {len(prompts)} images in: {OUTPUT_DIR}...")
config = create_dreamlayer_config()
save_config(config, OUTPUT_DIR)
generator = DreamLayerGenerator(config, local_model_path=LOCAL_MODEL_PATH)

results = []
for i,p in enumerate(prompts,1):
    try:
        img_path = generator.generate_image(p,i,OUTPUT_DIR)
        results.append({"prompt_id":i,"prompt":p,"image_path":img_path})
        print(f"âœ… {i}: {img_path}")
    except Exception as e:
        print(f"â�Œ Failed {i}: {e}")
        results.append({"prompt_id":i,"prompt":p,"image_path":None})

submission_csv = os.path.join(OUTPUT_DIR,"submission.csv")
pd.DataFrame(results)[["prompt_id","prompt"]].to_csv(submission_csv,index=False)
print(f"\nğŸ“„ submission.csv created at {submission_csv}")

# -------------------------------
# 5ï¸�âƒ£ Display images in grid
# -------------------------------
cols = 4
html="<div style='display:grid;grid-template-columns:repeat({0},1fr);gap:15px;'>".format(cols)
for r in results:
    pth=r["image_path"]
    if pth and os.path.exists(pth):
        with open(pth,"rb") as f:
            b64=base64.b64encode(f.read()).decode()
        html+=f"<div style='border:1px solid #ccc;padding:5px;text-align:center;'><img src='data:image/png;base64,{b64}' style='width:100%;height:200px;object-fit:cover;'><br><span style='font-size:10px;'>{r['prompt_id']}: {r['prompt'][:50]}...</span></div>"
html+="</div>"
display(HTML(html))

print("\nğŸ�‰ All images generated and displayed successfully!")



# =====================================================
# ğŸ–¼ï¸� DREAMLAYER IMAGE GENERATION + KAGGLE SUBMISSION
# =====================================================

!pip install diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm kaggle --quiet
!pip install diffusers==0.24.0 transformers==4.38.2 accelerate safetensors torch torchvision pandas Pillow tqdm kaggle --quiet
!pip install --upgrade diffusers transformers accelerate safetensors torch torchvision pandas Pillow tqdm kaggle --quiet

import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display, HTML
import base64

# -------------------------------
# 1ï¸�âƒ£ Setup paths
# -------------------------------
BASE_DIR = "/kaggle/working/text-to-image-challenge-ishitabahamnia/text-to-image-generation-challenge-nbooke6a7355747-/kaggle/working"
OUTPUT_DIR = os.path.join(BASE_DIR, "dreamlayer_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 2ï¸�âƒ£ Functions & Classes
# -------------------------------
def create_dreamlayer_config():
    return {
        "model": {"name":"runwayml/stable-diffusion-v1-5", "revision":"main", "torch_dtype":"float16"},
        "scheduler":{"num_inference_steps":25},
        "generation":{"width":512,"height":512,"guidance_scale":7.5,"seed":42},
        "device":"cuda" if torch.cuda.is_available() else "cpu"
    }

def save_config(config, output_dir):
    import json
    config_path = os.path.join(output_dir, "config-dreamlayer.json")
    with open(config_path,'w') as f:
        json.dump(config,f,indent=2)
    return config_path

class DreamLayerGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.setup_model()
    
    def setup_model(self):
        print("ğŸš€ Loading Stable Diffusion model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config["model"]["name"],
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        if self.device=="cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_memory_efficient_attention()
        print("âœ… Model loaded successfully!")

    def generate_image(self, prompt, prompt_id, output_dir):
        from PIL import Image
        output_path = os.path.join(output_dir,f"{prompt_id:04d}.png")
        if os.path.exists(output_path):
            return output_path
        with torch.autocast(self.device):
            img = self.pipe(
                prompt,
                width=self.config["generation"]["width"],
                height=self.config["generation"]["height"],
                guidance_scale=self.config["generation"]["guidance_scale"],
                num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                generator=torch.Generator(device=self.device).manual_seed(
                    self.config["generation"]["seed"] + prompt_id
                )
            ).images[0]
            img.save(output_path)
        return output_path
self.pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(self.device)
class DreamLayerGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.pipe = None
        self.setup_model()

    def setup_model(self):
        try:
            print("ğŸš€ Loading Stable Diffusion model...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_memory_efficient_attention()
            print("âœ… Model loaded successfully!")
        except Exception as e:
            print(f"â�Œ Failed to load model: {e}")
            raise

    def generate_image(self, prompt, prompt_id, output_dir):
        try:
            from PIL import Image
            output_path = os.path.join(output_dir, f"{prompt_id:04d}.png")
            if os.path.exists(output_path):
                return output_path

            with torch.autocast(self.device):
                img = self.pipe(
                    prompt,
                    width=self.config["generation"]["width"],
                    height=self.config["generation"]["height"],
                    guidance_scale=self.config["generation"]["guidance_scale"],
                    num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.config["generation"]["seed"] + prompt_id
                    )
                ).images[0]
                img.save(output_path)
            return output_path
        except Exception as e:
            print(f"â�Œ Failed to generate image for prompt {prompt_id}: {e}")
            return None
class DreamLayerGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.pipe = None
        self.setup_model()

    def setup_model(self):
        try:
            print("ğŸš€ Loading Stable Diffusion model...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config["model"]["name"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_memory_efficient_attention()
            print("âœ… Model loaded successfully!")
        except Exception as e:
            print(f"â�Œ Failed to load model: {e}")
            raise

    def generate_image(self, prompt, prompt_id, output_dir):
        try:
            from PIL import Image
            output_path = os.path.join(output_dir, f"{prompt_id:04d}.png")
            if os.path.exists(output_path):
                return output_path

            with torch.autocast(self.device):
                img = self.pipe(
                    prompt,
                    width=self.config["generation"]["width"],
                    height=self.config["generation"]["height"],
                    guidance_scale=self.config["generation"]["guidance_scale"],
                    num_inference_steps=self.config["scheduler"]["num_inference_steps"],
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.config["generation"]["seed"] + prompt_id
                    )
                ).images[0]
                img.save(output_path)
            return output_path
        except Exception as e:
            print(f"â�Œ Failed to generate image for prompt {prompt_id}: {e}")
            return None

# -------------------------------
# 3ï¸�âƒ£ Prompts
# -------------------------------
prompts = [
    # 49 prompts here...
    "A man baking and preparing donuts to sell at shop.",
    "A zebra chews a flower in a fenced in field.",
    "A person skiing down a mountain kicking his leg up.",
    "An airplane in route with a cloudy sky behind it.",
    "A white and blue truck parked in the middle of a dirt road.",
    "PIcked peach flowers sit in a vase with water.",
    "Sheep are on a grassy field and one of them is a white and black baby.",
    "A blue rusted train engine sitting on top of rail tracks.",
    "An open laptop computer sitting on top of a wooden table.",
    "A woman twirling an umbrella with flowers on it.",
    "A woman sitting in a restaurant with Mexican food on her plate",
    "A woman holding a purse and a cellphone.",
    "A cat reaching for a knife that has it's blade out.",
    "The man smiles with a slice of pizza while next to a friend.",
    "Night picture of a car parked and some parking lights in the distance.",
    "A professional photograph of a motorcycle rider in the air.",
    "A person putting doughnuts into a bag in a shop.",
    "A very ornately decorated and brightly colored clock.",
    "A young man that is having some wine and something to eat.",
    "A picture of pink bathroom sink and a mirror.",
    "people standing in line beside a food truck",
    "Single sheep in a field looking back at camera.",
    "A few pizza slices next to a couple of bread slices.",
    "Two giraffe standing in a field next to trees.",
    "A person in rubber boots and a rain coat seated on a bench.",
    "A giraffe standing next to a bamboo building.",
    "A large combination pizza with two pieces gone.",
    "A bearded man in a suit eating pizza.",
    "An old yellow train is waiting at the station.",
    "People flying kites in a park next to a lake.",
    "A young man is body surfing and paddling in the water.",
    "A very large stuffed giraffe posed looking out a window.",
    "A dim room with toilet bowls lined along the wall",
    "Two trains parked at a train station as passenger wait to board them.",
    "A slightly made/messy bed against the corner in a white room.",
    "A bench sit in front of a blue and yellow train.",
    "A man is skate boarding down a path and a dog is running by his side.",
    "A man on a long exposure picture riding an electric skateboard.",
    "A person on skis and with poles in the snow and facing the blue sky.",
    "a cake with a section missing sitting next to a burning candle",
    "a woman selling jewelry laid on a blanket on the sidewalk",
    "A group of people on a tennis court.",
    "Two large elephants walking behind a wire fence on green grass.",
    "a woman on a tennis court getting ready to serve the ball",
    "Tabby cat with green eyes wearing a hat",
    "A small, white formica kitchen with a refrigerator, sink and small electrical appliances",
    "A tall clock tower with a large clock on it's face.",
    "A bathroom with a toilet, tub, mirror, window and a shower pole.",
    "A group of people standing on a snow covered hill."
]

# -------------------------------
# 4ï¸�âƒ£ Generate images + submission.csv
# -------------------------------
print(f"ğŸ�¨ Generating {len(prompts)} images in: {OUTPUT_DIR}...")
config = create_dreamlayer_config()
save_config(config, OUTPUT_DIR)
generator = DreamLayerGenerator(config)

results = []
for i,p in enumerate(prompts,1):
    try:
        img_path = generator.generate_image(p,i,OUTPUT_DIR)
        results.append({"prompt_id":i,"prompt":p,"image_path":img_path})
        print(f"âœ… {i}: {img_path}")
    except Exception as e:
        print(f"â�Œ Failed {i}: {e}")
        results.append({"prompt_id":i,"prompt":p,"image_path":None})

submission_csv = os.path.join(OUTPUT_DIR,"submission.csv")
pd.DataFrame(results)[["prompt_id","prompt"]].to_csv(submission_csv,index=False)
print(f"\nğŸ“„ submission.csv created at {submission_csv}")

# -------------------------------
# 5ï¸�âƒ£ Display images in grid
# -------------------------------
cols = 4
html="<div style='display:grid;grid-template-columns:repeat({0},1fr);gap:15px;'>".format(cols)
for r in results:
    pth=r["image_path"]
    if pth and os.path.exists(pth):
        with open(pth,"rb") as f:
            b64=base64.b64encode(f.read()).decode()
        html+=f"<div style='border:1px solid #ccc;padding:5px;text-align:center;'><img src='data:image/png;base64,{b64}' style='width:100%;height:200px;object-fit:cover;'><br><span style='font-size:10px;'>{r['prompt_id']}: {r['prompt'][:50]}...</span></div>"
html+="</div>"
display(HTML(html))

print("\nğŸ�‰ All images generated and displayed successfully!")



import os
import pandas as pd

# Define the output directory
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of 49 prompts
prompts = [
    "A man baking and preparing donuts to sell at shop.",
    "A zebra chews a flower in a fenced in field.",
    "A person skiing down a mountain kicking his leg up.",
    "An airplane in route with a cloudy sky behind it.",
    "A white and blue truck parked in the middle of a dirt road.",
    "Picked peach flowers sit in a vase with water.",
    "Sheep are on a grassy field and one of them is a white and black baby.",
    "A blue rusted train engine sitting on top of rail tracks.",
    "An open laptop computer sitting on top of a wooden table.",
    "A woman twirling an umbrella with flowers on it.",
    "A woman sitting in a restaurant with Mexican food on her plate",
    "A woman holding a purse and a cellphone.",
    "A cat reaching for a knife that has its blade out.",
    "The man smiles with a slice of pizza while next to a friend.",
    "Night picture of a car parked and some parking lights in the distance.",
    "A professional photograph of a motorcycle rider in the air.",
    "A person putting doughnuts into a bag in a shop.",
    "A very ornately decorated and brightly colored clock.",
    "A young man that is having some wine and something to eat.",
    "A picture of pink bathroom sink and a mirror.",
    "People standing in line beside a food truck",
    "Single sheep in a field looking back at camera.",
    "A few pizza slices next to a couple of bread slices.",
    "Two giraffes standing in a field next to trees.",
    "A person in rubber boots and a rain coat seated on a bench.",
    "A giraffe standing next to a bamboo building.",
    "A large combination pizza with two pieces gone.",
    "A bearded man in a suit eating pizza.",
    "An old yellow train is waiting at the station.",
    "People flying kites in a park next to a lake.",
    "A young man is body surfing and paddling in the water.",
    "A very large stuffed giraffe posed looking out a window.",
    "A dim room with toilet bowls lined along the wall",
    "Two trains parked at a train station as passengers wait to board them.",
    "A slightly made/messy bed against the corner in a white room.",
    "A bench sits in front of a blue and yellow train.",
    "A man is skateboarding down a path and a dog is running by his side.",
    "A man on a long exposure picture riding an electric skateboard.",
    "A person on skis and with poles in the snow and facing the blue sky.",
    "A cake with a section missing sitting next to a burning candle",
    "A woman selling jewelry laid on a blanket on the sidewalk",
    "A group of people on a tennis court.",
    "Two large elephants walking behind a wire fence on green grass.",
    "A woman on a tennis court getting ready to serve the ball",
    "Tabby cat with green eyes wearing a hat",
    "A small, white formica kitchen with a refrigerator, sink, and small electrical appliances",
    "A tall clock tower with a large clock on its face.",
    "A bathroom with a toilet, tub, mirror, window, and a shower pole.",
    "A group of people standing on a snow-covered hill."
]

# Generate the results list
results = []
for prompt_id, prompt in enumerate(prompts, start=1):
    image_path = os.path.join(OUTPUT_DIR, f"{prompt_id:04d}.png")
    results.append({
        "prompt_id": prompt_id,
        "prompt": prompt,
        "image_path": image_path
    })

# Create a DataFrame and save to CSV
submission_df = pd.DataFrame(results)
submission_csv_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission_df.to_csv(submission_csv_path, index=False)

print(f"CSV file generated at: {submission_csv_path}")



!pip install clip-client  # Install CLIP client if not already installed
import clip
import torch
from PIL import Image
!pip install git+https://github.com/openai/CLIP.git
import clip
import torch
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to calculate text-image similarity
def calculate_similarity(text, image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    similarity = torch.cosine_similarity(image_features, text_features).item()
    return round(similarity, 2)  # Round to 2 decimal places

# Generate the results list with CLIP similarity scores
results = []
for prompt_id, prompt in enumerate(prompts, start=1):
    image_path = os.path.join(OUTPUT_DIR, f"{prompt_id:04d}.png")
    if os.path.exists(image_path):
        f1_score = calculate_similarity(prompt, image_path)
    else:
        f1_score = 0.0  # Default score if image doesn't exist

    results.append({
        "prompt_id": prompt_id,
        "prompt": prompt,
        "image_path": image_path,
        "F1_score": f1_score
    })

# Create a DataFrame and save to CSV
submission_df = pd.DataFrame(results)
submission_csv_path = os.path.join(OUTPUT_DIR, "submission_with_f1.csv")
submission_df.to_csv(submission_csv_path, index=False)

print(f"CSV file with CLIP similarity scores generated at: {submission_csv_path}")



from IPython.display import display, HTML
import base64

# Display images in a grid
cols = 4
html = f"<div style='display:grid;grid-template-columns:repeat({cols},1fr);gap:15px;'>"

for result in results:
    image_path = result["image_path"]
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode()
        html += f"""
        <div style='border:1px solid #ccc;padding:5px;text-align:center;'>
            <img src='data:image/png;base64,{b64_image}' style='width:100%;height:200px;object-fit:cover;'>
            <br>
            <span style='font-size:10px;'>{result['prompt_id']}: {result['prompt'][:50]}...</span>
        </div>
        """
html += "</div>"
display(HTML(html))



import pandas as pd
import os

# Example: Assume you have a list of results with prompt_id, prompt, and image_path
results = [
    {"prompt_id": 1, "prompt": "A man baking and preparing donuts to sell at shop.", "image_path": "/kaggle/working/dreamlayer_output/0001.png", "F1_score": 0.95},
    {"prompt_id": 2, "prompt": "A zebra chews a flower in a fenced in field.", "image_path": "/kaggle/working/dreamlayer_output/0002.png", "F1_score": 0.92},
    # Add more entries as needed
]

# Create a DataFrame
submission_df = pd.DataFrame(results)

# Save to CSV
submission_csv_path = os.path.join(OUTPUT_DIR, "submission_with_f1.csv")
submission_df.to_csv(submission_csv_path, index=False)

print(f"Submission CSV with F1 scores saved to: {submission_csv_path}")
from sklearn.metrics import f1_score

# Example ground truth and predicted labels
ground_truth = [1, 0, 1, 1, 0]  # Replace with actual ground truth
predicted = [1, 0, 1, 0, 0]      # Replace with predicted labels

# Calculate F1 score
f1 = f1_score(ground_truth, predicted, average='weighted')  # Use 'binary' for binary classification
print(f"F1 Score: {f1:.4f}")

# Add F1 score to results
for i, result in enumerate(results):
    result["F1_score"] = f1  # Assign the same F1 score to all entries (or calculate per image)



import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the output directory
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
image_files.sort()  # Sort the files to ensure they are displayed in order

# Define the number of rows and columns for the grid
rows = 10
cols = 5

# Create a figure
fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through the images and display them
for i, ax in enumerate(axes):
    if i < len(image_files):
        img_path = os.path.join(OUTPUT_DIR, image_files[i])
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"ID: {i+1}", fontsize=8)
    else:
        ax.axis('off')  # Hide empty subplots

plt.tight_layout()
plt.show()



import os
from IPython.display import display, HTML
import base64

# Define the output directory
OUTPUT_DIR = "/kaggle/working/dreamlayer_output"

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
image_files.sort()  # Sort the files to ensure they are displayed in order

# Display images in a grid
cols = 5
html = f"<div style='display:grid;grid-template-columns:repeat({cols},1fr);gap:15px;'>"

for i, image_file in enumerate(image_files):
    image_path = os.path.join(OUTPUT_DIR, image_file)
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode('utf-8')
        html += f"""
        <div style='border:1px solid #ccc;padding:5px;text-align:center;'>
            <img src='data:image/png;base64,{b64_image}' style='width:100%;height:200px;object-fit:cover;'>
            <br>
            <span style='font-size:10px;'>{i+1}</span>
        </div>
        """
html += "</div>"
display(HTML(html))


