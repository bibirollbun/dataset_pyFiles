# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install timm==1.0.17
!pip install transformers==4.53.2


# Import required libraries
import time
import torchaudio
import traceback
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
import logging
from typing import Union, Tuple
from dataclasses import dataclass
import os
import kagglehub
import json






# GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b-it")
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


# Configuration class
@dataclass
class Config:
    # Gemma3n model configuration
    MODEL_NAME: str = GEMMA_PATH
    
    # Generation parameters
    MAX_NEW_TOKENS: int = 512
    
    # Device configuration
    TORCH_DTYPE: str = torch.bfloat16
    DEVICE_MAP: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Image preprocessing
    IMAGE_SIZE: int = 512
    
    # Hugging Face token (if needed)
    HF_TOKEN: str = ""

# Initialize config
config = Config()
print(f"Model: {config.MODEL_NAME}")
print(f"Device: {config.DEVICE_MAP}")
print(f"Data type: {config.TORCH_DTYPE}")


class ClothesClassifer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the model and processor"""
        try:
            self.logger.info(f"Loading model: {self.config.MODEL_NAME}")

            # Load processor
            kwargs = {}
            if self.config.HF_TOKEN:
                kwargs["token"] = self.config.HF_TOKEN

            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_NAME, **kwargs
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=self.config.TORCH_DTYPE,
                device_map=self.config.DEVICE_MAP,
            )

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def process_audio(self,audio_path):
        waveform, sr = torchaudio.load(audio_path)  
        waveform = waveform.mean(dim=0).numpy()
        return waveform
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to meet Gemma3n requirements (512x512)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        target_size = (512, 512)

        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            # Width is larger
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            # Height is larger or equal
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        # Resize image maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with target size and paste the resized image
        processed_image = Image.new(
            "RGB", target_size, (255, 255, 255)
        )  # White background

        # Calculate position to center the image
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2

        processed_image.paste(image, (x_offset, y_offset))

        return processed_image
        
    def gemma_output(self,prompt,processed_image,audio_waveform_or_bytes):
        content = [
                        {"type": "image", "image": processed_image},
                        {"type": "audio", "audio": audio_waveform_or_bytes},  
                        {"type": "text", "text": prompt}
                   ]
        
        messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.MAX_NEW_TOKENS,
            disable_compile=True,
        )
        response = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
        )[0]
        # print('Reponse',response)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                response = response[response.index("{"):]
            parsed = json.loads(response)
            print("ğŸ§¾ Parsed JSON:", parsed)
        except Exception as e:
            print(f"âš ï¸� Could not parse JSON: {e}")
            print("Raw response:", response)
        return parsed
        
    def classify_image(self, image,audio):
        
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Load and process image
            if isinstance(image, str):
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a PIL Image or file path")
            
            # Preprocess image to meet Gemma3n requirements
            processed_image = self.preprocess_image(image)
            prompt1 = """
             You are a fashion image assistant. Is the image showing a clothing item?

                Respond in JSON only as:
                {
                  "relevant": true
                }
        
                or
        
                {
                  "relevant": false
                }
            
            """
            prompt1_response = self.gemma_output(prompt1,processed_image,audio)
            if not isinstance(prompt1_response, dict) or not prompt1_response.get("relevant", False):
                return "The image is not of a clothing item. Please provide a clothing article."
            
            prompt2 = """You are a fashion image assistant that classifies clothing for blind users and color blind user. Your task is to describe the clothing shown in an image in a short, structured, and informative way.

                INSTRUCTIONS:
                - If the image does not have any clothes related skip everything 
                - Classify the clothing as **formal** or **informal**
                - Specify the **type of clothing** (e.g., shirt, t-shirt, jeans, pants, dress, jacket, etc.)
                - Mention the **color** and **notable features** (like fabric, collar type, sleeve type, fit, etc.)
                - Your tone should be simple and helpful â€” like explaining to a blind person what they are wearing.
                - Keep your output **crisp and under 80 words**.
                
                RESPONSE FORMAT:
                Return your output strictly in this JSON format:
                ```json
                {
                    "description": "This is a formal white shirt. It has long sleeves, button-up front, classic collar, tailored fit. Appears to be made of a smooth fabric, likely cotton or a cotton blend. Cuffs have buttons."
                }
                """
            # Prepare messages with system prompt and user query
            prompt2_response = self.gemma_output(prompt2, processed_image,audio)
           
            prompt2_response = prompt2_response.get("description")

            # required_fields = ["formality", "type", "color", "features"]
            # if not all(field in prompt2_response for field in required_fields):
            #     print("ğŸ›‘ This doesn't appear to be a clothing item.")
            #     parsed = {
            #         "message": "This image does not seem to contain clothing. Please provide an image of a clothing article."
            #     }
            #     prompt2_response = parsed
            #     return prompt2_response
            # else:
                # {
                #   "formality": "<formal or informal>",
                #   "type": "<type of clothing>",
                #   "color": "<main color(s)>",
                #   "features": "<key features like fabric, collar, sleeve, etc.>"
                # }
            return prompt2_response

        except Exception as e:
            self.logger.error(f"Error during classification: {str(e)}")

            traceback.print_exc()
            return "Error", f"Classification failed: {str(e)}", 0

print("Clothes classifer class defined successfully!")


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    try:
        if url.startswith(('http://', 'https://')):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        else:
            image = Image.open(url)
            return image
    except Exception as e:
        logger.error(f"Error loading image from URL: {str(e)}")
        raise

print("Enhanced utility functions defined!")


print("Initializing classifier...")
classifier = ClothesClassifer(config)

print("Loading model... This may take a few minutes on first run.")
classifier.load_model()
print("âœ… Model loaded successfully!")


# !nvidia-smi




image_dir = "/kaggle/input/clothes-images/clothes_images"
audio_path = "/kaggle/input/voice-instructions/user_audio.wav"
start_time = time.time()
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        print(f"\n{'-'*50}")
        # print(f"ğŸ”� TESTING {material.upper()} FROM URL")
        print(f"Filename: {filename}")
        print('-'*50)
        
        try:
            # Load image from URL
            iter_start = time.time()
            image_path = os.path.join(image_dir, filename)
            image = load_image_from_url(image_path)
            processed_audio = classifier.process_audio(audio_path)
            display(image)
            
            # Classify the image
            response = classifier.classify_image(image,processed_audio)
            # print(f"Final response: {response}")
            iter_end = time.time()
            print(f"ğŸ•’ Iteration Time: {iter_end - iter_start:.2f} seconds")

        except Exception as e:
            print(f"Error: {str(e)}")
        torch.cuda.empty_cache()
end_time = time.time()
print(f"\nâœ… Total Execution Time: {end_time - start_time:.2f} seconds")





# import sounddevice as sd
# import numpy as np
# import scipy.io.wavfile

# fs = 16000  # sample rate
# duration = 6 # seconds

# print("Recording...")
# audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# sd.wait()
# print("Recording complete.")

# scipy.io.wavfile.write("user_audio.wav", fs, audio)

