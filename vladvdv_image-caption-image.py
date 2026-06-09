# Constants
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_LENGTH = 16
NUM_BEAMS = 4
GENERATION_KWARGS = {"max_length": MAX_LENGTH, "num_beams": NUM_BEAMS}
IMAGES_PATH = "/kaggle/input/gen-ai-competition/data/"
OUTPUT_CSV_PATH = "sample_sub.csv"


import os
import torch
import csv
import numpy as np
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel
%matplotlib inline
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
from typing import List
from diffusers import StableDiffusionXLPipeline


class Utils:
    """Utility class containing helper functions for image captioning and embedding extraction."""

    @staticmethod
    def get_all_files(folder_path: str) -> list:
        """Returns a list of all files from the specified folder."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Collect all files in the given folder
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    @staticmethod
    def save_embeddings_to_csv(image_paths: list, embeddings_list: list, output_csv: str):
        """Saves image captions and their corresponding embeddings to a CSV file."""
        
        embeddings_list = np.array(embeddings_list)  # Convert embeddings to a NumPy array
        image_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]  # Extract image names

        # Writing to the CSV file
        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write CSV header
            header = ["Embedding_id", "Embedding_value"]
            writer.writerow(header)

            # Writing image embeddings in long format
            for group_id, embedding in zip(image_names, embeddings_list):
                for embedding_id, embedding_value in enumerate(embedding):
                    writer.writerow([f"{group_id}_{embedding_id}", embedding_value])

        print(f"Embeddings saved to {output_csv}")

    @staticmethod
    def load_image(image_path: str) -> Image:
        """Loads an image from disk, ensuring it is in RGB format."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        return image
        
    @staticmethod
    def show_images(image_paths: list, captions: list = None):
        """Displays up to 8 images in a grid with optional captions shown below each image."""
        num_images = min(len(image_paths), 8)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for idx in range(num_images):
            row = idx // 4
            col = idx % 4
            ax = axes[row, col]

            image = Utils.load_image(image_paths[idx])
            ax.imshow(image)
            ax.axis("off")

            if captions and idx < len(captions):
                ax.text(
                    0.5, -0.1, captions[idx],
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=10, wrap=True
                )

        # Hide unused subplots if less than 8 images
        for idx in range(num_images, 8):
            row = idx // 4
            col = idx % 4
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()
    @staticmethod
    def display_images_with_captions_and_titles(list1, list2, list3, captions, row_titles):
        """
        Takes 3 lists of image paths, a list of captions, and row titles, and displays images horizontally with captions below
        and titles above each row of images. All images in a row will have the same title from the `row_titles` list.
        
        Args:
            list1 (list): List of image paths for the first set of images.
            list2 (list): List of image paths for the second set of images.
            list3 (list): List of image paths for the third set of images.
            captions (list): List of strings containing captions for the images.
            row_titles (list): List of strings containing titles for each row of images.
        """
        # Ensure the lists are of the same length and captions list length matches the number of rows
        min_length = min(len(list1), len(list2), len(list3))
        
        # Set up a figure for displaying the images
        fig, axes = plt.subplots(nrows=min_length, ncols=3, figsize=(15, 5 * min_length))
        
        for i in range(min_length):
            # Open images from each list
            img1 = Image.open(list1[i])
            img2 = Image.open(list2[i])
            img3 = Image.open(list3[i])
            
            # Display the images in the subplots
            axes[i, 0].imshow(img1)
            axes[i, 1].imshow(img2)
            axes[i, 2].imshow(img3)
            
            # Remove axis labels for a cleaner view
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')
            
       # Display captions below each image using fig.text
            axes[i, 0].text(0.5, -0.1, captions[i], ha='center', va='top', fontsize=8, transform=axes[i, 0].transAxes)
            axes[i, 1].text(0.5, -0.1, captions[i], ha='center', va='top', fontsize=8, transform=axes[i, 1].transAxes)
            axes[i, 2].text(0.5, -0.1, captions[i], ha='center', va='top', fontsize=8, transform=axes[i, 2].transAxes)
            
            # Display the row title above the images (same title for all images in the row)
            axes[i, 0].set_title(row_titles[0], fontsize=16, weight='bold', loc='center')
            axes[i, 1].set_title(row_titles[1], fontsize=16, weight='bold', loc='center')
            axes[i, 2].set_title(row_titles[2], fontsize=16, weight='bold', loc='center')
        
        # Display the plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # Increase this value to leave more space between rows
        plt.show()
    


# Step 1: Load and process images from the folder
image_paths = Utils.get_all_files(IMAGES_PATH)
print(f"Processing {len(image_paths)} images...")
# View some images
Utils.show_images(image_paths[:8])


class ImageCaptioningModel:
    """Handles image captioning using the VisionEncoderDecoderModel from HuggingFace Transformers."""

    def __init__(self, model_name: str):
        """Initializes the model, processor, and tokenizer."""
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def process_images(self, image_paths: list) -> torch.Tensor:
        """Processes the input image paths and returns the pixel values for model inference."""
        images = [Utils.load_image(image_path) for image_path in image_paths]
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        return pixel_values.to(self.device)

    def generate_caption(self, image_paths: list) -> list:
        """Generates captions for a list of input image paths."""
        pixel_values = self.process_images(image_paths)
        output_ids = self.model.generate(pixel_values, **GENERATION_KWARGS)
        captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [caption.strip() for caption in captions]


# Step 2: Generate captions using the image captioning model
captioning_model = ImageCaptioningModel(MODEL_NAME)
captions = captioning_model.generate_caption(image_paths)
Utils.show_images(image_paths[:8], captions[:8])



class SdTextToImageModel:
    """Generates images from text prompts using Stable Diffusion."""

    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Initializes the Stable Diffusion pipeline."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.pipe = self.pipe.to(self.device)

    def generate_images(self, captions: List[str], output_dir: str = "./generated_images_sd") -> List[str]:
        """
        Generates images from a list of captions and saves them.

        Args:
            captions (List[str]): A list of text prompts to generate images from.
            output_dir (str): Directory to save the generated images.

        Returns:
            List[str]: Paths to the saved images.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for idx, prompt in enumerate(captions):
            image: Image.Image = self.pipe(prompt).images[0]
            file_path = os.path.join(output_dir, f"generated_image_{idx}.png")
            image.save(file_path)
            saved_paths.append(file_path)

        return saved_paths


model = SdTextToImageModel("runwayml/stable-diffusion-v1-5")
image_paths_sd_15 = model.generate_images(captions[:8])
Utils.show_images(image_paths_sd_15[:8], captions[:8])


class SdXlTextToImageModel:
    """Generates images from text prompts using Stable Diffusion."""

    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Initializes the Stable Diffusion pipeline."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.pipe = self.pipe.to(self.device)

    def generate_images(self, captions: List[str], output_dir: str = "./generated_images_sdxl") -> List[str]:
        """
        Generates images from a list of captions and saves them.

        Args:
            captions (List[str]): A list of text prompts to generate images from.
            output_dir (str): Directory to save the generated images.

        Returns:
            List[str]: Paths to the saved images.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for idx, prompt in enumerate(captions):
            image: Image.Image = self.pipe(prompt).images[0]
            file_path = os.path.join(output_dir, f"generated_image_{idx}.png")
            image.save(file_path)
            saved_paths.append(file_path)

        return saved_paths


model = SdXlTextToImageModel("stabilityai/stable-diffusion-xl-base-1.0")
image_paths_sd_xl = model.generate_images(captions[:8])
Utils.show_images(image_paths_sd_xl[:8], captions[:8])


Utils.display_images_with_captions_and_titles(image_paths[:8],image_paths_sd_15,image_paths_sd_xl,captions[:8],["Original Images","Stable Diffusion 1.5","Stable Diffusion XL"])


class CLIPModelEvaluator:
    """Handles extracting text embeddings from CLIP model."""

    def __init__(self, clip_model_name: str):
        """Initializes the CLIP model and processor."""
        self.model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    def get_text_embeddings(self, captions: list):
        """Generates text embeddings for the provided captions using CLIP."""
        inputs = self.processor(text=captions, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs.cpu().numpy().tolist()  # Return text embeddings


# Step 3: Extract text embeddings for captions using the CLIP model
clip_evaluator = CLIPModelEvaluator(CLIP_MODEL_NAME)
captions_embeddings = clip_evaluator.get_text_embeddings(captions)


Utils.save_embeddings_to_csv(image_paths, captions_embeddings, OUTPUT_CSV_PATH)

