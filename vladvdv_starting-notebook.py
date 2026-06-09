import os
import torch
import csv
import numpy as np
import pandas as pd
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel
%matplotlib inline
import matplotlib.pyplot as plt


# Constants
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_LENGTH = 16
NUM_BEAMS = 4
GENERATION_KWARGS = {"max_length": MAX_LENGTH, "num_beams": NUM_BEAMS}
IMAGES_PATH = "/kaggle/input/gen-ai-competition/data/"
OUTPUT_CSV_PATH = "sample_sub.csv"


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

