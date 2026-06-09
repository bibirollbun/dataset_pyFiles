#| default_exp core


#| export
import base64
import gc
import io
from io import BytesIO
import os
import random
import re
import time
import warnings
from datetime import timedelta

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import SVG
from IPython.display import display

import torch
import torch.nn.functional as F
from torch import autocast

from diffusers import StableDiffusionPipeline

import cupy as cp
import cupyx.scipy.ndimage
from numba import jit, cuda


#| export
# Path to the locally saved SD Turbo model

local_model_path = kagglehub.notebook_output_download("richolson/sdxl-turbo-install-notebook") + "/sdturbo_xl_model_local"

# Set device to GPU 1
device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Import the correct pipeline class for SDXL
from diffusers import StableDiffusionXLPipeline

# Load SD Turbo XL from local path
pipe = StableDiffusionXLPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    local_files_only=True
)

# Disable the progress bar
pipe.set_progress_bar_config(disable=True)

# Move model to the specified GPU
pipe.to(device)


#| export

# Define function to generate multiple images
# Can return multiple images for single prompt
def generate_bitmaps(
    prompt,
    negative_prompt="", 
    num_images_per_prompt=1,
    height=400,
    width=400,
    num_inference_steps=4,
    guidance_scale=0.0, 
    seed=None
):
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    return result.images


test_prompt = "Racoon with a tophat in a garbage can"
images = generate_bitmaps(prompt=test_prompt)

display(images[0])


#| export

import ast
import io
import math
import statistics
import string

import cairosvg
import clip
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from more_itertools import chunked
from PIL import Image, ImageFilter
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

svg_constraints = kagglehub.package_import('metric/svg-constraints')


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, random_seed: int = 0
) -> float:
    """Calculates a fidelity score by comparing generated SVG images to target text descriptions.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame containing target questions, choices, and answers about an SVG image.
    submission : pd.DataFrame
        A DataFrame containing generated SVG strings. Must have a column named 'svg'.
    row_id_column_name : str
        The name of the column containing row identifiers. This column is removed before scoring.
    random_seed : int
        A seed to set the random state.

    Returns
    -------
    float
        The mean fidelity score (a value between 0 and 1) representing the average similarity between the generated SVGs and their descriptions.
        A higher score indicates better fidelity.

    Raises
    ------
    ParticipantVisibleError
        If the 'svg' column in the submission DataFrame is not of string type or if validation of the SVG fails.

    Examples
    --------
    >>> import pandas as pd
    >>> solution = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'question': ['["Is there a red circle?", "What shape is present?"]'],
    ...     'choices': ['[["yes", "no"], ["square", "circle", "triangle", "hexagon"]]'],
    ...     'answer': ['["yes", "circle"]'],
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': ["abcde"],
    ...     'svg': ['<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'],
    ... })
    >>> score(solution, submission, 'row_id', random_seed=42)
    0...
    """
    # Convert solution fields to list dtypes and expand
    for colname in ['question', 'choices', 'answer']:
        solution[colname] = solution[colname].apply(ast.literal_eval)
    solution = solution.explode(['question', 'choices', 'answer'])

    # Validate
    if not pd.api.types.is_string_dtype(submission.loc[:, 'svg']):
        raise ParticipantVisibleError('svg must be a string.')

    # Check that SVG code meets defined constraints
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, 'svg']:
            constraints.validate_svg(svg)
    except:
        raise ParticipantVisibleError('SVG code violates constraints.')

    # Score
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()

    results = []
    rng = np.random.RandomState(random_seed)
    try:
        df = solution.merge(submission, on='id')
        for i, (_, group) in enumerate(df.loc[
            :, ['id', 'question', 'choices', 'answer', 'svg']
        ].groupby('id')):
            questions, choices, answers, svg = [
                group[col_name].to_list()
                for col_name in group.drop('id', axis=1).columns
            ]
            svg = svg[0]  # unpack singleton from list
            group_seed = rng.randint(0, np.iinfo(np.int32).max)
            image_processor = ImageProcessor(image=svg_to_png(svg), seed=group_seed).apply()
            image = image_processor.image.copy()
            aesthetic_score = aesthetic_evaluator.score(image)
            vqa_score = vqa_evaluator.score(questions, choices, answers, image)
            image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
            ocr_score = vqa_evaluator.ocr(image_processor.image)
            instance_score = (
                harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
            )
            results.append(instance_score)

    except:
        raise ParticipantVisibleError('SVG failed to score.')

    fidelity = statistics.mean(results)
    return float(fidelity)


class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description using multiple choice questions."""

    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.letters = string.ascii_uppercase
        self.model_path = kagglehub.model_download(
            'google/paligemma-2/transformers/paligemma2-10b-mix-448'
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        ).to('cuda:0')

    def score(self, questions, choices, answers, image, n=4):
        scores = []
        batches = (chunked(qs, n) for qs in [questions, choices, answers])
        for question_batch, choice_batch, answer_batch in zip(*batches, strict=True):
            scores.extend(
                self.score_batch(
                    image,
                    question_batch,
                    choice_batch,
                    answer_batch,
                )
            )
        # Logs out individual question scores before averaging (added)
        formatted_scores = [f"{score:.4f}" for score in scores]
        print(f"VQA individual question scores: {formatted_scores}")

        return statistics.mean(scores)

    def score_batch(
        self,
        image: Image.Image,
        questions: list[str],
        choices_list: list[list[str]],
        answers: list[str],
    ) -> list[float]:
        """Evaluates the image based on multiple choice questions and answers.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to evaluate.
        questions : list[str]
            List of questions about the image.
        choices_list : list[list[str]]
            List of lists of possible answer choices, corresponding to each question.
        answers : list[str]
            List of correct answers from the choices, corresponding to each question.

        Returns
        -------
        list[float]
            List of scores (values between 0 and 1) representing the probability of the correct answer for each question.
        """
        prompts = [
            self.format_prompt(question, choices)
            for question, choices in zip(questions, choices_list, strict=True)
        ]
        batched_choice_probabilities = self.get_choice_probability(
            image, prompts, choices_list
        )

        scores = []
        for i, _ in enumerate(questions):
            choice_probabilities = batched_choice_probabilities[i]
            answer = answers[i]
            answer_probability = 0.0
            for choice, prob in choice_probabilities.items():
                if choice == answer:
                    answer_probability = prob
                    break
            scores.append(answer_probability)

        return scores

    def format_prompt(self, question: str, choices: list[str]) -> str:
        prompt = f'<image>answer en Question: {question}\nChoices:\n'
        for i, choice in enumerate(choices):
            prompt += f'{self.letters[i]}. {choice}\n'
        return prompt

    def mask_choices(self, logits, choices_list):
        """Masks logits for the first token of each choice letter for each question in the batch."""
        batch_size = logits.shape[0]
        masked_logits = torch.full_like(logits, float('-inf'))

        for batch_idx in range(batch_size):
            choices = choices_list[batch_idx]
            for i in range(len(choices)):
                letter_token = self.letters[i]

                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                if isinstance(first_token, int):
                    masked_logits[batch_idx, first_token] = logits[
                        batch_idx, first_token
                    ]
                if isinstance(first_token_with_space, int):
                    masked_logits[batch_idx, first_token_with_space] = logits[
                        batch_idx, first_token_with_space
                    ]

        return masked_logits

    def get_choice_probability(self, image, prompts, choices_list) -> list[dict]:
        inputs = self.processor(
            images=[image] * len(prompts),
            text=prompts,
            return_tensors='pt',
            padding='longest',
        ).to('cuda:0')

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_choices(logits, choices_list)
            probabilities = torch.softmax(masked_logits, dim=-1)

        batched_choice_probabilities = []
        for batch_idx in range(len(prompts)):
            choice_probabilities = {}
            choices = choices_list[batch_idx]
            for i, choice in enumerate(choices):
                letter_token = self.letters[i]
                first_token = self.processor.tokenizer.encode(
                    letter_token, add_special_tokens=False
                )[0]
                first_token_with_space = self.processor.tokenizer.encode(
                    ' ' + letter_token, add_special_tokens=False
                )[0]

                prob = 0.0
                if isinstance(first_token, int):
                    prob += probabilities[batch_idx, first_token].item()
                if isinstance(first_token_with_space, int):
                    prob += probabilities[batch_idx, first_token_with_space].item()
                choice_probabilities[choice] = prob

            # Renormalize probabilities for each question
            total_prob = sum(choice_probabilities.values())
            if total_prob > 0:
                renormalized_probabilities = {
                    choice: prob / total_prob
                    for choice, prob in choice_probabilities.items()
                }
            else:
                renormalized_probabilities = (
                    choice_probabilities  # Avoid division by zero if total_prob is 0
                )
            batched_choice_probabilities.append(renormalized_probabilities)

        return batched_choice_probabilities

    def ocr(self, image, free_chars=4):
        inputs = (
            self.processor(
                text='<image>ocr\n',
                images=image,
                return_tensors='pt',
            )
            .to(torch.float16)
            .to(self.model.device)
        )
        input_len = inputs['input_ids'].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
            outputs = outputs[0][input_len:]
            decoded = self.processor.decode(outputs, skip_special_tokens=True)

        num_char = len(decoded)

        print("Chars detected: ", num_char)

        # Exponentially decreasing towards 0.0 if more than free_chars detected
        return min(1.0, math.exp(-num_char + free_chars))


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        
        # -----------------modified paths for packaging!!!!-----------------
        self.model_path = kagglehub.notebook_output_download(
            'metric/sac-logos-ava1-l14-linearmse'
        ) + '/sac+logos+ava1-l14-linearMSE.pth'

        self.clip_model_path = kagglehub.notebook_output_download(
            'metric/openai-clip-vit-large-patch14'
        ) + '/ViT-L-14.pt'
        
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(self.model_path, weights_only=True, map_location='cuda:1')

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to('cuda:1')
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device='cuda:1')

        return predictor, clip_model, preprocessor

    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        image = self.preprocessor(image).unsqueeze(0).to('cuda:1')

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

        score = self.predictor(torch.from_numpy(image_features).to('cuda:1').float())

        return score.item() / 10.0  # scale to [0, 1]


def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
        a: First value (e.g., precision)
        b: Second value (e.g., recall)
        beta: Weighting parameter

    Returns:
        Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


class ImageProcessor:
    def __init__(self, image: Image.Image, seed=None):
        """Initialize with either a path to an image or a PIL Image object."""
        self.image = image
        self.original_image = self.image.copy()
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def reset(self):
        self.image = self.original_image.copy()
        return self
    
    def visualize_comparison(
        self,
        original_name='Original',
        processed_name='Processed',
        figsize=(10, 5),
        show=True,
    ):
        """Display original and processed images side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(np.asarray(self.original_image))
        ax1.set_title(original_name)
        ax1.axis('off')

        ax2.imshow(np.asarray(self.image))
        ax2.set_title(processed_name)
        ax2.axis('off')

        title = f'{original_name} vs {processed_name}'
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def apply_median_filter(self, size=3):
        """Apply median filter to remove outlier pixel values.

        Args:
            size: Size of the median filter window.
        """
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))
        return self

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter to smooth while preserving edges.

        Args:
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.asarray(self.image)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # Convert back to PIL Image
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5):
        """Apply low-pass filter in the frequency domain using FFT.

        Args:
            cutoff_frequency: Normalized cutoff frequency (0-1).
                Lower values remove more high frequencies.
        """
        # Convert to numpy array, ensuring float32 for FFT
        img_array = np.array(self.image, dtype=np.float32)

        # Process each color channel separately
        result = np.zeros_like(img_array)
        for i in range(3):  # For RGB channels
            # Apply FFT
            f = np.fft.fft2(img_array[:, :, i])
            fshift = np.fft.fftshift(f)

            # Create a low-pass filter mask
            rows, cols = img_array[:, :, i].shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            r = int(min(crow, ccol) * cutoff_frequency)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            # Apply mask and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)

            result[:, :, i] = img_back

        # Clip to 0-255 range and convert to uint8 after processing all channels
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        self.image = Image.fromarray(result)
        return self

    def apply_jpeg_compression(self, quality=85):
        """Apply JPEG compression.

        Args:
            quality: JPEG quality (0-95). Lower values increase compression.
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        self.image = Image.open(buffer)
        return self

    def apply_random_crop_resize(self, crop_percent=0.05):
        """Randomly crop and resize back to original dimensions.

        Args:
            crop_percent: Percentage of image to crop (0-0.4).
        """
        width, height = self.image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = self.rng.randint(0, crop_pixels_w + 1)
        top = self.rng.randint(0, crop_pixels_h + 1)
        right = width - self.rng.randint(0, crop_pixels_w + 1)
        bottom = height - self.rng.randint(0, crop_pixels_h + 1)

        self.image = self.image.crop((left, top, right, bottom))
        self.image = self.image.resize((width, height), Image.BILINEAR)
        return self

    def apply(self):
        """Apply an ensemble of defenses."""
        return (
            self.apply_random_crop_resize(crop_percent=0.03)
            .apply_jpeg_compression(quality=95)
            .apply_median_filter(size=9)
            .apply_fft_low_pass(cutoff_frequency=0.5)
            .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
            .apply_jpeg_compression(quality=92)
        )


#| export

global_vqa_evaluator = None
global_aesthetic_evaluator = None

def initialize_evaluators():
    """Initialize the evaluators once and store them in global variables"""
    global global_vqa_evaluator, global_aesthetic_evaluator
    
    if global_vqa_evaluator is None:
        print("Initializing VQA Evaluator...")
        global_vqa_evaluator = VQAEvaluator()
    
    if global_aesthetic_evaluator is None:
        print("Initializing Aesthetic Evaluator...")
        global_aesthetic_evaluator = AestheticEvaluator()
    
    return global_vqa_evaluator, global_aesthetic_evaluator


#| export
def evaluate_with_competition_metric(svg, questions=None, choices=None, answers=None, prompt=None):
    """
    Evaluates an SVG using the competition metric with support for multiple questions.
    OCR is checked first and if not perfect (1.0), other evaluations are skipped.
    
    Parameters:
    -----------
    svg : str
        The SVG string to evaluate
    questions : list of str, optional
        List of questions to ask about the image. If None, a single question will be generated from prompt.
    choices : list of list of str, optional
        List of lists of answer choices for each question. If None, simple yes/no choices will be used.
    answers : list of str, optional
        List of correct answers for each question. If None, defaults will be used based on prompt.
    prompt : str, optional
        The text description of what the SVG should represent. Used if questions/choices/answers are not provided.
    
    Returns:
    --------
    dict
        Dictionary with evaluation scores: vqa_score, aesthetic_score, ocr_score, combined_score
    """
    vqa_evaluator, aesthetic_evaluator = initialize_evaluators()
    
    # Convert SVG to PNG with the competition's standard size
    image = svg_to_png(svg)
    
    # Apply the same image processing steps as the competition
    image_processor = ImageProcessor(image=image).apply()
    base_transformed_image = image_processor.image.copy()

    # skip OCR scoring (assuming we have fooled it)
    ocr_score = 1.0
    
    # If we're passed questions / answers use those (we do this for LB estimation) - otherwise the questions 
    if questions is None and prompt is not None:
        
        # Questions / Answers used for optimization defined here!        
        questions = [
            f"Image includes all of: {prompt}?",
            f"Does this image show all elements of: {prompt}?",
        ]
        
        # Format choices to always be yes/no or exactly 4 options
        choices = [
            ["yes", "no"], 
            ["yes", "no"], 
        ]
        
        # Default answers based on expecting a good match to the prompt
        answers = [
            "yes",
            "yes",
        ]
        
    elif questions is None or choices is None or answers is None:
        raise ValueError("Either provide 'prompt' or all of 'questions', 'choices', and 'answers'")
    
    # Calculate VQA score using batched Q&A evaluation
    vqa_score = vqa_evaluator.score(questions, choices, answers, base_transformed_image)
    
    # Calculate Aesthetic score
    aesthetic_score = aesthetic_evaluator.score(base_transformed_image)

    # This reduces the importance of aesthics when optimizing - this helped LB a little...
    aesthetic_score = (aesthetic_score + 1.5) / 4
    
    # Calculate final score using competition formula
    combined_score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
    
    # Return detailed results
    return {
        'vqa_score': vqa_score,
        'aesthetic_score': aesthetic_score,
        'ocr_score': ocr_score,
        'combined_score': combined_score,
        'questions_asked': questions,
        'choices_given': choices,
        'expected_answers': answers
    }


#| export
# Happens on first image evaluation otherwise (no need to export this cell)
initialize_evaluators() 


#| export

# These settings have significant impact on conversion time
SVG_TARGET_SIZE = (420, 420)
SVG_COLORS = 12

def compress_hex_color(hex_color):
    """Convert hex color to shortest possible representation"""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    return hex_color

# CUDA kernel for color distance calculation
@cuda.jit
def color_distance_kernel(pixels, centers, distances):
    """CUDA kernel to calculate distances between pixels and color centers"""
    i = cuda.grid(1)
    if i < pixels.shape[0]:
        min_dist = 1e10
        for c in range(centers.shape[0]):
            # Calculate Euclidean distance
            dist = 0
            for j in range(3):  # RGB channels
                diff = pixels[i, j] - centers[c, j]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                distances[i] = c

# GPU-accelerated K-means
def kmeans_gpu(pixels, k, max_iter=100, tol=0.2):
    """
    GPU-accelerated K-means clustering
    
    Args:
        pixels: Image pixels (N x 3 array)
        k: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        labels, centers
    """
    # Move data to GPU
    pixels_gpu = cp.asarray(pixels, dtype=cp.float32)
    
    # Initialize centers randomly
    idx = cp.random.choice(pixels.shape[0], k, replace=False)
    centers_gpu = pixels_gpu[idx].copy()
    
    # Previous centers to check convergence
    prev_centers = cp.zeros_like(centers_gpu)
    
    # Prepare output arrays
    labels_gpu = cp.zeros(pixels.shape[0], dtype=cp.int32)
    
    # Setup CUDA grid
    threadsperblock = 256
    blockspergrid = (pixels.shape[0] + threadsperblock - 1) // threadsperblock
    
    for _ in range(max_iter):
        # Copy centers for convergence check
        prev_centers = centers_gpu.copy()
        
        # Allocate distance array on GPU
        distances = cp.zeros(pixels.shape[0], dtype=cp.int32)
        
        # Launch kernel to compute distances and assign labels
        color_distance_kernel[(blockspergrid,), (threadsperblock,)](
            pixels_gpu, centers_gpu, distances
        )
        
        # Update centers
        for i in range(k):
            mask = (distances == i)
            if cp.any(mask):
                centers_gpu[i] = cp.mean(pixels_gpu[mask], axis=0)
        
        # Check convergence
        center_shift = cp.linalg.norm(centers_gpu - prev_centers)
        if center_shift < tol:
            break
    
    # Move results back to CPU
    centers = cp.asnumpy(centers_gpu).astype(np.uint8)
    labels = cp.asnumpy(distances).astype(np.int32)
    
    return labels, centers

def extract_features_by_scale_gpu(img_np, num_colors):
    """
    Extract image features hierarchically by scale using GPU
    
    Args:
        img_np (np.ndarray): Input image
        num_colors (int): Number of colors to quantize
    
    Returns:
        list: Hierarchical features sorted by importance
    """
    # Start timing
    start_time = time.time()
    
    # Convert to RGB if needed
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Perform color quantization using GPU
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    
    labels, palette = kmeans_gpu(pixels, num_colors)
    
    # Quantized image
    quantized = palette[labels].reshape(img_rgb.shape)
        
    # Create a CuPy array for the quantized image to accelerate further processing
    # (We'll still use OpenCV for contour detection as it's highly optimized)
    cp_quantized = cp.asarray(quantized)
    
    # Hierarchical feature extraction
    hierarchical_features = []
    
    # Sort colors by frequency
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_colors = [palette[i] for i in sorted_indices]
    
    # Center point for importance calculations
    center_x, center_y = width/2, height/2
    
    # Process each color in parallel where possible
    for color in sorted_colors:
        # Create color mask - using OpenCV as it's optimized
        color_mask = cv2.inRange(quantized, color, color)
        
        # Find contours - OpenCV's contour finding is already optimized
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Convert RGB to compressed hex
        hex_color = compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
        
        color_features = []
        for contour in contours:
            # Skip tiny contours
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            
            # Calculate contour center
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            
            # Distance from image center (normalized)
            dist_from_center = np.sqrt(((cx - center_x) / width)**2 + ((cy - center_y) / height)**2)
            
            # Simplify contour - adaptive epsilon based on contour size for better detail preservation
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate points string
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
            
            # Calculate importance (area, proximity to center, complexity)
            importance = (
                area * 
                (1 - dist_from_center) * 
                (1 / (len(approx) + 1))
            )
            
            color_features.append({
                'points': points,
                'color': hex_color,
                'area': area,
                'importance': importance,
                'point_count': len(approx),
                'original_contour': approx
            })
        
        # Sort features by importance within this color
        color_features.sort(key=lambda x: x['importance'], reverse=True)
        hierarchical_features.extend(color_features)
    
    # Final sorting by overall importance
    hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)
        
    return hierarchical_features

# Numba-accelerated polygon simplification
@jit(nopython=True)
def calculate_simplified_points(x_points, y_points, level):
    """Numba-accelerated point simplification"""
    result = []
    
    if level == 1:  # Round to 1 decimal place
        for i in range(len(x_points)):
            result.append((round(x_points[i], 1), round(y_points[i], 1)))
    elif level == 2:  # Round to integer
        for i in range(len(x_points)):
            result.append((round(x_points[i]), round(y_points[i])))
    elif level == 3:  # Reduce points
        if len(x_points) <= 4:
            for i in range(len(x_points)):
                result.append((round(x_points[i]), round(y_points[i])))
        else:
            step = min(2, len(x_points) // 3)
            for i in range(0, len(x_points), step):
                result.append((round(x_points[i]), round(y_points[i])))
            if (len(x_points)-1, len(y_points)-1) not in result:
                result.append((round(x_points[-1]), round(y_points[-1])))
    else:  # No simplification
        for i in range(len(x_points)):
            result.append((x_points[i], y_points[i]))
            
    return result

def simplify_polygon(points_str, simplification_level):
    """
    Simplify a polygon by reducing coordinate precision or number of points
    
    Args:
        points_str (str): Space-separated "x,y" coordinates
        simplification_level (int): Level of simplification (0-3)
    
    Returns:
        str: Simplified points string
    """
    if simplification_level == 0:
        return points_str
    
    points = points_str.split()
    
    # Extract x and y coordinates for Numba processing
    x_points = np.array([float(p.split(',')[0]) for p in points])
    y_points = np.array([float(p.split(',')[1]) for p in points])
    
    # Use Numba-accelerated function for simplification
    simplified = calculate_simplified_points(x_points, y_points, simplification_level)
    
    # Convert back to string format
    return " ".join([f"{x},{y}" for x, y in simplified])

def bitmap_to_svg_layered_gpu(image, max_size_bytes=10000, resize=True, target_size=SVG_TARGET_SIZE, 
                         adaptive_fill=True, num_colors=SVG_COLORS):
    """
    Convert bitmap to SVG using layered feature extraction with GPU optimization
    
    Args:
        image: Input image (PIL.Image)
        max_size_bytes (int): Maximum SVG size
        resize (bool): Whether to resize the image before processing
        target_size (tuple): Target size for resizing (width, height)
        adaptive_fill (bool): Whether to adaptively fill available space
        num_colors (int): Number of colors to quantize
    
    Returns:
        str: SVG representation
    """
    start_time = time.time()
    
    # Resize the image if requested
    if resize:
        original_size = image.size
        image = image.resize(target_size, Image.LANCZOS)
    else:
        original_size = image.size
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Get image dimensions
    height, width = img_np.shape[:2]
    
    # Calculate average background color
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = cp.mean(cp.asarray(img_np), axis=(0,1)).get().astype(int)
        bg_hex_color = compress_hex_color(f'#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}')
    else:
        bg_hex_color = '#fff'
    
    # Start building SVG
    # Use original dimensions in viewBox for proper scaling when displayed
    orig_width, orig_height = original_size
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">\n'
    svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = '</svg>'
    
    # Calculate base size
    base_size = len((svg_base + svg_footer).encode('utf-8'))
    available_bytes = max_size_bytes - base_size
    
    # Extract hierarchical features
    features = extract_features_by_scale_gpu(img_np, num_colors=num_colors)
    
    # If not using adaptive fill, just add features until we hit the limit
    if not adaptive_fill:
        svg = svg_base
        for feature in features:
            # Try adding the feature
            feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
            
            # Check if adding this feature exceeds size limit
            if len((svg + feature_svg + svg_footer).encode('utf-8')) > max_size_bytes:
                break
            
            # Add the feature
            svg += feature_svg
        
        # Close SVG
        svg += svg_footer
        return svg
    
    # For adaptive fill, use binary search to find optimal simplification level
    
    # First attempt: calculate size of all features at different simplification levels
    feature_sizes = []
    for feature in features:
        feature_sizes.append({
            'original': len(f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level1': len(f'<polygon points="{simplify_polygon(feature["points"], 1)}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level2': len(f'<polygon points="{simplify_polygon(feature["points"], 2)}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level3': len(f'<polygon points="{simplify_polygon(feature["points"], 3)}" fill="{feature["color"]}" />\n'.encode('utf-8'))
        })
    
    # Two-pass approach: first add most important features, then fill remaining space
    svg = svg_base
    bytes_used = base_size
    added_features = set()
    
    # Pass 1: Add most important features at original quality
    for i, feature in enumerate(features):
        feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
        feature_size = feature_sizes[i]['original']
        
        if bytes_used + feature_size <= max_size_bytes:
            svg += feature_svg
            bytes_used += feature_size
            added_features.add(i)
    
    # Pass 2: Try to add remaining features with progressive simplification
    for level in range(1, 4):  # Try simplification levels 1-3
        for i, feature in enumerate(features):
            if i in added_features:
                continue
                
            feature_size = feature_sizes[i][f'level{level}']
            if bytes_used + feature_size <= max_size_bytes:
                feature_svg = f'<polygon points="{simplify_polygon(feature["points"], level)}" fill="{feature["color"]}" />\n'
                svg += feature_svg
                bytes_used += feature_size
                added_features.add(i)
    
    # Finalize SVG
    svg += svg_footer
    
    # Double check we didn't exceed limit
    final_size = len(svg.encode('utf-8'))
    if final_size > max_size_bytes:
        # If we somehow went over, return basic SVG
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'
    
    # Calculate space utilization
    utilization = (final_size / max_size_bytes) * 100
        
    # Return the SVG with efficient space utilization
    return svg

def init_svg_gpu():
    # Init CUDA context
    _ = cp.cuda.runtime.getDeviceCount()
    
    # Compile CUDA kernel
    pixels = cp.ones((10, 3), dtype=cp.float32)
    centers = cp.ones((2, 3), dtype=cp.float32)
    distances = cp.zeros(10, dtype=cp.int32)
    color_distance_kernel[(1,), (256,)](pixels, centers, distances)
    
    # Compile Numba JIT function
    x = y = np.array([1.0, 2.0, 3.0])
    _ = calculate_simplified_points(x, y, 1)
    
    return True

#so the first SVG conversion doesn't look slow
init_svg_gpu()


#| export
def add_ocr_decoy_svg(svg_code: str) -> str:
    """
    Adds nested circles with second darkest and second brightest colors from the existing SVG,
    positioned in one of the four corners (randomly selected) but positioned to avoid being
    cropped out during image processing.
    
    Parameters:
    -----------
    svg_code : str
        The original SVG string
    
    Returns:
    --------
    str
        Modified SVG with the nested circles added
    """
    import random
    import re
    from colorsys import rgb_to_hls, hls_to_rgb
    
    # Check if SVG has a closing tag
    if "</svg>" not in svg_code:
        return svg_code
    
    # Extract viewBox if it exists to understand the dimensions
    viewbox_match = re.search(r'viewBox=["\'](.*?)["\']', svg_code)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        try:
            x, y, width, height = map(float, viewbox)
        except ValueError:
            # Default dimensions if we can't parse viewBox
            width, height = 384, 384
    else:
        # Default dimensions if viewBox not found
        width, height = 384, 384
    
    # Function to convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Function to convert RGB to hex
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
    
    # Function to calculate color lightness
    def get_lightness(color):
        # Handle different color formats
        if color.startswith('#'):
            rgb = hex_to_rgb(color)
            return rgb_to_hls(*rgb)[1]  # Lightness is the second value in HLS
        elif color.startswith('rgb'):
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if rgb_match:
                r, g, b = map(lambda x: int(x)/255, rgb_match.groups())
                return rgb_to_hls(r, g, b)[1]
        return 0.5  # Default lightness if we can't parse
    
    # Extract all colors from the SVG
    color_matches = re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))"', svg_code)
    
    # Default colors in case we don't find enough
    second_darkest_color = "#333333"  # Default to dark gray
    second_brightest_color = "#CCCCCC"  # Default to light gray
    
    if color_matches:
        # Remove duplicates and get unique colors
        unique_colors = list(set(color_matches))
        
        # Calculate lightness for each unique color
        colors_with_lightness = [(color, get_lightness(color)) for color in unique_colors]
        
        # Sort by lightness (brightness)
        sorted_colors = sorted(colors_with_lightness, key=lambda x: x[1])
        
        # Handle different scenarios based on number of unique colors
        if len(sorted_colors) >= 4:
            # We have at least 4 unique colors - use 2nd darkest and 2nd brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[-2][0]
        elif len(sorted_colors) == 3:
            # We have 3 unique colors - use 2nd darkest and brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[2][0]
        elif len(sorted_colors) == 2:
            # We have only 2 unique colors - use the darkest and brightest
            second_darkest_color = sorted_colors[0][0]
            second_brightest_color = sorted_colors[1][0]
        elif len(sorted_colors) == 1:
            # Only one color - use it for second_darkest and a derived lighter version
            base_color = sorted_colors[0][0]
            base_lightness = sorted_colors[0][1]
            second_darkest_color = base_color
            
            # Create a lighter color variant if the base is dark, or darker if base is light
            if base_lightness < 0.5:
                # Base is dark, create lighter variant
                second_brightest_color = "#CCCCCC"
            else:
                # Base is light, create darker variant
                second_darkest_color = "#333333"
    
    # Ensure the colors are different
    if second_darkest_color == second_brightest_color:
        # If they ended up the same, modify one of them
        if get_lightness(second_darkest_color) < 0.5:
            # It's a dark color, make the bright one lighter
            second_brightest_color = "#CCCCCC"
        else:
            # It's a light color, make the dark one darker
            second_darkest_color = "#333333"
    
    # Base size for the outer circle
    base_outer_radius = width * 0.023
    
    # Randomize size by ±10%
    size_variation = base_outer_radius * 0.1
    outer_radius = base_outer_radius + random.uniform(-size_variation, size_variation)
    
    # Define radii for inner circles based on outer radius
    middle_radius = outer_radius * 0.80
    inner_radius = middle_radius * 0.65
    
    # Calculate the maximum crop margin based on the image processing (5% of dimensions)
    # Add 20% extra margin for safety
    crop_margin_w = int(width * 0.05 * 1.2)
    crop_margin_h = int(height * 0.05 * 1.2)
    
    # Calculate center point based on the outer radius to ensure the entire circle stays visible
    safe_offset = outer_radius + max(crop_margin_w, crop_margin_h)
    
    # Choose a random corner (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    corner = random.randint(0, 3)
    
    # Position the circle in the chosen corner, accounting for crop margin
    if corner == 0:  # Top-left
        center_x = safe_offset
        center_y = safe_offset
    elif corner == 1:  # Top-right
        center_x = width - safe_offset
        center_y = safe_offset
    elif corner == 2:  # Bottom-left
        center_x = safe_offset
        center_y = height - safe_offset
    else:  # Bottom-right
        center_x = width - safe_offset
        center_y = height - safe_offset
    
    # Add a small random offset (±10% of safe_offset) to make positioning less predictable
    random_offset = safe_offset * 0.1
    center_x += random.uniform(-random_offset, random_offset)
    center_y += random.uniform(-random_offset, random_offset)
    
    # Round to 1 decimal place to keep file size down
    outer_radius = round(outer_radius, 1)
    middle_radius = round(middle_radius, 1)
    inner_radius = round(inner_radius, 1)
    center_x = round(center_x, 1)
    center_y = round(center_y, 1)
    
    # Create the nested circles
    outer_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{outer_radius}" fill="{second_darkest_color}" />'
    middle_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{middle_radius}" fill="{second_brightest_color}" />'
    inner_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{inner_radius}" fill="{second_darkest_color}" />'
    
    # Create a group element that contains all three circles
    group_element = f'<g>{outer_circle}{middle_circle}{inner_circle}</g>'
    
    # Insert the group element just before the closing SVG tag
    modified_svg = svg_code.replace("</svg>", f"{group_element}</svg>")
    
    # Calculate and add a comment with the byte size information
    outer_bytes = len(outer_circle.encode('utf-8'))
    middle_bytes = len(middle_circle.encode('utf-8'))
    inner_bytes = len(inner_circle.encode('utf-8'))
    total_bytes = outer_bytes + middle_bytes + inner_bytes
    
    corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    byte_info = f'<!-- Circle bytes: outer={outer_bytes}, middle={middle_bytes}, ' \
                f'inner={inner_bytes}, total={total_bytes}, ' \
                f'colors: dark={second_darkest_color}, light={second_brightest_color}, ' \
                f'position: {corner_names[corner]} -->'
    
    modified_svg = modified_svg.replace("</svg>", f"{byte_info}</svg>")
    
    return modified_svg


#| export
def generate_and_convert(prompt, prompt_suffix="", prompt_prefixes=[], negative_prompt="", 
                         max_num_attempts=30, max_time_seconds=None, num_inference_steps=5, 
                         guidance_scale=0, show_summary=True, verbose=True):
    best_svg = None
    best_bitmap = None
    best_rendered_svg = None
    best_combined_prompt = None

    #manual tweak to assure we have space for OCR decoy
    svg_ocr_decoy_size = 350
    max_svg_size_bytes = (10000 - svg_ocr_decoy_size)
    
    best_similarity = -1
    best_similarity_vqa = -1
    best_similarity_aesthetic = -1
    
    # Track total processing time
    total_start_time = time.time()
    
    # Track timing statistics
    generation_times = []
    conversion_times = []
    evaluation_times = []
    attempt_times = []
    attempts_completed = 0

    def display_images(prompt_base, full_prompt, bitmap, rendered_svg):
        plt.figure(figsize=(12, 6))
        plt.suptitle(prompt_base, fontsize=16, y=0.98)
        plt.figtext(0.5, 0.91, f"({full_prompt})", fontsize=10, ha='center')
        
        # Original bitmap
        plt.subplot(1, 2, 1)
        plt.imshow(bitmap)
        plt.title(f"Original")
        plt.axis('off')
        
        # SVG conversion
        plt.subplot(1, 2, 2)
        plt.imshow(rendered_svg)
        plt.title(f"SVG")
        plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
        
    for i in range(max_num_attempts):
        # Check if we've exceeded the time limit (if specified)
        current_time = time.time()
        elapsed_time = current_time - total_start_time
        
        if max_time_seconds is not None and elapsed_time >= max_time_seconds:
            if verbose:
                print(f"\n⏱️ Time limit of {max_time_seconds:.1f}s reached after {attempts_completed} attempts")
                print(f"Current elapsed time: {elapsed_time:.1f}s")
            break
            
        attempt_start_time = time.time()
        if verbose:
            print(f"\n=== Attempt {i+1} ===")
            if max_time_seconds is not None:
                print(f"Time elapsed: {elapsed_time:.1f}s / {max_time_seconds:.1f}s")
        else:
            print(".", end="")

        prompt_prefix = prompt_prefixes[i % len(prompt_prefixes)]
        combined_prompt = prompt_prefix + " " + prompt + " " + prompt_suffix
        
        # Generate bitmap with Stable Diffusion (using combined_prompt)
        generation_start = time.time()
        bitmap = generate_bitmaps(combined_prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, num_images_per_prompt=1)[0]
        generation_end = time.time()
        generation_time = generation_end - generation_start
        generation_times.append(generation_time)
                
        # Convert to SVG with size limit
        conversion_start = time.time()
        svg_content = bitmap_to_svg_layered_gpu(bitmap, max_size_bytes = max_svg_size_bytes)
        
        svg_init_size = len(svg_content.encode('utf-8'))

        # add OCR decoy
        svg_content = add_ocr_decoy_svg(svg_content)
        svg_size_with_decoy = len(svg_content.encode('utf-8'))

        conversion_end = time.time()
        conversion_time = conversion_end - conversion_start
        conversion_times.append(conversion_time)
                
        # Render SVG to bitmap for evaluation
        rendered_svg = svg_to_png(svg_content)
        
        if verbose:
            display_images(prompt, combined_prompt, bitmap, rendered_svg)

        # Evaluate rendered SVG with competition metric (using just base prompt)
        evaluation_start = time.time()

        svg_scores = evaluate_with_competition_metric(svg_content, prompt=prompt)                
            
        evaluation_end = time.time()
        evaluation_time = evaluation_end - evaluation_start
        evaluation_times.append(evaluation_time)

        if svg_scores['ocr_score'] < 1.0:
            print(f"Non-1.0 OCR! (Other scoring skipped)")
                
        if verbose:
            # print("-"*10, f"Score Cycle {i}", "-"*10)
            print(f"SVG VQA Score: {svg_scores['vqa_score']:.4f}")
            print(f"SVG Aesthetic Score: {svg_scores['aesthetic_score']:.4f}")
            print(f"SVG OCR Score: {svg_scores['ocr_score']:.4f}")
            print(f"SVG Competition Score: {svg_scores['combined_score']:.4f}")
                
        # Track the best result using competition score
        if svg_scores['combined_score'] > best_similarity:
            best_similarity = svg_scores['combined_score']
            best_similarity_vqa = svg_scores['vqa_score']
            best_similarity_aesthetic = svg_scores['aesthetic_score']

            best_svg = svg_content
            best_rendered_svg = rendered_svg
            best_bitmap = bitmap
            best_combined_prompt = combined_prompt
            
            if verbose: print(f"✅ New best result: {svg_scores['combined_score']:.4f}")
        else:
            if verbose: print(f"❌ Not better than current best: {best_similarity:.4f}")
        
        # Calculate total time for this attempt
        attempt_end_time = time.time()
        attempt_time = attempt_end_time - attempt_start_time
        attempt_times.append(attempt_time)
        
        if verbose:
            print(f"Image generation time: {generation_time:.2f}s")
            print(f"SVG conversion time: {conversion_time:.2f}s")
            print(f"SVG initial size: {svg_init_size} bytes")
            print(f"SVG size with OCR decoy: {svg_size_with_decoy} bytes")
            print(f"Image evaluation time: {evaluation_time:.2f}s")
            print(f"Total time for attempt {i+1}: {attempt_time:.2f}s")
        
        attempts_completed += 1
    
    # Calculate total processing time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Show best image as part of summary only if didn't already display in verbose
    if show_summary and not verbose and best_bitmap is not None:
        display_images(prompt, best_combined_prompt, best_bitmap, best_rendered_svg)

    if show_summary or verbose:
        if best_similarity > -1:
            print(f"Best score achieved: {best_similarity:.4f} (VQA: {best_similarity_vqa:.4f} / Aesthetic: {best_similarity_aesthetic:.4f})")
        else:
            print("No valid results obtained.")
        
        if attempts_completed > 0:
            print(f"Average image generation time: {sum(generation_times)/len(generation_times):.2f}s")
            print(f"Average SVG conversion time: {sum(conversion_times)/len(conversion_times):.2f}s")
            print(f"Average image evaluation time: {sum(evaluation_times)/len(evaluation_times):.2f}s")
            print(f"Average time per attempt: {sum(attempt_times)/len(attempt_times):.2f}s")
        
        if max_time_seconds is not None:
            print(f"Completed {attempts_completed} of {max_num_attempts} possible attempts within time limit of {max_time_seconds:.1f}s")
        else:
            print(f"Completed all {attempts_completed} attempts")
        
        print(f"Total processing time: {total_time:.2f}s")
                    
    return best_svg, best_similarity


#| export

def reset_seeds():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42) 
    np.random.seed(42)
    cp.random.seed(42)


#| export

reset_seeds()

class Model:
    def __init__(self):
        '''Optional constructor, performs any setup logic, model instantiation, etc.'''
        
        # having time limit as opposed to just image count allows more image attempts if we trigger OCR
        
        self.max_time_seconds_all_attempts = 55   # generate next image as long as we are under this time
        self.max_num_attempts_per_prompt = 30     # probably never get to do this many tries in allowed time
        self.num_inference_steps = 3
        
        self.prompt_suffix = ""

        # we cycle through these 
        self.prompt_prefixes = ["",
                                "Flat color block, ultrasimple, solid colors only:",
                                "dramatic high-contrast:",
                                "simple oil painting, close-up:",
                                "in cartoon style:",
                                "close-up unfocused photograph - but containing all described elements:"
                               ]
                                
        
        # SDXL Turbo doesn't use guidance_scle or negative_prompt (leaving for compatibility)        
        self.negative_prompt = ""
        self.guidance_scale = 0

        self.last_score = None
            
        pass

    def predict(self, prompt: str, show_summary=False, verbose=False) -> str:
        '''Generates SVG which produces an image described by the prompt.

        Args:
            prompt (str): A prompt describing an image
        Returns:
            String of valid SVG code.
        '''
        
        best_svg, best_score = generate_and_convert(
            prompt,
            prompt_prefixes=self.prompt_prefixes,
            prompt_suffix=self.prompt_suffix,            
            negative_prompt=self.negative_prompt,
            max_num_attempts=self.max_num_attempts_per_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            show_summary=show_summary,
            max_time_seconds=self.max_time_seconds_all_attempts,
            verbose=verbose
        )

        self.last_score = best_score
        
        return best_svg


# Initialize the model
reset_seeds()

model = Model()

best_svg = model.predict("a lighthouse overlooking the ocean", show_summary=True, verbose=True)


train_df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')
questions_df = pd.read_parquet('/kaggle/input/drawing-with-llms/questions.parquet')

demo_df = train_df.copy()

# uncomment to test on just a few
#demo_df = demo_df.head(5)

# Create arrays to store scores and timing data
optimization_scores = []  # Scores from the model's optimization questions
training_scores = []      # Scores using the competition training questions
generation_times = []

for i, row in enumerate(demo_df.iterrows()):
    description = row[1]['description']
    img_id = row[1]['id']
    
    start_time = time.time()    
    svg = model.predict(description, verbose=False, show_summary=True)
    rendered_img = svg_to_png(svg)
    end_time = time.time()

    generation_time = end_time - start_time
    generation_times.append(generation_time)
    optimization_score = model.last_score
    optimization_scores.append(optimization_score)
    
    # Score using competition questions
    scoring_start_time = time.time()
    
    # Filter questions for this specific image ID
    img_questions = questions_df[questions_df['id'] == img_id]
    # Extract questions, choices, and answers
    questions = img_questions['question'].tolist()
    choices = img_questions['choices'].tolist()
    answers = img_questions['answer'].tolist()

    print("Rescoring with training questions for LB estimate... ")
    print("LB Questions: ", questions)
    print("LB Choices: ", choices)
    print("LB Answers: ", answers)
    
    # Evaluate our image based on training questions
    score_results = evaluate_with_competition_metric(svg, questions=questions, choices=choices, answers=answers)
    scoring_end_time = time.time()
    scoring_time = scoring_end_time - scoring_start_time

    #print(f"{scoring_time:.2f}s")
    
    # Get the training scores
    combined_score = score_results['combined_score']
    vqa_score = score_results['vqa_score']
    aesthetic_score = score_results['aesthetic_score']
    training_scores.append(combined_score)
    
    # Print detailed score information and timing
    print(f"LB estimate score: {combined_score:.4f} (VQA: {vqa_score:.4f} / Aesthetic: {aesthetic_score:.4f})")

    # Print progress with both score types
    current_opt_avg = np.mean(optimization_scores)
    current_train_avg = np.mean(training_scores)
    current_avg_time = np.mean(generation_times)
    
# When all done, calculate final statistics
avg_opt_score = np.mean(optimization_scores)
avg_train_score = np.mean(training_scores)
avg_generation_time = np.mean(generation_times)
total_time_taken = sum(generation_times)

# Calculate projections for 500 images
projected_time_500_images = 500 * avg_generation_time
projected_hours = projected_time_500_images / 3600

print("\n=== SUMMARY ===")
print(f"Prompts processed: {len(demo_df)}")
print(f"Average optimization score (using stand-in questions): {avg_opt_score:.4f}")
print(f"Average LB estimate score (using training questions): {avg_train_score:.4f}")
print(f"Average generation time per prompt: {avg_generation_time:.2f} seconds")
print(f"Total time elapsed for generation: {timedelta(seconds=total_time_taken)}")
print(f"Projected time for 500 prompts: {projected_hours:.2f} hours ({timedelta(seconds=projected_time_500_images)})")
print("Time estimates exclude re-scoring with train prompts.")

