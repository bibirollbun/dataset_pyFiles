#| export
# Import necessary packages for model downloading
import kagglehub

# Download the SVG scoring metric package
svg_scoring_package = kagglehub.package_import('richolson/stable-diffusion-svg-scoring-metric/versions/17')


#| export
# Basic libraries
import os
import io
import re
import random
import base64
from io import BytesIO
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

# Deep learning and image processing
import torch
import torch.nn.functional as F
from IPython.display import SVG
from PIL import Image
import cv2
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, AutoModel

# Additional libraries for SVG scoring and evaluation
import io
from math import prod
from statistics import mean
from IPython.display import SVG
import cairosvg
import clip
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

# Import SVG constraints package for validation
svg_constraints = kagglehub.package_import('metric/svg-constraints')

# Define error class for competition submissions
class ParticipantVisibleError(Exception):
    pass


#| export
def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """
    Calculates a fidelity score by comparing generated SVG images to target text descriptions.
    
    Parameters:
    ----------
    solution : pd.DataFrame
        DataFrame with target text descriptions (must have 'description' column)
    submission : pd.DataFrame
        DataFrame with generated SVG strings (must have 'svg' column)
    row_id_column_name : str
        Column name for row identifiers
        
    Returns:
    -------
    float
        Mean fidelity score (0-1) representing similarity between SVGs and descriptions
    """
    # Validation steps
    del solution[row_id_column_name], submission[row_id_column_name]
    if not pd.api.types.is_string_dtype(submission.loc[:, 'svg']):
        raise ParticipantVisibleError('svg must be a string.')
        
    # Check SVG constraints
    constraints = svg_constraints.SVGConstraints()
    try:
        for svg in submission.loc[:, 'svg']:
            constraints.validate_svg(svg)
    except:
        raise ParticipantVisibleError('SVG code violates constraints.')
    
    # Initialize evaluators
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()
    results = []
    
    # Score each SVG against its description
    try:
        for svg, description in zip(
            submission.loc[:, 'svg'], solution.loc[:, 'description'], strict=True
        ):
            image = svg_to_png(svg)
            vqa_score = vqa_evaluator.score(image, 'SVG illustration of ' + description)
            aesthetic_score = aesthetic_evaluator.score(image)
            instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=2.0)
            results.append(instance_score)
    except:
        raise ParticipantVisibleError('SVG failed to score.')
        
    # Calculate final fidelity score
    fidelity = mean(results)
    return float(fidelity)


#| export
class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description."""
    
    def __init__(self):
        # Configure quantization for efficient loading
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load PaliGemma model for visual question answering
        self.model_path = kagglehub.model_download(
            'google/paligemma-2/transformers/paligemma2-10b-mix-448'
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        )
        
        # Define evaluation questions
        self.questions = {
            'fidelity': 'Does <image> portray "{}" without any lettering? Answer yes or no.',
            'text': '<image> Text present: yes or no?',
        }
    
    def score(self, image: Image.Image, description: str) -> float:
        """
        Evaluates image-to-description fidelity using VQA
        
        Parameters:
        ----------
        image : PIL.Image.Image
            Image to evaluate
        description : str
            Target description
            
        Returns:
        -------
        float
            Fidelity score (0-1)
        """
        p_fidelity = self.get_yes_probability(image, self.questions['fidelity'].format(description))
        p_text = self.get_yes_probability(image, self.questions['text'])
        
        # Penalize images with text
        return p_fidelity * (1 - p_text)
    
    def mask_yes_no(self, logits):
        """Masks logits to only consider 'yes' or 'no' responses"""
        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')
        
        # Create mask with very negative values (will become ~0 after softmax)
        mask = torch.full_like(logits, float('-inf'))
        
        # Allow only yes/no tokens
        mask[:, yes_token_id] = logits[:, yes_token_id]
        mask[:, no_token_id] = logits[:, no_token_id]
        mask[:, yes_with_space_token_id] = logits[:, yes_with_space_token_id]
        mask[:, no_with_space_token_id] = logits[:, no_with_space_token_id]
        
        return mask
    
    def get_yes_probability(self, image, prompt) -> float:
        """Calculate probability of 'yes' answer to prompted question about image"""
        # Process image and text prompt
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to('cuda:0')
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Get logits for the last token (predicted token)
            logits = outputs.logits[:, -1, :]
            
            # Apply yes/no mask and get probabilities
            masked_logits = self.mask_yes_no(logits)
            probabilities = torch.softmax(masked_logits, dim=-1)
        
        # Get token IDs for yes/no responses
        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')
        
        # Get probabilities for each token
        prob_yes = probabilities[0, yes_token_id].item()
        prob_no = probabilities[0, no_token_id].item()
        prob_yes_space = probabilities[0, yes_with_space_token_id].item()
        prob_no_space = probabilities[0, no_with_space_token_id].item()
        
        # Combine probabilities for yes/no with/without space
        total_yes_prob = prob_yes + prob_yes_space
        total_no_prob = prob_no + prob_no_space
        total_prob = total_yes_prob + total_no_prob
        
        # Renormalize yes probability
        renormalized_yes_prob = total_yes_prob / total_prob
        
        return renormalized_yes_prob


#| export
class AestheticPredictor(nn.Module):
    """Neural network for predicting aesthetic score from CLIP embeddings"""
    
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        # MLP architecture with dropout for regularization
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
    """Evaluates aesthetic quality of images using a pre-trained model"""
    
    def __init__(self):
        # Load pre-trained aesthetic predictor model and CLIP model
        self.model_path = kagglehub.notebook_output_download(
            'metric/sac-logos-ava1-l14-linearmse'
        ) + '/sac+logos+ava1-l14-linearMSE.pth'
        
        self.clip_model_path = kagglehub.notebook_output_download(
            'metric/openai-clip-vit-large-patch14'
        ) + '/ViT-L-14.pt'
        
        self.predictor, self.clip_model, self.preprocessor = self.load()
    
    def load(self):
        """Loads the aesthetic predictor and CLIP models"""
        # Load aesthetic predictor model
        state_dict = torch.load(self.model_path, weights_only=True, map_location='cuda:1')
        
        # Create model (CLIP ViT-L/14 has 768-dim embeddings)
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to('cuda:1')
        predictor.eval()
        
        # Load CLIP model
        clip_model, preprocessor = clip.load(self.clip_model_path, device='cuda:1')
        
        return predictor, clip_model, preprocessor
    
    def score(self, image: Image.Image) -> float:
        """Predicts aesthetic score for an image using CLIP features"""
        # Preprocess image for CLIP
        image = self.preprocessor(image).unsqueeze(0).to('cuda:1')
        
        with torch.no_grad():
            # Extract image features using CLIP
            image_features = self.clip_model.encode_image(image)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()
        
        # Get aesthetic score prediction
        score = self.predictor(torch.from_numpy(image_features).to('cuda:1').float())
        
        # Scale to [0, 1] range
        return score.item() / 10.0


#| export
def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate weighted harmonic mean of two values
    
    Parameters:
    ----------
    a : float
        First value (e.g., precision)
    b : float
        Second value (e.g., recall)
    beta : float
        Weighting parameter (beta > 1 emphasizes b)
        
    Returns:
    -------
    float
        Weighted harmonic mean
    """
    # Handle zero values
    if a <= 0 or b <= 0:
        return 0.0
    
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)

def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Convert SVG string to PNG image
    
    Parameters:
    ----------
    svg_code : str
        SVG code to convert
    size : tuple
        Output image dimensions
        
    Returns:
    -------
    PIL.Image.Image
        Rendered PNG image
    """
    # Ensure SVG has proper viewBox attribute
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')
    
    # Convert SVG to PNG using cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    
    # Load as PIL image, ensure RGB mode, and resize
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


#| export
# Global evaluator instances
global_vqa_evaluator = None
global_aesthetic_evaluator = None

def initialize_evaluators():
    """Initialize evaluators once and store them globally"""
    global global_vqa_evaluator, global_aesthetic_evaluator
    
    if global_vqa_evaluator is None:
        print("Initializing VQA Evaluator...")
        global_vqa_evaluator = VQAEvaluator()
    
    if global_aesthetic_evaluator is None:
        print("Initializing Aesthetic Evaluator...")
        global_aesthetic_evaluator = AestheticEvaluator()
    
    return global_vqa_evaluator, global_aesthetic_evaluator

def evaluate_with_competition_metric(svg, prompt):
    """
    Evaluate SVG against prompt using competition metric
    
    Parameters:
    ----------
    svg : str
        SVG code to evaluate
    prompt : str
        Text description
        
    Returns:
    -------
    dict
        Dictionary with evaluation scores
    """
    # Get or initialize evaluators
    vqa_evaluator, aesthetic_evaluator = initialize_evaluators()
    
    # Convert SVG to image
    image = svg_to_png(svg)
    
    # Calculate scores
    vqa_score = vqa_evaluator.score(image, 'SVG illustration of ' + prompt)
    aesthetic_score = aesthetic_evaluator.score(image)
    combined_score = harmonic_mean(vqa_score, aesthetic_score, beta=2.0)
    
    return {
        'vqa_score': vqa_score,
        'aesthetic_score': aesthetic_score,
        'combined_score': combined_score
    }

# Initialize evaluators to prevent load times from affecting benchmarking
initialize_evaluators()


#| export
# Ensure GPU is available and set up device
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Import necessary libraries
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Download and initialize Stable Diffusion model with optimizations
stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

# Use DDIM scheduler for better quality and speed
scheduler = DDIMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

# Load pipeline with half precision for faster inference
pipe = StableDiffusionPipeline.from_pretrained(
    stable_diffusion_path,
    scheduler=scheduler,
    torch_dtype=torch.float16,  # Use half precision
    safety_checker=None         # Disable safety checker for speed
)

# Move to GPU
pipe.to(device)


#| export
def compress_hex_color(hex_color):
    """
    Convert hex color to shortest possible representation
    
    Example: #ff0099 -> #f09 if possible
    """
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    
    # Check if color can be compressed (all components are multiples of 17)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    
    return hex_color

def extract_features_by_scale(img_np, num_colors=16):
    """
    Extract image features hierarchically by scale
    
    Parameters:
    ----------
    img_np : np.ndarray
        Input image as NumPy array
    num_colors : int
        Number of colors to extract
        
    Returns:
    -------
    list
        Hierarchical features sorted by importance
    """
    # Ensure image is in RGB format
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    # Perform color quantization using k-means
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Create quantized image with color palette
    palette = centers.astype(np.uint8)
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)
    
    # Prepare to extract hierarchical features
    hierarchical_features = []
    
    # Sort colors by frequency
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_colors = [palette[i] for i in sorted_indices]
    
    # Calculate center point for importance calculations
    center_x, center_y = width/2, height/2
    
    # Process each color
    for color in sorted_colors:
        # Create mask for current color
        color_mask = cv2.inRange(quantized, color, color)
        
        # Find contours in the color mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Convert RGB to compressed hex
        hex_color = compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
        
        # Process each contour for this color
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
            
            # Calculate normalized distance from image center
            dist_from_center = np.sqrt(((cx - center_x) / width)**2 + ((cy - center_y) / height)**2)
            
            # Simplify contour for fewer points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Generate points string for SVG polygon
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
            
            # Calculate importance based on area, center proximity, and complexity
            importance = (
                area * 
                (1 - dist_from_center) * 
                (1 / (len(approx) + 1))
            )
            
            # Store feature data
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

def simplify_polygon(points_str, simplification_level):
    """
    Simplify polygon by reducing precision or number of points
    
    Parameters:
    ----------
    points_str : str
        Space-separated point coordinates
    simplification_level : int
        Level of simplification (0-3)
        
    Returns:
    -------
    str
        Simplified points string
    """
    if simplification_level == 0:
        return points_str
    
    points = points_str.split()
    
    # Level 1: Round to 1 decimal place
    if simplification_level == 1:
        return " ".join([f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}" for p in points])
    
    # Level 2: Round to integer
    if simplification_level == 2:
        return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in points])
    
    # Level 3: Reduce number of points and round to integer
    if simplification_level == 3:
        if len(points) <= 4:
            # Keep all points for simple polygons
            return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in points])
        else:
            # For complex polygons, keep fewer points
            step = min(2, len(points) // 3)
            reduced_points = [points[i] for i in range(0, len(points), step)]
            
            # Ensure minimum 3 points and include last point
            if len(reduced_points) < 3:
                reduced_points = points[:3]
            if points[-1] not in reduced_points:
                reduced_points.append(points[-1])
                
            return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in reduced_points])
    
    return points_str

def bitmap_to_svg_layered(image, max_size_bytes=10000, resize=True, target_size=(384, 384), 
                         adaptive_fill=True, num_colors=None):
    """
    Convert bitmap to SVG using layered approach with size optimization
    
    Parameters:
    ----------
    image : PIL.Image
        Input image
    max_size_bytes : int
        Maximum SVG size in bytes
    resize : bool
        Whether to resize image before processing
    target_size : tuple
        Target size for resizing
    adaptive_fill : bool
        Whether to use adaptive space filling
    num_colors : int
        Number of colors to extract (adaptive if None)
        
    Returns:
    -------
    str
        SVG representation
    """
    # Determine optimal number of colors based on image complexity
    if num_colors is None:
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]
        
        # Adaptive color selection
        if pixel_count < 65536:  # 256x256
            num_colors = 8
        elif pixel_count < 262144:  # 512x512
            num_colors = 12
        else:
            num_colors = 16
    
    # Resize image if requested
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
        avg_bg_color = np.mean(img_np, axis=(0,1)).astype(int)
        bg_hex_color = compress_hex_color(f'#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}')
    else:
        bg_hex_color = '#fff'
    
    # Start building SVG structure
    orig_width, orig_height = original_size
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">\n'
    svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = '</svg>'
    
    # Calculate base size
    base_size = len((svg_base + svg_footer).encode('utf-8'))
    available_bytes = max_size_bytes - base_size
    
    # Extract hierarchical features
    features = extract_features_by_scale(img_np, num_colors=num_colors)
    
    # Simple approach without adaptive fill
    if not adaptive_fill:
        svg = svg_base
        for feature in features:
            feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
            
            # Check size limit
            if len((svg + feature_svg + svg_footer).encode('utf-8')) > max_size_bytes:
                break
            
            svg += feature_svg
        
        svg += svg_footer
        return svg
    
    # Adaptive fill approach: calculate sizes at different simplification levels
    feature_sizes = []
    for feature in features:
        feature_sizes.append({
            'original': len(f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level1': len(f'<polygon points="{simplify_polygon(feature["points"], 1)}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level2': len(f'<polygon points="{simplify_polygon(feature["points"], 2)}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level3': len(f'<polygon points="{simplify_polygon(feature["points"], 3)}" fill="{feature["color"]}" />\n'.encode('utf-8'))
        })
    
    # Two-pass approach for optimal space utilization
    svg = svg_base
    bytes_used = base_size
    added_features = set()
    
    # Pass 1: Add most important features at full quality
    for i, feature in enumerate(features):
        feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
        feature_size = feature_sizes[i]['original']
        
        if bytes_used + feature_size <= max_size_bytes:
            svg += feature_svg
            bytes_used += feature_size
            added_features.add(i)
    
    # Pass 2: Add remaining features with progressive simplification
    for level in range(1, 4):
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
    
    # Verify size constraint
    final_size = len(svg.encode('utf-8'))
    if final_size > max_size_bytes:
        # Fallback to basic SVG if size exceeded
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'
    
    return svg


#| export
def generate_bitmap(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=15):
    """
    Generate image using Stable Diffusion
    
    Parameters:
    ----------
    prompt : str
        Text prompt for image generation
    negative_prompt : str
        Negative prompt to guide what to avoid
    num_inference_steps : int
        Number of denoising steps
    guidance_scale : float
        Higher values adhere more closely to prompt
        
    Returns:
    -------
    PIL.Image
        Generated image
    """
    # Generate image with Stable Diffusion
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale,
    ).images[0]
    
    return image


#| export
def generate_and_convert(prompt, prompt_prefix="", prompt_suffix="", negative_prompt="", 
                          num_attempts=3, num_inference_steps=20, guidance_scale=15, verbose=True):
    """
    Generate image with Stable Diffusion, convert to SVG, and evaluate
    
    Parameters:
    ----------
    prompt : str
        Base text prompt
    prompt_prefix : str
        Text to prepend to prompt
    prompt_suffix : str
        Text to append to prompt
    negative_prompt : str
        What to avoid in generation
    num_attempts : int
        Number of attempts to generate
    num_inference_steps : int
        Denoising steps
    guidance_scale : float
        Prompt adherence strength
    verbose : bool
        Whether to show detailed output
        
    Returns:
    -------
    tuple
        (best_svg, best_score)
    """
    best_svg = None
    best_bitmap = None
    best_similarity = -1
    
    # Track timing statistics
    total_start_time = time.time()
    generation_times = []
    conversion_times = []
    evaluation_times = []
    attempt_times = []
    
    # Construct full prompt
    combined_prompt = f"{prompt_prefix} {prompt} {prompt_suffix}".strip()
        
    for i in range(num_attempts):
        attempt_start_time = time.time()
        if verbose: 
            print(f"\n=== Attempt {i+1}/{num_attempts} ===")
        
        # Generate bitmap with Stable Diffusion
        generation_start = time.time()
        bitmap = generate_bitmap(
            combined_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale
        )
        generation_end = time.time()
        generation_time = generation_end - generation_start
        generation_times.append(generation_time)
                
        # Convert to SVG with size limit
        if verbose: 
            print(f"Converting to SVG... ", end="")
        conversion_start = time.time()
        svg_content = bitmap_to_svg_layered(bitmap)
        conversion_end = time.time()
        conversion_time = conversion_end - conversion_start
        conversion_times.append(conversion_time)
                
        # Render SVG to bitmap for evaluation
        rendered_svg = svg_to_png(svg_content)
        svg_size = len(svg_content.encode('utf-8'))
        
        if verbose: 
            print(f"SVG size: {svg_size} bytes")
            # Display original and SVG side by side
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(bitmap)
            plt.title(f"Original Image {i+1}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(rendered_svg)
            plt.title(f"SVG Conversion {i+1}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Evaluate rendered SVG with competition metric
        evaluation_start = time.time()
        svg_scores = evaluate_with_competition_metric(svg_content, prompt)
        evaluation_end = time.time()
        evaluation_time = evaluation_end - evaluation_start
        evaluation_times.append(evaluation_time)
                
        if verbose:
            print(f"VQA Score: {svg_scores['vqa_score']:.4f}")
            print(f"Aesthetic Score: {svg_scores['aesthetic_score']:.4f}")
            print(f"Combined Score: {svg_scores['combined_score']:.4f}")
                
        # Track the best result based on competition score
        if svg_scores['combined_score'] > best_similarity:
            best_similarity = svg_scores['combined_score']
            best_svg = svg_content
            best_bitmap = bitmap
            if verbose: 
                print(f"✅ New best result: {svg_scores['combined_score']:.4f}")
        else:
            if verbose: 
                print(f"❌ Not better than current best: {best_similarity:.4f}")
        
        # Calculate timing for this attempt
        attempt_end_time = time.time()
        attempt_time = attempt_end_time - attempt_start_time
        attempt_times.append(attempt_time)
        
        if verbose:
            print(f"Image generation: {generation_time:.2f}s")
            print(f"SVG conversion: {conversion_time:.2f}s")
            print(f"Evaluation: {evaluation_time:.2f}s")
            print(f"Total for attempt {i+1}: {attempt_time:.2f}s")
    
    # Calculate total processing time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Print timing summary if verbose
    if verbose:
        print("\n=== Timing Summary ===")
        print(f"Average generation time: {sum(generation_times)/len(generation_times):.2f}s")
        print(f"Average conversion time: {sum(conversion_times)/len(conversion_times):.2f}s")
        print(f"Average evaluation time: {sum(evaluation_times)/len(evaluation_times):.2f}s")
        print(f"Average per attempt: {sum(attempt_times)/len(attempt_times):.2f}s")
        print(f"Total processing time ({num_attempts} attempts): {total_time:.2f}s")
        print(f"Best score achieved: {best_similarity:.4f}")
                    
    return best_svg, best_similarity


#| export
# Example parameters for testing
prompt_prefix = "Simple, classic image of"
prompt = "a lighthouse overlooking the ocean"
prompt_suffix = "with flat color blocks, beautiful, minimal details, solid colors only"
negative_prompt = "lines, framing, hatching, background, textures, patterns, details, outlines"

# Generate SVG and evaluate
best_svg, best_score = generate_and_convert(
    prompt, 
    prompt_prefix=prompt_prefix, 
    prompt_suffix=prompt_suffix, 
    negative_prompt=negative_prompt, 
    num_inference_steps=25, 
    guidance_scale=20, 
    num_attempts=5
)


#| export
class Model:
    def __init__(self):
        """
        Initialize model with optimized parameters for SVG generation
        """
        # Configure generation parameters
        self.num_attempts_per_prompt = 3
        self.num_inference_steps = 25
        self.guidance_scale = 20
        
        # Configure prompt engineering
        self.prompt_prefix = "Simple, classic image of"
        self.prompt_suffix = "with flat color blocks, beautiful, minimal details, solid colors only"
        self.negative_prompt = "lines, framing, hatching, background, textures, patterns, details, outlines"
        
        # Track last generated score
        self.last_score = None
            
    def predict(self, prompt: str) -> str:
        """
        Generate SVG image from text description
        
        Parameters:
        ----------
        prompt : str
            Text description of image to generate
            
        Returns:
        -------
        str
            SVG code string
        """
        # Generate image and convert to SVG
        best_svg, best_score = generate_and_convert(
            prompt,
            prompt_prefix=self.prompt_prefix,
            prompt_suffix=self.prompt_suffix,            
            negative_prompt=self.negative_prompt,
            num_attempts=self.num_attempts_per_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            verbose=False  # No verbose output for submission
        )
        
        # Store score for reporting
        self.last_score = best_score
        
        return best_svg


#| export
# Read training data
df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

# Uncomment to test on fewer samples
# df = df.head(3)

# Initialize the model
model = Model()

# Create arrays to store scores and timing data
scores = []
generation_times = []

# Process each description
for i, row in enumerate(df.iterrows()):
    description = row[1]['description']
    
    # Start timing
    start_time = time.time()
    
    # Generate SVG from description
    svg = model.predict(description)
    rendered_img = svg_to_png(svg)
    
    # End timing
    end_time = time.time()
    generation_time = end_time - start_time
    generation_times.append(generation_time)
    
    # Get the score
    score = model.last_score
    scores.append(score)
        
    # Display the generated image
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"Generated for: {description}\nScore: {score:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print progress and statistics
    current_avg_score = np.mean(scores)
    current_avg_time = np.mean(generation_times)
    
    print(f"Processed {i+1}/{len(df)} prompts")
    print(f"Current average score: {current_avg_score:.2f}")
    print(f"Time for this prompt: {generation_time:.2f}s")
    print(f"Current average generation time: {current_avg_time:.2f}s")
    
# Calculate final statistics
avg_score = np.mean(scores)
avg_generation_time = np.mean(generation_times)
total_time_taken = sum(generation_times)

# Project timings for full dataset
projected_time_500_images = 500 * avg_generation_time
projected_hours = projected_time_500_images / 3600

# Output summary statistics
print("\n=== SUMMARY ===")
print(f"Prompts processed: {len(df)}")
print(f"Final average score: {avg_score:.2f}")
print(f"Average generation time per prompt: {avg_generation_time:.2f} seconds")
print(f"Total time elapsed: {timedelta(seconds=total_time_taken)}")
print(f"Projected time for 500 prompts: {projected_hours:.2f} hours ({timedelta(seconds=projected_time_500_images)})")




