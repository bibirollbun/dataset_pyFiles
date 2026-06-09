!pip install -q rasterio


import time
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from io import BytesIO
import pickle

import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import rasterio
from rasterio.windows import from_bounds
from PIL import Image
import folium
import ipywidgets as widgets

from IPython.display import display, clear_output, HTML

# API imports
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


# Data classes
@dataclass
class Coordinate:
    lat: float
    lon: float


class TifImageExtractor:
    """Extract image from TIF file given coordinates"""
    
    def __init__(self, dataset: rasterio.DatasetReader):
        self.dataset = dataset
        self.bounds = self.dataset.bounds
        self.transform = self.dataset.transform
    
    def extract_image(self, coord: Coordinate, degree_offset: float = 0.025, 
                     size: int = 512) -> Optional[Image.Image]:
        """Extract image from coordinates"""
        # Check bounds
        if not (self.bounds.left <= coord.lon <= self.bounds.right and 
               self.bounds.bottom <= coord.lat <= self.bounds.top):
            return None
        
        # Create bounding box
        min_lon = coord.lon - degree_offset
        max_lon = coord.lon + degree_offset
        min_lat = coord.lat - degree_offset
        max_lat = coord.lat + degree_offset
        
        # Create window and read data
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, self.transform)
        data = self.dataset.read(window=window, fill_value=0)
        
        # Convert to PIL image
        if data.shape[0] >= 3:  # Check RGB channels
            img_array = np.transpose(data[:3], (1, 2, 0))  # (C,H,W) → (H,W,C)
            
            # Normalize
            if img_array.dtype == np.uint8:
                normalized = img_array
            elif img_array.dtype == np.uint16:
                normalized = (img_array / 256).astype(np.uint8)
            else:
                min_val, max_val = img_array.min(), img_array.max()
                if max_val > min_val:
                    normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(img_array, dtype=np.uint8)
            
            pil_image = Image.fromarray(normalized, mode='RGB')
            
            # Resize
            if size:
                pil_image = pil_image.resize((size, size), Image.Resampling.LANCZOS)
            
            return pil_image
        
        return None
    
    def close(self):
        """Release resources"""
        if self.dataset:
            self.dataset.close()


# Example
sentinel_tif_path = "/kaggle/input/2-2-plan-research-area/2_2_plan_research_area/data/focus_rgb_swir1_nir_red.tif"
dataset = rasterio.open(sentinel_tif_path)
png_patch_extractor = TifImageExtractor(dataset)

coordinates = [
    Coordinate(-10.1679, -73.9568),    # Original point
    Coordinate(-10.1579, -73.9468),    # Northeast (+0.01, +0.01)
    Coordinate(-10.1779, -73.9468),    # Southeast (-0.01, +0.01)
    Coordinate(-10.1579, -73.9668)     # Northwest (+0.01, -0.01)
]

# Extract images from each coordinate
patches = []
for i, coord in enumerate(coordinates):
    print(f"Generating patch {i+1}... Coordinate: ({coord.lon:.1f}, {coord.lat:.1f})")
    
    result = png_patch_extractor.extract_image(
        coord=coord,
        degree_offset=0.01,
        size=384
    )
    patches.append(result)


# 2x2 visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('4 Satellite Image Patches', fontsize=16)

for i, patch in enumerate(patches):
    row = i // 2
    col = i % 2
    
    axes[row, col].imshow(patch)
    axes[row, col].set_title(f'Patch {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Results
print(f"\nTotal {len(patches)} patches generated!")


class Tool:
    """Base tool class"""
    def __init__(self, name: str, description: str, parameters: dict, fn: Callable[..., Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn
    
    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class QuestionManager:
    """Question registration and management"""
    
    def __init__(self):
        self.questions = {}
        self.tools = {}
    
    def register_question(self, question_id: str, question_text: str, 
                         response_categories: List[str], weights: Dict[str, float]):
        """Register a question
        
        Example:
        manager.register_question(
            question_id="object_detection",
            question_text="Can you detect the object?",
            response_categories=["visible", "unclear", "not_visible"],
            weights={"visible": 1.0, "unclear": 0.5, "not_visible": 0.0}
        )
        """
        self.questions[question_id] = {
            "text": question_text,
            "categories": response_categories,
            "weights": weights
        }
        
        # Create tool
        properties = {}
        for category in response_categories:
            properties[category] = {
                "type": "number", 
                "description": f"{category} confidence score (0.0-1.0)"
            }
            
        parameters = {
            "type": "object",
            "properties": properties,
            "required": [response_categories[0]]
        }
        
        def evaluate_fn(**scores):
            return self.calculate_probability(question_id, **scores)
        
        tool = Tool(
            name=f"evaluate_{question_id}",
            description=f"Evaluate: {question_text}",
            parameters=parameters,
            fn=evaluate_fn
        )
        
        self.tools[tool.name] = tool
    
    def calculate_probability(self, question_id: str, **scores) -> float:
        """Convert scores to probability"""
        if question_id not in self.questions:
            return 0.5
        
        config = self.questions[question_id]
        weights = config["weights"]
        
        # Format scores (missing categories default to 0)
        formatted_scores = {}
        for category in weights.keys():
            score = scores.get(category, 0)
            formatted_scores[category] = score if isinstance(score, (int, float)) else 0
        
        # Weighted average
        weighted_sum = sum(score * weights[key] for key, score in formatted_scores.items())
        total_score = max(sum(formatted_scores.values()), 1)
        
        return round(weighted_sum / total_score, 3) if total_score > 0 else 0.5
    
    def get_tool_specs(self) -> List[dict]:
        """Get all registered tool specs"""
        return [tool.spec() for tool in self.tools.values()]
    
    def list_questions(self) -> List[str]:
        return list(self.questions.keys())


class VQAEvaluator:
    """VQA evaluator"""
    
    def __init__(self, api_key: str, question_manager: QuestionManager, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.question_manager = question_manager
    
    def _image_to_data_uri(self, image: Image.Image, jpeg_quality: int = 100) -> str:
        """Convert image to data URI"""
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    
    def _extract_all_scores(self, message) -> Dict[str, Dict[str, float]]:
        """Extract all question scores from message"""
        if not (hasattr(message, 'tool_calls') and message.tool_calls):
            return {}
        
        all_scores = {}
        
        for call in message.tool_calls:
            try:
                tool_name = call.function.name
                if not tool_name.startswith('evaluate_'):
                    continue
                
                question_id = tool_name.replace('evaluate_', '')
                if question_id not in self.question_manager.questions:
                    continue
                
                args = json.loads(call.function.arguments)
                config = self.question_manager.questions[question_id]
                
                if question_id not in all_scores:
                    all_scores[question_id] = {cat: 0 for cat in config["categories"]}
                
                for k, v in args.items():
                    if isinstance(v, (int, float)) and k in all_scores[question_id]:
                        all_scores[question_id][k] += v
                        
            except (json.JSONDecodeError, KeyError):
                continue
        
        return all_scores

    def evaluate_image(self, image: Image.Image, question: str = None) -> Tuple[float, float, Dict]:
        """Image evaluation → (score, confidence, vqa_details)"""
        if image is None:
            return 0.0, 0.0, {}
        
        # Use general questions if no specific question provided
        if question is None:
            question = "Please analyze this satellite/aerial image. Evaluate features such as terrain, vegetation, artificial structures, etc."
        
        try:
            result = self.evaluate(question, image)
            score = result["average_probability"]
            confidence = score  # In VQA, probability value is the confidence
            return score, confidence, result
        except Exception as e:
            print(f"VQA evaluation failed: {e}")
            return 0.0, 0.0, {"error": str(e)}
  
    def evaluate(self, question: str, image: Optional[Image.Image] = None, 
                         default_prob: float = 0.5, jpeg_quality: int = 100) -> Dict:
        """Detailed evaluation results"""
        if not question.strip():
            return {"average_probability": default_prob, "question_results": {}}
        
        # Compose message
        content = [{"type": "text", "text": question}]
        if image:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_uri(image, jpeg_quality)}
            })
        
        # API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            tools=self.question_manager.get_tool_specs()
        )
        
        # Build detailed results
        all_scores = self._extract_all_scores(response.choices[0].message)
        
        question_results = {}
        probabilities = []
        
        for qid, scores in all_scores.items():
            probability = self.question_manager.calculate_probability(qid, **scores)
            question_results[qid] = {
                "probability": probability,
                "scores": scores,
                "question_text": self.question_manager.questions[qid]["text"]
            }
            probabilities.append(probability)
        
        average_prob = round(sum(probabilities) / len(probabilities), 3) if probabilities else default_prob
        
        return {
            "average_probability": average_prob,
            "question_results": question_results
        }


# Question management system for agricultural suitability assessment

# Create QuestionManager instance
question_manager = QuestionManager()

# Register questions
question_manager.register_question(
    question_id="river_for_farming", 
    question_text="Do you see a river that would help farming?",  
    response_categories=["yes", "no"],  
    weights={"yes": 1.0, "no": 0.0} 
)

question_manager.register_question(
    question_id="fertile_red_soil",  
    question_text="Do you see red soil that looks fertile?",  
    response_categories=["yes", "no"], 
    weights={"yes": 1.0, "no": 0.0} 
)

question_manager.register_question(
    question_id="farm_access_road", 
    question_text="Do you see dirt roads for farm access?", 
    response_categories=["yes", "no"], 
    weights={"yes": 1.0, "no": 0.0}  
)

question_manager.register_question(
    question_id="nearby_housing",
    question_text="Do you see residential areas nearby?",
    response_categories=["yes", "no"],
    weights={"yes": 1.0, "no": 0.0}
)


# Create VQA (Visual Question Answering) evaluator
evaluator = VQAEvaluator(
    api_key=UserSecretsClient().get_secret("OpenAI_API_KEY"),  # OpenAI API key (retrieved from secure storage)
    question_manager=question_manager  # Pass the question manager configured above
)

# Set prompt for image analysis
question_prompt = "This is 5km satellite image. Please use all tools to correct and save the image questions."

# Set coordinates for the area to analyze
coord = Coordinate(-10.1679, -73.9568)

# Extract satellite image using PNG patch extractor
png_patch = png_patch_extractor.extract_image(
    coord=coord,  
    degree_offset=0.025,  # ±0.025 degree range from coordinates (approximately 2.8km x 2.8km)
    size=384  # Image size: 384x384 pixels
)

# Execute image analysis using VQA evaluator
results = evaluator.evaluate(question_prompt, image=png_patch)


# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Satellite image
if hasattr(png_patch, 'convert'):
    patch_array = np.array(png_patch.convert('RGB'))
else:
    patch_array = np.array(png_patch)

ax1.imshow(patch_array)
ax1.set_title(f'Satellite Image\nCoordinate: ({coord.lon:.4f}, {coord.lat:.4f})', fontsize=12)
ax1.axis('off')

# Right: Analysis results
questions = list(results['question_results'].keys())
probabilities = [results['question_results'][q]['probability'] for q in questions]
labels = [results['question_results'][q]['question_text'].replace('Do you see ', '').replace('?', '') for q in questions]

# Color coding based on probability
colors = ['#2E8B57' if p >= 0.7 else '#FFB347' if p >= 0.5 else '#F08080' for p in probabilities]

bars = ax2.barh(range(len(questions)), probabilities, color=colors)
ax2.set_yticks(range(len(questions)))
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel('Probability', fontsize=12)
ax2.set_title(f'Analysis Results\nAverage: {results["average_probability"]:.1f}', fontsize=12)
ax2.set_xlim(0, 1)

# Add probability values on bars
for i, (bar, prob) in enumerate(zip(bars, probabilities)):
    ax2.text(prob + 0.02, i, f'{prob:.1f}', va='center', fontsize=10, fontweight='bold')

# Add grid for better readability
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Analysis Summary:")
print(f"• Average probability: {results['average_probability']:.1f}")
print(f"• Highest score: {max(probabilities):.1f} (River for farming)")
print(f"• Lowest score: {min(probabilities):.1f} (Residential areas)")

