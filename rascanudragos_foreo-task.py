# Normally i would have to create a wrapper with this package
# The competition does not allow online access for the notebooks submitted
!pip install -q vtracer ftfy regex cairosvg tqdm git+https://github.com/openai/CLIP.git


!nvidia-smi


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import traceback
import json
import time
import io

from skimage import measure, color
from PIL import Image
from IPython.display import SVG, display
from vtracer import convert_raw_image_to_svg
from io import BytesIO

from diffusers import StableDiffusionPipeline, DDIMScheduler
import clip

from lxml import etree
from tqdm import tqdm

import ast
import cairosvg

import kagglehub
from IPython.display import display
import polars as pl
import torch
print(torch.cuda.device_count())


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

scheduler = DDIMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

pipe_sdv2 = StableDiffusionPipeline.from_pretrained(
    stable_diffusion_path,
    scheduler=scheduler,
    torch_dtype=torch.float32,  
    safety_checker=None    
)

# Move to GPU and apply optimizations
device_sdv2 = torch.device("cuda:1")
pipe_sdv2.to(device_sdv2)


prompt = "A smiling sun with sunglasses. simple illustration. vivid colors. minimal style. geometrical shapes."


generator = torch.Generator(device=device).manual_seed(42)

with torch.autocast(device):
    result = pipe_sdv2(prompt, 
                       height=512,
                       width=512,
                       guidance_scale=7.5, 
                       num_inference_steps=25,
                       generator=generator)

image_sdv2 = result.images[0]


display(image_sdv2)


def raster_to_svg(image: Image.Image, threshold=0.5) -> str:
    image = image.convert("L")
    img_array = np.array(image) / 255.0

    contours = measure.find_contours(img_array, threshold)

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">'.format(*img_array.shape[::-1])]

    for contour in contours:
        points = " ".join(f"{x:.2f},{y:.2f}" for y, x in contour)  # note y,x to match image
        svg.append(f'<polyline fill="none" stroke="black" stroke-width="1" points="{points}" />')

    svg.append('</svg>')
    return "\n".join(svg)


svg_image_sdv2 = raster_to_svg(image_sdv2, 0.7) #with bigger threshold more details are visible

display(SVG(svg_image_sdv2))


def convert_bitmap_with_vectorizer(gen_image: str):
    '''
    Takes a raster image and converts it to svg
    '''
    buffer = BytesIO()
    gen_image.convert("RGBA").save(buffer, format="PNG")
    gen_image_bytes = buffer.getvalue()
    
    # Run VTracer
    result = convert_raw_image_to_svg(
        gen_image_bytes,
        img_format='png',
        mode='polygon', 
        colormode='color',
        hierarchical='cutout',
        color_precision=16,
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=120,
        path_precision=3,
    )
    return result


svg_image_sdv2 = convert_bitmap_with_vectorizer(image_sdv2)

display(SVG(svg_image_sdv2))


def generate_bitmap_sdv2(prompt, negative_prompt='', height=768, width=768, guidance_scale=15, num_inference_steps=25):    
    with torch.autocast(device):
        result = pipe_sdv2(prompt, 
                           negative_prompt=negative_prompt,
                           height=height,
                           width=width,
                           guidance_scale=guidance_scale, 
                           num_inference_steps=num_inference_steps,
                           )
    return result.images[0]


train_path = kagglehub.competition_download('drawing-with-llms', 'train.csv')
train_df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

train_question_df = pd.read_parquet('/kaggle/input/drawing-with-llms/questions.parquet')

train_df.head()


train_question_df.head()


train_question_df = train_question_df.groupby('id').apply(lambda df: df.to_dict(orient='list'))
train_question_df = train_question_df.reset_index(name='qa')

train_question_df['question'] = train_question_df.qa.apply(lambda qa: json.dumps(qa['question'], ensure_ascii=False))

train_question_df['choices'] = train_question_df.qa.apply(
    lambda qa: json.dumps(
        [x.tolist() for x in qa['choices']], ensure_ascii=False
    )
)

train_question_df['answer'] = train_question_df.qa.apply(lambda qa: json.dumps(qa['answer'], ensure_ascii=False))

merged_df = pd.merge(train_df, train_question_df, how='left', on='id')


merged_df.head(5)


!nvidia-smi


# Load CLIP model to first GPU
device_clip = "cuda:0"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device_clip)

class CLIPVQAEvaluator:
    def __init__(self):
        self.model = clip_model
        self.preprocess = clip_preprocess
        self.device = device_clip

    
    def svg_to_image(self, svg_string, size=(384, 384)):
        png_data = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"))
        return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)

    
    def get_yes_probability(self, image: Image.Image, prompt: str) -> float:
        """
        Compute cosine similarity between the image and text prompt using CLIP.

        Returns float: The cosine similarity
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

        return similarity

    
    def score_multiple_choice_set(self, svg_str, questions, choices_list, answers):
        """
        Evaluate how well an image matches a multiple-choice question set using CLIP.

        Returns float: Average similarity score across all questions.
        """
        image = self.svg_to_image(svg_str)
        score = 0
        valid = 0

        for q, choices, a in zip(questions, choices_list, answers):
            prompt = (
                f"In response to the question: '{q}' "
                f"with options: {', '.join(choices)}, does the correct answer appear to be '{a}'?"
            )
            try:
                score += self.get_yes_probability(image, prompt)
                valid += 1
            except Exception as e:
                print(f"[!] Skipping due to error: {e}")

        return score / valid if valid > 0 else 0.0



evaluator = CLIPVQAEvaluator()
best_svgs = []
best_scores = []
generation_times = []

# We will only iterate over the first 10 entries
for i, (_, row) in enumerate(merged_df.head(10).iterrows()):
    description = row["description"]
    # Improve prompt
    prompt = f"Simple, classic image of {description} with flat color blocks, beautiful, minimal details, solid colors only"
    best_svg = None
    best_score = -1

    print(f"\n=== Prompt {i+1}/10: {description} ===")

    # Try generating and evaluating the image 3 times
    for attempt in range(1, 4):
        start_time = time.time()

        
        img_sd = generate_bitmap_sdv2(prompt, negative_prompt="lines, framing, hatching, background, textures, patterns, details, outlines")
        img_sd = img_sd.resize((384, 384), Image.LANCZOS)
        svg_image_sdv2 = convert_bitmap_with_vectorizer(img_sd)

        
        # Evaluate the SVG using the VQA model and multiple choice questions
        score = evaluator.score_multiple_choice_set(
            svg_str=svg_image_sdv2,
            questions=row.question,
            choices_list=row.choices,
            answers=row.answer
        )

        elapsed = time.time() - start_time
        generation_times.append(elapsed)

        if score > best_score:
            best_score = score
            best_svg = svg_image_sdv2

        print(f"[Attempt {attempt}] Score: {score:.4f}, Time: {elapsed:.2f}s")

    best_svgs.append(best_svg)
    best_scores.append(best_score)

    # Render the best SVG into a bitmap image for visualization
    rendered_img = evaluator.svg_to_image(best_svg)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"Best of 3 for: {description}\nScore: {best_score:.2f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Best Score for Prompt {i+1}: {best_score:.4f}")
    print(f"Avg Time (so far): {np.mean(generation_times):.2f}s")
    print(f"Avg Score (so far): {np.mean(best_scores):.4f}")



!nvidia-smi

