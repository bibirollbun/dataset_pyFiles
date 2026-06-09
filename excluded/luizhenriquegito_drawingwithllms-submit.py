#| default_exp core


#| export
import kagglehub
drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')


#| export
import torch

import io
import os
import cv2
import json
import pandas as pd
import numpy as np
import cairosvg
import tempfile
import vtracer

import matplotlib.pyplot as plt

from pathlib import Path
from IPython.display import SVG
from PIL import Image
from pprint import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline, DDIMScheduler


#| export
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

SEED = 42
if SEED is not None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

STABLE_DIFFUSION_PATH = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")


#| export
class ImageGenerator(ABC):
    @abstractmethod
    def generate(self) -> str:
        pass

class StableDiffusionGenerator(ImageGenerator):
    def __init__(self):
        self.scheduler = DDIMScheduler.from_pretrained(
            STABLE_DIFFUSION_PATH,
            subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            STABLE_DIFFUSION_PATH,
            scheduler=self.scheduler,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(DEVICE)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: int,
        *args,
        **kwargs
    ):
        print(f'PROMPT: {prompt}')
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,  # (mais para melhor qualidade / mais lento)
            guidance_scale=guidance_scale,  # (quão rigorosamente seguir as instruções)
            generator=torch.Generator(device=DEVICE).manual_seed(42),
        ).images[0]
        return image

class ImageGeneratorFactory:
    @classmethod
    def get_image_generator(self, image_generator: str) -> ImageGenerator:
        match image_generator:
            case 'stable_diffusion':
                return StableDiffusionGenerator()
            case _:
                raise ValueError(f'Image Generator does not exist.')

image_generator = ImageGeneratorFactory.get_image_generator('stable_diffusion')


#| export
IMAGE_SIZE = (75 ,75)
def svg_conversion(img, image_size=IMAGE_SIZE):
        temp_dir = tempfile.TemporaryDirectory()
        # Open the image, resize it, and save it to the temporary directory
        resized_img = img.resize(image_size)
        temp_file_path = os.path.join(temp_dir.name, "abc.png")
        resized_img = resized_img.convert("RGB")
        resized_img.save(temp_file_path)
    
        svg_path = os.path.join(temp_dir.name, "gen_svg.svg")
        vtracer.convert_image_to_svg_py(
                    temp_file_path,
                    svg_path,
                    colormode="color",  # ["color"] or "binary"
                    hierarchical="cutout",  # ["stacked"] or "cutout"
                    mode="polygon",  # ["spline"] "polygon", or "none"
                    filter_speckle=4,  # default: 4
                    color_precision=6,  # default: 6
                    layer_difference=16,  # default: 16
                    corner_threshold=60,  # default: 60
                    length_threshold=10,  # in [3.5, 10] default: 4.0
                    max_iterations=100,  # default: 10
                    splice_threshold=45,  # default: 45
                    path_precision=8,  # default: 8
                )
        if os.path.getsize(svg_path) < 10_000:
            is_within_size=True
        else:
            is_within_size=False
    
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_string = f.read()
        print(f'MENOR QUE 10K BYTES: {is_within_size}')
    
        return svg_string


def svg_to_pil_image(svg_string: str) -> Image.Image:
    png_buffer = io.BytesIO()
    cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), write_to=png_buffer)
    png_buffer.seek(0)
    return Image.open(png_buffer)


#| export

PROMPT = 'Minimalist flat vector of {description}, with solid colors, geometric shapes, no textures or gradients, and a clean background.'
NEGATIVE_PROMPT = 'textured background, patterns, noise, gradients, shadows, rough surface, grain, fabric, wallpaper, complex background, abstract background'

class Model:
    def __init__(self):
        return None

    def predict(self, prompt: str) -> str:
        image = image_generator.generate(
            prompt=PROMPT.format(description=prompt),
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=20, 
            guidance_scale=15,
        )
        return svg_conversion(image)


model = Model()

IMAGE = model.predict('a yellow dog')

svg_to_pil_image(IMAGE)

