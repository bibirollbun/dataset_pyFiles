

import kagglehub
drawing_with_llms_path = kagglehub.competition_download('drawing-with-llms')
svg_constraints = kagglehub.package_import('metric/svg-constraints')


import subprocess, os, json, torch, psutil, platform, typing, sys, time, pathlib, math
!nvidia-smi
print(f"Torch version : {torch.__version__}\nGPU count : {torch.cuda.device_count()}\nSystem RAM : {psutil.virtual_memory().total/1e9:.1f} GB")


import torch
import io
import os
import cv2
import cairosvg
import tempfile
import vtracer
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import SVG
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler



PROMPT = prompt = "simple, minimalistic icon, vector, white background,{description}"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model:
    def __init__(self):
        self.model_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
        self.constraints = svg_constraints.SVGConstraints()
        self.load_model()
            
    def load_model(self):
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_path,
            subfolder="scheduler"
        )
        self.model = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            scheduler=self.scheduler,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(DEVICE)
        self.generator = torch.Generator(device=DEVICE).manual_seed(42)
        
    def generate(self, prompt: str):
        image = self.model(prompt, height=512, width=512,generator=self.generator).images[0]
        return image
    
    def svg_conversion(self, img, image_size=(100,100)):
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
            is_within_size = True
        else:
            is_within_size = False
    
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_string = f.read()
        print(f'MENOR QUE 10K BYTES: {is_within_size}')
    
        return svg_string
    
    def predict(self, prompt: str) -> str:
        image = self.generate(
            prompt=PROMPT.format(description=prompt)
        )
        return self.svg_conversion(image)


model = Model()

IMAGE = model.predict('a yellow dog')

