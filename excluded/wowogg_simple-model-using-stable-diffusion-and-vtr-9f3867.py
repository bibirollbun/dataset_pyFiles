#| default_exp core


#| export
import kagglehub

import concurrent
import io
import os
import vtracer
import tempfile
import re
import re2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm import tqdm
import pandas as pd
from io import BytesIO
import math
from lxml import etree
import logging

# Set up device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#| export

# Load Stable Diffusion pipeline
stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
scheduler = DDIMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained(
        stable_diffusion_path,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
        safety_checker=None   # Disable safety checker for speed
    )
pipeline.to(device)
print("Stable Diffusion model loaded successfully")


def generate_image(prompt, height=512, width=512, num_inference_steps=25, guidance_scale=12):
    """
    Generate an image using Stable Diffusion
        
    Args:
        prompt (str): Text description for image generation
        height (int): Height of the output image
        width (int): Width of the output image
        num_inference_steps (int): Number of denoising steps
            
    Returns:
        PIL.Image.Image: Generated image as a PIL image
    """
    # logging.debug('Generating image for prompt: %s', prompt)
        
    # Generate image
    image = pipeline(
        prompt, 
        height=height, 
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
        
    return image


# prompt = "gray wool coat with a faux fur collar"
# image = generate_image(prompt)
# print(type(image))
# image


#| export

def svg_conversion(img, image_size=(128,128)):
    tmp_dir = tempfile.TemporaryDirectory()
    # Open the image, resize it, and save it to the temporary directory
    resized_img = img.resize(image_size)
    tmp_file_path = os.path.join(tmp_dir.name, "tmp.png")
    resized_img = resized_img.convert("RGB")
    resized_img.save(tmp_file_path)
    
    svg_path = os.path.join(tmp_dir.name, "gen_svg.svg")
    vtracer.convert_image_to_svg_py(
                tmp_file_path,
                svg_path,
                colormode="color",  # ["color"] or "binary"
                hierarchical="cutout",  # ["stacked"] or "cutout"
                mode="polygon",  # ["spline"] "polygon", or "none"
                filter_speckle=4,  # default: 4
                color_precision=5,  # default: 6
                layer_difference=16,  # default: 16
                corner_threshold=60,  # default: 60
                length_threshold=10,  # in [3.5, 10] default: 4.0
                max_iterations=10,  # default: 10
                splice_threshold=45,  # default: 45
                path_precision=4,  # default: 8
            )
    print(os.path.getsize(svg_path))
    if os.path.getsize(svg_path) < 10000:
        is_within_size=True
    else:
        is_within_size=False
    
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_str = f.read()
    
    # print(svg_str)
    return svg_str, is_within_size



#| export

# Setting
svg_constraints = kagglehub.package_import('metric/svg-constraints')
constraints = svg_constraints.SVGConstraints()

def enforce_constraints(svg_string):
    """Enforces constraints on an SVG string, removing disallowed elements and attributes.

    Args:
        svg_string : str 
            The SVG string to process.

    Returns:
        svg_string : str
            The processed SVG string, or the default SVG if constraints
            cannot be satisfied.
    """
    # logging.info('Sanitizing SVG...')

    try:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        root = etree.fromstring(svg_string, parser=parser)
    except etree.ParseError as e:
        logging.error('SVG Parse Error: %s. Returning default SVG.', e)
        return default_svg
    
    elements_to_remove = []
    for element in root.iter():
        tag_name = etree.QName(element.tag).localname
    
        # Remove disallowed elements
        if tag_name not in constraints.allowed_elements:
            elements_to_remove.append(element)
            continue  # Skip attribute checks for removed elements
    
        # Remove disallowed attributes
        attrs_to_remove = []
        for attr in element.attrib:
            attr_name = etree.QName(attr).localname
            if (
                attr_name
                not in constraints.allowed_elements[tag_name]
                and attr_name
                not in constraints.allowed_elements['common']
            ):
                attrs_to_remove.append(attr)
    
        for attr in attrs_to_remove:
            logging.debug(
                'Attribute "%s" for element "%s" not allowed. Removing.',
                attr,
                tag_name,
            )
            del element.attrib[attr]
    
        # Check and remove invalid href attributes
        for attr, value in element.attrib.items():
            if etree.QName(attr).localname == 'href' and not value.startswith('#'):
                logging.debug(
                    'Removing invalid href attribute in element "%s".', tag_name
                )
                del element.attrib[attr]

        # Validate path elements to help ensure SVG conversion
        if tag_name == 'path':
            d_attribute = element.get('d')
            if not d_attribute:
                logging.warning('Path element is missing "d" attribute. Removing path.')
                elements_to_remove.append(element)
                continue # Skip further checks for this removed element
            # Use regex to validate 'd' attribute format
            path_regex = re2.compile(
                r'^'  # Start of string
                r'(?:'  # Non-capturing group for each command + numbers block
                r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                r'\s*'  # Optional whitespace after command
                r'(?:'  # Non-capturing group for optional numbers
                r'-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?'  # First number
                r'(?:[\s,]+-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)*'  # Subsequent numbers with mandatory separator(s)
                r')?'  # Numbers are optional (e.g. for Z command)
                r'\s*'  # Optional whitespace after numbers/command block
                r')+'  # One or more command blocks
                r'\s*'  # Optional trailing whitespace
                r'$'  # End of string
            )
            if not path_regex.match(d_attribute):
                logging.warning(
                    'Path element has malformed "d" attribute format. Removing path.'
                )
                elements_to_remove.append(element)
                continue
            logging.debug('Path element "d" attribute validated (regex check).')
        
    # Remove elements marked for removal
    for element in elements_to_remove:
        if element.getparent() is not None:
            element.getparent().remove(element)
            logging.debug('Removed element: %s', element.tag)

    try:
        cleaned_svg_string = etree.tostring(root, encoding='unicode')
        return cleaned_svg_string
    except ValueError as e:
        logging.error(
            'SVG could not be sanitized to meet constraints: %s', e
        )
        return default_svg


#| export

class Model:
    def __init__(self):
        """
        Initialize the pipeline with Stable Diffusion model
        
        Args:
            sd_model (str): Stable Diffusion model ID to load
        """
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""

    def predict(self, description: str) -> str:
        """
        Process a prompt through the full pipeline
        
        Args:
            prompt (str): Text description
            
        Returns:
            dict: Dictionary containing paths to outputs and the SVG string
        """
        self.prefix = ""
        self.suffix = ""

        prompt = self.prefix + description + self.suffix
        img = generate_image(prompt)
            
        # Convert to SVG
        svg, is_within_size = svg_conversion(img,image_size=(128,128))
        if not is_within_size:
            svg, is_within_size = svg_conversion(img,image_size=(96,96))
        if not is_within_size:
            svg, is_within_size = svg_conversion(img,image_size=(64,64))

        del img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not is_within_size:
            svg = self.default_svg

        matches = re.findall(r"<svg.*?</svg>", svg, re.DOTALL | re.IGNORECASE)
        if matches:
            svg = matches[-1]
        else:
            svg = self.default_svg
        svg = enforce_constraints(svg)
        return svg


import kaggle_evaluation

logging.basicConfig(level=logging.INFO, force=True)
kaggle_evaluation.test(Model)


import cairosvg
from PIL import Image
import matplotlib.pyplot as plt

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


# Read the CSV file
df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

# uncomment to test on just a few
df = df.head(3)

# Initialize the model
model = Model()

for i, row in enumerate(df.iterrows()):
    # print_kaggle_memory_status()
    description = row[1]['description']
    
    # Generate image from description
    svg = model.predict(description)
    rendered_img = svg_to_png(svg)
        
    # Display the image being processed
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"Prompt: {description}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


svg_val_path = kagglehub.dataset_download('raresbarbantan/draw-svg-validation')
val_df = pd.read_csv(f'{svg_val_path}/validation.csv')
val_df = val_df.groupby('id').apply(lambda df: df.to_dict(orient='list'), include_groups=False)
val_df = val_df.reset_index(name='qa')
val_df['description'] = val_df.qa.apply(lambda qa: qa['description'][0])
val_df = val_df.drop("qa", axis=1)
val_df = val_df.head(5)
val_df.head()


for i, row in enumerate(val_df.iterrows()):
    # print_kaggle_memory_status()
    description = row[1]['description']
    
    # Generate image from description
    svg = model.predict(description)
    rendered_img = svg_to_png(svg)
        
    # Display the image being processed
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"Prompt: {description}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

