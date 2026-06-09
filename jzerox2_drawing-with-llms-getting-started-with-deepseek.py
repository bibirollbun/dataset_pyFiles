#| default_exp core


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")


# https://www.kaggle.com/code/metric/svg-image-fidelity/notebook
import gc
import torch

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel

class SVGEvaluator:
    """Evaluates SVG images based on their similarity to a given text description using CLIP.

    This class handles SVG validation, conversion to PNG, and CLIP-based scoring.

    Attributes
    ----------
    device : str
        The device used for CLIP (either 'cuda' if available or 'cpu').
    model : model
    preprocess : callable
        The preprocessing function for images used by the CLIP model.

    Parameters
    ----------
    model_name : str, default='google-siglip-so400m-patch14-384'
        The name of the CLIP model to load.
    constraints : SVGConstraints, optional
        The constraints to use for SVG validation. If None, default constraints are used.
    """

    def __init__(
        self,
        model_name: str = 'google/siglip-so400m-patch14-384'
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model_path = self.model_name
        self.model = AutoModel.from_pretrained(self.model_path, token=HF_TOKEN).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.size = (384, 384)
        
    def svg_to_png(self, svg_code: str) -> Image.Image:
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
            svg_code = svg_code.replace(
                '<svg', f'<svg viewBox="0 0 {self.size[0]} {self.size[1]}"'
            )

        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB').resize(self.size)

    def evaluate_svg(self, svg_code: str, target_class: str) -> float:
        """
        Evaluates the fidelity of an SVG image against a target description using SigLIP.

        This method validates the SVG, converts it to a PNG, preprocesses it, and calculates the SigLIP-based similarity score with the provided description.

        Parameters
        ----------
        svg_code : str
            The SVG string to evaluate.
        target_class : str
            The text description that the SVG should represent.

        Returns
        -------
        float
            The mean similarity score (a value between 0 and 1) representing the match between the SVG and its description.

        """
        target_class = "SVG illustration of " +  target_class # add
        # Convert SVG to PNG
        image = self.svg_to_png(svg_code)
        # Preprocess image and text
        inputs = self.processor(
            text=[target_class], images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)

        # Get features and normalize
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores
            similarities = (image_features @ text_features.T).squeeze()

        return similarities.item()

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model if it exists
        if hasattr(self, 'model'):
            del self.model

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

    def evaluate_svg(self, svg_code: str, target_class: str) -> float:
        """Evaluates the fidelity of an SVG image against a target description using SigLIP."""
        target_class = "SVG illustration of " + target_class
        image = self.svg_to_png(svg_code)
        inputs = self.processor(
            text=[target_class], images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze()

        return similarities.item()

evaluator = SVGEvaluator()


#| export
import concurrent
import io
import logging
import re
import re2

import cairosvg
import kagglehub
import torch
from lxml import etree
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

svg_constraints = kagglehub.package_import('metric/svg-constraints')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self):
         # Quantization Configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model_path = kagglehub.model_download('deepseek-ai/deepseek-r1/Transformers/deepseek-r1-distill-qwen-7b/1')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.evaluator = SVGEvaluator()
        self.prompt_template = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90

    def predict(self, description: str, max_new_tokens=768) -> str:
        def generate_svg():
            try:
                prompt = self.prompt_template.format(description)
                inputs = self.tokenizer(text=prompt, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                    )

                output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                logging.debug('Output decoded from model: %s', output_decoded)

                matches = re.findall(r"<svg.*?</svg>", output_decoded, re.DOTALL | re.IGNORECASE)
                if matches:
                    svg = matches[-1]
                else:
                    return self.default_svg
                    
                logging.debug('Unprocessed SVG: %s', svg)
                svg = self.enforce_constraints(svg)
                logging.debug('Processed SVG: %s', svg)
                validation_score = self.evaluator.evaluate_image(svg,description)
                if validation_score < 0.7:  # Retry if poor quality
                    return self.predict(description)
                # Ensure the generated code can be converted by cairosvg
                cairosvg.svg2png(bytestring=svg.encode('utf-8'))
                return svg
            except Exception as e:
                logging.error('Exception during SVG generation: %s', e)
                return self.default_svg

        # Execute SVG generation in a new thread to enforce time constraints
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_svg)
            try:
                return future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                logging.warning("Prediction timed out after %s seconds.", self.timeout_seconds)
                return self.default_svg
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                return self.default_svg
                
    def enforce_constraints(self, svg_string: str) -> str:
        """Enforces constraints on an SVG string, removing disallowed elements
        and attributes.

        Parameters
        ----------
        svg_string : str
            The SVG string to process.

        Returns
        -------
        str
            The processed SVG string, or the default SVG if constraints
            cannot be satisfied.
        """
        logging.info('Sanitizing SVG...')
        try:
            parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
            root = etree.fromstring(svg_string, parser=parser)
        except etree.ParseError as e:
            logging.error('SVG Parse Error: %s. Returning default SVG.', e)
            return self.default_svg
    
        elements_to_remove = []
        for element in root.iter():
            tag_name = etree.QName(element.tag).localname
    
            # Remove disallowed elements
            if tag_name not in self.constraints.allowed_elements:
                elements_to_remove.append(element)
                continue
    
            # Remove disallowed attributes and check attribute values
            attrs_to_remove = []
            for attr, value in element.attrib.items():
                attr_name = etree.QName(attr).localname
                if (
                    attr_name not in self.constraints.allowed_elements[tag_name]
                    and attr_name not in self.constraints.allowed_elements['common']
                ):
                    attrs_to_remove.append(attr)
                else:
                    # Check if color attributes are valid CSS colors
                    if attr_name in ['fill', 'stroke'] and not self.is_valid_css_color(value):
                        attrs_to_remove.append(attr)
                    # Check if dimensions are positive numbers
                    if attr_name in ['width', 'height', 'r', 'x', 'y', 'cx', 'cy', 'rx', 'ry'] and not self.is_positive_number(value):
                        attrs_to_remove.append(attr)
                    # Check if opacity is within the valid range
                    if attr_name == 'opacity' and not self.is_valid_opacity(value):
                        attrs_to_remove.append(attr)
    
            for attr in attrs_to_remove:
                logging.debug('Attribute "%s" for element "%s" not allowed. Removing.', attr, tag_name)
                del element.attrib[attr]
    
        # Remove elements marked for removal
        for element in elements_to_remove:
            if element.getparent() is not None:
                element.getparent().remove(element)
                logging.debug('Removed element: %s', element.tag)
    
        try:
            cleaned_svg_string = etree.tostring(root, encoding='unicode')
            return cleaned_svg_string
        except ValueError as e:
            logging.error('SVG could not be sanitized to meet constraints: %s', e)
            return self.default_svg

    def is_valid_css_color(self, color: str) -> bool:
        # Implement a simple check for valid CSS color values
        return re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color) is not None or color in ['red', 'blue', 'green', 'black', 'white']
    
    def is_positive_number(self, value: str) -> bool:
        try:
            return float(value) > 0
        except ValueError:
            return False
    
    def is_valid_opacity(self, value: str) -> bool:
        try:
            return 0 <= float(value) <= 1
        except ValueError:
            return False
  


import kaggle_evaluation

logging.basicConfig(level=logging.INFO, force=True)
kaggle_evaluation.test(Model)


def generate():
    import polars as pl
    from IPython.display import SVG
    import time  # Import the time module
    
    logging.basicConfig(level=logging.DEBUG, force=True)
    
    train = pl.read_csv('/kaggle/input/drawing-with-llms/train.csv')
    display(train.head())
    
    model = Model()
    svgs = []
    for desc in train.get_column('description'):
        start_time = time.time()  # Record start time
        svg = model.predict(desc)
        end_time = time.time()    # Record end time
        elapsed_time = end_time - start_time # Calculate elapsed time
        print(f"Prediction time for description '{desc[:20]}...': {elapsed_time:.4f} seconds") # Print time
    
        try:
            display(SVG(svg))
        except Exception as e:
            print(e)
            continue

# Uncomment and run the line below to see some generated images
generate()

