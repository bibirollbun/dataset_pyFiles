#| default_exp core


# !pip install -U -q transformers
# !pip install -q cairosvg
# !pip install -q scour
# !pip install -q lxml
# !python-pip openmpi-bin libopenmpi-dev
# !pip install -q lmdeploy


#| export
try:
    import os
    import sys
    import re
    
    import re
    from xml.parsers.expat import ExpatError
    from scour.scour import scourString, sanitizeOptions
    import logging
    import os
    import PIL
    import torch
    import re
    import kagglehub
    import time
    import signal
    import os
    from contextlib import contextmanager
    from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
except Exception as e:
    print(f"Error in import: {e}")


#| export

# import cairosvg
import os
import sys
import re

import re
from xml.parsers.expat import ExpatError
import logging

def fix_svg_code(svg_code):
    """
    Fix common SVG issues, particularly focusing on namespace issues and unbound prefixes.
    Takes a more conservative approach to avoid introducing new errors.
    
    Args:
        svg_code (str): The SVG code to fix
        
    Returns:
        str: Fixed SVG code
    """
    # If SVG is empty or not a string, return as is
    if not svg_code or not isinstance(svg_code, str):
        return svg_code
    
    # Fix 1: Add missing namespaces
    svg_tag_pattern = re.compile(r'<svg\s+([^>]*)>')
    svg_tag_match = svg_tag_pattern.search(svg_code)
    
    if svg_tag_match:
        attributes = svg_tag_match.group(1)
        updated_attributes = attributes
        
        # Add svg namespace if missing
        if 'xmlns="http://www.w3.org/2000/svg"' not in attributes:
            updated_attributes += ' xmlns="http://www.w3.org/2000/svg"'
        
        # Add xlink namespace if it's used but missing
        if 'xlink:' in svg_code and 'xmlns:xlink="http://www.w3.org/1999/xlink"' not in attributes:
            updated_attributes += ' xmlns:xlink="http://www.w3.org/1999/xlink"'
        
        # Replace the SVG tag
        new_svg_tag = f'<svg {updated_attributes}>'
        svg_code = svg_code.replace(svg_tag_match.group(0), new_svg_tag)
    
    # Fix 2: Fix defs appearing after use
    # Only if we have both defs and use tags
    defs_pattern = re.compile(r'<defs>(.*?)</defs>', re.DOTALL)
    defs_match = defs_pattern.search(svg_code)
    
    use_pattern = re.compile(r'<use\s')
    use_match = use_pattern.search(svg_code)
    
    if defs_match and use_match:
        defs_pos = defs_match.start()
        first_use_pos = use_match.start()
        
        if defs_pos > first_use_pos:
            # Move defs section to the beginning, right after the SVG tag
            defs_content = defs_match.group(1)
            svg_code = svg_code.replace(defs_match.group(0), '')  # Remove current defs
            
            # Find position to insert the defs (right after svg opening tag)
            svg_end_pos = svg_tag_pattern.search(svg_code).end()
            svg_code = (
                svg_code[:svg_end_pos] + 
                f'\n<defs>{defs_content}</defs>\n' + 
                svg_code[svg_end_pos:]
            )
    
    # Fix 3: Handle specific unbound prefix errors for 'xlink' and other namespaces
    # This is a critical fix for your error
    if 'xlink:href' in svg_code:
        if 'xmlns:xlink=' not in svg_code:
            svg_code = svg_code.replace('<svg ', '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ')
    
    # Fix 4: Fix unclosed tags (very specific to common SVG fixes)
    unclosed_tags = ['defs', 'g', 'svg']
    for tag in unclosed_tags:
        opening_count = len(re.findall(f'<{tag}[^>]*>', svg_code))
        closing_count = len(re.findall(f'</{tag}>', svg_code))
        
        if opening_count > closing_count:
            svg_code += f'</{tag}>' * (opening_count - closing_count)
    
    return svg_code
    
def extract_svg(input_string:str):
    pattern = r'<svg.*?</svg>'
    match = re.search(pattern, input_string, re.DOTALL)
    
    if match:
        return match.group(0)
    else:
        # print("from extract svg")
        return """<svg width="100" height="100" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""

from scour.scour import scourString, sanitizeOptions
def optimize_svg_preserve_ids(svg_code: str, precision: int = 1) -> str:
    """
    Optimizes SVG string using scour, shortening IDs to preserve references
    (like <use href="#id">), while applying other optimizations.

    Args:
        svg_code: The input SVG code as a string.
        precision: The number of decimal places for numeric values.

    Returns:
        The optimized SVG code as a string, or the original if optimization fails.
    """
    if not svg_code:
        return ""

    options = sanitizeOptions() # Start with default options

    # --- Preservation-Focused Options ---
    options.remove_metadata = True
    options.remove_descriptive_elements = True
    options.remove_comments = True
    options.strip_xml_prolog = True
    options.enable_viewboxing = True

    # ID Handling (Preserve & Shorten)
    options.strip_ids = False        # Do NOT strip all IDs
    options.shorten_ids = True       # Shorten necessary IDs instead
    # options.shorten_ids_prefix = '' # Optional prefix if needed

    # Whitespace
    options.remove_newlines = True
    options.indent_depth = -1
    options.indent_type = 'none'

    # Optimizations
    options.keep_editor_data = False
    options.remove_defs = False
    options.optimize_gradients = True
    options.optimize_colors = True
    options.optimize_transforms = True
    options.optimize_elements = True
    options.optimize_paths = True
    options.create_groups = True

    # Precision
    options.digits = precision

    try:
        # Ensure input is string
        if isinstance(svg_code, bytes):
            svg_code = svg_code.decode('utf-8')

        optimized_svg = scourString(svg_code, options=options)
        return optimized_svg if optimized_svg is not None else svg_code
    except Exception as e:
        # print(f"Error during SVG optimization (preserve_ids): {e}", file=sys.stderr)
        return svg_code

@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")
    
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


#| export

SYSTEM_PROMPT = """
You are a svg code writer. who always write the svg code based on the description given by the user.

Strictly Follow these points, otherwise you will get penalty:
1. Think how this image exists, write it's features, aesthetics.
2. Then you will start writing svg-code, meeting criterion defined below.
3. Each SVG may only include elements and attributes from an allowlist. Note that CSS style elements are not allowed.
4. No SVG may include any rasterized image data or data from external sources.
5. it should have good aesthetics following description.
6. You can write long svg-code
7. Make sure there are no missing namespace declaration in the <svg> tag
8. Make sure to avoid any unbound prefix error
allowlist = allowed_elements: dict[str, set[str]] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
            },
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feFlood': {'result'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }

Remember to follow proper description aesthetics, you can write long svg-code.
You can't use any element beyond given allowedlist, Strict penalty for not doing so.
Don't use any external links, that require internet

* Remember you can't use any elements outside of allowed_elements, it is utmost strict conditions,
You can write long svg-code if you need.
"""


#| export

import os
import PIL

# Patch PIL._util.is_directory manually
import types
if not hasattr(PIL, '_util'):
    PIL._util = types.SimpleNamespace()
PIL._util.is_directory = os.path.isdir


#| export

import kagglehub
import time



#| export
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(
    top_p=0.9,
    top_k=40,
    temperature=0.7,
    max_new_tokens=8192,
    do_sample=True
)

# pipe = pipeline(model_path, backend_config=backend_config)



#| export

import torch
import re
import kagglehub
import time
import signal
import os
from contextlib import contextmanager

# Load the SVG constraints package
package = kagglehub.package_import('metric/svg-constraints/versions/1')
cont = package.SVGConstraints()

class Model:
    def __init__(self):
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.default_svg = """<svg width="100" height="100" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        try:
            
            # it's actually internLM
            model_path = kagglehub.model_download("arpit1bansal/qwen-2.5-7b-instruct/transformers/default/4")
            backend_config = TurbomindEngineConfig(tp=2)
            
            self.pipe = pipeline(model_path, backend_config=backend_config)

        except Exception as e:
            print(f"Failed to initialize LMdeploy: {e}")
            raise RuntimeError(f"LMdeploy initialization failed: {e}")


    def predict(self, prompt: str) -> str:
        '''Generates SVG which produces an image described by the prompt.
        Args:
            prompt (str): A prompt describing an image
        Returns:
            String of valid SVG code.
        '''
        # Define messages for the model
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": 
                    f"""write svg code for {prompt}, think about the images, how it should be designed.
             following proper aesthetics.
                        then write svg-code for it.\n There is no need to give any explaination after writing svg-code. stop there."""
            }
        ]
        # print("user message:", messages[-1])
        try:
            with timeout_context(75):
                outputs = self.pipe(messages, gen_config=gen_config)
                content = outputs.text
                response = extract_svg(content)
                response = optimize_svg_preserve_ids(response)
                try:
                    cont.validate_svg(response)
                except Exception as e:
                    print("exception validation", e)
                    try:
                        response = fix_svg_code(response)
                        cont.validate_svg(response)
                    except Exception as e:
                        print("exception after fix validation", e)
                        response = self.default_svg

        except TimeoutError as e:
            response = self.default_svg
            print("Timeout Error:", e)
        except Exception as e:
            response = self.default_svg
            print("error in generation", e)
        torch.cuda.empty_cache()

        return response


model = Model()


import pandas as pd
df = pd.read_csv("/kaggle/input/drawing-with-llms/train.csv")


# ques = pd.read_parquet("/kaggle/input/drawing-with-llms/questions.parquet")


# ques


# import time
# t1 = time.time()
# l = []
# for desc in df['description']:
#     svg_code = model.predict(desc)
#     l.append(svg_code)

# submission = pd.DataFrame({"svg":l})
# t2 = time.time()
# print(t2-t1)


# submission.head(15)


# from IPython.display import SVG
# for i in l:
#     time.sleep(4)
#     SVG(i)


import time
t1 = time.time()
svg_code = model.predict(df['description'][3])
t2 = time.time()
print(t2-t1)


# print(svg_code)


# from IPython.display import SVG


# SVG(svg_code)


# df['description'][3]


# del model
# import gc
# gc.collect()
# torch.cuda.empty_cache()




