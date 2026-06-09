#| default_exp core


#| export

import concurrent
import io
import logging
import re
import re2
import pandas as pd
import numpy as np
import cairosvg
import kagglehub
import torch
from lxml import etree
import vllm

svg_constraints = kagglehub.package_import('metric/svg-constraints/versions/1')


#| export

num_attempt = 3

llm_config = {
    "quantization": "awq",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9, 
    "trust_remote_code": True,
    "dtype": "half",
    "enforce_eager": True,
    "max_model_len": 5120,  
    "disable_log_stats": True,
}

sampling_config = {
    "n": 1,
    "top_k": 15,
    "top_p": 0.9,
    "temperature": 0.6,
    "repetition_penalty": 1.1,
    "seed": 777,
    "skip_special_tokens": True,  # 修改1
    "max_tokens": 1024           # 修改2
}

prompt_template = """
Generate SVG code to visually represent the following text description, while respecting the given constraints.

<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A multicolored landscape with trees, mountains, and a sun in a circular formation representing a picturesque view."</description>
```svg
<svg viewBox="0 0 200 200" width="200" height="200">
  <path fill="#E6FAFF" fill-opacity="1.0" d="M200.0 100.0 C200.0 155.228515625 155.228515625 200.0 100.0 200.0 C44.771480560302734 200.0 0.0 155.228515625 0.0 100.0 C0.0 44.771480560302734 44.771480560302734 0.0 100.0 0.0 C155.228515625 0.0 200.0 44.771480560302734 200.0 100.0 Z"/>
  <path fill="#FF4682" fill-opacity="1.0" d="M178.2277374267578 27.34375 L144.53125 27.34375 L144.53125 3.90625 L178.2277374267578 3.90625 C178.8515625 3.90625 179.22381591796875 4.6015625 178.8777313232422 5.120697021484375 L171.875 15.625 L178.8777313232422 26.129295349121094 C179.22381591796875 26.6484375 178.8515625 27.34375 178.2277374267578 27.34375 Z"/>
  <path fill="#A00046" fill-opacity="1.0" d="M143.7707061767578 97.65625 L140.2550811767578 97.65625 C139.1765594482422 97.65625 138.3019561767578 96.7816390991211 138.3019561767578 95.703125 L138.3019561767578 1.953125 C138.3019561767578 0.8746109008789062 139.1765594482422 0.0 140.2550811767578 0.0 L143.7707061767578 0.0 C144.84921264648438 0.0 145.7238311767578 0.8746109008789062 145.7238311767578 1.953125 L145.7238311767578 95.703125 C145.7238311767578 96.7816390991211 144.849609375 97.65625 143.7707061767578 97.65625 Z"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""


#| export

path = kagglehub.model_download('qwen-lm/qwen2.5/Transformers/32b-instruct-awq/1')

class Model:
    def __init__(self):
        self.model_path = path
        
        self.llm = vllm.LLM(self.model_path,**llm_config)
        self.sampling_params = vllm.SamplingParams(**sampling_config)
        
        self.tokenizer = self.llm.get_tokenizer()
        self.prompt_template = prompt_template
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90*3

    # You could try increasing `max_new_tokens`
    def predict(self, description: str, max_new_tokens=600) -> str:
        def apply_template(prompt, tokenizer):
            messages = [
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return text
        
        def parse_svg_from_response(response):
            matchs = re.findall(r'<svg.*?</svg>', response, re.S)
            if matchs:
                return matchs[-1].strip()
            else:
                return ''
        
        def check_svg_valid(svg):
            try:
                cairosvg.svg2png(bytestring=svg.encode('utf-8'))
                return True
            except:
                return False
        
        def generate_svg():
            try:
                prompt = self.prompt_template.format(description)
                inputs = [apply_template(prompt, self.tokenizer)] * num_attempt
                responses = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
                responses = [x.outputs[0].text for x in responses]
                svgs = [parse_svg_from_response(x) for x in responses]
                # use the first valid svg
                choosen_svg = None
                for svg in svgs:
                    if check_svg_valid(svg):
                        svg = self.enforce_constraints(svg)
                        if check_svg_valid(svg):
                            choosen_svg = svg
                            break
                
                assert choosen_svg is not None
                return choosen_svg

            except Exception as e:
                logging.error('Exception during SVG generation: %s', e)
                return self.default_svg

        return generate_svg()

        # # # Execute SVG generation in a new thread to enforce time constraints
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        #     future = executor.submit(generate_svg)
        #     try:
        #         return future.result(timeout=self.timeout_seconds)
        #     except concurrent.futures.TimeoutError:
        #         logging.warning("Prediction timed out after %s seconds.", self.timeout_seconds)
        #         return self.default_svg
        #     except Exception as e:
        #         logging.error(f"An unexpected error occurred: {e}")
        #         return self.default_svg

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
                continue  # Skip attribute checks for removed elements
    
            # Remove disallowed attributes
            attrs_to_remove = []
            for attr in element.attrib:
                attr_name = etree.QName(attr).localname
                if (
                    attr_name
                    not in self.constraints.allowed_elements[tag_name]
                    and attr_name
                    not in self.constraints.allowed_elements['common']
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
            return self.default_svg


print('done')


%%time
model = Model()


train = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')
solution = pd.read_csv('/kaggle/input/solution/solution.csv')
mymetric = kagglehub.package_import('lorenzonyyy/zekin-metric/versions/4')


def get_submission():
    submission = []
    for row in train.itertuples():
        svg = model.predict(row.description)
        submission.append({'id':row.id,'svg':svg})
    submission = pd.DataFrame(submission)
    return submission

def display(train,submission):

    def svg_to_png(svg_code: str, size: tuple = (384, 384)):
        # Ensure SVG has proper size attributes
        if 'viewBox' not in svg_code:
            svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

        # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)
    
    # get df
    df = train.merge(submission,on='id')

    # display
    fig = plt.figure(figsize=(12, 20),dpi=600)
    # i从1开始（而非0）
    for i, r in enumerate(df.itertuples(), 1):
        plt.subplot(5, 3, i)
        img = svg_to_png(r.svg)
        plt.imshow(img)
        plt.axis('off')
        plt.title(r.description, fontdict={'fontsize': 8})
        
    return fig


print('done')


%%time
submission = get_submission()


submission.to_csv('Model 2-Try 2.csv')


submission = pd.read_csv('Model 2-Try 2.csv')


%%time
mymetric.score(solution,submission,'id',101)


import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image


display = display(train,submission)
display.savefig('Image-Model-2-try-2.png')


display

