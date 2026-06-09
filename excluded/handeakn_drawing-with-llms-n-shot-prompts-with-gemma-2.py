#| default_exp core


#| export
import concurrent
import io
import logging
import re
import re2
import json 
import pandas as pd
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
        self.model_path = kagglehub.model_download('google/gemma-2/Transformers/gemma-2-9b-it/2')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.prompt_template = """
Generate SVG code to visually represent the following text description, while respecting the given constraints.
For designing the SVG, break it down into basic shapes, use coordinates (x, y) to place elements correctly, the size of each element, position relative to others. 
Also, Use paths for complex parts, apply transformations. Please always use SVG elements which are defined in the allowed elements section below.

<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"a moving car on a road at midnight"</description>
<svg viewBox="0 0 256 256" width="256" height="256" fill="none">
  <!-- Night Sky Background -->
  <rect width="256" height="256" fill="black"/>
  <!-- Stars -->
  <circle cx="40" cy="50" r="2" fill="white"/>
  <circle cx="80" cy="20" r="1.5" fill="white"/>
  <circle cx="120" cy="70" r="2.5" fill="white"/>
  <circle cx="160" cy="30" r="2" fill="white"/>
  <circle cx="200" cy="90" r="1.5" fill="white"/>
  <circle cx="220" cy="40" r="2" fill="white"/>
  <circle cx="60" cy="130" r="2" fill="white"/>
  <circle cx="100" cy="180" r="1.5" fill="white"/>
  <circle cx="150" cy="200" r="2.5" fill="white"/>
  <circle cx="180" cy="160" r="2" fill="white"/>
  <circle cx="210" cy="190" r="1.5" fill="white"/>
  <!-- Moon -->
  <circle cx="200" cy="50" r="20" fill="lightgray"/>
  <!-- Road -->
  <rect x="0" y="200" width="256" height="40" fill="gray"/>
  <line x1="0" y1="220" x2="256" y2="220" stroke="white" stroke-width="4" stroke-dasharray="10,10"/>
  <!-- Car Body -->
  <rect x="80" y="170" width="100" height="30" fill="blue" stroke="black" stroke-width="2"/>
  <rect x="95" y="155" width="70" height="20" fill="blue" stroke="black" stroke-width="2"/>
  <!-- Car Windows -->
  <rect x="100" y="160" width="20" height="15" fill="lightblue" stroke="black" stroke-width="1"/>
  <rect x="130" y="160" width="20" height="15" fill="lightblue" stroke="black" stroke-width="1"/>
  <!-- Car Wheels -->
  <circle cx="100" cy="200" r="10" fill="black" stroke="gray" stroke-width="2"/>
  <circle cx="160" cy="200" r="10" fill="black" stroke="gray" stroke-width="2"/>
  <!-- Motion Lines -->
  <line x1="60" y1="185" x2="50" y2="185" stroke="white" stroke-width="2"/>
  <line x1="60" y1="195" x2="45" y2="195" stroke="white" stroke-width="2"/>
</svg>
</example>
<example>
<description>"a tower surrounded by a forest under blue sky"</description>
<svg viewBox="0 0 256 256" width="256" height="256" fill="none">
  <!-- Blue Sky Background -->
  <rect width="256" height="256" fill="skyblue"/>
  
  <!-- Sun -->
  <circle cx="200" cy="50" r="20" fill="yellow"/>
  
  <!-- Trees with Trunks -->
  <rect x="34" y="220" width="12" height="20" fill="saddlebrown" stroke="black" stroke-width="2"/>
  <polygon points="40,200 20,240 60,240" fill="darkgreen" stroke="black" stroke-width="2"/>
  
  <rect x="74" y="210" width="12" height="20" fill="saddlebrown" stroke="black" stroke-width="2"/>
  <polygon points="80,190 60,230 100,230" fill="darkgreen" stroke="black" stroke-width="2"/>
  
  <rect x="114" y="215" width="12" height="20" fill="saddlebrown" stroke="black" stroke-width="2"/>
  <polygon points="120,195 100,235 140,235" fill="darkgreen" stroke="black" stroke-width="2"/>
  
  <rect x="154" y="205" width="12" height="20" fill="saddlebrown" stroke="black" stroke-width="2"/>
  <polygon points="160,185 140,225 180,225" fill="darkgreen" stroke="black" stroke-width="2"/>
  
  <rect x="194" y="210" width="12" height="20" fill="saddlebrown" stroke="black" stroke-width="2"/>
  <polygon points="200,190 180,230 220,230" fill="darkgreen" stroke="black" stroke-width="2"/>
  
  <!-- Tower Base -->
  <rect x="110" y="120" width="40" height="80" fill="gray" stroke="black" stroke-width="2"/>
  
  <!-- Tower Roof -->
  <polygon points="130,80 100,120 160,120" fill="brown" stroke="black" stroke-width="2"/>
  
  <!-- Tower Windows -->
  <rect x="125" y="140" width="10" height="15" fill="yellow" stroke="black" stroke-width="1"/>
  
  <!-- Ground -->
  <rect x="0" y="240" width="256" height="16" fill="darkgreen"/>
</svg>

</example>
<example>
<description>"a goose winning a gold medal"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <!-- Goose Body -->
  <ellipse cx="128" cy="140" rx="50" ry="70" fill="white" stroke="black" stroke-width="2"/>
  <!-- Goose Head -->
  <circle cx="128" cy="70" r="25" fill="white" stroke="black" stroke-width="2"/>
  <!-- Goose Beak -->
  <polygon points="138,65 158,75 138,85" fill="orange" stroke="black" stroke-width="2"/>
  <!-- Goose Eye -->
  <circle cx="120" cy="65" r="4" fill="black"/>
  <!-- Gold Medal Ribbon -->
  <polyline points="100,120 128,100 156,120" stroke="red" stroke-width="6" fill="none"/>
  <!-- Gold Medal -->
  <circle cx="128" cy="130" r="15" fill="gold" stroke="black" stroke-width="2"/>
</svg>
```
</example>


Focus on a clear and concise representation of the input description within the given limitations, be mindful about the object shapes, 
colors, and background details from the description. Unless mentioned in the description, background 
may have a different color than the object. For instance, if a car body is a rectangle at (50, 100), and the wheels are below it, you must position the circles accordingly
If basic shapes aren’t enough, <path> provides powerful commands:
M x y → Move to (x, y)
L x y → Line to (x, y)
C (Cubic Bezier) or Q (Quadratic) → Curves
Z → Close the path
Apply Transformations where it is needed.
Rotation (rotate(degrees x y))
Scaling (scale(sx sy))
Translation (translate(dx dy))
Skewing (skewX(degrees), skewY(degrees))
Always give the complete SVG code with nothing omitted. Never use an ellipsis.
<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90

    # You could try increasing `max_new_tokens`
    def predict(self, description: str, max_new_tokens=512) -> str:
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


data_path = kagglehub.competition_download('drawing-with-llms')
train = pd.read_csv(f'{data_path}/train.csv')
train_questions = pd.read_parquet(f'{data_path}/questions.parquet')
train = pd.merge(train, train_questions, how='left', on='id')
train_df = train.groupby(['id', 'description']).agg({
    'question': list,
    'choices': list,
    'answer': list
}).reset_index()
train_df.head()


train_df['description']


model =  Model()



def generate():
    from IPython.display import SVG
    import time  # Import the time module
    
    logging.basicConfig(level=logging.DEBUG, force=True)

    for desc in train_df['description']:
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



generate()



import kaggle_evaluation

logging.basicConfig(level=logging.INFO, force=True)
kaggle_evaluation.test(Model)


# import shutil

# shutil.rmtree('/kaggle/working', ignore_errors=True)

