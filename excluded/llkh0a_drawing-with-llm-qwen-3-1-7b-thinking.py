from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import kagglehub
import random
import torch


#| default_exp core


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
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

# Add debugging environment variable
import os

# Add CUDA_LAUNCH_BLOCKING=1 to the environment for debugging


svg_constraints = kagglehub.package_import('metric/svg-constraints')




class Model:
    def __init__(self):
        self.model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/1.7b/1")
        
         # Quantization Configuration
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"":0},
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            
        )
       
        self.prompt_template = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
        <constraints>
        * **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
        * **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
        </constraints>

        <example>
        <description>"A red circle with a blue square inside"</description>
        <think>
        The user wants an SVG representation of "A red circle with a blue square inside".
        I need to create an SVG with a red circle and a blue square placed inside it.
        I will use the svg element as the root.
        I will use the circle element for the red circle.
        I will use the rect element for the blue square.
        I will set the fill attribute for the circle to "red".
        I will set the fill attribute for the square to "blue".
        I need to choose coordinates and dimensions. A viewBox="0 0 100 100" might be simpler than 256x256.
        Let's center the circle in the viewBox. cx="50" cy="50".
        Let's give the circle a radius, say r="40".
        Now, the blue square needs to be inside the circle.
        Let's center the square as well. x and y should be calculated based on the desired size.
        If the square has side length s, the top-left corner (x, y) should be (50 - s/2, 50 - s/2).
        Let's choose a side length for the square, say s=40.
        So, x = 50 - 40/2 = 30, y = 50 - 40/2 = 30.
        The square dimensions will be width="40" and height="40".
        This square will fit inside the circle with radius 40 (the diagonal of the square is sqrt(40^2 + 40^2) = 40 * sqrt(2) â‰ˆ 56.5, which is less than the diameter 2 * 40 = 80).
        So the square centered at (50, 50) with side 40 fits within the circle centered at (50, 50) with radius 40.

        The SVG structure will be:
        <svg viewBox="0 0 100 100" width="100" height="100">
        <circle cx="50" cy="50" r="40" fill="red"/>
        <rect x="30" y="30" width="40" height="40" fill="blue"/>
        </svg>

        Let's double-check the constraints:

        Allowed Elements: svg, circle, rect are used. OK.

        Allowed Attributes: viewBox, width, height, fill, cx, cy, r, x, y are used. OK.

        Well-formed and valid: Yes.

        Strictly adheres: Yes.

        Complete SVG code: Yes.

        No ellipsis: Yes.

        The example uses a viewBox="0 0 256 256". I should probably stick to that to be consistent with the example format provided by the user.
        Let's recalculate for viewBox="0 0 256 256".
        Center: (128, 128).
        Circle: cx="128" cy="128". Let's choose a radius, say r="100". fill="red".
        Square: Centered at (128, 128). Let's choose a side length, say s=100.
        x = 128 - 100/2 = 78.
        y = 128 - 100/2 = 78.
        width="100" height="100". fill="blue".
        Check if the square fits: Diagonal = 100 * sqrt(2) â‰ˆ 141.4. Diameter of circle = 2 * 100 = 200. Yes, it fits comfortably.

        The revised SVG:
        <svg viewBox="0 0 256 256" width="256" height="256">
        <circle cx="128" cy="128" r="100" fill="red"/>
        <rect x="78" y="78" width="100" height="100" fill="blue"/>
        </svg>

        This looks correct and matches the description and constraints. The user provided the start of the SVG, so I just need to complete it.

        <svg viewBox="0 0 256 256" width="256" height="256">
        <circle cx="128" cy="128" r="100" fill="red"/>
        <rect x="78" y="78" width="100" height="100" fill="blue"/>
        </svg>
        </think>
        ```svg
        <svg viewBox="0 0 256 256" width="256" height="256">
        <circle cx="128" cy="128" r="100" fill="red"/>
        <rect x="78" y="78" width="100" height="100" fill="blue"/>
        </svg>
        ```
        </example>


        Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

        <description>{}</description>
        ```svg
        <svg viewBox="0 0 256 256" width="256" height="256">
        """
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 300

    # You could try increasing `max_new_tokens`
    def __del__(self):
        # Clean up the temporary directory when the model is destroyed
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    def predict(self, description: str, max_new_tokens=512) -> str:
        def generate_svg():
            try:
                
                prompt = self.prompt_template.format(description)
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )

                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                # # Check for NaNs or infinities in tensor values
                # if torch.isnan(model_inputs.input_ids).any() or torch.isinf(model_inputs.input_ids).any():
                #     logging.error("Input tensor contains NaNs or infinities.")
                #     return self.default_svg

                # # Add an option to switch to CPU for debugging
                # if DEVICE.type == "cuda":
                #     logging.info("Switching to CPU for debugging.")
                #     self.model = self.model.to("cpu")
                #     model_inputs = model_inputs.to("cpu")

                # # Temporarily reduce max_new_tokens for debugging
                with torch.no_grad():
                    generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=32768  # Reduced for debugging
                    )



                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
                # parsing thinking content
                try:
                    # rindex finding 151668 ()
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                logging.debug('Thinking content from del: %s', thinking_content)
                logging.debug('Output content from model: %s', content)
                

                matches = re.findall(r"<svg.*?</svg>", content, re.DOTALL | re.IGNORECASE)
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


model = Model()


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

