#| default_exp core


#!nvidia-smi


import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, AutoProcessor
from PIL import Image

svg_constraints = kagglehub.package_import('metric/svg-constraints')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE', DEVICE)

class SVGSanitizer:
    def __init__(self, constraints, default_svg):
        self.constraints = constraints
        self.default_svg = default_svg
    
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

class SVGProcessor:
    @staticmethod
    def clean_and_extract_svgs(text, default_svg):
        text = re.sub(r'^.*?(<svg\b)', r'\1', text, flags=re.DOTALL)
        svg_blocks = re.findall(r'<svg\b.*?</svg>', text, re.DOTALL)
    
        if svg_blocks:
            tmp = re.findall(r'<svg\b.*?', svg_blocks[-1], re.DOTALL)
            if len(tmp) > 1:
                tmp2 = svg_blocks[-1].split('<svg')
                return '<svg ' + tmp2[-1]
            else:
                return svg_blocks[-1]
        else:
            if "<svg" in text and "</svg>" not in text:
                text += "</svg>"
                return text
            return default_svg
    
    @staticmethod
    def svg_conversion_check(topic, base_svg_code, default_svg):
        try:
            cairosvg.svg2png(bytestring=base_svg_code.encode('utf-8'), write_to="temp.png")
            return base_svg_code
        except Exception as e:
            print(f"Failed to convert {topic} due to {str(e)}, Returning default SVG.")
            return default_svg
            


class Model:
    def __init__(self):
        self.model_path = kagglehub.model_download('vinothkumarsekar89/qwen3_4b_svg_code_generation/transformers/01') 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.sanitizer = SVGSanitizer(self.constraints, self.default_svg)
    
    def get_response(self, description):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
        ### Instruction:
        Generate a SVG code for the given input:
    
        ### Input:
        {}
    
        ### Response:
        """
        formatted_input = alpaca_prompt.format(description)
        inputs = self.tokenizer([formatted_input], return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def predict(self, description: str) -> str:
        # Get the response and clean, extract, and check SVG code
        output_decoded = self.get_response(description)
        base_svg_code = SVGProcessor.clean_and_extract_svgs(output_decoded, self.default_svg)
        clean_svg_code = self.sanitizer.enforce_constraints(base_svg_code)
        clean_svg_code = SVGProcessor.svg_conversion_check(description, clean_svg_code, self.default_svg)
        
        return clean_svg_code



import sys
#sys.path.append('/kaggle/input/drawing-with-llms/published/')
import kaggle_evaluation

logging.basicConfig(level=logging.INFO, force=True)
kaggle_evaluation.test(Model)

