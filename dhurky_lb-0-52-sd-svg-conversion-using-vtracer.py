#| default_exp core


#| export
import warnings
warnings.filterwarnings("ignore")
import concurrent
import io
import logging
import re
import re2
import cairosvg
import time
import os
from lxml import etree
import torch
import kagglehub
import vtracer
import tempfile
from PIL import Image
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler

svg_constraints = kagglehub.package_import('metric/svg-constraints')
class Model:
    def __init__(self):
        self.model_name = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
        self.default_svg= """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds=250
        self.load_model()
        self.generator = torch.Generator("cuda:0").manual_seed(1024)

    def load_model(self):
        scheduler = DDIMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.model = StableDiffusionPipeline.from_pretrained(self.model_name,scheduler=scheduler,torch_dtype=torch.float16,safety_checker=None)
        self.model = self.model.to("cuda:0")
        self.model.safety_checker=None  # working

    def predict(self, description: str) -> str:
        def generate_svg():
            try:
                output_decoded = self._generating_svg(description)

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

    def _generating_svg(self, description:str):
        prompt = "simple, minimalistic icon, vector, white background,'{}'"
        prompt = prompt.format(description)
        image = self.model(prompt,height=512, width=512,  generator=self.generator).images[0]
        svg_string, is_within_size = self.svg_conversion(image)
        if not is_within_size:
            svg_string, is_within_size = self.svg_conversion(image,image_size=(50,50))

        if not is_within_size:
            return self.default_svg
        return svg_string
            
        
    def svg_conversion(self,img, image_size=(100,100)):
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
                    max_iterations=10,  # default: 10
                    splice_threshold=45,  # default: 45
                    path_precision=8,  # default: 8
                )
        if os.path.getsize(svg_path) < 10000:
            is_within_size=True
        else:
            is_within_size=False
    
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_string = f.read()
    
    
        return svg_string, is_within_size
        
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


# svg_gen=Model()


# import polars as pl
# import os
# from IPython.display import SVG, display

# import time  # Import the time module

# # logging.basicConfig(level=logging.DEBUG, force=True)

# train = pl.read_csv('/kaggle/input/drawing-with-llms/train.csv')[:10]


# for desc in train.get_column('description'):
#     svg=svg_gen.predict(desc)
#     display(SVG(data=svg))
# #     # prompt = """Generate a simple vector and icon based on the descrption. Where please use 6 colurs and make the background as white which is given below Description:'{}'"""
# #     # prompt = """Generate a simple, minimalistic vector and icon based on the description provided below. Description: '{}'"""
# #     # prompt = "Generate a simple,minimalistic vector and icon based on the descrption which is given below Description:'{}'"
# #     # # prompt = "Generate a simple and minimalistic images like a vector without any gradient colours based on the descrption which is given below Description: orange_corduroy_overalls"
    


    


# #     # prompt = prompt.format(desc)
# #     # image = pipe(prompt, generator=generator).images[0]
# #     # image_name=desc[:50].replace(" ","_")
# #     # print(image_name)
# #     # os.makedirs("/kaggle/working/inference_result",exist_ok=True)
# #     # image.save(f"/kaggle/working/inference_result/{image_name}.jpg")
# #     # svg=svg_conversion(image)
# #     display(SVG(data=svg))
# #     # display(image)
    
# #     print("/n/n","="*20)
# #     break
# #     # image = pipe(prompt).images[0]  




