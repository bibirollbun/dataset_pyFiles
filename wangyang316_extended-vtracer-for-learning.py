#| default_exp core


#| export
import kagglehub
import os
import torch
import numpy as np
from PIL import Image
import cv2
import vtracer
from diffusers import StableDiffusionPipeline
from typing import Optional, Union, List, Tuple
import logging
import tempfile

device = "cuda:1" if torch.cuda.is_available() else "cpu"


#| export
# Load models globally
print("Loading models...")
model_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

# Initialize Stable Diffusion pipeline on first CUDA device
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to(device)

print("Models loaded successfully!")


#| export

# Setting
svg_constraints = kagglehub.package_import('metric/svg-constraints')
constraints = svg_constraints.SVGConstraints()

from lxml import etree
import re2

default_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="384" height="384"><rect width="100%" height="100%"/><path d="M80 82h224v224H80z"/><g fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="#FF4700" stroke-width="12" d="M150 199q131-7 45-73"/><path stroke="#000010" stroke-opacity=".1" stroke-width="8" d="M270 220q-165-138-99 43"/><path stroke="#FF5E00" stroke-width="4" d="M88 83q-4 70 150 54"/><path stroke="#000" stroke-opacity=".6" stroke-width="8" d="M239 232q-80-23 5-136"/><path stroke="#0000C7" stroke-opacity=".7" stroke-width="16" d="M202 286q76-84 12-25"/><path d="M223 228q-143-98 56-43"/><path stroke="#00002A" stroke-opacity=".6" stroke-width="16" d="M121 268q129 18 124-8"/><path stroke="#8AACFF" stroke-opacity=".5" stroke-width="4" d="M304 85q-13 94-138 12"/><path stroke="#04F" stroke-width="12" d="M242 95q-84 96-47 9"/><path stroke="#13FFFF" stroke-width="6" d="M94 166q41-51 72 57"/><path d="M150 134q75 70-5 126"/><path stroke="#FF6300" stroke-width="14" d="M160 103q24-17 80 9"/><path stroke="#446AFF" stroke-width="2" d="M129 232q91 50 156 74"/><path stroke="#00f" stroke-opacity=".3" stroke-width="2" d="M120 131q-21-43 67 50"/><path stroke="#fff" stroke-width="12" d="M124 135q26-48-19 29"/><path stroke="#FDA1F6" stroke-opacity=".5" stroke-width="10" d="M280 175q23 27-200 100"/><path stroke="#C77145" stroke-opacity=".7" stroke-width="16" d="M133 256q132 0 120-87"/><path stroke="#FF0400" stroke-width="12" d="M123 175q-8-33 18 62"/><path d="M135 111q119 59 139 55"/><path stroke="#FF7C00" stroke-width="10" d="M166 250q-30-95 20-168"/><path stroke="#FFE415" stroke-opacity=".8" stroke-width="16" d="M169 108q11 77 27 93"/><path stroke="#470C00" stroke-opacity=".6" stroke-width="16" d="M234 99q-131 166 2 141"/><path stroke="#040000" stroke-opacity=".4" stroke-width="4" d="M92 172q50 11 87 78"/><path d="M143 112q-9 116 93 150"/><path stroke="#fff" stroke-width="12" d="M195 82q-40 0-92 88"/><path d="M169 103q-84 152 42 151"/><path stroke="#fff" stroke-width="14" d="M252 193q18 43-165 38"/><path stroke="#0A61FF" stroke-opacity=".6" stroke-width="6" d="M117 176q131-82 132 0"/><path stroke="#09A7E4" stroke-width="14" d="M141 223q123-78 65 29"/><path stroke="#34009E" stroke-opacity=".3" stroke-width="16" d="M232 237q43 37 44-72"/><path d="M215 280q88-20 63-88"/><path stroke="#FF4000" stroke-width="10" d="M246 223q-140-71-89 50"/><path stroke="#FFB000" stroke-width="6" d="M187 82q-66 51-103 10"/><path d="M131 168q37 50-27-66"/><path stroke="#FF614F" stroke-width="2" d="M282 248q-158 55 22-22"/><path stroke="#FF9D7A" stroke-width="16" d="M176 153q-44-31-74 109"/><path stroke="#57001C" stroke-opacity=".2" stroke-width="16" d="M206 151q66 57-126 131"/><path stroke="#F50048" stroke-opacity=".4" stroke-width="16" d="M265 199q-133-75-121-7"/><path stroke="#AE1600" stroke-opacity=".7" stroke-width="12" d="M291 256Q84 155 91 165"/><path stroke="#34FFFF" stroke-width="12" d="M95 162q36-67 112-12"/><path stroke="#7B9D6C" stroke-opacity=".6" stroke-width="2" d="M271 158q-62 18-177 111"/><path stroke="red" stroke-width="6" d="M211 210q-101-13-128-51"/><path stroke="#005478" stroke-opacity=".9" stroke-width="4" d="M118 118Q89 241 232 242"/><path d="M147 230q-58-98-51 25"/><path stroke="#0074FF" stroke-opacity=".9" stroke-width="4" d="M154 172q60 68-35 47"/><path stroke="#000574" stroke-opacity=".4" stroke-width="16" d="M101 277q182-67 163-142"/><path stroke="#00005E" stroke-opacity=".1" stroke-width="14" d="M275 168q-87 27-48-66"/><path stroke="#00B6AB" stroke-width="2" d="M209 84q-115 20 72 83"/><path stroke="#00718B" stroke-opacity=".9" stroke-width="14" d="M111 105q93 32-19 70"/><path d="M188 265q-90 16-69 35"/><path stroke="#FDFF00" stroke-width="2" d="M205 214q-82-20-31-132"/><path d="M267 127q33 82 14 135"/><path stroke="#FFFFB7" stroke-width="8" d="M98 255q153-123 52-149"/><path stroke="#BD003F" stroke-opacity=".3" stroke-width="16" d="M270 188Q202 84 80 262"/><path stroke="#EC8500" stroke-opacity=".8" stroke-width="16" d="M254 121q-131-35-20 46"/><path stroke="#950200" stroke-opacity=".4" stroke-width="10" d="M203 162q87-38 7 103"/><path stroke="#FFD670" stroke-opacity=".7" stroke-width="12" d="M249 214q-169-48-131 42"/><path d="M210 226q29-131-75-133"/><path stroke="#FFC559" stroke-opacity=".7" stroke-width="6" d="M201 83q76 56 96 38"/><path d="M150 117q70 160 48 131"/><path stroke="#000020" stroke-width="4" d="M264 221q37-65-13-38"/><path stroke="#595987" stroke-opacity=".6" stroke-width="12" d="M248 246q-38-90 13-81"/><path stroke="#0344A0" stroke-opacity=".6" stroke-width="16" d="M80 265q204-51 1-26"/><path stroke="#F60" stroke-width="10" d="M298 239q-168-31-160-12"/><path stroke="#A47686" stroke-opacity=".5" stroke-width="14" d="M273 152q-92-70-136 32"/><path stroke="#FFAF67" stroke-width="16" d="M166 129q1-46-47 79"/><path stroke="#00000A" stroke-opacity=".5" stroke-width="6" d="M304 162q-35-36-89-14"/><path stroke="#1E167F" stroke-width="10" d="M302 242q-86-26-49-56"/><path stroke="#620114" stroke-opacity=".9" stroke-width="2" d="M248 257q-14-145-56-160"/><path stroke="#C6000D" stroke-opacity=".3" stroke-width="2" d="M217 82q-39 116-128 85"/><path stroke="#000" stroke-opacity=".7" stroke-width="4" d="M109 306q-28-3-20-224"/><path stroke="#ff0" stroke-opacity=".2" stroke-width="6" d="M165 96q6 141 57 117"/><path stroke="#00636C" stroke-width="14" d="M118 165q67 89 16 77"/><path stroke="#F03A00" stroke-opacity=".5" stroke-width="6" d="M253 162q-136-61 51-27"/><path stroke="#275E99" stroke-opacity=".6" stroke-width="16" d="M227 306q-6-72-111-145"/><path d="M250 167q-22-15-169 61"/><path stroke="#01F0C3" stroke-opacity=".6" stroke-width="14" d="M217 199q8 1-118 53"/><path stroke="#FDFF80" stroke-opacity=".3" stroke-width="14" d="M147 134q108-52 65-33"/><path stroke="#5E448B" stroke-opacity=".4" stroke-width="16" d="M198 306q-57-143-60-129"/><path d="M112 135q148 110 13 96"/><path stroke="#798CBA" stroke-opacity=".6" stroke-width="4" d="M245 231q-115 2 34-72"/><path stroke="#FFFFD4" stroke-width="4" d="M92 158q195 102 157 90"/><path stroke="#0093FF" stroke-opacity=".8" stroke-width="12" d="M131 228q-38-33 77-4"/><path stroke="#F25A27" stroke-opacity=".3" stroke-width="14" d="M204 211q72-115-83 0"/><path stroke="#1A4BE6" stroke-opacity=".5" stroke-width="14" d="M227 306q-122-190 77-12"/><path stroke="#04737C" stroke-opacity=".5" stroke-width="16" d="M227 123q-56 42-123 41"/><path stroke="#FF9F10" stroke-opacity=".9" stroke-width="4" d="M135 106q-48 69 144 133"/><path stroke="#8C8086" stroke-opacity=".4" stroke-width="8" d="M296 116q-182 47-102 5"/><path stroke="#00F6FF" stroke-width="2" d="M243 159q-63 24 61-22"/><path stroke="#F97658" stroke-opacity=".4" stroke-width="12" d="M209 120q-91 68-110 134"/><path stroke="#FF3600" stroke-width="12" d="M174 206q20-109 73-30"/><path stroke="#8A0000" stroke-width="4" d="M106 92q46 98 116 122"/><path stroke="#817E60" stroke-opacity=".9" stroke-width="12" d="M188 158q34 82-67-20"/><path stroke="#000040" stroke-width="16" d="M115 274q189-68 35-118"/><path stroke="#976000" stroke-opacity=".4" stroke-width="12" d="M277 169q-38-26-182 73"/><path stroke="#59693F" stroke-opacity=".6" stroke-width="10" d="M158 158q129 98-25 30"/></g><rect width="100%" height="80"/><rect width="100%" height="80" y="304"/><rect width="80" height="100%"/><rect width="80" height="100%" x="304"/><g stroke="#616161"><path d="m140 320.5 4 7m0 0v5.5m0-5.5 4-7"/><path id="a" d="M172.6 324v3.5m0 0v9.5m0-9.5c8-13 7.4 14.5 0 1.5"/><use x="21" y="36" href="#a"/><path id="b" d="M164.774 324.408s-9.472-.045-3 4c8 5-3 4-3 4"/><use x="60" y="36" href="#b"/><use x="71" href="#b"/><use x="77" y="18" href="#b"/><path id="c" d="M149.5 364.5c0 2.5-1.5 4-3 4s-3-1.5-3-4 1.5-4 3-4 3 1.5 3 4Z"/><use x="20" y="-18" href="#c"/><use x="39" y="-36" href="#c"/><path id="d" d="M150 328c1-8 7-2 6 0zm0 0c0 6 6 4 6 4"/><use x="78" y="36" href="#d"/><use x="60" y="36" href="#d"/><use x="14" y="36" href="#d"/><use x="49" y="18" href="#d"/><use x="72" y="18" href="#d"/><path id="f" d="M192 323v3m0 7v-7m0 0 4-3"/><use x="11" y="36" href="#f"/><use x="39" y="18" href="#f"/><use x="14" href="#f"/><path id="g" d="M146 343h2m4 0h-4m0 0v-3m0 3v7.5l3.5.5"/><use x="52" y="-18" href="#g"/><use x="38" href="#g"/><use x="10" y="18" href="#g"/><use x="101" y="18" href="#g"/><use x="36" y="18" href="#g"/><use x="61" href="#g"/><use x="68" href="#g"/><path d="M218 333v-3m-5-6c6-2 5 4 5 4m0 0s-6-2-6 2c1 5 6 0 6 0m0-2v2M221 324l4 9m0 0 3-9m-3 9-4 4M128.5 341.5l2.5 8.5 3-7.5 3.5 7.5 2.5-8.5M196 338v14M153.5 351v-7m0-7v7m0 0s7-7.5 6.5 7"/><path id="h" d="M180.5 342v6.5m0 2.5v-2.5m0 0s-7.5 7.5-6.5-7"/><use x="64" y="18" href="#h" transform="rotate(180 241 364)"/><path d="M143 342v9m0-11v-2M131 369v-11l8 11v-12M173 369l3-5m4-4-4 4m0 0-3-4m3 4 3 5M239.5 331l-1 4M246.5 349l-1 4"/></g></svg>"""

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
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSVG:
    def __init__(
        self,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
        svg_params: Optional[dict] = None
    ):
        """
        Initialize the TextToSVG converter.
        
        Args:
            model_id: The Stable Diffusion model ID to use
            device: The device to run the model on
            svg_params: Parameters for SVG conversion
        """
        self.device = device
        self.pipe = pipe
        
        # Default SVG conversion parameters
        self.svg_params = svg_params or {
            "colormode": "color",
            "mode": "spline",
            "filter_speckle": 8,
            "color_precision": 6,
            "corner_threshold": 60,
            "length_threshold": 4.0,
            "splice_threshold": 45,
            "path_precision": 3
        }

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt using Stable Diffusion.
        
        Args:
            prompt: The text prompt to generate the image from
            negative_prompt: Optional negative prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate the image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        return image

    def split_image_to_grid(self, image: Image.Image, grid_size) -> Tuple[List[Image.Image], str]:
        """
        Split image into a grid of cells.
        
        Args:
            image: Input PIL Image
            grid_size: Size of the grid (e.g., 3 for 3x3 grid)
            
        Returns:
            Tuple of (list of cell images, temporary directory path)
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Convert to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Calculate cell dimensions
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        cells = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract cell
                y_start = i * cell_height
                y_end = (i + 1) * cell_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width
                
                cell = img_array[y_start:y_end, x_start:x_end]
                cell_image = Image.fromarray(cell)
                
                # Save cell
                cell_path = os.path.join(temp_dir, f'cell_{i*grid_size + j + 1}.png')
                cell_image.save(cell_path)
                cells.append(cell_image)
        
        return cells, temp_dir

    def process_consecutive_images(
        self,
        img1_path: str,
        img2_path: str,
        index: int,
        output_dir: str,
        thresh: int = 5
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process consecutive images to find differences.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            index: Index of the image pair
            output_dir: Output directory
            thresh: Threshold for difference detection
            
        Returns:
            Tuple of (result image path, SVG path)
        """
        try:
            # Read images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return None, None
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(gray1, gray2)
            _, thresh_diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
            
            # Create RGBA image with transparency
            result_rgba = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
            result_rgba[:, :, 3] = thresh_diff
            
            # Save results
            result_path = os.path.join(output_dir, f'difference_{index}.png')
            svg_path = os.path.join(output_dir, f'difference_{index}.svg')
            
            cv2.imwrite(result_path, result_rgba)
            
            return result_path, svg_path
            
        except Exception as e:
            logger.error(f"Error processing consecutive images: {e}")
            return None, None

    def process_cell_to_svg(
        self,
        cell_image: Image.Image,
        cell_index: int,
        grid_size: int,
        output_dir: str
    ) -> Optional[str]:
        """
        Process a single cell image to SVG.
        
        Args:
            cell_image: Cell image to process
            cell_index: Index of the cell (0-based)
            grid_size: Size of the grid
            output_dir: Output directory
            
        Returns:
            SVG content for the cell
        """
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(cell_image)
            
            # Convert RGB to BGR (OpenCV format)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Save temporary PNG
            temp_png = os.path.join(output_dir, f'cell_{cell_index}.png')
            temp_svg = os.path.join(output_dir, f'cell_{cell_index}.svg')
            
            cv2.imwrite(temp_png, img_bgr)
            
            # Convert to SVG
            vtracer.convert_image_to_svg_py(
                temp_png,
                temp_svg,
                **self.svg_params
            )
            
            # Read SVG content
            with open(temp_svg, 'r') as f:
                svg_content = f.read()
            
            # Calculate cell position
            row = cell_index // grid_size
            col = cell_index % grid_size
            
            # Extract SVG content between <svg> tags, removing XML declaration
            start = svg_content.find('<svg')
            end = svg_content.find('</svg>')
            if start != -1 and end != -1:
                # Remove XML declaration if present
                svg_content = svg_content[start:end].split('>', 1)[1]
                
                # Add transform to position the cell correctly
                transform = f'<g transform="translate({col * cell_image.width}, {row * cell_image.height})">\n{svg_content}\n</g>'
                return transform
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing cell {cell_index}: {e}")
            return None

    def merge_svgs(self, svg_contents: List[str]) -> str:
        """
        Merge multiple SVG contents into one.
        
        Args:
            svg_contents: List of SVG content strings
            
        Returns:
            Merged SVG content
        """
        # Start with SVG header without XML declaration
        merged_svg = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        
        # Add all cell contents
        for content in svg_contents:
            if content:
                merged_svg += content + '\n'
        
        merged_svg += '</svg>'
        return merged_svg

    def image_to_svg(self, image: Image.Image, grid_size: int) -> str:
        """
        Convert a PIL Image to SVG string using LayerTracer's grid-based approach.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            SVG string
        """
        # Split image into grid
        cells, temp_dir = self.split_image_to_grid(image, grid_size)
        
        try:
            # Process each cell
            svg_contents = []
            for i, cell in enumerate(cells):
                svg_content = self.process_cell_to_svg(
                    cell,
                    i,
                    grid_size,
                    temp_dir
                )
                if svg_content:
                    svg_contents.append(svg_content)
            
            # Merge all SVGs
            merged_svg = self.merge_svgs(svg_contents)
            return merged_svg
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def text_to_svg(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        grid_size: int = 3,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> str:
        """
        Convert text prompt directly to SVG string.
        
        Args:
            prompt: The text prompt to generate the image from
            negative_prompt: Optional negative prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            
        Returns:
            SVG string
        """
        # Generate image from text
        image = self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # Convert image to SVG using LayerTracer's approach
        svg_content = self.image_to_svg(image, grid_size)
        
        return svg_content, image


#| export
import cairosvg

def svg_to_png(svg_code: str, size: tuple = (512, 512)) -> Image.Image:

    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)



# Example usage
converter = TextToSVG()
    
# Example prompt
# prompt = "A beautiful sunset over mountains, digital art"
# prompt = "a lighthouse overlooking the ocean"
# prompt = "a purple forest at dusk"
prompt = "gray wool coat with a faux fur collar"
negative_prompt = "blurry, low quality"
    
# Convert text to SVG
svg_content, image = converter.text_to_svg(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,
    width=512,
    grid_size=4,
    num_inference_steps=30,
    guidance_scale=7.5
)

svg_content = enforce_constraints(svg_content)
    
logger.info("SVG file has been generated successfully!")

import io
import matplotlib.pyplot as plt

# Render SVG to bitmap for evaluation
rendered_svg = svg_to_png(svg_content)
svg_size = len(svg_content.encode('utf-8'))

print(f"SVG size: {svg_size} bytes")
# Display the images side by side
plt.figure(figsize=(12, 6))
            
# Original bitmap
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title(f"Original Image")
plt.axis('off')
            
# SVG conversion
plt.subplot(1, 2, 2)
plt.imshow(rendered_svg)
plt.title(f"SVG Conversion")
plt.axis('off')
            
plt.tight_layout()
plt.show()


#| export

class Model:
    def __init__(self):
        '''Optional constructor, performs any setup logic, model instantiation, etc.'''
        
        # Set number of image attempts per prompt for competition here...

        self.num_inference_steps = 20
        self.guidance_scale = 7.5

        self.prompt_prefix = ""
        self.prompt_suffix = ", flat color blocks, solid colors only"
        self.negative_prompt = ""
        
        self.verbose=False

    def predict(self, prompt: str) -> str:
        '''Generates SVG which produces an image described by the prompt.

        Args:
            prompt (str): A prompt describing an image
        Returns:
            String of valid SVG code.
        '''
        
        # Convert text to SVG
        svg_content, image = converter.text_to_svg(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            grid_size=5,
            num_inference_steps=30,
            guidance_scale=7.5
        )

        svg_content = enforce_constraints(svg_content)
        return svg_content


# Read the CSV file
import pandas as pd
import time

df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

# uncomment to test on just a few
df = df.head(5)

# Initialize the model
model = Model()

# Create arrays to store scores and timing data
scores = []
generation_times = []

for i, row in enumerate(df.iterrows()):
    description = row[1]['description']
    
    # Start timing
    start_time = time.time()
    
    # Generate image from description
    svg = model.predict(description)
    rendered_img = svg_to_png(svg)
    svg_size = len(svg.encode('utf-8'))
    print(f"SVG size: {svg_size} bytes")
    
    # End timing
    end_time = time.time()
    generation_time = end_time - start_time
    generation_times.append(generation_time)
        
    # Display the image being processed
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"Image for: {description}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print progress, current average score, and timing info
    current_avg_time = np.mean(generation_times)
    
    print(f"Processed {i+1}/{len(df)} prompts")
    print(f"Time for this prompt: {generation_time:.2f}s")
    print(f"Current average generation time: {current_avg_time:.2f}s")
    
# When all done, calculate final statistics
avg_generation_time = np.mean(generation_times)
total_time_taken = sum(generation_times)

# Calculate projections for 500 images
projected_time_500_images = 500 * avg_generation_time
projected_hours = projected_time_500_images / 3600

print("\n=== SUMMARY ===")
print(f"Prompts processed: {len(df)}")
print(f"Average generation time per prompt: {avg_generation_time:.2f} seconds")

