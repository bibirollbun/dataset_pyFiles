#| default_exp core


#| export
# copied from https://www.kaggle.com/code/richolson/let-s-defeat-ocr-easy-lb-boost/notebook and modded
def add_ocr_decoy_svg(svg_code: str, corner: int = -1) -> str:
    """
    Adds nested circles with second darkest and second brightest colors from the existing SVG,
    positioned in one of the four corners (randomly selected) but positioned to avoid being
    cropped out during image processing.
    
    Parameters:
    -----------
    svg_code : str
        The original SVG string
    
    Returns:
    --------
    str
        Modified SVG with the nested circles added
    """
    import random
    import re
    from colorsys import rgb_to_hls, hls_to_rgb

    x, y, width, height = 0, 0, 512, 512 ### modded: simplified

    # Function to convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Function to convert RGB to hex
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
    
    # Function to calculate color lightness
    def get_lightness(color):
        # Handle different color formats
        if color.startswith('#'):
            rgb = hex_to_rgb(color)
            return rgb_to_hls(*rgb)[1]  # Lightness is the second value in HLS
        elif color.startswith('rgb'):
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if rgb_match:
                r, g, b = map(lambda x: int(x)/255, rgb_match.groups())
                return rgb_to_hls(r, g, b)[1]
        return 0.5  # Default lightness if we can't parse
    
    # Extract all colors from the SVG
    color_matches = re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))"', svg_code)
    
    # Default colors in case we don't find enough
    second_darkest_color = "#333333"  # Default to dark gray
    second_brightest_color = "#CCCCCC"  # Default to light gray
    
    if color_matches:
        # Remove duplicates and get unique colors
        unique_colors = list(set(color_matches))
        
        # Calculate lightness for each unique color
        colors_with_lightness = [(color, get_lightness(color)) for color in unique_colors]
        
        # Sort by lightness (brightness)
        sorted_colors = sorted(colors_with_lightness, key=lambda x: x[1])
        
        # Handle different scenarios based on number of unique colors
        if len(sorted_colors) >= 4:
            # We have at least 4 unique colors - use 2nd darkest and 2nd brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[-2][0]
        elif len(sorted_colors) == 3:
            # We have 3 unique colors - use 2nd darkest and brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[2][0]
        elif len(sorted_colors) == 2:
            # We have only 2 unique colors - use the darkest and brightest
            second_darkest_color = sorted_colors[0][0]
            second_brightest_color = sorted_colors[1][0]
        elif len(sorted_colors) == 1:
            # Only one color - use it for second_darkest and a derived lighter version
            base_color = sorted_colors[0][0]
            base_lightness = sorted_colors[0][1]
            second_darkest_color = base_color
            
            # Create a lighter color variant if the base is dark, or darker if base is light
            if base_lightness < 0.5:
                # Base is dark, create lighter variant
                second_brightest_color = "#CCCCCC"
            else:
                # Base is light, create darker variant
                second_darkest_color = "#333333"
    
    # Ensure the colors are different
    if second_darkest_color == second_brightest_color:
        # If they ended up the same, modify one of them
        if get_lightness(second_darkest_color) < 0.5:
            # It's a dark color, make the bright one lighter
            second_brightest_color = "#CCCCCC"
        else:
            # It's a light color, make the dark one darker
            second_darkest_color = "#333333"
    
    # Base size for the outer circle
    base_outer_radius = width * 0.023
    
    # Randomize size by ±10%
    size_variation = base_outer_radius * 0.1
    outer_radius = 12 # base_outer_radius + random.uniform(-size_variation, size_variation)
    
    # Define radii for inner circles based on outer radius
    middle_radius = outer_radius * 0.80
    inner_radius = middle_radius * 0.65
    
    # Calculate the maximum crop margin based on the image processing (5% of dimensions)
    # Add 20% extra margin for safety
    crop_margin_w = int(width * 0.05 * 1.2)
    crop_margin_h = int(height * 0.05 * 1.2)
    
    # Calculate center point based on the outer radius to ensure the entire circle stays visible
    safe_offset = outer_radius + max(crop_margin_w, crop_margin_h)
    
    # Choose a random corner (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    if corner == -1:
        corner = random.randint(0, 3)
    
    # Position the circle in the chosen corner, accounting for crop margin
    if corner == 0:  # Top-left
        center_x = safe_offset
        center_y = safe_offset
    elif corner == 1:  # Top-right
        center_x = width - safe_offset
        center_y = safe_offset
    elif corner == 2:  # Bottom-left
        center_x = safe_offset
        center_y = height - safe_offset
    else:  # Bottom-right
        center_x = width - safe_offset
        center_y = height - safe_offset
    
    # Add a small random offset (±10% of safe_offset) to make positioning less predictable
    random_offset = safe_offset * 0.1
    center_x += random.uniform(-random_offset, random_offset)
    center_y += random.uniform(-random_offset, random_offset)
    
    # Round to 1 decimal place to keep file size down
    outer_radius = round(outer_radius, 1)
    middle_radius = round(middle_radius, 1)
    inner_radius = round(inner_radius, 1)
    center_x = int(center_x)
    center_y = int(center_y)
    
    # Create the nested circles
    outer_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{outer_radius}" fill="{second_darkest_color}"/>' ### modded: remove space before /
    middle_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{middle_radius}" fill="{second_brightest_color}"/>' ### modded: remove space before /
    inner_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{inner_radius}" fill="{second_darkest_color}"/>' ### modded: remove space before /
    
    # Create a group element that contains all three circles
    group_element = f'{outer_circle}{middle_circle}{inner_circle}' ### modded: remove <g> and </g>
    
    # Insert the group element just before the closing SVG tag
    modified_svg = svg_code.replace("</svg>", f"{group_element}</svg>")
    
    # Calculate and add a comment with the byte size information ### modded: remove comments
#    outer_bytes = len(outer_circle.encode('utf-8'))
#    middle_bytes = len(middle_circle.encode('utf-8'))
#    inner_bytes = len(inner_circle.encode('utf-8'))
#    total_bytes = outer_bytes + middle_bytes + inner_bytes
    
#    corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
#    byte_info = f'<!-- Circle bytes: outer={outer_bytes}, middle={middle_bytes}, ' \
#                f'inner={inner_bytes}, total={total_bytes}, ' \
#                f'colors: dark={second_darkest_color}, light={second_brightest_color}, ' \
#                f'position: {corner_names[corner]} -->'
#    
#    modified_svg = modified_svg.replace("</svg>", f"{byte_info}</svg>")
    
    return modified_svg


#| export
import os
import vtracer
from typing import List
from scour.scour import scourString
from scour.scour import parse_args as parseScourArgs
from kagglehub import notebook_output_download
import kagglehub
import re

import io
import cairosvg
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

def simplify_background(svg_str):
    return svg_str.replace('<path transform="translate(0)" d="m0 0h512v512h-512z" fill="', '<circle r="1e5" fill="')

def simplify_fill_color(svg_content):
    return re.sub(r'fill="#([0-9A-F])[0-9A-F]([0-9A-F])[0-9A-F]([0-9A-F])[0-9A-F]', r'fill="#\1\2\3', svg_content)

def add_sign_for_defeat_ocr(svg_str):
    return svg_str.replace('</svg>', '<path d="M1 9H30M15 9V40" stroke="red" stroke-width="8"/></svg>')

def expand_frame(svg_str):
    return svg_str.replace('<svg viewBox="0 0 512 512"', '<svg viewBox="-10 -10 532 532"')

def expand_black_frame(svg_str):
    return svg_str.replace('</svg>', '<path d="M-5-5H517V517H-5Z" fill="none" stroke="black" stroke-width="10"/></svg>')

def convert_path_to_rect(svg_content):
    pattern = re.compile(
        r'<path\s+transform="translate\((?P<x>[\d.]+),(?P<y>[\d.]+)\)"\s+'
        r'd="m0 0h(?P<width>[\d.]+)v(?P<height>[\d.]+)h-(?P=width)z"\s+'
        r'fill="(?P<fill>#[0-9A-Fa-f]{3,6})"\s*/?>'
    )
    def replacement(m):
        return (
            f'<rect x="{m.group("x")}" y="{m.group("y")}" '
            f'width="{m.group("width")}" height="{m.group("height")}" '
            f'fill="{m.group("fill")}"/>'
        )
    return pattern.sub(replacement, svg_content)


#try:
#    del Model
#    import gc
#    gc.collect()
#except:
#    pass

class Model:
    def __init__(self):
        self.model_dir = notebook_output_download("toshimitsukimura/stable-diffusion-cpp-python-0-2-7-usage-alt-clip")
        svgcp = kagglehub.package_import('abhasm/svg-constraints/versions/1')
        self.svgc = svgcp.SVGConstraints()
        self.scouroptions = parseScourArgs([
            '--enable-viewboxing',
            "--enable-id-stripping",
            "--enable-comment-stripping",
            "--shorten-ids",
            "--indent=none",
            "--no-line-breaks", # add
            "--strip-xml-prolog" # add
            ])
    def predict(self, prompt: str) -> str:
        if not hasattr(self, "stable_diffusion"):
            whl_dir = notebook_output_download("toshimitsukimura/stable-diffusion-cpp-python-0-2-7-cuda-enabled")
            os.system("pip install " + whl_dir + "/stable_diffusion_cpp_python-0.2.7-cp311-cp311-linux_x86_64.whl")


#            whl_dir2 = notebook_output_download("toshimitsukimura/diffvg-cuda-enabled")
#            os.system("pip install " + whl_dir2 + "/diffvg/diffvg-0.0.1-cp311-cp311-linux_x86_64.whl")

            from stable_diffusion_cpp import StableDiffusion
            self.stable_diffusion = StableDiffusion(
            diffusion_model_path = self.model_dir+"/flux1-schnell-Q6_K.gguf",
            clip_l_path = self.model_dir+"/ViT-L-14-BEST-smooth-GmP-TE-only-HF-format.safetensors",
            t5xxl_path = self.model_dir+"/t5xxl_fp16.safetensors",
            vae_path = self.model_dir+"/ae.safetensors",
            vae_decode_only = True,
            keep_clip_on_cpu = True,
            )

#        prompt = "A single square image divided into a 2x2 grid by thin black lines. Each quadrant features a distinct version of " + prompt
#        prompt += ", simple toon vector image, monoline, icon style, limited color palette, best quality, beautiful, masterpiece, aesthetic"
        prompt += ", simple toon vector image, monoline, icon style, limited color palette, aesthetic"

#        prompt = "simple toon vector image of " + prompt
#        prompt += ", monoline, icon style, limited color palette, aesthetic"
        outputs = self.stable_diffusion.txt_to_img(
              prompt = prompt,
              sample_steps = 4,
              cfg_scale = 1.0,
              sample_method = "euler",
              batch_count = 1,
              seed = 372, # tweak seed
        )
        prev2_ssim_score = 0
        prev2_best_svg = None
        for output in outputs:
            prev_ssim_score = 0
            prev_best_svg = "<svg></svg>"
            output_data = np.array(output.convert('RGB'))

            pixels: list[tuple[int, int, int, int]] = list(output.getdata())
            for cp in range(8, 0, -1): # 8..1
                low = 0
                high = 100
                best_svg_opt_str = None
                while low <= high: # binary search
                    mid = (low + high) // 2
#                    output = process(compression_level=mid)

                    svg_str: str = vtracer.convert_pixels_to_svg(pixels, size=(512, 512), path_precision=1, mode="polygon",
                                                                 color_precision=cp, filter_speckle=mid)

#                    print(svg_str)
                    svg_opt_str = scourString(svg_str, self.scouroptions)
                    svg_opt_str = svg_opt_str.replace(' version="1.1"', "")
#                    svg_opt_str = refine_svg(svg_opt_str, output) # diffvg
# TODO: simplify with SVGCompress (Ramer–Douglas–Peucker algorithm), curve fitting w/ fit_to_bezpath_opt

                    svg_opt_str = simplify_background(svg_opt_str)
#                    svg_opt_str = convert_path_to_rect(svg_opt_str) # added after final submission

#                    svg_opt_str = simplify_fill_color(svg_opt_str)
#                    svg_opt_str = add_sign_for_defeat_ocr(svg_opt_str)

#                    svg_opt_str = expand_frame(svg_opt_str)
#                    svg_opt_str = expand_black_frame(svg_opt_str)

                    svg_opt_str = add_ocr_decoy_svg(svg_opt_str, 0)
                    svg_opt_str = add_ocr_decoy_svg(svg_opt_str, 3)
                    svg_opt_str = add_ocr_decoy_svg(svg_opt_str, 2)
#                    svg_opt_str = add_ocr_decoy_svg(svg_opt_str, 1)

                    length = len(svg_opt_str)

                    if length <= 10000:
                        best_svg_opt_str = svg_opt_str
                        high = mid - 1
                    else:
                        low = mid + 1

                if not best_svg_opt_str:
                    continue

                png_data = cairosvg.svg2png(bytestring=best_svg_opt_str)
                img = Image.open(io.BytesIO(png_data)).convert('RGB')
#                display(img)
                ssim_score, _ = ssim(output_data, np.array(img), channel_axis=-1, full=True)
#                print("score: %s"%ssim_score)

                if ssim_score < prev_ssim_score:
#                    print("final: %s"%prev_ssim_score)
                    if prev2_ssim_score < prev_ssim_score:
                        prev2_ssim_score = prev_ssim_score
                        prev2_best_svg = prev_best_svg
                    break
                prev_ssim_score = ssim_score
                prev_best_svg = best_svg_opt_str

        return prev2_best_svg



# Test Code (not included in submission)
import time
from IPython.display import SVG
mod = Model()

l = ["a purple forest at dusk",
"gray wool coat with a faux fur collar",
"a lighthouse overlooking the ocean",
"burgundy corduroy pants with patch pockets and silver buttons",
"orange corduroy overalls",
"a purple silk scarf with tassel trim",
"a green lagoon under a cloudy sky",
"crimson rectangles forming a chaotic grid",
"purple pyramids spiraling around a bronze cone",
"magenta trapezoids layered on a transluscent silver sheet",
"a snowy plain",
"black and white checkered pants",
"a starlit night over snow-covered peaks",
"khaki triangles and azure crescents",
"a maroon dodecahedron interwoven with teal threads",
]

r = []

for i in l:
    start = time.perf_counter()
    
    svg = mod.predict(i)
    end = time.perf_counter()
    print(f"処理時間: {end - start:.4f} 秒")
    print(svg)
    display(SVG(svg))
    r.append([i, svg])

print(r)

