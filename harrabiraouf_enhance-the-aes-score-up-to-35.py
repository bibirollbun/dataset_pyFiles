import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses INFO and WARNING logs


import kagglehub
import matplotlib.pyplot as plt
from PIL import Image
import torch
import xml.etree.ElementTree as ET
import math
import pandas as pd
from IPython.display import display, HTML
import io
import base64

# Import the metric package
metric = kagglehub.package_import('jiazhuang/svg-image-fidelity')

# Access the pre-initialized aesthetic evaluator 
aesthetic_evaluator = metric.aesthetic_evaluator

# Access the pre-initialized svg_to_png function for converting SVG to PNG images
svg_to_png = metric.svg_to_png

def evaluate_aes(svg_string):
    """
    Evaluates the aesthetic score (AES) of an SVG image and displays it.

    Parameters:
    svg_string (str): The SVG content as a string.
    
    Returns:
    float: Aesthetic score (range 0 to 1)
    """
    # Convert SVG to PNG image using the pre-initialized function
    try:
        image = svg_to_png(svg_string)
    except Exception as e:
        raise ValueError(f"Invalid SVG: {e}")

    # Compute the AES score using the pre-initialized aesthetic evaluator
    with torch.no_grad():
        aes_score = aesthetic_evaluator.score(image)

    # Plot the image along with the AES score
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Aesthetic Score: {aes_score:.2f}', fontsize=12)
    plt.show()

    return aes_score

def get_score_and_image(svg_string):
    """
    Returns the aesthetic score and the image for a given SVG string.
    
    Parameters:
    svg_string (str): The SVG content as a string.
    
    Returns:
    tuple: (score, image) where score is the AES score and image is the converted PNG image.
    """
    if not isinstance(svg_string, str) or not svg_string.strip():
        return "Invalid SVG", None
    try:
        image = svg_to_png(svg_string)
        if image is None:
            return "Render Failed", None
        with torch.no_grad():
            aes_score = aesthetic_evaluator.score(image)
        return float(aes_score), image
    except Exception as e:
        return f"Error: {str(e)}", None

def image_to_base64_str(image):
    """
    Converts a PIL Image object to a base64-encoded PNG string.
    
    Parameters:
    image (PIL Image): The image to convert.
    
    Returns:
    tuple: (base64 string, width, height) of the image.
    """
    if image is None:
        return "", 0, 0
    try:
        # Get image dimensions (width and height)
        width, height = image.size

        # Convert the image to base64 format
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return base64_str, width, height
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return "", 0, 0

def resize_svg_code(svg_code: str, new_size: int = 384) -> str:
    """
    Resizes the SVG code to a new size (default 384x384), preserving the layout and aspect ratio.
    
    Parameters:
    svg_code (str): The original SVG code as a string.
    new_size (int): The new size for the SVG (default is 384).
    
    Returns:
    str: The resized SVG code.
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")  # Prevent xmlns from being added
    root = ET.fromstring(svg_code)

    # Update width and height of the SVG to the new size
    root.set("width", str(new_size))
    root.set("height", str(new_size))

    # Add a viewBox if not present
    if "viewBox" not in root.attrib:
        root.set("viewBox", "0 0 96 96")

    return ET.tostring(root, encoding="unicode")

def high_score_svg_resize(
    svg_code: str,
    new_size: int = 384,
    padding_ratio: float = 0.1,
    min_stroke: float = 1.5,
    max_stroke: float = 16,
    preserve_aspect: bool = True
) -> str:
    """
    Enhances the SVG code by adjusting the stroke width, font size, and scaling it to the new size.
    
    Parameters:
    svg_code (str): The original SVG code.
    new_size (int): The target size for the SVG (default 384).
    padding_ratio (float): Padding to apply around the SVG (default 0.1).
    min_stroke (float): Minimum stroke width (default 1.5).
    max_stroke (float): Maximum stroke width (default 16).
    preserve_aspect (bool): Whether to preserve the aspect ratio (default True).
    
    Returns:
    str: The enhanced SVG code.
    """
    root = ET.fromstring(svg_code)
    viewBox = root.get("viewBox")
    if viewBox is None:
        viewBox = "0 0 96 96"
        root.set("viewBox", viewBox)
    vb_x, vb_y, vb_w, vb_h = map(float, viewBox.strip().split())

    scale = (1 - 2 * padding_ratio) * new_size / max(vb_w, vb_h)
    translate_x = (new_size - vb_w * scale) / 2
    translate_y = (new_size - vb_h * scale) / 2

    # Transform the SVG content to adjust scaling and positioning
    g = ET.Element("g")
    transform = f"translate({translate_x:.2f},{translate_y:.2f}) scale({scale:.4f}) translate({-vb_x:.6f},{-vb_y:.6f})"
    g.set("transform", transform)

    # Remove structural elements and append to the main group
    structural_tags = {
        ET.QName("http://www.w3.org/2000/svg", ln)
        for ln in ['defs', 'style', 'title', 'metadata', 'script']
    }
    for child in list(root):
        qname = ET.QName(child.tag)
        if qname in structural_tags:
            continue
        g.append(child)
        root.remove(child)
    root.append(g)

    # Update width, height, and viewBox to the new size
    root.set("width", str(new_size))
    root.set("height", str(new_size))
    root.set("viewBox", f"0 0 {new_size} {new_size}")
    if preserve_aspect:
        root.set("preserveAspectRatio", "xMidYMid meet")

    # Adjust stroke-width and font-size to fit the new scale
    def scale_visuals(el, scale_factor):
        for attr in ("stroke-width", "font-size"):
            if attr in el.attrib:
                try:
                    original = float(el.attrib[attr])
                    effective = original * scale_factor
                    clamped = max(min_stroke, min(max_stroke, effective))
                    new_val = clamped / scale_factor
                    el.attrib[attr] = f"{new_val:.2f}"
                except ValueError:
                    pass
        for child in el:
            scale_visuals(child, scale_factor)

    scale_visuals(g, scale)
    return ET.tostring(root, encoding="unicode")

# Load the dataset
df = pd.read_csv('/kaggle/input/scored-svg-11k/svg_dataset_scored_11k.csv')

# Drop rows with missing 'svg_code' or 'sentence'
df = df.dropna(subset=['svg_code', 'sentence']).reset_index(drop=True)

# Sample 10 rows for demonstration
samples = df.sample(10, random_state=2026)

def format_score(score):
    """
    Formats the AES score for display.
    
    Parameters:
    score (str or float): The AES score or an error message to format.
    
    Returns:
    str: Formatted score as a string (e.g., 'N/A' or '0.75').
    """
    try:
        score_float = float(score)
        if math.isnan(score_float):
            return "N/A"
        return f"{score_float:.2f}"
    except (TypeError, ValueError):
        return str(score)

# Build HTML table to display the samples with AES scores
html_table = """
<table border="1" style="width:100%; text-align:center; font-family:Arial;">
  <tr>
    <th>Sentence</th>
    <th>Original SVG<br>(AES Score)</th>
    <th>Resized SVG<br>(AES Score)</th>
    <th>Enhanced SVG<br>(AES Score)</th>
  </tr>
"""

for idx, row in samples.iterrows():
    sentence = row['sentence']
    svg_code = row['svg_code']

    # Process Original SVG and compute AES score
    original_score, original_img = get_score_and_image(svg_code)
    original_b64, _, _ = image_to_base64_str(original_img)  # Original size (96x96)

    # Process Resized SVG and compute AES score
    resized_svg = resize_svg_code(svg_code)
    resized_score, resized_img = get_score_and_image(resized_svg)
    resized_b64, _, _ = image_to_base64_str(resized_img)  # Resized size (384x384)

    # Process Enhanced SVG and compute AES score
    enhanced_svg = high_score_svg_resize(svg_code)
    enhanced_score, enhanced_img = get_score_and_image(enhanced_svg)
    enhanced_b64, _, _ = image_to_base64_str(enhanced_img)  # Enhanced size (384x384)

    # Add each row to the HTML table
    html_table += f"""
    <tr>
        <td style="vertical-align: top; width: 25%;">{sentence}</td>
        <td><img src="data:image/png;base64,{original_b64 or ''}" width="96" height="96"><br><b>{format_score(original_score)}</b></td>
        <td><img src="data:image/png;base64,{resized_b64 or ''}" width="384" height="384"><br><b>{format_score(resized_score)}</b></td>
        <td><img src="data:image/png;base64,{enhanced_b64 or ''}" width="384" height="384"><br><b>{format_score(enhanced_score)}</b></td>
    </tr>
    """

html_table += "</table>"

# Display the HTML table
display(HTML(html_table))


# Function to process a chunk of the dataset
def process_chunk(chunk):
    new_data = []
    for idx, row in chunk.iterrows():
        sentence = row['sentence']
        svg_code = row['svg_code']
        original_score = row['best_image_score']
        
        # Skip if svg_code is not a string
        if not isinstance(svg_code, str) or not svg_code.strip():
            print(f"Skipping row {idx}: Invalid SVG code")
            continue
            
        # Process Original
        original_score_aes, original_img = get_score_and_image(svg_code)

        # Process Resized
        resized_svg = resize_svg_code(svg_code)
        resized_score_aes, resized_img = get_score_and_image(resized_svg)

        # Process Enhanced
        enhanced_svg = high_score_svg_resize(svg_code)
        enhanced_score_aes, enhanced_img = get_score_and_image(enhanced_svg)

        # Choose the best scoring SVG
        best_score = max(resized_score_aes, enhanced_score_aes)
        if best_score == resized_score_aes:
            best_svg = resized_svg
        else:
            best_svg = enhanced_svg

        # Store the new data row (original columns + best score + best svg)
        new_data.append({
            'sentence': sentence,
            'svg_code': svg_code,
            'score': original_score,  # Retain the original score
            'original_score': original_score_aes,
            'resized_score': resized_score_aes,
            'enhanced_score': enhanced_score_aes,
            'aes_best_score': best_score,
            'svg_enhanced': best_svg  # Store the best SVG based on score
        })
    
    # Convert the list of new rows into a DataFrame and return it
    return pd.DataFrame(new_data)


# Please uncomment this part to process and save the new enhanced dataset
# I comment it to save some gpu hours :)

"""
# Initialize an empty list to collect processed chunks
processed_chunks = []

# Read the dataset in chunks and process each chunk
chunk_size = 1000  # Adjust the chunk size based on your system's memory capacity
csv_path = '/kaggle/input/scored-svg-11k/svg_dataset_scored_11k.csv'

for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    processed_chunk = process_chunk(chunk)
    processed_chunks.append(processed_chunk)

# Concatenate all processed chunks into a single DataFrame
final_df = pd.concat(processed_chunks, ignore_index=True)

# Save the final DataFrame to a CSV file
final_df.to_csv('/kaggle/working/new_svg_dataset_with_aes_and_score.csv', index=False)

# Display the first few rows of the final dataset
final_df.head()

"""

