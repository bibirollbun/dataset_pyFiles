!pip install svgwrite cairosvg


import svgwrite
from cairosvg import svg2png
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def generate_svg_overlap(y_overlap=1.0, width=100, height=200):
    """
    Generates an SVG with two vertically stacked rectangles that slightly overlap.
    y_overlap: amount of vertical overlap in pixels (can be fractional).
    """
    dwg = svgwrite.Drawing(size=(width, height))
    half_h = height / 2

    # Top rectangle (green), extending below the midpoint by y_overlap
    dwg.add(dwg.rect(insert=(0, 0),
                     size=(width, half_h + y_overlap),
                     fill='#00FF00'))

    # Bottom rectangle (magenta), starting above the midpoint by y_overlap
    dwg.add(dwg.rect(insert=(0, half_h - y_overlap),
                     size=(width, half_h + y_overlap),
                     fill='#FF00FF'))

    return dwg.tostring()

def render_png_from_svg(svg_code):
    """
    Renders an SVG string to a PNG image using CairoSVG.
    Output size is fixed at 100x200 pixels.
    """
    width, height = 100, 200
    png_bytes = svg2png(bytestring=svg_code, output_width=width, output_height=height)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return image


overlap = 1.2
svg_code = generate_svg_overlap(y_overlap=overlap)
full_image = render_png_from_svg(svg_code)
full_image


def horizontal_boundary_crop(image, crop_height=10):
    """
    Crops a horizontal strip centered around the vertical midpoint of the image.
    """
    w, h = image.size
    cy = h // 2
    return image.crop((0, cy - crop_height // 2, w, cy + crop_height // 2))

def vertical_pixel_repeat(image: Image.Image, zoom_factor: int) -> Image.Image:
    """
    Stretches an image vertically by repeating each horizontal row zoom_factor times.
    This replicates the pixel values without any interpolation, preserving exact color values.
    """
    arr = np.array(image)
    stretched = np.repeat(arr, zoom_factor, axis=0)  # repeat rows (height-wise)
    return Image.fromarray(stretched)


def show_combined_visualization(y_overlaps, crop_height=6, zoom_factor=4):
    """
    Displays the full image and zoomed-in boundary regions using vertical pixel replication.
    A single title is used for the zoomed section; each zoom view is labeled with the overlap value only.
    """
    num_zoom = len(y_overlaps)
    fig, axes = plt.subplots(1, num_zoom + 1, figsize=(2 + num_zoom * 2, 6))

    # Render full image using the first overlap value
    overlap = y_overlaps[0]
    svg_code = generate_svg_overlap(y_overlap=overlap)
    full_image = render_png_from_svg(svg_code)

    # Left: Full image
    axes[0].imshow(np.array(full_image))
    axes[0].set_title('Full Image', fontsize=12)
    axes[0].axis('off')

    # Right: Zoomed-in views
    for ax, overlap in zip(axes[1:], y_overlaps):
        svg_code = generate_svg_overlap(y_overlap=overlap)
        image = render_png_from_svg(svg_code)
        strip = horizontal_boundary_crop(image, crop_height=crop_height)
        zoomed_strip = vertical_pixel_repeat(strip, zoom_factor=zoom_factor)

        ax.imshow(np.array(zoomed_strip))
        ax.set_title(f'±{overlap}px', fontsize=11)
        ax.axis('off')

    # Overall title for zoomed region
    fig.suptitle("Zoomed Boundary Comparison", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('result.png', dpi=300)
    plt.show()


y_overlaps = [0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0]
crop_height = 6
zoom_factor = 20

show_combined_visualization(y_overlaps=y_overlaps, 
                            crop_height=crop_height, 
                            zoom_factor=zoom_factor)

