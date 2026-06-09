# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%writefile model.py
import os
import numpy as np
import random
from tqdm.auto import tqdm

class EnhancedSVGGenerator:
    """
    Enhanced SVG Generator for Kaggle competition.
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.colors = {
            "blue": ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
            "green": ["#14532d", "#166534", "#22c55e", "#4ade80", "#86efac", "#bbf7d0"],
            "red": ["#7f1d1d", "#b91c1c", "#ef4444", "#f87171", "#fca5a5", "#fee2e2"],
            "yellow": ["#713f12", "#a16207", "#eab308", "#facc15", "#fde047", "#fef9c3"],
            "purple": ["#4c1d95", "#6d28d9", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"],
            "gray": ["#1f2937", "#374151", "#4b5563", "#6b7280", "#9ca3af", "#d1d5db"],
            "orange": ["#7c2d12", "#c2410c", "#f97316", "#fb923c", "#fdba74", "#fed7aa"],
            "teal": ["#134e4a", "#0f766e", "#14b8a6", "#2dd4bf", "#5eead4", "#99f6e4"],
            "pink": ["#831843", "#be185d", "#ec4899", "#f472b6", "#f9a8d4", "#fbcfe8"],
            "sky": ["#0c4a6e", "#0369a1", "#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd"],
            "sunset": ["#7c2d12", "#9a3412", "#c2410c", "#ea580c", "#f97316", "#fb923c", "#fdba74"],
            "forest": ["#064e3b", "#065f46", "#047857", "#059669", "#10b981", "#34d399", "#6ee7b7"]
        }
        self.shapes = ["rect", "circle", "ellipse", "polygon", "path", "line"]
        
    def generate_random_shape(self, canvas_width=400, canvas_height=400, palette=None):
        if palette is None:
            color_palette = random.choice(list(self.colors.values()))
        else:
            color_palette = palette
            
        shape_type = random.choice(self.shapes)
        fill_color = random.choice(color_palette)
        stroke_color = random.choice(color_palette)
        stroke_width = random.randint(0, 5)
        opacity = round(random.uniform(0.3, 1.0), 2)
        
        if shape_type == "rect":
            x = random.randint(0, int(canvas_width * 0.8))
            y = random.randint(0, int(canvas_height * 0.8))
            width = random.randint(20, int(canvas_width * 0.5))
            height = random.randint(20, int(canvas_height * 0.5))
            rx = random.randint(0, 20) if random.random() > 0.5 else 0
            
            return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" opacity="{opacity}" />'
                   
        elif shape_type == "circle":
            cx = random.randint(50, canvas_width - 50)
            cy = random.randint(50, canvas_height - 50)
            r = random.randint(10, 100)
            
            return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" opacity="{opacity}" />'
                   
        elif shape_type == "ellipse":
            cx = random.randint(50, canvas_width - 50)
            cy = random.randint(50, canvas_height - 50)
            rx = random.randint(20, 150)
            ry = random.randint(20, 150)
            
            return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" opacity="{opacity}" />'
                   
        elif shape_type == "polygon":
            points = []
            num_points = random.randint(3, 8)
            for _ in range(num_points):
                x = random.randint(50, canvas_width - 50)
                y = random.randint(50, canvas_height - 50)
                points.append(f"{x},{y}")
                
            return f'<polygon points="{" ".join(points)}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" opacity="{opacity}" />'
                   
        elif shape_type == "path":
            path_type = random.choice(["zigzag", "curve"])
            start_x = random.randint(50, canvas_width - 200)
            start_y = random.randint(50, canvas_height - 200)
            
            if path_type == "zigzag":
                segments = random.randint(3, 8)
                path_data = f"M {start_x} {start_y} "
                
                for i in range(segments):
                    next_x = start_x + (i + 1) * random.randint(20, 60)
                    next_y = start_y + ((-1) ** i) * random.randint(20, 60)
                    path_data += f"L {next_x} {next_y} "
            else:  # curve
                path_data = f"M {start_x} {start_y} "
                path_data += f"Q {start_x + random.randint(50, 150)} {start_y - random.randint(50, 150)}, "
                path_data += f"{start_x + random.randint(100, 200)} {start_y} "
                
            return f'<path d="{path_data}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" opacity="{opacity}" />'
                   
        elif shape_type == "line":
            x1 = random.randint(0, canvas_width)
            y1 = random.randint(0, canvas_height)
            x2 = random.randint(0, canvas_width)
            y2 = random.randint(0, canvas_height)
            
            return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke_color}" stroke-width="{stroke_width + 2}" opacity="{opacity}" />'
        
        return ""

    def create_gradient(self, id, type="linear", angle=None, colors=None):
        if angle is None:
            angle = random.randint(0, 360)
        if colors is None:
            color_family = random.choice(list(self.colors.values()))
            colors = random.sample(color_family, min(3, len(color_family)))
        
        if type == "linear":
            x1, y1, x2, y2 = 0, 0, 0, 0
            
            if angle == 0 or angle == 180:  # horizontal
                x1, y1, x2, y2 = 0, 0, 100, 0
            elif angle == 90 or angle == 270:  # vertical
                x1, y1, x2, y2 = 0, 0, 0, 100
            elif angle == 45 or angle == 225:  # diagonal
                x1, y1, x2, y2 = 0, 0, 100, 100
            elif angle == 135 or angle == 315:  # diagonal
                x1, y1, x2, y2 = 100, 0, 0, 100
            else:
                # Convert angle to radians
                rad = np.radians(angle)
                # Calculate end point
                x2 = 100 * np.cos(rad)
                y2 = 100 * np.sin(rad)
            
            gradient = f'<linearGradient id="{id}" x1="{x1}%" y1="{y1}%" x2="{x2}%" y2="{y2}%">'
            
            num_colors = len(colors)
            for i, color in enumerate(colors):
                offset = i * (100 / (num_colors - 1)) if num_colors > 1 else 50
                gradient += f'<stop offset="{offset}%" stop-color="{color}" />'
                
            gradient += '</linearGradient>'
            return gradient
            
        elif type == "radial":
            gradient = f'<radialGradient id="{id}" cx="50%" cy="50%" r="50%" fx="{random.randint(30, 70)}%" fy="{random.randint(30, 70)}%">'
            
            num_colors = len(colors)
            for i, color in enumerate(colors):
                offset = i * (100 / (num_colors - 1)) if num_colors > 1 else 50
                gradient += f'<stop offset="{offset}%" stop-color="{color}" />'
                
            gradient += '</radialGradient>'
            return gradient
        
        return ""
    
    def generate_better_landscape(self, width=400, height=400):
        """Generate a more aesthetically pleasing landscape scene"""
        defs = []
        
        # Create SVG header
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        
        # Add sky gradient background
        sky_gradient_id = "sky-gradient"
        # Choose between day and night sky
        is_day = random.random() > 0.3
        
        if is_day:
            # Day sky (blue gradient)
            sky_colors = self.colors["sky"]
            # Sort from dark to light (top to bottom)
            sky_colors.sort(key=lambda x: int(x[1:], 16))
            sky_gradient = self.create_gradient(sky_gradient_id, "linear", 90, sky_colors)
        else:
            # Night sky (dark blue to black)
            night_colors = ["#000000", "#111827", "#1e3a8a", "#1e40af"]
            sky_gradient = self.create_gradient(sky_gradient_id, "linear", 90, night_colors)
        
        defs.append(sky_gradient)
        svg += f'<rect width="{width}" height="{height}" fill="url(#{sky_gradient_id})" />'
        
        # Add sun or moon
        celestial_body_x = random.randint(width // 4, width - (width // 4))
        celestial_body_y = random.randint(height // 6, height // 3)
        celestial_body_r = random.randint(30, 50)
        
        if is_day:
            # Add sun with gradient
            sun_gradient_id = "sun-gradient"
            sun_colors = [self.colors["yellow"][5], self.colors["yellow"][3], self.colors["orange"][2]]
            sun_gradient = self.create_gradient(sun_gradient_id, "radial", None, sun_colors)
            defs.append(sun_gradient)
            
            svg += f'<circle cx="{celestial_body_x}" cy="{celestial_body_y}" r="{celestial_body_r}" fill="url(#{sun_gradient_id})" />'
            
            # Add sun rays
            ray_count = random.randint(8, 16)
            ray_length = random.randint(celestial_body_r, celestial_body_r * 2)
            for i in range(ray_count):
                angle = 2 * np.pi * i / ray_count
                x1 = celestial_body_x + celestial_body_r * np.cos(angle)
                y1 = celestial_body_y + celestial_body_r * np.sin(angle)
                x2 = celestial_body_x + (celestial_body_r + ray_length) * np.cos(angle)
                y2 = celestial_body_y + (celestial_body_r + ray_length) * np.sin(angle)
                svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.colors["yellow"][3]}" stroke-width="2" opacity="0.7" />'
            
            # Add some clouds
            cloud_count = random.randint(2, 5)
            for _ in range(cloud_count):
                cloud_x = random.randint(0, width)
                cloud_y = random.randint(height // 10, height // 2)
                self.add_cloud(svg, cloud_x, cloud_y)
        else:
            # Add moon with gradient
            moon_gradient_id = "moon-gradient"
            moon_colors = ["#f9fafb", "#f3f4f6", "#e5e7eb", "#d1d5db"]
            moon_gradient = self.create_gradient(moon_gradient_id, "radial", None, moon_colors)
            defs.append(moon_gradient)
            
            svg += f'<circle cx="{celestial_body_x}" cy="{celestial_body_y}" r="{celestial_body_r}" fill="url(#{moon_gradient_id})" />'
            
            # Add stars
            star_count = random.randint(50, 100)
            for _ in range(star_count):
                star_x = random.randint(5, width - 5)
                star_y = random.randint(5, height // 2)
                star_r = random.randint(1, 3) / 2
                star_opacity = random.uniform(0.5, 1.0)
                svg += f'<circle cx="{star_x}" cy="{star_y}" r="{star_r}" fill="white" opacity="{star_opacity}" />'
        
        # Draw mountain range in the background
        mountain_colors = self.colors["gray"] if not is_day else self.colors["forest"]
        mountain_count = random.randint(4, 8)
        
        # Sort mountains by distance (furthest first)
        mountains = []
        for i in range(mountain_count):
            # Base parameters for this mountain
            height_factor = random.uniform(0.3, 0.5)
            base_y = height * 0.65
            peak_height = height * height_factor
            
            # Create a mountain with random parameters
            mountains.append({
                "base_y": base_y,
                "peak_height": peak_height,
                "color_idx": i % len(mountain_colors),
                "opacity": 0.7 + (i / mountain_count) * 0.3  # Further mountains are more transparent
            })
        
        # Draw mountains from back to front
        for i, mountain in enumerate(mountains):
            base_y = mountain["base_y"]
            peak_height = mountain["peak_height"]
            color = mountain_colors[mountain["color_idx"]]
            opacity = mountain["opacity"]
            
            # Calculate mountain position and size
            peak_x = width * (i + 0.5) / mountain_count
            left_x = max(0, peak_x - random.randint(width // 8, width // 4))
            right_x = min(width, peak_x + random.randint(width // 8, width // 4))
            peak_y = base_y - peak_height
            
            # Draw the mountain
            svg += f'<polygon points="{left_x},{base_y} {peak_x},{peak_y} {right_x},{base_y}" fill="{color}" opacity="{opacity}" />'
            
            # Add snow cap to taller mountains
            if peak_height > height * 0.35:
                snow_height = peak_height * random.uniform(0.1, 0.2)
                snow_left_x = peak_x - (peak_x - left_x) * random.uniform(0.2, 0.4)
                snow_right_x = peak_x + (right_x - peak_x) * random.uniform(0.2, 0.4)
                snow_left_y = peak_y + snow_height
                snow_right_y = peak_y + snow_height
                
                svg += f'<polygon points="{snow_left_x},{snow_left_y} {peak_x},{peak_y} {snow_right_x},{snow_right_y}" fill="white" opacity="0.9" />'
        
        # Add foreground ground
        ground_y = height * 0.65
        ground_height = height - ground_y
        
        # Create a gradient for the ground
        ground_gradient_id = "ground-gradient"
        if is_day:
            ground_colors = [self.colors["green"][2], self.colors["green"][3], self.colors["green"][4]]
        else:
            ground_colors = [self.colors["green"][0], self.colors["green"][1], self.colors["green"][2]]
        
        ground_gradient = self.create_gradient(ground_gradient_id, "linear", 90, ground_colors)
        defs.append(ground_gradient)
        
        svg += f'<rect x="0" y="{ground_y}" width="{width}" height="{ground_height}" fill="url(#{ground_gradient_id})" />'
        
        # Add some trees or elements in the foreground
        tree_count = random.randint(5, 10)
        for _ in range(tree_count):
            tree_x = random.randint(0, width)
            tree_base_y = random.uniform(ground_y, ground_y + ground_height * 0.2)
            tree_height = random.uniform(height * 0.1, height * 0.2)
            self.add_tree(svg, tree_x, tree_base_y, tree_height, is_day)
        
        # Add defs section
        if defs:
            svg += '<defs>'
            for def_item in defs:
                svg += f'{def_item}'
            svg += '</defs>'
        
        # Close SVG
        svg += '</svg>'
        
        return svg
    
    def add_cloud(self, svg, x, y):
        """Add a stylized cloud to the SVG"""
        cloud_color = "#ffffff"
        cloud_opacity = random.uniform(0.7, 0.9)
        
        # Create a group of overlapping circles to form a cloud
        circles = []
        circle_count = random.randint(3, 6)
        base_radius = random.randint(15, 25)
        
        for i in range(circle_count):
            cx = x + random.randint(-base_radius, base_radius)
            cy = y + random.randint(-base_radius // 2, base_radius // 2)
            r = random.randint(base_radius // 2, base_radius)
            circles.append((cx, cy, r))
        
        # Add circles to SVG
        for cx, cy, r in circles:
            svg += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{cloud_color}" opacity="{cloud_opacity}" />'
    
    def add_tree(self, svg, x, base_y, height, is_day=True):
        """Add a stylized tree to the SVG"""
        # Tree trunk
        trunk_width = height / 8
        trunk_height = height * 0.4
        trunk_color = "#7c2d12" if is_day else "#3f1f0e"
        
        svg += f'<rect x="{x - trunk_width/2}" y="{base_y - trunk_height}" width="{trunk_width}" height="{trunk_height}" fill="{trunk_color}" />'
        
        # Tree foliage
        if is_day:
            foliage_colors = [self.colors["green"][2], self.colors["green"][3], self.colors["green"][4]]
        else:
            foliage_colors = [self.colors["green"][0], self.colors["green"][1]]
        
        # Add multiple layers of foliage
        foliage_layers = random.randint(2, 4)
        max_radius = height * 0.6
        
        for i in range(foliage_layers):
            layer_y = base_y - trunk_height - (i * max_radius / foliage_layers / 2)
            layer_radius = max_radius * (foliage_layers - i) / foliage_layers
            layer_color = random.choice(foliage_colors)
            
            svg += f'<circle cx="{x}" cy="{layer_y}" r="{layer_radius}" fill="{layer_color}" />'
    
    def generate_scene(self, num_elements=None, scene_type=None, width=400, height=400):
        """Generate a complete SVG scene with multiple elements."""
        if scene_type == "landscape":
            return self.generate_better_landscape(width, height)
            
        if num_elements is None:
            num_elements = random.randint(3, 15)
            
        if scene_type is None:
            scene_type = random.choice(["abstract", "landscape", "geometry", "pattern"])
        
        defs = []
        elements = []
        
        # Create SVG header
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        
        # Choose a cohesive color palette based on scene type
        if scene_type == "abstract":
            palette_name = random.choice(list(self.colors.keys()))
            palette = self.colors[palette_name]
        elif scene_type == "geometry":
            palette_name = random.choice(["blue", "purple", "teal", "gray"])
            palette = self.colors[palette_name]
        else:
            palette = random.choice(list(self.colors.values()))
            
        # Add background
        bg_type = random.choice(["solid", "gradient"])
        if bg_type == "solid":
            bg_color = random.choice(palette)
            svg += f'<rect width="{width}" height="{height}" fill="{bg_color}" />'
        else:
            gradient_id = "bg-gradient"
            gradient_type = random.choice(["linear", "radial"])
            colors = random.sample(palette, min(3, len(palette)))
            gradient_def = self.create_gradient(gradient_id, gradient_type, None, colors)
            defs.append(gradient_def)
            svg += f'<rect width="{width}" height="{height}" fill="url(#{gradient_id})" />'
            
        # Process scene types
        if scene_type == "geometry":
            # Add geometric patterns
            pattern_count = random.randint(3, 8)
            for i in range(pattern_count):
                pattern_type = random.choice(["grid", "circles", "concentric"])
                
                if pattern_type == "grid":
                    grid_size = random.randint(20, 60)
                    grid_color = random.choice(palette)
                    grid_opacity = round(random.uniform(0.2, 0.6), 2)
                    
                    # Create a more interesting grid
                    for x in range(0, width, grid_size):
                        svg += f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{grid_color}" stroke-width="1" opacity="{grid_opacity}" />'
                    
                    for y in range(0, height, grid_size):
                        svg += f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{grid_color}" stroke-width="1" opacity="{grid_opacity}" />'
                
                elif pattern_type == "circles":
                    # Create a field of circles
                    circle_count = random.randint(10, 30)
                    for _ in range(circle_count):
                        cx = random.randint(0, width)
                        cy = random.randint(0, height)
                        r = random.randint(5, 30)
                        fill_color = random.choice(palette)
                        opacity = round(random.uniform(0.3, 0.8), 2)
                        
                        svg += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill_color}" opacity="{opacity}" />'
                
                elif pattern_type == "concentric":
                    # Create concentric shapes
                    center_x = random.randint(width // 3, width * 2 // 3)
                    center_y = random.randint(height // 3, height * 2 // 3)
                    shape_count = random.randint(5, 12)
                    max_size = min(width, height) * 0.8
                    
                    shape_type = random.choice(["circle", "square"])
                    
                    for i in range(shape_count):
                        size = max_size * (shape_count - i) / shape_count
                        fill_color = random.choice(palette)
                        opacity = 0.3 + (i / shape_count) * 0.7
                        
                        if shape_type == "circle":
                            svg += f'<circle cx="{center_x}" cy="{center_y}" r="{size/2}" fill="none" stroke="{fill_color}" stroke-width="2" opacity="{opacity}" />'
                        else:
                            svg += f'<rect x="{center_x - size/2}" y="{center_y - size/2}" width="{size}" height="{size}" fill="none" stroke="{fill_color}" stroke-width="2" opacity="{opacity}" />'
        
        # Add random elements
        for _ in range(num_elements):
            element = self.generate_random_shape(width, height, palette)
            elements.append(element)
        
        # Add defs section if needed
        if defs:
            svg += '<defs>'
            for def_item in defs:
                svg += f'{def_item}'
            svg += '</defs>'
        
        # Add elements
        for element in elements:
            svg += f'{element}'
        
        # Close SVG
        svg += '</svg>'
        
        # Ensure size constraints (under 10,000 bytes)
        if len(svg.encode('utf-8')) > 9800:  # Buffer for safety
            # Generate simpler scene if too large
            return self.generate_scene(num_elements=min(5, num_elements if num_elements else 5), 
                                     scene_type=scene_type)
        
        return svg

class Model:
    """
    Kaggle SVG Generator Model
    """
    
    def __init__(self):
        self.generator = EnhancedSVGGenerator(seed=42)
    
    def predict(self, prompts):
        """
        Generate SVGs based on prompts.
        
        Args:
            prompts (list): List of text prompts
            
        Returns:
            list: List of SVG strings
        """
        results = []
        
        for prompt in tqdm(prompts, desc="Generating SVGs"):
            # Default scene configuration
            scene_type = "abstract"
            num_elements = 10
            
            # Parse prompt if available
            if isinstance(prompt, str) and prompt:
                prompt_lower = prompt.lower()
                
                # Extract scene type
                if any(word in prompt_lower for word in ["landscape", "nature", "outdoor", "mountain", "sky", "sun", "moon", "tree"]):
                    scene_type = "landscape"
                elif any(word in prompt_lower for word in ["geometric", "geometry", "pattern", "shape", "circle", "square", "grid"]):
                    scene_type = "geometry"
                
                # Extract complexity hint
                if any(word in prompt_lower for word in ["simple", "minimal", "clean"]):
                    num_elements = random.randint(3, 7)
                elif any(word in prompt_lower for word in ["complex", "detailed", "elaborate", "rich"]):
                    num_elements = random.randint(12, 20)
            
            # Generate SVG
            svg = self.generator.generate_scene(num_elements=num_elements, scene_type=scene_type)
            results.append(svg)
        
        return results


%%writefile package.yaml
name: kaggle-svg-generator
version: 1.0.0
description: SVG generator for Kaggle competition


from IPython.display import HTML, display
from model import Model

# Create a model instance
model = Model()

# Helper function to display SVG using HTML
def display_svg(svg_content, prompt):
    html = f"""
    <div style="border: 1px solid #ccc; padding: 10px; margin: 15px 0; background: #f9f9f9; border-radius: 8px;">
      <h3 style="margin-top: 0; color: #333;">{prompt}</h3>
      <div style="background: white; border: 1px solid #eee; border-radius: 4px; padding: 10px;">
        {svg_content}
      </div>
    </div>
    """
    display(HTML(html))

# Your test descriptions
test_descriptions = [
    "A mountain landscape with a lake and sunset",
    "An abstract pattern with colorful circles",
    "A simple house with a red roof",
    "A blue dress with floral pattern",
    "A yellow flower with green stem"
]

# Generate and display SVGs for each description
print("Generating SVGs for test descriptions:")
print("-" * 50)

for i, prompt in enumerate(test_descriptions):
    print(f"Processing: {prompt}")
    
    # Generate SVG
    svg_result = model.predict([prompt])[0]
    
    # Display the SVG
    display_svg(svg_result, prompt)
    
    # Save the SVG to a file
    filename = f"test_image_{i+1}.svg"
    with open(filename, 'w') as f:
        f.write(svg_result)
    
    print(f"Saved as: {filename}")
    print("-" * 50)

print("All SVGs have been generated and saved to files!")


import os
import random
import pandas as pd
from tqdm import tqdm
import numpy as np

# For Kaggle competition structure
try:
    from kaggle_evaluation import svg_gateway
except ImportError:
    print("Warning: kaggle_evaluation not found. Competition evaluation will not work.")

class SVGGenerator:
    def __init__(self):
        """
        Initialize the SVG generator with color palettes and settings.
        """
        print("SVG Generator initialized for Kaggle competition.")
        # Color palettes for different themes
        self.nature_colors = ["#228B22", "#8B4513", "#87CEEB", "#4682B4", "#F5F5F5", "#FFD700"]
        self.abstract_colors = ["#FF5733", "#33FFF5", "#F533FF", "#3380FF", "#FF3380", "#80FF33"]
        self.pastel_colors = ["#FFB6C1", "#FFD700", "#ADFF2F", "#87CEEB", "#DDA0DD", "#FFFACD"]
    
    def generate_svg(self, description):
        """
        Generate an SVG based on a text description.
        
        Args:
            description (str): Text description of what to draw
            
        Returns:
            str: SVG code as a string
        """
        description = description.lower()
        
        # Check for different categories and generate appropriate SVG
        if any(word in description for word in ["landscape", "mountain", "lake", "nature", "sunset", "forest", "beach"]):
            return self._create_landscape_svg(description)
        
        elif any(word in description for word in ["abstract", "pattern", "geometric", "shape", "design", "art"]):
            return self._create_abstract_svg(description)
        
        elif any(word in description for word in ["flower", "tree", "house", "building", "car", "animal"]):
            return self._create_object_svg(description)
            
        elif any(word in description for word in ["fashion", "dress", "clothing", "outfit", "shirt", "skirt"]):
            return self._create_fashion_svg(description)
        
        # Default geometric pattern
        else:
            return self._create_default_svg(description)
    
    def _create_landscape_svg(self, description):
        """Create a landscape SVG based on description keywords"""
        # Determine colors based on description
        sky_color = "#87CEEB"  # Default sky blue
        ground_color = "#8B4513"  # Default brown
        
        if "sunset" in description or "sunrise" in description:
            sky_color = "#FF7F50"  # Coral for sunset
        elif "night" in description:
            sky_color = "#191970"  # Midnight blue
            
        if "grass" in description or "green" in description:
            ground_color = "#228B22"  # Forest green
        elif "snow" in description or "winter" in description:
            ground_color = "#F5F5F5"  # White smoke
        elif "beach" in description or "sand" in description:
            ground_color = "#F4A460"  # Sandy brown
            
        # Create the SVG
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <rect width="400" height="250" fill="{sky_color}"/>
            <rect y="250" width="400" height="150" fill="{ground_color}"/>'''
            
        # Add mountains if mentioned
        if "mountain" in description:
            svg += '''
            <polygon points="50,250 150,100 250,250" fill="#808080"/>
            <polygon points="200,250 300,150 400,250" fill="#696969"/>'''
            
        # Add sun/moon if mentioned
        if "sun" in description or "sunset" in description:
            svg += '''
            <circle cx="320" cy="80" r="40" fill="#FFD700"/>'''
        elif "moon" in description or "night" in description:
            svg += '''
            <circle cx="320" cy="80" r="30" fill="#F0F0F0"/>'''
            
        # Add water features
        if "lake" in description:
            svg += '''
            <ellipse cx="200" cy="300" rx="150" ry="30" fill="#4682B4"/>'''
        elif "river" in description:
            svg += '''
            <path d="M0,300 C100,280 200,320 400,290" stroke="#4682B4" stroke-width="30" fill="none"/>'''
        elif "ocean" in description or "sea" in description:
            svg += '''
            <rect y="250" width="400" height="150" fill="#1E90FF"/>
            <path d="M0,260 C50,250 100,270 150,260 C200,250 250,270 300,260 C350,250 400,270 450,260" fill="none" stroke="#FFFFFF" stroke-width="5"/>'''
            
        # Add trees if mentioned
        if "tree" in description or "forest" in description:
            svg += '''
            <rect x="120" y="230" width="10" height="30" fill="#8B4513"/>
            <polygon points="110,230 140,230 125,180" fill="#228B22"/>
            <rect x="320" y="220" width="15" height="40" fill="#8B4513"/>
            <polygon points="300,220 355,220 327.5,150" fill="#228B22"/>'''
            
        # Close the SVG
        svg += '\n</svg>'
        return svg
    
    def _create_abstract_svg(self, description):
        """Create an abstract pattern SVG based on description keywords"""
        # Set a seed based on the description for reproducibility
        seed = sum(ord(char) for char in description)
        random.seed(seed)
        np.random.seed(seed)
        
        # Choose colors based on description mood
        colors = self.abstract_colors
        bg_color = "#f0f0f0"  # Default light gray
        
        if "colorful" in description or "vibrant" in description:
            colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33F5", "#F5FF33"]
        elif "dark" in description or "bold" in description:
            colors = ["#8B0000", "#006400", "#00008B", "#8B008B", "#000000"]
            bg_color = "#2F2F2F"
        elif "pastel" in description or "soft" in description:
            colors = self.pastel_colors
            
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <rect width="400" height="400" fill="{bg_color}"/>'''
            
        # Add shapes based on description
        if "circle" in description:
            for i in range(5):
                cx = random.randint(50, 350)
                cy = random.randint(50, 350)
                r = random.randint(20, 80)
                color = random.choice(colors)
                opacity = round(random.uniform(0.5, 1.0), 1)
                svg += f'''
                <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="{opacity}"/>'''
                
        elif "triangle" in description:
            for i in range(5):
                x1 = random.randint(50, 350)
                y1 = random.randint(50, 150)
                x2 = x1 - random.randint(30, 100)
                y2 = y1 + random.randint(100, 200)
                x3 = x1 + random.randint(30, 100)
                y3 = y2
                color = random.choice(colors)
                opacity = round(random.uniform(0.5, 1.0), 1)
                svg += f'''
                <polygon points="{x1},{y1} {x2},{y2} {x3},{y3}" fill="{color}" opacity="{opacity}"/>'''
                
        elif "square" in description or "rectangle" in description:
            for i in range(5):
                x = random.randint(50, 300)
                y = random.randint(50, 300)
                size = random.randint(50, 150)
                color = random.choice(colors)
                opacity = round(random.uniform(0.5, 1.0), 1)
                svg += f'''
                <rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{color}" opacity="{opacity}"/>'''
                
        # Default - mix of shapes
        else:
            # Circles
            for i in range(3):
                cx = random.randint(50, 350)
                cy = random.randint(50, 350)
                r = random.randint(30, 80)
                color = random.choice(colors)
                svg += f'''
                <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.7"/>'''
                
            # Rectangles
            for i in range(2):
                x = random.randint(50, 250)
                y = random.randint(50, 250)
                w = random.randint(50, 150)
                h = random.randint(50, 150)
                color = random.choice(colors)
                svg += f'''
                <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" opacity="0.5"/>'''
                
            # Triangles
            for i in range(2):
                x1 = random.randint(100, 300)
                y1 = random.randint(50, 150)
                x2 = x1 - random.randint(50, 100)
                y2 = y1 + random.randint(100, 200)
                x3 = x1 + random.randint(50, 100)
                y3 = y2
                color = random.choice(colors)
                svg += f'''
                <polygon points="{x1},{y1} {x2},{y2} {x3},{y3}" fill="{color}" opacity="0.6"/>'''
                
        # Close the SVG
        svg += '\n</svg>'
        return svg
    
    def _create_object_svg(self, description):
        """Create an object SVG based on description keywords"""
        # Base SVG with background
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <rect width="400" height="400" fill="#f5f5f5"/>'''
            
        # Flower
        if "flower" in description:
            # Determine flower color
            petal_color = "#FF69B4"  # Default pink
            if "red" in description:
                petal_color = "#FF0000"
            elif "blue" in description:
                petal_color = "#0000FF"
            elif "yellow" in description:
                petal_color = "#FFFF00"
            elif "purple" in description:
                petal_color = "#800080"
                
            svg += f'''
            <circle cx="200" cy="200" r="30" fill="#FFD700"/>
            <circle cx="160" cy="160" r="25" fill="{petal_color}"/>
            <circle cx="240" cy="160" r="25" fill="{petal_color}"/>
            <circle cx="160" cy="240" r="25" fill="{petal_color}"/>
            <circle cx="240" cy="240" r="25" fill="{petal_color}"/>
            <circle cx="150" cy="200" r="25" fill="{petal_color}"/>
            <circle cx="250" cy="200" r="25" fill="{petal_color}"/>
            <circle cx="200" cy="150" r="25" fill="{petal_color}"/>
            <circle cx="200" cy="250" r="25" fill="{petal_color}"/>
            <line x1="200" y1="230" x2="200" y2="350" stroke="#228B22" stroke-width="10"/>'''
            
        # House
        elif "house" in description or "building" in description:
            # House color
            house_color = "#CD853F"  # Default wood brown
            if "red" in description:
                house_color = "#B22222"  # Firebrick
            elif "blue" in description:
                house_color = "#4682B4"  # Steel blue
            elif "white" in description:
                house_color = "#F5F5F5"  # White smoke
                
            svg += f'''
            <rect x="100" y="200" width="200" height="150" fill="{house_color}"/>
            <polygon points="100,200 200,100 300,200" fill="#8B4513"/>
            <rect x="150" y="250" width="40" height="100" fill="#4682B4"/>
            <rect x="240" y="230" width="40" height="40" fill="#4682B4"/>'''
            
        # Tree
        elif "tree" in description:
            # Standard deciduous tree
            svg += '''
            <rect x="185" y="250" width="30" height="100" fill="#8B4513"/>
            <circle cx="200" cy="180" r="80" fill="#228B22"/>'''
                
        # Car
        elif "car" in description:
            # Car color
            car_color = "#FF0000"  # Default red
            if "blue" in description:
                car_color = "#0000FF"
            elif "green" in description:
                car_color = "#008000"
            elif "yellow" in description:
                car_color = "#FFFF00"
            elif "black" in description:
                car_color = "#000000"
                
            svg += f'''
            <rect x="100" y="240" width="200" height="60" rx="10" fill="{car_color}"/>
            <rect x="120" y="200" width="160" height="40" rx="5" fill="{car_color}"/>
            <circle cx="140" cy="300" r="20" fill="#333333"/>
            <circle cx="260" cy="300" r="20" fill="#333333"/>
            <circle cx="140" cy="300" r="10" fill="#FFFFFF"/>
            <circle cx="260" cy="300" r="10" fill="#FFFFFF"/>'''
            
        # Default - simple smiley face
        else:
            svg += '''
            <circle cx="200" cy="200" r="100" fill="#FFFF00"/>
            <circle cx="160" cy="160" r="15" fill="#000000"/>
            <circle cx="240" cy="160" r="15" fill="#000000"/>
            <path d="M 150 220 Q 200 280 250 220" stroke="#000000" stroke-width="10" fill="none"/>'''
            
        # Close the SVG
        svg += '\n</svg>'
        return svg
    
    def _create_fashion_svg(self, description):
        """Create a fashion-related SVG based on description keywords with improved dress rendering"""
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <rect width="400" height="400" fill="#f5f5f5"/>'''
            
        # Determine clothing color
        clothing_color = "#FF69B4"  # Default pink
        if "red" in description:
            clothing_color = "#FF0000"
        elif "blue" in description:
            clothing_color = "#0000FF"  # Pure blue
        elif "navy" in description:
            clothing_color = "#000080"  # Navy blue
        elif "light blue" in description:
            clothing_color = "#ADD8E6"  # Light blue
        elif "green" in description:
            clothing_color = "#008000"
        elif "black" in description:
            clothing_color = "#000000"
        elif "white" in description:
            clothing_color = "#FFFFFF"
        elif "yellow" in description:
            clothing_color = "#FFFF00"
        elif "purple" in description:
            clothing_color = "#800080"
            
        # Dress
        if "dress" in description:
            # Long dress or short dress determination
            if "long" in description or "maxi" in description:
                # Long dress
                svg += f'''
                <path d="M160,100 L240,100 L260,350 L140,350 Z" fill="{clothing_color}"/>
                <path d="M160,100 C160,80 180,60 200,60 C220,60 240,80 240,100" fill="{clothing_color}"/>'''
            else:
                # Regular dress (shorter)
                svg += f'''
                <path d="M160,100 L240,100 L255,280 L145,280 Z" fill="{clothing_color}"/>
                <path d="M160,100 C160,80 180,60 200,60 C220,60 240,80 240,100" fill="{clothing_color}"/>'''
                
            # Add neckline
            svg += f'''
            <path d="M180,100 C180,110 220,110 220,100" fill="none" stroke="{clothing_color}" stroke-width="2"/>'''
                
            # Add pattern if mentioned
            if "floral" in description or "flower" in description:
                # Set seed for reproducible patterns
                seed = sum(ord(char) for char in description)
                random.seed(seed)
                
                for i in range(15):
                    cx = random.randint(160, 240)
                    cy = random.randint(120, 280 if "long" not in description else 350)
                    r = random.randint(3, 8)
                    
                    # For floral patterns, use contrasting colors
                    if clothing_color == "#0000FF":  # If blue dress
                        c = random.choice(["#FFFFFF", "#FFFF00", "#FF69B4"])  # White, yellow, pink flowers
                    else:
                        c = random.choice(["#FF0000", "#FFFF00", "#FFFFFF", "#0000FF", "#800080"])
                        
                    svg += f'''
                    <circle cx="{cx}" cy="{cy}" r="{r}" fill="{c}"/>'''
                    # Add small center to the flower
                    center_color = "#FFFF00" if c != "#FFFF00" else "#FFFFFF"
                    svg += f'''
                    <circle cx="{cx}" cy="{cy}" r="{r/3}" fill="{center_color}"/>'''
                    
            elif "polka" in description or "dot" in description:
                # Set seed for reproducible patterns
                seed = sum(ord(char) for char in description)
                random.seed(seed)
                
                for i in range(20):
                    cx = random.randint(160, 240)
                    cy = random.randint(120, 280 if "long" not in description else 350)
                    r = random.randint(3, 6)
                    
                    # For polka dots, usually white or a light color on darker backgrounds
                    dot_color = "#FFFFFF"  # Default white dots
                    if clothing_color == "#FFFFFF":  # If white dress
                        dot_color = "#000000"  # Black dots
                    
                    svg += f'''
                    <circle cx="{cx}" cy="{cy}" r="{r}" fill="{dot_color}"/>'''
                    
            elif "stripe" in description:
                # Horizontal or vertical stripes
                if "horizontal" in description:
                    max_y = 280 if "long" not in description else 350
                    stripe_gap = (max_y - 100) / 10  # Divide the dress height into 10 stripes
                    
                    for i in range(5):  # Draw 5 stripes
                        y = 100 + i * stripe_gap * 2
                        stripe_color = "#FFFFFF" if clothing_color != "#FFFFFF" else "#000000"
                        svg += f'''
                        <path d="M160,{y} L240,{y} L{240 + (i*2)},{y + stripe_gap} L{160 - (i*2)},{y + stripe_gap} Z" fill="{stripe_color}"/>'''
                else:
                    # Vertical stripes
                    max_width = 240 - 160
                    stripe_count = 5
                    stripe_width = max_width / (stripe_count * 2)
                    
                    for i in range(stripe_count):
                        x = 160 + (i * stripe_width * 2)
                        max_y = 280 if "long" not in description else 350
                        
                        stripe_color = "#FFFFFF" if clothing_color != "#FFFFFF" else "#000000"
                        svg += f'''
                        <path d="M{x},{100} L{x + stripe_width},{100} L{x + stripe_width},{max_y} L{x},{max_y} Z" fill="{stripe_color}"/>'''
        
        # Blouse or top
        elif "blouse" in description or "top" in description:
            svg += f'''
            <rect x="160" y="100" width="80" height="100" fill="{clothing_color}"/>
            <path d="M160,100 C160,80 180,60 200,60 C220,60 240,80 240,100" fill="{clothing_color}"/>
            <path d="M160,100 L140,130 L140,200 L160,200" fill="{clothing_color}"/>
            <path d="M240,100 L260,130 L260,200 L240,200" fill="{clothing_color}"/>'''
            
            # Add collar if mentioned
            if "collar" in description:
                svg += f'''
                <path d="M180,100 L200,120 L220,100" fill="none" stroke="#FFFFFF" stroke-width="3"/>'''
        
        # Skirt
        elif "skirt" in description:
            if "long" in description or "maxi" in description:
                svg += f'''
                <path d="M160,100 C160,80 240,80 240,100 L260,350 L140,350 Z" fill="{clothing_color}"/>'''
            else:
                svg += f'''
                <path d="M160,100 C160,80 240,80 240,100 L260,200 L140,200 Z" fill="{clothing_color}"/>'''
                
            # Add pleats if mentioned
            if "pleat" in description:
                max_y = 200 if "long" not in description else 350
                for i in range(6):
                    x = 160 + i * 13
                    svg += f'''
                    <line x1="{x}" y1="100" x2="{x}" y2="{max_y}" stroke="#FFFFFF" stroke-width="1" opacity="0.7"/>'''
        
        # Generic clothing item if nothing specific is mentioned
        else:
            svg += f'''
            <path d="M160,100 L240,100 L255,280 L145,280 Z" fill="{clothing_color}"/>
            <path d="M160,100 C160,80 180,60 200,60 C220,60 240,80 240,100" fill="{clothing_color}"/>'''
            
        # Close the SVG
        svg += '\n</svg>'
        return svg
    
    def _create_default_svg(self, description):
        """Create a default SVG based on general description keywords"""
        # Set a seed based on the description for reproducibility
        seed = sum(ord(char) for char in description)
        random.seed(seed)
        
        # Choose random colors
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6"]
        bg_color = "#f0f0f0"
            
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <rect width="400" height="400" fill="{bg_color}"/>
            <circle cx="200" cy="200" r="100" fill="{colors[0]}" opacity="0.7"/>
            <rect x="120" y="120" width="160" height="160" fill="{colors[1]}" opacity="0.5"/>
            <polygon points="200,80 120,280 280,280" fill="{colors[2]}" opacity="0.6"/>
        </svg>'''
        return svg
    
    def batch_generate(self, descriptions):
        """
        Generate SVGs for multiple descriptions.
        
        Args:
            descriptions (list): List of text descriptions
            
        Returns:
            list: List of SVG code strings
        """
        results = []
        for desc in tqdm(descriptions, desc="Generating SVGs"):
            svg = self.generate_svg(desc)
            results.append(svg)
        return results


# Kaggle competition class that follows the required structure
class SVGModel:
    def __init__(self):
        self.generator = SVGGenerator()
        print("SVG Model initialized for Kaggle competition.")
        
    def predict(self, descriptions):
        """
        Generate SVGs for test set descriptions.
        
        This method matches the expected interface for the Kaggle competition.
        
        Args:
            descriptions (list): List of text descriptions from the test set
            
        Returns:
            list: List of SVG code strings
        """
        return self.generator.batch_generate(descriptions)


# For local testing
if __name__ == "__main__":
    print("Testing Kaggle SVG Generator...")
    
    # Create model
    model = SVGModel()
    
    # Test with example descriptions
    test_descriptions = [
        "A mountain landscape with a lake and sunset",
        "An abstract pattern with colorful circles",
        "A simple house with a red roof",
        "A blue dress with floral pattern",
        "A yellow flower with green stem"
    ]
    
    # Generate SVGs
    results = model.predict(test_descriptions)
    
    # Print results
    for i, svg in enumerate(results):
        print(f"\nSVG {i+1} (first 100 chars):")
        print(svg[:100] + "...")
    
    # Try to run the competition test if possible
    try:
        print("\nTrying to run competition test...")
        from kaggle_evaluation import test
        test(SVGModel)
        print("Competition test completed successfully!")
    except Exception as e:
        print(f"Error running competition test: {str(e)}")
        print("This is normal if you're not in the Kaggle competition environment.")
    
    print("\nTesting completed!")


# Add this to your notebook after you've generated the SVGs
from IPython.display import display, HTML

# Generate some example SVGs
model = SVGModel()
test_descriptions = [
    "A mountain landscape with a lake and sunset",
    "An abstract pattern with colorful circles",
    "A simple house with a red roof",
    "A blue dress with floral pattern",
    "A yellow flower with green stem"
]
svg_results = model.predict(test_descriptions)

# Display each SVG with its description
for i, (desc, svg) in enumerate(zip(test_descriptions, svg_results)):
    print(f"Description {i+1}: {desc}")
    display(HTML(svg))
    print("\n" + "-"*50 + "\n")

