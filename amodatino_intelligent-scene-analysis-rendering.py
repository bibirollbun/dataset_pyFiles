# Drawing with LLMs - Comprehensive SVG Generator
# ================================================

# Import necessary libraries
import re
import random
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import nbdev
import nbdev.showdoc as sd
from nbdev.export import nb_export

# @title Install and import required libraries
# You can keep this section if you need any additional libraries

# @title Define Model class
class Model:
    """
    Model class for the Drawing with LLMs competition.
    
    This model converts textual descriptions into SVG images by:
    1. Analyzing the description to extract key objects, colors, and scene types
    2. Generating appropriate SVG elements based on the analysis
    3. Ensuring the generated SVG adheres to competition constraints
    """
    
    def __init__(self):
        """Initialize the model with necessary components"""
        # Color palette with web colors
        self.colors = {
            # Basic colors
            "red": "#FF0000", "blue": "#0066CC", "green": "#00AA55", 
            "yellow": "#FFDD00", "purple": "#8800AA", "orange": "#FF8800",
            "black": "#000000", "white": "#FFFFFF", "brown": "#8B4513",
            "pink": "#FF55AA", "gray": "#888888", "cyan": "#00DDFF",
            
            # Extended palette
            "lightblue": "#ADD8E6", "skyblue": "#87CEEB", "navy": "#000080",
            "teal": "#008080", "lime": "#00FF00", "olive": "#808000",
            "darkgreen": "#006400", "forestgreen": "#228B22", "gold": "#FFD700",
            "beige": "#F5F5DC", "tan": "#D2B48C", "maroon": "#800000",
            "crimson": "#DC143C", "coral": "#FF7F50", "lavender": "#E6E6FA",
            "plum": "#DDA0DD", "violet": "#EE82EE", "indigo": "#4B0082",
            "turquoise": "#40E0D0", "magenta": "#FF00FF", "salmon": "#FA8072",
            "khaki": "#F0E68C", "darkgray": "#A9A9A9", "lightgray": "#D3D3D3",
            "silver": "#C0C0C0"
        }
        
        # Scene type detectors and generators
        self.scene_detectors = {
            "landscape": self._is_landscape_scene,
            "abstract": self._is_abstract_scene,
            "fashion": self._is_fashion_scene,
            "food": self._is_food_scene,
            "geometric": self._is_geometric_scene,
            "building": self._is_building_scene,
            "animal": self._is_animal_scene,
            "floral": self._is_floral_scene,
            "vehicle": self._is_vehicle_scene
        }
        
        self.scene_generators = {
            "landscape": self._generate_landscape,
            "abstract": self._generate_abstract,
            "fashion": self._generate_fashion,
            "food": self._generate_food_scene,
            "geometric": self._generate_geometric,
            "building": self._generate_building,
            "animal": self._generate_animal,
            "floral": self._generate_floral,
            "vehicle": self._generate_vehicle,
            "default": self._generate_default
        }
        
        # Initialize shape generators
        self._initialize_shape_generators()
        
        # SVG validation parameters
        self.max_svg_size = 9900  # Adding a safety margin below 10,000 bytes

    def predict(self, description: str) -> str:
        """
        The main prediction method required by the competition.
        Converts a text description into an SVG representation.
        
        Args:
            description: Text description of the image to generate
            
        Returns:
            SVG code representing the described image
        """
        try:
            # 1. Analyze the description
            scene_info = self._analyze_description(description)
            
            # 2. Determine scene type and generate appropriate SVG
            scene_type = scene_info.get("scene_type", "default")
            generator = self.scene_generators.get(scene_type, self.scene_generators["default"])
            
            # 3. Generate SVG code
            svg_code = generator(scene_info)
            
            # 4. Validate and optimize SVG
            svg_code = self._optimize_svg(svg_code)
            
            return svg_code
        except Exception as e:
            # Fallback to simple SVG if anything goes wrong
            return self._generate_fallback_svg(str(e))

    def _analyze_description(self, description: str) -> Dict[str, Any]:
        """
        Analyzes the text description to extract key information.
        
        Args:
            description: The text description to analyze
            
        Returns:
            Dictionary containing scene type, objects, colors, and other attributes
        """
        # Initialize with default values
        result = {
            "original_text": description,
            "scene_type": "default",
            "objects": [],
            "colors": [],
            "attributes": {},
            "positions": {},
            "size_modifiers": {}
        }
        
        # Normalize description
        description = description.lower()
        
        # Extract colors
        for color in self.colors.keys():
            if color in description:
                result["colors"].append(color)
        
        # Default colors if none specified
        if not result["colors"]:
            result["colors"] = ["blue", "green"]
        
        # Extract common objects
        objects = [
            "circle", "square", "triangle", "star", "rectangle", "line",
            "mountain", "tree", "sun", "cloud", "moon", "river", "lake", "ocean",
            "dress", "shirt", "hat", "shoes", "coat", "flower", "pattern", 
            "building", "house", "tower", "bird", "cat", "dog", "heart", 
            "car", "boat", "plane", "road", "apple", "fruit", "cake"
        ]
        
        for obj in objects:
            if obj in description:
                result["objects"].append(obj)
        
        # Detect modifiers (size, position)
        size_modifiers = ["small", "large", "big", "tiny", "huge", "medium"]
        for obj in result["objects"]:
            for modifier in size_modifiers:
                if f"{modifier} {obj}" in description:
                    result["size_modifiers"][obj] = modifier
        
        # Detect positions
        position_words = ["above", "below", "under", "on top of", "beside", "next to", 
                        "left", "right", "center", "middle", "inside", "outside"]
        
        for pos in position_words:
            match = re.search(f"({pos} the \\w+)", description)
            if match:
                result["positions"][match.group(1)] = True
        
        # Detect scene type
        for scene_type, detector in self.scene_detectors.items():
            if detector(description):
                result["scene_type"] = scene_type
                break
        
        return result

    # Scene type detectors
    def _is_landscape_scene(self, description: str) -> bool:
        landscape_keywords = ["mountain", "tree", "forest", "sun", "sky", "cloud", 
                             "nature", "river", "lake", "ocean", "sea", "beach", 
                             "hill", "valley", "landscape", "sunset", "sunrise"]
        return any(keyword in description for keyword in landscape_keywords)
    
    def _is_abstract_scene(self, description: str) -> bool:
        abstract_keywords = ["abstract", "geometric", "pattern", "shapes", "random", 
                            "non-representational", "modern art", "contemporary"]
        return any(keyword in description for keyword in abstract_keywords)
    
    def _is_fashion_scene(self, description: str) -> bool:
        fashion_keywords = ["dress", "clothing", "fashion", "wear", "outfit", "shirt", 
                           "pants", "hat", "coat", "jacket", "sweater", "scarf", "shoes"]
        return any(keyword in description for keyword in fashion_keywords)
    
    def _is_food_scene(self, description: str) -> bool:
        food_keywords = ["food", "fruit", "vegetable", "meal", "dish", "cake", 
                        "pizza", "bread", "cheese", "apple", "banana", "plate"]
        return any(keyword in description for keyword in food_keywords)
    
    def _is_geometric_scene(self, description: str) -> bool:
        geometric_keywords = ["circle", "square", "triangle", "rectangle", "polygon", 
                             "geometric", "shapes", "pattern", "symmetric"]
        return any(keyword in description for keyword in geometric_keywords)
    
    def _is_building_scene(self, description: str) -> bool:
        building_keywords = ["building", "house", "architecture", "tower", "castle", 
                            "skyscraper", "church", "temple", "dome", "roof"]
        return any(keyword in description for keyword in building_keywords)
    
    def _is_animal_scene(self, description: str) -> bool:
        animal_keywords = ["animal", "cat", "dog", "bird", "fish", "pet", "wild animal",
                          "butterfly", "elephant", "lion", "tiger", "bear"]
        return any(keyword in description for keyword in animal_keywords)
    
    def _is_floral_scene(self, description: str) -> bool:
        floral_keywords = ["flower", "floral", "bouquet", "garden", "rose", "tulip", 
                          "daisy", "petal", "bloom", "blossom"]
        return any(keyword in description for keyword in floral_keywords)
    
    def _is_vehicle_scene(self, description: str) -> bool:
        vehicle_keywords = ["car", "vehicle", "truck", "bus", "bicycle", "bike", 
                           "boat", "ship", "plane", "airplane", "motorcycle"]
        return any(keyword in description for keyword in vehicle_keywords)

    # SVG generation methods
    def _generate_landscape(self, scene_info: Dict) -> str:
        """Generate a landscape scene"""
        colors = scene_info.get("colors", ["blue", "green"])
        objects = scene_info.get("objects", [])
        
        # Choose background colors
        sky_color = self._get_color_from_list(colors, ["blue", "lightblue", "skyblue", "cyan"], "skyblue")
        ground_color = self._get_color_from_list(colors, ["green", "darkgreen", "forestgreen", "brown"], "forestgreen")
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add sky
        svg += f'  <rect x="0" y="0" width="300" height="120" fill="{sky_color}" />\n'
        
        # Add ground
        svg += f'  <rect x="0" y="120" width="300" height="80" fill="{ground_color}" />\n'
        
        # Add sun or moon
        if "sun" in objects or (random.random() > 0.5 and "moon" not in objects):
            sun_color = self._get_color_from_list(colors, ["yellow", "gold", "orange"], "gold")
            svg += f'  <circle cx="240" cy="40" r="20" fill="{sun_color}" />\n'
            
            # Add sun rays
            for i in range(8):
                angle = i * math.pi / 4
                x1 = 240 + 22 * math.cos(angle)
                y1 = 40 + 22 * math.sin(angle)
                x2 = 240 + 30 * math.cos(angle)
                y2 = 40 + 30 * math.sin(angle)
                svg += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{sun_color}" stroke-width="2" />\n'
        
        if "moon" in objects:
            svg += f'  <circle cx="240" cy="40" r="20" fill="{self.colors["lightgray"]}" />\n'
            svg += f'  <circle cx="230" cy="35" r="18" fill="{sky_color}" />\n'
        
        # Add clouds
        if "cloud" in objects or random.random() > 0.6:
            cloud_positions = [(80, 40), (170, 30), (250, 60)]
            for x, y in cloud_positions[:2]:  # Limit to 2 clouds
                svg += self._draw_cloud(x, y)
        
        # Add mountains
        if "mountain" in objects or random.random() > 0.5:
            mountain_color1 = self._get_color_from_list(colors, ["gray", "darkgray", "brown"], "darkgray")
            mountain_color2 = self._get_color_from_list(colors, ["gray", "darkgray", "brown"], "brown")
            
            # First mountain range
            svg += f'  <polygon points="30,120 100,50 170,120" fill="{mountain_color1}" />\n'
            
            # Second mountain range
            svg += f'  <polygon points="130,120 200,60 270,120" fill="{mountain_color2}" />\n'
            
            # Snow caps
            svg += f'  <polygon points="90,60 100,50 110,60" fill="{self.colors["white"]}" />\n'
            svg += f'  <polygon points="190,70 200,60 210,70" fill="{self.colors["white"]}" />\n'
        
        # Add trees
        if "tree" in objects or "forest" in objects or random.random() > 0.3:
            num_trees = 5 if "forest" in objects else 3
            for _ in range(num_trees):
                x = random.randint(30, 270)
                y = random.randint(130, 150)
                size = random.randint(10, 20)
                svg += self._draw_tree(x, y, size)
        
        # Add river or lake
        if "river" in objects:
            river_color = self._get_color_from_list(colors, ["blue", "lightblue", "cyan"], "lightblue")
            svg += f'  <path d="M0,140 C50,130 100,150 150,140 C200,130 250,150 300,140" fill="none" stroke="{river_color}" stroke-width="10" />\n'
        
        if "lake" in objects or "ocean" in objects:
            lake_color = self._get_color_from_list(colors, ["blue", "lightblue", "cyan"], "lightblue")
            svg += f'  <ellipse cx="150" cy="160" rx="80" ry="25" fill="{lake_color}" />\n'
        
        svg += '</svg>'
        return svg
    
    def _generate_abstract(self, scene_info: Dict) -> str:
        """Generate an abstract art scene"""
        colors = scene_info.get("colors", ["blue", "red", "yellow", "purple"])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background
        background_color = self._get_color_from_list(colors, ["white", "black", "gray"], "white")
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{background_color}" />\n'
        
        # Add a linear gradient for sophistication
        svg += f'''  <defs>
    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{self._get_color(colors[0])}" />
      <stop offset="100%" stop-color="{self._get_color(colors[-1])}" />
    </linearGradient>
  </defs>\n'''
        
        # Create various abstract elements
        # Large background shape
        svg += f'  <rect x="50" y="50" width="200" height="100" fill="url(#gradient1)" opacity="0.7" />\n'
        
        # Add geometric shapes with variety
        shapes = ["circle", "rect", "polygon", "ellipse", "path"]
        for i in range(12):
            shape = random.choice(shapes)
            color = self._get_color(random.choice(colors))
            opacity = random.uniform(0.4, 0.9)
            stroke_width = random.randint(0, 3)
            stroke_color = self._get_color(random.choice(colors))
            
            x = random.randint(20, 280)
            y = random.randint(20, 180)
            
            if shape == "circle":
                r = random.randint(5, 30)
                svg += f'  <circle cx="{x}" cy="{y}" r="{r}" fill="{color}" opacity="{opacity:.1f}" stroke="{stroke_color}" stroke-width="{stroke_width}" />\n'
            
            elif shape == "rect":
                width = random.randint(10, 60)
                height = random.randint(10, 60)
                rotation = random.randint(0, 45)
                svg += f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}" opacity="{opacity:.1f}" stroke="{stroke_color}" stroke-width="{stroke_width}" transform="rotate({rotation}, {x}, {y})" />\n'
            
            elif shape == "polygon":
                num_points = random.randint(3, 6)
                points = []
                for j in range(num_points):
                    angle = j * 2 * math.pi / num_points
                    distance = random.randint(15, 30)
                    px = x + distance * math.cos(angle)
                    py = y + distance * math.sin(angle)
                    points.append(f"{px},{py}")
                svg += f'  <polygon points="{" ".join(points)}" fill="{color}" opacity="{opacity:.1f}" stroke="{stroke_color}" stroke-width="{stroke_width}" />\n'
            
            elif shape == "ellipse":
                rx = random.randint(10, 40)
                ry = random.randint(5, 25)
                svg += f'  <ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{color}" opacity="{opacity:.1f}" stroke="{stroke_color}" stroke-width="{stroke_width}" />\n'
            
            elif shape == "path":
                # Create a simple cubic bezier curve
                x1 = random.randint(20, 280)
                y1 = random.randint(20, 180)
                x2 = random.randint(20, 280)
                y2 = random.randint(20, 180)
                svg += f'  <path d="M {x} {y} C {x1} {y1}, {x2} {y2}, {x+random.randint(-50, 50)} {y+random.randint(-50, 50)}" stroke="{color}" fill="none" stroke-width="{stroke_width+2}" opacity="{opacity:.1f}" />\n'
        
        svg += '</svg>'
        return svg
    
    def _generate_fashion(self, scene_info: Dict) -> str:
        """Generate a fashion-related image"""
        colors = scene_info.get("colors", ["red", "black"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background
        background_color = self.colors["white"]
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{background_color}" />\n'
        
        # Determine the type of clothing to draw
        if "dress" in objects:
            primary_color = self._get_color_from_list(colors, ["red", "blue", "purple", "black"], "red")
            secondary_color = self._get_color_from_list(colors, ["gold", "silver", "white"], "gold")
            
            # Draw a dress
            svg += f'''  <path d="M150,40 C120,40 110,60 110,70 L110,100 L190,100 L190,70 C190,60 180,40 150,40 Z" fill="{primary_color}" />
  <path d="M110,100 L90,180 L210,180 L190,100 Z" fill="{primary_color}" />
  <path d="M150,40 L150,180" stroke="{secondary_color}" stroke-width="2" />
  <path d="M130,100 L130,160" stroke="{secondary_color}" stroke-width="1" />
  <path d="M170,100 L170,160" stroke="{secondary_color}" stroke-width="1" />\n'''
            
        elif "shirt" in objects or "coat" in objects:
            primary_color = self._get_color_from_list(colors, ["blue", "white", "green", "red"], "blue")
            
            # Draw a shirt or coat
            svg += f'''  <path d="M120,60 L180,60 L170,40 L130,40 Z" fill="{primary_color}" />
  <rect x="120" y="60" width="60" height="80" fill="{primary_color}" />
  <rect x="120" y="60" width="15" height="80" fill="{self._darken_color(primary_color)}" />
  <rect x="165" y="60" width="15" height="80" fill="{self._darken_color(primary_color)}" />\n'''
            
            # Add buttons
            for i in range(4):
                y_pos = 70 + i * 20
                svg += f'  <circle cx="150" cy="{y_pos}" r="3" fill="{self.colors["gold"]}" />\n'
                
        elif "hat" in objects:
            primary_color = self._get_color_from_list(colors, ["brown", "black", "red"], "brown")
            
            # Draw a hat
            svg += f'''  <ellipse cx="150" cy="130" rx="70" ry="15" fill="{primary_color}" />
  <path d="M120,130 L120,90 C120,70 180,70 180,90 L180,130" fill="{primary_color}" />\n'''
            
            # Add hat band
            band_color = self._get_color_from_list(colors, ["gold", "red", "black"], "gold")
            svg += f'  <path d="M120,95 L180,95" stroke="{band_color}" stroke-width="5" />\n'
            
        else:
            # Default - draw a simple clothing item
            primary_color = self._get_color(colors[0])
            
            svg += f'''  <path d="M100,60 C100,40 200,40 200,60 L200,170 L100,170 Z" fill="{primary_color}" />
  <path d="M150,60 L150,170" stroke="{self._lighten_color(primary_color)}" stroke-width="2" />\n'''
        
        svg += '</svg>'
        return svg
    
    def _generate_food_scene(self, scene_info: Dict) -> str:
        """Generate a food-related image"""
        colors = scene_info.get("colors", ["red", "green", "yellow"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{self.colors["white"]}" />\n'
        
        # Draw a plate
        svg += f'  <ellipse cx="150" cy="120" rx="100" ry="60" fill="{self.colors["lightgray"]}" />\n'
        svg += f'  <ellipse cx="150" cy="120" rx="90" ry="54" fill="{self.colors["white"]}" />\n'
        
        if "cake" in objects:
            # Draw a cake
            cake_color = self._get_color_from_list(colors, ["brown", "tan", "pink"], "tan")
            frosting_color = self._get_color_from_list(colors, ["white", "pink", "crimson"], "white")
            
            svg += f'''  <rect x="100" y="80" width="100" height="60" rx="5" fill="{cake_color}" />
  <rect x="90" y="80" width="120" height="10" rx="5" fill="{frosting_color}" />
  <ellipse cx="150" cy="80" rx="60" ry="10" fill="{frosting_color}" />\n'''
            
            # Add candles
            for i in range(3):
                x_pos = 120 + i * 30
                candle_color = self._get_color(random.choice(colors))
                svg += f'''  <rect x="{x_pos}" y="60" width="4" height="20" fill="{candle_color}" />
  <circle cx="{x_pos+2}" cy="60" r="2" fill="{self.colors["yellow"]}" />\n'''
                
        elif "apple" in objects or "fruit" in objects:
            # Draw fruits
            fruit_types = ["apple", "orange", "banana"]
            for i in range(3):
                x_offset = (i - 1) * 50
                fruit = "apple" if "apple" in objects else random.choice(fruit_types)
                
                if fruit == "apple":
                    apple_color = self._get_color_from_list(colors, ["red", "green", "crimson"], "red")
                    svg += f'''  <circle cx="{150+x_offset}" cy="120" r="20" fill="{apple_color}" />
  <path d="M{150+x_offset-5},100 Q{150+x_offset},{90} {150+x_offset+5},100" stroke="{self.colors["brown"]}" stroke-width="2" fill="none" />
  <path d="M{150+x_offset},100 C{150+x_offset-10},{115} {150+x_offset+10},{115} {150+x_offset},100" stroke="{self.colors["darkgreen"]}" stroke-width="2" fill="{self.colors["darkgreen"]}" />\n'''
                
                elif fruit == "orange":
                    svg += f'''  <circle cx="{150+x_offset}" cy="120" r="20" fill="{self.colors["orange"]}" />
  <circle cx="{150+x_offset}" cy="110" r="2" fill="{self.colors["darkgreen"]}" />\n'''
                
                elif fruit == "banana":
                    svg += f'''  <path d="M{130+x_offset},120 C{150+x_offset},{90} {170+x_offset},{110} {170+x_offset},130" fill="{self.colors["yellow"]}" />
  <path d="M{130+x_offset},120 C{150+x_offset},{90} {170+x_offset},{110} {170+x_offset},130" fill="none" stroke="{self.colors["gold"]}" stroke-width="1" />\n'''
            
        else:
            # Draw a generic meal
            svg += f'''  <circle cx="150" cy="120" r="30" fill="{self._get_color(colors[0])}" />
  <circle cx="120" cy="100" r="15" fill="{self._get_color(colors[1] if len(colors) > 1 else colors[0])}" />
  <rect x="160" y="90" width="30" height="20" fill="{self._get_color(colors[2] if len(colors) > 2 else colors[0])}" />\n'''
        
        svg += '</svg>'
        return svg
    
    def _generate_geometric(self, scene_info: Dict) -> str:
        """Generate a geometric pattern"""
        colors = scene_info.get("colors", ["blue", "red", "yellow"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{self.colors["white"]}" />\n'
        
        # Add gradient definition
        svg += f'''  <defs>
    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{self._get_color(colors[0])}" />
      <stop offset="100%" stop-color="{self._get_color(colors[-1])}" />
    </linearGradient>
  </defs>\n'''
        
        # Determine which shapes to create
        shape_types = []
        for shape in ["circle", "square", "triangle", "rectangle"]:
            if shape in objects:
                shape_types.append(shape)
        
        if not shape_types:
            shape_types = ["circle", "square", "triangle", "rectangle"]
            
        # Create a geometric pattern
        if "pattern" in objects:
            # Grid pattern
            for row in range(5):
                for col in range(7):
                    x = 25 + col * 40
                    y = 25 + row * 40
                    shape = shape_types[random.randint(0, len(shape_types)-1)] if random.random() > 0.3 else random.choice(shape_types)
                    color = self._get_color(colors[random.randint(0, len(colors)-1)])
                    
                    if shape == "circle":
                        svg += f'  <circle cx="{x}" cy="{y}" r="15" fill="{color}" />\n'
                    elif shape == "square":
                        svg += f'  <rect x="{x-15}" y="{y-15}" width="30" height="30" fill="{color}" />\n'
                    elif shape == "triangle":
                        svg += f'  <polygon points="{x},{y-15} {x+15},{y+15} {x-15},{y+15}" fill="{color}" />\n'
                    elif shape == "rectangle":
                        svg += f'  <rect x="{x-20}" y="{y-10}" width="40" height="20" fill="{color}" />\n'
        else:
            # Concentric or radial pattern
            center_x, center_y = 150, 100
            
            # Background shape
            svg += f'  <circle cx="{center_x}" cy="{center_y}" r="80" fill="url(#gradient1)" opacity="0.3" />\n'
            
            # Create concentric shapes
            for i in range(5):
                size = 70 - i * 15
                color = self._get_color(colors[i % len(colors)])
                rotation = i * 30
                
                if "circle" in shape_types:
                    svg += f'  <circle cx="{center_x}" cy="{center_y}" r="{size}" fill="none" stroke="{color}" stroke-width="3" />\n'
                
                if "square" in shape_types:
                    svg += f'  <rect x="{center_x-size}" y="{center_y-size}" width="{size*2}" height="{size*2}" fill="none" stroke="{color}" stroke-width="3" transform="rotate({rotation}, {center_x}, {center_y})" />\n'
                
                if "triangle" in shape_types:
                    points = []
                    for j in range(3):
                        angle = j * 2 * math.pi / 3 + math.radians(rotation)
                        px = center_x + size * math.cos(angle)
                        py = center_y + size * math.sin(angle)
                        points.append(f"{px},{py}")
                    svg += f'  <polygon points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" />\n'
        
        svg += '</svg>'
        return svg
    
    def _generate_building(self, scene_info: Dict) -> str:
        """Generate a building or architectural scene"""
        colors = scene_info.get("colors", ["gray", "brown", "red"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add sky
        sky_color = self._get_color_from_list(colors, ["lightblue", "skyblue", "blue"], "skyblue")
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{sky_color}" />\n'
        
        # Add ground
        ground_color = self._get_color_from_list(colors, ["green", "brown", "gray"], "brown")
        svg += f'  <rect x="0" y="160" width="300" height="40" fill="{ground_color}" />\n'
        
        if "house" in objects:
            # Draw a house
            wall_color = self._get_color_from_list(colors, ["red", "tan", "brown", "white"], "tan")
            roof_color = self._get_color_from_list(colors, ["brown", "red", "gray"], "brown")
            
            svg += f'''  <rect x="100" y="90" width="100" height="70" fill="{wall_color}" />
  <polygon points="100,90 150,50 200,90" fill="{roof_color}" />
  <rect x="135" y="120" width="30" height="40" fill="{self.colors["brown"]}" />
  <rect x="110" y="100" width="20" height="20" fill="{self.colors["lightblue"]}" />
  <rect x="170" y="100" width="20" height="20" fill="{self.colors["lightblue"]}" />\n'''
            
        elif "castle" in objects:
            # Draw a castle
            wall_color = self._get_color_from_list(colors, ["gray", "lightgray", "tan"], "gray")
            roof_color = self._get_color_from_list(colors, ["darkgray", "blue", "red"], "darkgray")
            
            svg += f'''  <rect x="80" y="70" width="140" height="90" fill="{wall_color}" />
  <rect x="70" y="60" width="20" height="20" fill="{wall_color}" />
  <rect x="120" y="60" width="20" height="20" fill="{wall_color}" />
  <rect x="170" y="60" width="20" height="20" fill="{wall_color}" />
  <rect x="210" y="60" width="20" height="20" fill="{wall_color}" />
  <rect x="130" y="110" width="40" height="50" fill="{self.colors["brown"]}" />
  <rect x="90" y="90" width="20" height="30" fill="{self.colors["lightblue"]}" />
  <rect x="190" y="90" width="20" height="30" fill="{self.colors["lightblue"]}" />\n'''
            
            # Add castle tops
            for x in [80, 130, 180]:
                svg += f'  <polygon points="{x},70 {x+10},50 {x+20},70" fill="{roof_color}" />\n'
            
        elif "tower" in objects or "skyscraper" in objects:
            # Draw a tower or skyscraper
            building_color = self._get_color_from_list(colors, ["gray", "silver", "blue"], "gray")
            window_color = self._get_color_from_list(colors, ["lightblue", "gold", "white"], "lightblue")
            
            svg += f'  <rect x="120" y="40" width="60" height="120" fill="{building_color}" />\n'
            
            # Add windows
            for row in range(8):
                for col in range(4):
                    x = 125 + col * 15
                    y = 50 + row * 13
                    svg += f'  <rect x="{x}" y="{y}" width="10" height="8" fill="{window_color}" />\n'
            
            # Add tower top
            svg += f'  <polygon points="120,40 150,20 180,40" fill="{self._lighten_color(building_color)}" />\n'
            
        else:
            # Draw a default building
            building_color = self._get_color_from_list(colors, ["gray", "brown", "red"], "gray")
            
            svg += f'''  <rect x="100" y="60" width="100" height="100" fill="{building_color}" />
  <rect x="130" y="120" width="40" height="40" fill="{self.colors["brown"]}" />\n'''
            
            # Add windows
            for row in range(3):
                for col in range(3):
                    if not (row == 2 and col == 1):  # Skip middle bottom (where door is)
                        x = 110 + col * 30
                        y = 70 + row * 30
                        svg += f'  <rect x="{x}" y="{y}" width="20" height="20" fill="{self.colors["lightblue"]}" />\n'
        
        svg += '</svg>'
        return svg
    
    def _generate_animal(self, scene_info: Dict) -> str:
        """Generate an animal scene"""
        colors = scene_info.get("colors", ["brown", "orange", "white"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background - sky and ground
        sky_color = self._get_color_from_list(colors, ["lightblue", "skyblue", "white"], "lightblue")
        ground_color = self._get_color_from_list(colors, ["green", "brown", "tan"], "green")
        
        svg += f'  <rect x="0" y="0" width="300" height="140" fill="{sky_color}" />\n'
        svg += f'  <rect x="0" y="140" width="300" height="60" fill="{ground_color}" />\n'
        
        if "bird" in objects:
            # Draw a bird
            bird_color = self._get_color_from_list(colors, ["blue", "red", "orange", "yellow"], "blue")
            
            svg += f'''  <ellipse cx="150" cy="100" rx="25" ry="15" fill="{bird_color}" />
  <circle cx="170" cy="92" r="7" fill="{self.colors["white"]}" />
  <circle cx="172" cy="90" r="3" fill="{self.colors["black"]}" />
  <polygon points="175,92 185,95 175,98" fill="{self.colors["yellow"]}" />
  <ellipse cx="120" cy="100" rx="15" ry="10" transform="rotate(-20, 120, 100)" fill="{bird_color}" />
  <ellipse cx="180" cy="100" rx="15" ry="10" transform="rotate(20, 180, 100)" fill="{bird_color}" />\n'''
            
            # Add some clouds
            svg += self._draw_cloud(50, 40)
            svg += self._draw_cloud(200, 60)
            
        elif "cat" in objects:
            # Draw a cat
            cat_color = self._get_color_from_list(colors, ["orange", "gray", "brown", "black"], "orange")
            
            svg += f'''  <ellipse cx="150" cy="130" rx="40" ry="25" fill="{cat_color}" />
  <circle cx="150" cy="100" r="25" fill="{cat_color}" />
  <polygon points="130,85 140,95 120,95" fill="{cat_color}" />
  <polygon points="170,85 180,95 160,95" fill="{cat_color}" />
  <circle cx="140" cy="95" r="5" fill="{self.colors["green"]}" />
  <circle cx="160" cy="95" r="5" fill="{self.colors["green"]}" />
  <circle cx="142" cy="93" r="2" fill="{self.colors["black"]}" />
  <circle cx="162" cy="93" r="2" fill="{self.colors["black"]}" />
  <polygon points="150,100 145,110 155,110" fill="{self.colors["pink"]}" />
  <line x1="140" y1="110" x2="130" y2="115" stroke="{self.colors["black"]}" stroke-width="1" />
  <line x1="142" y1="112" x2="132" y2="117" stroke="{self.colors["black"]}" stroke-width="1" />
  <line x1="144" y1="114" x2="134" y2="119" stroke="{self.colors["black"]}" stroke-width="1" />
  <line x1="160" y1="110" x2="170" y2="115" stroke="{self.colors["black"]}" stroke-width="1" />
  <line x1="158" y1="112" x2="168" y2="117" stroke="{self.colors["black"]}" stroke-width="1" />
  <line x1="156" y1="114" x2="166" y2="119" stroke="{self.colors["black"]}" stroke-width="1" />\n'''
            
            # Add tail
            svg += f'  <path d="M110,130 C90,120 80,110 90,90" stroke="{cat_color}" stroke-width="10" fill="none" />\n'
            
        elif "dog" in objects:
            # Draw a dog
            dog_color = self._get_color_from_list(colors, ["brown", "tan", "gray"], "brown")
            
            svg += f'''  <ellipse cx="150" cy="130" rx="40" ry="25" fill="{dog_color}" />
  <circle cx="150" cy="100" r="30" fill="{dog_color}" />
  <ellipse cx="180" cy="100" rx="15" ry="10" fill="{dog_color}" />
  <polygon points="185,95 195,85 195,105" fill="{dog_color}" />
  <polygon points="130,80 140,90 125,90" fill="{dog_color}" />
  <polygon points="170,80 180,90 165,90" fill="{dog_color}" />
  <circle cx="140" cy="95" r="5" fill="{self.colors["black"]}" />
  <circle cx="160" cy="95" r="5" fill="{self.colors["black"]}" />
  <ellipse cx="150" cy="110" rx="10" ry="5" fill="{self.colors["black"]}" />\n'''
            
            # Add tail
            svg += f'  <path d="M110,120 C95,110 95,95 105,85" stroke="{dog_color}" stroke-width="10" fill="none" />\n'
            
        elif "butterfly" in objects:
            # Draw a butterfly
            wing_color1 = self._get_color_from_list(colors, ["purple", "blue", "cyan"], "purple")
            wing_color2 = self._get_color_from_list(colors, ["orange", "pink", "yellow"], "orange")
            
            svg += f'''  <path d="M150,80 L150,120" stroke="{self.colors["black"]}" stroke-width="2" />
  <ellipse cx="130" cy="90" rx="20" ry="15" fill="{wing_color1}" transform="rotate(-20, 130, 90)" />
  <ellipse cx="170" cy="90" rx="20" ry="15" fill="{wing_color1}" transform="rotate(20, 170, 90)" />
  <ellipse cx="130" cy="110" rx="20" ry="15" fill="{wing_color2}" transform="rotate(20, 130, 110)" />
  <ellipse cx="170" cy="110" rx="20" ry="15" fill="{wing_color2}" transform="rotate(-20, 170, 110)" />
  <circle cx="145" cy="80" r="3" fill="{self.colors["black"]}" />
  <circle cx="155" cy="80" r="3" fill="{self.colors["black"]}" />\n'''
            
            # Add some flowers
            for i in range(3):
                x = 50 + i * 100
                y = 160
                svg += self._draw_flower(x, y, random.choice(colors))
            
        else:
            # Draw a generic animal silhouette
            animal_color = self._get_color(colors[0])
            
            svg += f'''  <ellipse cx="150" cy="130" rx="40" ry="20" fill="{animal_color}" />
  <circle cx="150" cy="100" r="25" fill="{animal_color}" />
  <circle cx="135" cy="95" r="5" fill="{self.colors["black"]}" />
  <circle cx="165" cy="95" r="5" fill="{self.colors["black"]}" />\n'''
        
        svg += '</svg>'
        return svg
    
    def _generate_floral(self, scene_info: Dict) -> str:
        """Generate a floral or garden scene"""
        colors = scene_info.get("colors", ["red", "yellow", "green", "pink"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background - sky and ground
        sky_color = self._get_color_from_list(colors, ["lightblue", "white", "skyblue"], "lightblue")
        ground_color = self._get_color_from_list(colors, ["green", "darkgreen", "brown"], "green")
        
        svg += f'  <rect x="0" y="0" width="300" height="140" fill="{sky_color}" />\n'
        svg += f'  <rect x="0" y="140" width="300" height="60" fill="{ground_color}" />\n'
        
        # Add the sun
        svg += f'  <circle cx="260" cy="40" r="20" fill="{self.colors["yellow"]}" />\n'
        
        # Draw multiple flowers
        flower_colors = []
        for color in colors:
            if color not in ["green", "darkgreen", "brown", "black", "white", "lightblue", "skyblue"]:
                flower_colors.append(color)
        
        if not flower_colors:
            flower_colors = ["red", "pink", "yellow", "purple"]
        
        # Generate flower positions
        num_flowers = random.randint(5, 10)
        positions = []
        for _ in range(num_flowers):
            x = random.randint(30, 270)
            y = random.randint(140, 180)
            positions.append((x, y))
        
        # Sort positions from back to front
        positions.sort(key=lambda pos: pos[1])
        
        # Add stems first
        for x, y in positions:
            height = random.randint(40, 100)
            svg += f'  <line x1="{x}" y1="{y}" x2="{x}" y2="{y-height}" stroke="{self.colors["green"]}" stroke-width="2" />\n'
            
            # Add leaves
            leaf_positions = random.randint(1, 3)
            for i in range(leaf_positions):
                leaf_y = y - random.randint(10, height-10)
                leaf_direction = -1 if random.random() > 0.5 else 1
                svg += f'  <path d="M{x},{leaf_y} C{x+15*leaf_direction},{leaf_y-10} {x+25*leaf_direction},{leaf_y+10} {x},{leaf_y+5}" fill="{self.colors["green"]}" />\n'
        
        # Add flowers
        for x, y in positions:
            height = random.randint(40, 100)
            flower_color = random.choice(flower_colors)
            flower_y = y - height
            
            # Different flower types
            flower_type = random.randint(0, 2)
            
            if flower_type == 0:
                # Basic daisy-like flower
                petal_size = random.randint(10, 15)
                svg += f'  <circle cx="{x}" cy="{flower_y}" r="{petal_size}" fill="{self._get_color(flower_color)}" />\n'
                svg += f'  <circle cx="{x}" cy="{flower_y}" r="{petal_size/3}" fill="{self.colors["yellow"]}" />\n'
                
            elif flower_type == 1:
                # Multi-petal flower
                petal_size = random.randint(5, 10)
                num_petals = random.randint(5, 8)
                
                for i in range(num_petals):
                    angle = i * 2 * math.pi / num_petals
                    petal_x = x + petal_size * 1.5 * math.cos(angle)
                    petal_y = flower_y + petal_size * 1.5 * math.sin(angle)
                    svg += f'  <circle cx="{petal_x}" cy="{petal_y}" r="{petal_size}" fill="{self._get_color(flower_color)}" />\n'
                
                svg += f'  <circle cx="{x}" cy="{flower_y}" r="{petal_size}" fill="{self.colors["yellow"]}" />\n'
                
            elif flower_type == 2:
                # Tulip-like flower
                svg += f'''  <path d="M{x-10},{flower_y} C{x-10},{flower_y-20} {x+10},{flower_y-20} {x+10},{flower_y}" fill="{self._get_color(flower_color)}" />
  <path d="M{x-10},{flower_y} C{x-5},{flower_y+5} {x+5},{flower_y+5} {x+10},{flower_y}" fill="{self._get_color(flower_color)}" />\n'''
        
        # Add some grass tufts
        for _ in range(10):
            x = random.randint(20, 280)
            y = 140
            height = random.randint(5, 15)
            svg += f'  <path d="M{x},{y} C{x-5},{y-height} {x},{y-height*1.5} {x},{y-height}" stroke="{self.colors["green"]}" stroke-width="1" fill="none" />\n'
            svg += f'  <path d="M{x},{y} C{x+5},{y-height} {x},{y-height*1.5} {x},{y-height}" stroke="{self.colors["green"]}" stroke-width="1" fill="none" />\n'
        
        svg += '</svg>'
        return svg
    
    def _generate_vehicle(self, scene_info: Dict) -> str:
        """Generate a vehicle scene"""
        colors = scene_info.get("colors", ["red", "blue", "black"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background - sky and road
        sky_color = self._get_color_from_list(colors, ["lightblue", "skyblue", "white"], "lightblue")
        road_color = self._get_color_from_list(colors, ["gray", "darkgray", "black"], "gray")
        
        svg += f'  <rect x="0" y="0" width="300" height="130" fill="{sky_color}" />\n'
        svg += f'  <rect x="0" y="130" width="300" height="70" fill="{road_color}" />\n'
        
        # Add road markings
        svg += f'  <line x1="0" y1="165" x2="300" y2="165" stroke="{self.colors["white"]}" stroke-width="5" stroke-dasharray="30,10" />\n'
        
        if "car" in objects:
            # Draw a car
            car_color = self._get_color_from_list(colors, ["red", "blue", "green", "yellow"], "red")
            
            svg += f'''  <rect x="80" y="120" width="140" height="25" rx="5" fill="{car_color}" />
  <rect x="100" y="100" width="100" height="30" rx="10" fill="{car_color}" />
  <rect x="110" y="105" width="25" height="20" fill="{self.colors["lightblue"]}" />
  <rect x="145" y="105" width="25" height="20" fill="{self.colors["lightblue"]}" />
  <circle cx="100" cy="145" r="15" fill="{self.colors["black"]}" />
  <circle cx="100" cy="145" r="8" fill="{self.colors["gray"]}" />
  <circle cx="200" cy="145" r="15" fill="{self.colors["black"]}" />
  <circle cx="200" cy="145" r="8" fill="{self.colors["gray"]}" />\n'''
            
            # Add headlights
            svg += f'''  <rect x="80" y="115" width="5" height="10" fill="{self.colors["yellow"]}" />
  <rect x="215" y="115" width="5" height="10" fill="{self.colors["red"]}" />\n'''
            
        elif "boat" in objects or "ship" in objects:
            # Draw a boat or ship
            boat_color = self._get_color_from_list(colors, ["blue", "red", "white"], "blue")
            
            # Replace road with water
            water_color = self._get_color_from_list(colors, ["blue", "cyan", "lightblue"], "blue")
            svg = svg.replace(f'  <rect x="0" y="130" width="300" height="70" fill="{road_color}" />\n', 
                            f'  <rect x="0" y="130" width="300" height="70" fill="{water_color}" />\n')
            
            # Remove road markings
            svg = svg.replace(f'  <line x1="0" y1="165" x2="300" y2="165" stroke="{self.colors["white"]}" stroke-width="5" stroke-dasharray="30,10" />\n', '')
            
            # Add water waves
            svg += f'''  <path d="M0,140 C30,130 60,150 90,140 C120,130 150,150 180,140 C210,130 240,150 270,140 C300,130 330,150 360,140" stroke="{self._lighten_color(water_color)}" stroke-width="3" fill="none" />
  <path d="M0,160 C30,150 60,170 90,160 C120,150 150,170 180,160 C210,150 240,170 270,160 C300,150 330,170 360,160" stroke="{self._lighten_color(water_color)}" stroke-width="3" fill="none" />\n'''
            
            # Draw the boat
            svg += f'''  <path d="M100,140 L200,140 L180,170 L120,170 Z" fill="{boat_color}" />
  <rect x="140" y="110" width="10" height="30" fill="{self.colors["brown"]}" />
  <path d="M150,110 L150,130 L180,120 Z" fill="{self.colors["white"]}" />\n'''
            
        elif "plane" in objects or "airplane" in objects:
            # Draw a plane
            plane_color = self._get_color_from_list(colors, ["white", "blue", "silver"], "white")
            
            svg += f'''  <ellipse cx="150" cy="80" rx="100" ry="15" fill="{plane_color}" />
  <polygon points="230,80 260,70 250,80 260,90" fill="{plane_color}" />
  <polygon points="120,80 120,60 140,70 160,70 180,60 180,80" fill="{plane_color}" />
  <polygon points="70,80 50,110 70,110 80,90" fill="{plane_color}" />
  <polygon points="230,80 210,110 230,110 240,90" fill="{plane_color}" />
  <rect x="100" y="70" width="10" height="20" fill="{self.colors["lightblue"]}" />
  <rect x="120" y="70" width="10" height="20" fill="{self.colors["lightblue"]}" />
  <rect x="140" y="70" width="10" height="20" fill="{self.colors["lightblue"]}" />
  <rect x="160" y="70" width="10" height="20" fill="{self.colors["lightblue"]}" />
  <rect x="180" y="70" width="10" height="20" fill="{self.colors["lightblue"]}" />\n'''
            
            # Add clouds
            svg += self._draw_cloud(50, 40)
            svg += self._draw_cloud(200, 30)
            
        elif "bicycle" in objects or "bike" in objects:
            # Draw a bicycle
            frame_color = self._get_color_from_list(colors, ["red", "blue", "black"], "red")
            
            svg += f'''  <circle cx="110" cy="145" r="20" fill="none" stroke="{self.colors["black"]}" stroke-width="3" />
  <circle cx="190" cy="145" r="20" fill="none" stroke="{self.colors["black"]}" stroke-width="3" />
  <path d="M110,145 L150,110 L190,145" stroke="{frame_color}" stroke-width="5" fill="none" />
  <path d="M150,110 L130,145" stroke="{frame_color}" stroke-width="5" fill="none" />
  <path d="M150,110 L170,85" stroke="{frame_color}" stroke-width="5" fill="none" />
  <circle cx="110" cy="145" r="3" fill="{self.colors["black"]}" />
  <circle cx="190" cy="145" r="3" fill="{self.colors["black"]}" />
  <circle cx="150" cy="110" r="3" fill="{self.colors["black"]}" />
  <line x1="170" y1="85" x2="180" y2="70" stroke="{frame_color}" stroke-width="5" />
  <line x1="180" y1="70" x2="190" y2="70" stroke="{self.colors["black"]}" stroke-width="3" />\n'''
            
        else:
            # Draw a generic vehicle
            vehicle_color = self._get_color(colors[0])
            
            svg += f'''  <rect x="100" y="130" width="100" height="20" rx="5" fill="{vehicle_color}" />
  <circle cx="120" cy="150" r="15" fill="{self.colors["black"]}" />
  <circle cx="180" cy="150" r="15" fill="{self.colors["black"]}" />\n'''
        
        svg += '</svg>'
        return svg
    
    def _generate_default(self, scene_info: Dict) -> str:
        """Generate a default scene when no specific type is detected"""
        colors = scene_info.get("colors", ["blue", "green", "red"])
        objects = scene_info.get("objects", [])
        
        # Start SVG
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        
        # Add background
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{self.colors["white"]}" />\n'
        
        # Add a gradient for visual interest
        svg += f'''  <defs>
    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{self._get_color(colors[0])}" />
      <stop offset="100%" stop-color="{self._get_color(colors[-1])}" />
    </linearGradient>
  </defs>\n'''
        
        # Draw basic shapes based on the objects mentioned
        shape_positions = [(100, 100), (200, 100), (150, 150)]
        
        for i, obj in enumerate(objects[:3]):
            if i >= len(shape_positions):
                break
                
            x, y = shape_positions[i]
            color = self._get_color(colors[i % len(colors)])
            
            if obj == "circle":
                svg += f'  <circle cx="{x}" cy="{y}" r="40" fill="{color}" />\n'
            elif obj == "square":
                svg += f'  <rect x="{x-30}" y="{y-30}" width="60" height="60" fill="{color}" />\n'
            elif obj == "triangle":
                svg += f'  <polygon points="{x},{y-35} {x+40},{y+35} {x-40},{y+35}" fill="{color}" />\n'
            elif obj == "star":
                svg += self._draw_star(x, y, 40, color)
            elif obj == "heart":
                svg += self._draw_heart(x, y, 40, color)
            else:
                # For unrecognized objects, just draw a circle
                svg += f'  <circle cx="{x}" cy="{y}" r="40" fill="{color}" />\n'
        
        # If no specific objects were mentioned, create a balanced composition
        if not objects:
            svg += f'  <rect x="50" y="50" width="200" height="100" fill="url(#gradient1)" />\n'
            svg += f'  <circle cx="100" cy="100" r="40" fill="{self._get_color(colors[0])}" opacity="0.7" />\n'
            
            if len(colors) > 1:
                svg += f'  <rect x="150" y="70" width="80" height="80" fill="{self._get_color(colors[1])}" opacity="0.7" />\n'
            
            if len(colors) > 2:
                svg += f'  <polygon points="120,150 180,150 150,90" fill="{self._get_color(colors[2])}" opacity="0.7" />\n'
        
        svg += '</svg>'
        return svg
    
    def _generate_fallback_svg(self, error_msg: str) -> str:
        """Generate a simple fallback SVG in case of errors"""
        svg = f'<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">\n'
        svg += f'  <rect x="0" y="0" width="300" height="200" fill="{self.colors["white"]}" />\n'
        svg += f'  <circle cx="150" cy="100" r="50" fill="{self.colors["blue"]}" opacity="0.5" />\n'
        svg += f'  <rect x="100" y="70" width="100" height="60" fill="{self.colors["red"]}" opacity="0.5" />\n'
        svg += '</svg>'
        return svg

    # Helper methods
    def _initialize_shape_generators(self):
        """Initialize shape generator methods"""
        self.shape_generators = {
            "circle": self._draw_circle,
            "square": self._draw_square,
            "triangle": self._draw_triangle,
            "rectangle": self._draw_rectangle,
            "star": self._draw_star,
            "heart": self._draw_heart,
            "cloud": self._draw_cloud,
            "tree": self._draw_tree,
            "flower": self._draw_flower
        }
    
    def _get_color(self, color_name: str) -> str:
        """Get color hex code from name"""
        return self.colors.get(color_name, self.colors["blue"])
    
    def _get_color_from_list(self, color_list: List[str], preferred_colors: List[str], default: str) -> str:
        """Get the first matching color from preferences"""
        for preferred in preferred_colors:
            if preferred in color_list:
                return self.colors[preferred]
        
        # If no preferred color found, use the first available or default
        if color_list:
            return self.colors.get(color_list[0], self.colors[default])
        return self.colors[default]
    
    def _lighten_color(self, color: str) -> str:
        """Lighten a hex color"""
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            # Lighten
            r = min(255, r + 40)
            g = min(255, g + 40)
            b = min(255, b + 40)
            
            return f'#{r:02x}{g:02x}{b:02x}'
        return color
    
    def _darken_color(self, color: str) -> str:
        """Darken a hex color"""
        if color.startswith('#'):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            
            # Darken
            r = max(0, r - 40)
            g = max(0, g - 40)
            b = max(0, b - 40)
            
            return f'#{r:02x}{g:02x}{b:02x}'
        return color
    
    def _draw_circle(self, x: int, y: int, radius: int, color: str) -> str:
        """Draw a circle"""
        return f'  <circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" />\n'
    
    def _draw_square(self, x: int, y: int, size: int, color: str) -> str:
        """Draw a square"""
        return f'  <rect x="{x-size/2}" y="{y-size/2}" width="{size}" height="{size}" fill="{color}" />\n'
    
    def _draw_rectangle(self, x: int, y: int, width: int, height: int, color: str) -> str:
        """Draw a rectangle"""
        return f'  <rect x="{x-width/2}" y="{y-height/2}" width="{width}" height="{height}" fill="{color}" />\n'
    
    def _draw_triangle(self, x: int, y: int, size: int, color: str) -> str:
        """Draw a triangle"""
        return f'  <polygon points="{x},{y-size/2} {x+size/2},{y+size/2} {x-size/2},{y+size/2}" fill="{color}" />\n'
    
    def _draw_star(self, x: int, y: int, size: int, color: str) -> str:
        """Draw a five-pointed star"""
        points = []
        for i in range(10):
            # Alternate between outer and inner radius points
            angle = i * math.pi / 5
            radius = size if i % 2 == 0 else size * 0.4
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append(f"{px},{py}")
        
        return f'  <polygon points="{" ".join(points)}" fill="{color}" />\n'
    
    def _draw_heart(self, x: int, y: int, size: int, color: str) -> str:
        """Draw a heart shape"""
        return f'''  <path d="M{x},{y+size*0.4} C{x-size*0.8},{y-size*0.5} {x-size*0.2},{y-size*0.8} {x},{y-size*0.2} C{x+size*0.2},{y-size*0.8} {x+size*0.8},{y-size*0.5} {x},{y+size*0.4}" fill="{color}" />\n'''
    
    def _draw_cloud(self, x: int, y: int) -> str:
        """Draw a cloud"""
        cloud_color = self.colors["white"]
        return f'''  <circle cx="{x}" cy="{y}" r="20" fill="{cloud_color}" />
  <circle cx="{x+15}" cy="{y-10}" r="15" fill="{cloud_color}" />
  <circle cx="{x-15}" cy="{y-5}" r="15" fill="{cloud_color}" />
  <circle cx="{x+10}" cy="{y+8}" r="15" fill="{cloud_color}" />
  <circle cx="{x-10}" cy="{y+5}" r="17" fill="{cloud_color}" />\n'''
    
    def _draw_tree(self, x: int, y: int, size: int) -> str:
        """Draw a tree"""
        trunk_color = self.colors["brown"]
        leaves_color = self.colors["forestgreen"]
        
        return f'''  <rect x="{x-size/4}" y="{y-size}" width="{size/2}" height="{size}" fill="{trunk_color}" />
  <circle cx="{x}" cy="{y-size*1.5}" r="{size}" fill="{leaves_color}" />
  <circle cx="{x-size*0.7}" cy="{y-size*1.1}" r="{size*0.7}" fill="{leaves_color}" />
  <circle cx="{x+size*0.7}" cy="{y-size*1.1}" r="{size*0.7}" fill="{leaves_color}" />\n'''
    
    def _draw_flower(self, x: int, y: int, color: str) -> str:
        """Draw a simple flower"""
        petal_color = self._get_color(color)
        center_color = self.colors["yellow"]
        stem_color = self.colors["green"]
        
        stem_height = random.randint(30, 50)
        petal_size = random.randint(8, 12)
        
        flower = f'  <line x1="{x}" y1="{y}" x2="{x}" y2="{y-stem_height}" stroke="{stem_color}" stroke-width="2" />\n'
        
        # Add leaves
        leaf_y = y - random.randint(10, stem_height-15)
        leaf_direction = -1 if random.random() > 0.5 else 1
        flower += f'  <path d="M{x},{leaf_y} C{x+10*leaf_direction},{leaf_y-5} {x+15*leaf_direction},{leaf_y+5} {x},{leaf_y+5}" fill="{stem_color}" />\n'
        
        # Add petals
        for i in range(5):
            angle = i * 2 * math.pi / 5
            px = x + petal_size * 1.5 * math.cos(angle)
            py = (y - stem_height) + petal_size * 1.5 * math.sin(angle)
            flower += f'  <circle cx="{px}" cy="{py}" r="{petal_size}" fill="{petal_color}" />\n'
        
        # Add center
        flower += f'  <circle cx="{x}" cy="{y-stem_height}" r="{petal_size/2}" fill="{center_color}" />\n'
        
        return flower
    
    def _optimize_svg(self, svg_code: str) -> str:
        """Optimize SVG code to reduce size and ensure compliance with competition constraints"""
        # Check if SVG has become too large
        while len(svg_code.encode('utf-8')) > self.max_svg_size:
            # First attempt: Remove whitespace and newlines
            svg_code = re.sub(r'\s+', ' ', svg_code)
            svg_code = re.sub(r'> <', '><', svg_code)
            
            if len(svg_code.encode('utf-8')) <= self.max_svg_size:
                break
            
            # Second attempt: Reduce precision of floating point numbers
            svg_code = re.sub(r'(\d+\.\d{2})\d*', r'\1', svg_code)
            
            if len(svg_code.encode('utf-8')) <= self.max_svg_size:
                break
            
            # Third attempt: Start removing elements from the end (least important first)
            element_pattern = r'<(circle|rect|polygon|ellipse|path|line)[^>]+/>'
            match = re.search(element_pattern, svg_code)
            if match:
                svg_code = re.sub(element_pattern, '', svg_code, count=1)
            else:
                # Final attempt: Just create a minimal valid SVG
                svg_code = '<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="300" height="200" fill="#FFFFFF"/></svg>'
                break
        
        return svg_code

# @title Test the Model with some sample descriptions
# Here we'll test our model with some example descriptions

from kaggle_evaluation import test

if __name__ == "__main__":
    # This line is required for Kaggle competition packaging
    test(Model)

