#| default_exp core


#| export

import re
import random

class Model:
    def __init__(self):
        pass  # No need for online models, runs fully offline

    def extract_features(self, prompt: str):
        """
        Extracts objects, colors, sizes and pos using simple pattern matching.
        """
        objects = re.findall(r"\b(\w+)\b", prompt.lower())  # Extract all words as potential objects
        colors = re.findall(r"(gold|silver|red|blue|green|yellow|black|white|gray|purple|orange)", prompt, re.IGNORECASE) #Detects color names in the prompt ignoring case
        sizes = re.findall(r"(small|medium|large|tiny|huge|gigantic)", prompt, re.IGNORECASE) # Detects size-related adjectives in the prompt ignoring case
        positions = re.findall(r"at (\d+,\d+)", prompt) # Extracts position coordinates (x, y) if specified in the format "at x,y"

        return {"objects": list(set(objects)), "colors": colors, "sizes": sizes, "positions": positions} # Returning Extracted Features in a dictionary

    def generate_svg_element(self, obj_name, color, size, position):
        """
        Dynamically generates an SVG shape for a given object.
        """
        x, y = position if position else (random.randint(100, 800), random.randint(100, 800))
        size_map = {"small": 30, "medium": 60, "large": 100}
        size_value = size_map.get(size, 50)  # Default size if not specified
        shape_type = random.choice(["circle", "rectangle", "triangle"])  # Random shape mapping
        
        if shape_type == "circle":
            return f'<circle cx="{x}" cy="{y}" r="{size_value}" fill="{color}" opacity="0.8"/>'
        elif shape_type == "rectangle":
            return f'<rect x="{x}" y="{y}" width="{size_value * 2}" height="{size_value}" fill="{color}" opacity="0.8"/>'
        elif shape_type == "triangle":
            return f'<polygon points="{x},{y} {x+size_value},{y+size_value} {x-size_value},{y+size_value}" fill="{color}" opacity="0.8"/>'

    def generate_svg(self, features):
        """
        Combines multiple extracted features and generates SVG code.
        Also, if no colors/sizes/positions are provided, it randomizes them.
        """
        svg_elements = ['<rect width="1000" height="1000" fill="lightgray"/>']  # Background

        for obj in features["objects"]:
            color = random.choice(features["colors"]) if features["colors"] else "black"
            size = random.choice(features["sizes"]) if features["sizes"] else "medium"
            position = random.choice(features["positions"]) if features["positions"] else None
            svg_elements.append(self.generate_svg_element(obj, color, size, position))

        return f"""
        <svg width="1000" height="1000" viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg">
          {''.join(svg_elements)}
        </svg>
        """

    def predict(self, prompt: str) -> str:
        """
        Generates SVG code dynamically for any input prompt.
        """
        features = self.extract_features(prompt)
        return self.generate_svg(features)


from IPython.display import SVG

model = Model()
svg = model.predict('a goose winning a gold medal')

print(svg)
display(SVG(svg))


import kaggle_evaluation
kaggle_evaluation.test(Model)

