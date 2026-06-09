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


#| default_exp core


#| export
# Allows passing of context between painters.
class SceneContext:
    def __init__(self, prompt):
        self.prompt = prompt.lower()
        self.color = None
        self.shape = None
        self.svg_elements = []


#| export
# Base painter class
class Painter:
    def apply(self, context: SceneContext):
        """Modify the context in-place with painter-specific logic."""
        pass


#| export
class PainterColor(Painter):
    # Full list of all 147 CSS/SVG color names (lowercase)
    COLORS = {
        "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond", 
        "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue", 
        "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkgrey", 
        "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", 
        "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", 
        "deepskyblue", "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", 
        "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green", "greenyellow", "grey", "honeydew", "hotpink", 
        "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", 
        "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey", "lightpink", 
        "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue", 
        "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", 
        "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", 
        "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", 
        "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", 
        "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple", 
        "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", 
        "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", 
        "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"
    }

    def extract_color(self, prompt: str) -> str:
        for word in prompt.split():
            if word in self.COLORS:
                return word
        return "gray"

    def apply(self, context: SceneContext):
        context.color = self.extract_color(context.prompt)
        # No drawing here — that happens after painter chain finishes


#Dataset created for use in next code cell's visualization script. No functional fine-tuning application, unless your LLM can interpret color vibes.
from datasets import Dataset

def generate_unique_color_prompt_svg_pairs():
    painter = PainterColor()
    examples = []

    for color in sorted(PainterColor.COLORS):
        prompt = f"a {color} object"
        context = SceneContext(prompt)
        painter.apply(context)

        # Intentionally no fallback shape
        svg = "<svg width='200' height='200'>" + "".join(context.svg_elements) + "</svg>"
        examples.append({"prompt": prompt, "svg": svg})

    return examples

# Build the dataset
examples = generate_unique_color_prompt_svg_pairs()
color_dataset = Dataset.from_list(examples)


#Uses temporary fallback to visalize color
from IPython.display import SVG, display

for i in range(5):
    ex = color_dataset[i]
    prompt = ex["prompt"]
    context = SceneContext(prompt)
    PainterColor().apply(context)

    # TEMPORARY fallback just for inspection
    x, y, r = 100, 100, 30
    color = context.color
    circle = f"<circle cx='{x}' cy='{y}' r='{r}' fill='{color}' />"
    svg = f"<svg width='200' height='200'>{circle}</svg>"

    print(f"{prompt} → {color}")
    display(SVG(svg))


#| export
# Defines basic shapes, applies context.color from SceneContext to call color fill
class PainterShape(Painter):
    SHAPES = ["circle", "square", "rectangle", "triangle", "ellipse", "line"]

    def extract_shape(self, prompt: str) -> str:
        for shape in self.SHAPES:
            if shape in prompt:
                return shape
        return None

    def apply(self, context: SceneContext):
        shape = self.extract_shape(context.prompt)
        if not shape:
            return

        context.shape = shape  # Now paintercolor can avoid double-rendering

        x, y = 100, 100
        color = context.color or "gray"

        if shape == "circle":
            svg_element = f"<circle cx='{x}' cy='{y}' r='30' fill='{color}' />"
        elif shape == "square":
            svg_element = f"<rect x='{x - 15}' y='{y - 15}' width='30' height='30' fill='{color}' />"
        elif shape == "rectangle":
            svg_element = f"<rect x='{x - 20}' y='{y - 10}' width='40' height='20' fill='{color}' />"
        elif shape == "triangle":
            points = f"{x},{y-30} {x-25},{y+15} {x+25},{y+15}"
            svg_element = f"<polygon points='{points}' fill='{color}' />"
        elif shape == "ellipse":
            svg_element = f"<ellipse cx='{x}' cy='{y}' rx='30' ry='15' fill='{color}' />"
        elif shape == "line":
            x1, y1 = x - 30, y
            x2, y2 = x + 30, y
            svg_element = f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='4' />"
        else:
            return

        context.svg_elements.append(svg_element)


# Funciton to generates dataset primitives for fine-tuning
def generate_color_shape_prompt_svg_pairs():
    painter_color = PainterColor()
    painter_shape = PainterShape()
    examples = []

    for color in sorted(PainterColor.COLORS):
        for shape in sorted(PainterShape.SHAPES):
            prompt = f"a {color} {shape}"
            context = SceneContext(prompt)
            painter_color.apply(context)
            painter_shape.apply(context)
            svg = "<svg width='200' height='200'>" + "".join(context.svg_elements) + "</svg>"
            examples.append({"prompt": prompt, "svg": svg})

    return examples

# Build the dataset
examples = generate_color_shape_prompt_svg_pairs()
color_shape_dataset = Dataset.from_list(examples)


# Displays a sample of shape/color primitives
from IPython.display import SVG, display
import random

examples = random.sample(list(color_shape_dataset), 5)

for ex in examples:
    print(ex["prompt"])
    display(SVG(ex["svg"]))


#| export
# Theoretical application for employing collaborative painterchain. Needs refinement to incorporate additional painters. #Comment out if adding additional painters without refinement to prevent issues.
def route_prompt(context: SceneContext) -> list[str]:
    prompt = context.prompt  # already lowercased by SceneContext
    active_painters = []

    # Color painter always active (no keyword needed)
    active_painters.append("color")

    # Shape detection based on known SVG-compatible keywords
    if any(word in prompt for word in [
        "circle", "square", "rectangle", "triangle", "ellipse", "line"
    ]):
        active_painters.append("shapes")

    return active_painters


#| export
# PainterChain rendering sequence to sequence overlays (no future backgrounds over foreground shapes)
class PainterChain:
    def __init__(self):
        self.registry = {
            "shapes": PainterShape(),
            "color": PainterColor(),  # still must be in registry
        }
    def render(self, prompt: str) -> str:
        context = SceneContext(prompt)

        for key in route_prompt(context):
            painter = self.registry.get(key)
            if painter:
                painter.apply(context)

        # Final fallback: if no shape rendered, add a color-only circle
        if context.shape is None and context.color:
            x, y, r = 100, 100, 30
            svg_element = f"<circle cx='{x}' cy='{y}' r='{r}' fill='{context.color}' />"
            context.svg_elements.append(svg_element)

        return f'<svg width="300" height="300">{"".join(context.svg_elements)}</svg>'


#| export
# Substantiate model for use as fallback or improved baseline
class Model:
    def __init__(self):
        self.painters = PainterChain()

    def predict(self, prompt: str) -> str:
        return self.painters.render(prompt)


import kaggle_evaluation
kaggle_evaluation.test(Model)

