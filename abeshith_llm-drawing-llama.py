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


!pip install --target=/kaggle/input/svgwrite svgwrite


import os
print(os.listdir("/kaggle/working/"))


import importlib.util

spec = importlib.util.spec_from_file_location("svgwrite", "/kaggle/working/svgwrite/__init__.py")
svgwrite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svgwrite)


import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import svgwrite

# Add Kaggle evaluation package
sys.path.append("/kaggle/input/drawing-with-llms/kaggle_evaluation")  

from kaggle_evaluation import test 


import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")

os.environ["HF_TOKEN"] = hf_token


import os
print(os.listdir("/kaggle/working/llama_model"))


%time 
model = AutoModelForCausalLM.from_pretrained("/kaggle/working/llama_model")
tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/llama_model")

print("âœ… Model loaded from local storage!")


def generate_svg(description):
    """
    Uses LLaMA-2 to generate an SVG representation of the text description.
    """
    
    prompt = f"Generate a simple SVG image for: {description}\n\n<svg"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate SVG code
    with torch.no_grad():
        output = model.generate(**inputs, max_length=300, temperature=0.7)
    
    # Decode output
    svg_code = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Ensure valid SVG format
    if "<svg" not in svg_code:
        svg_code = f'<svg width="256" height="256"><text x="10" y="20" font-size="14">{description[:10]}</text></svg>'
    
    return svg_code



import random

class Model:
    def __init__(self):
        print("Model initialized!")

    def predict(self, input_data):
        print("\nğŸ“Œ Received input_data:", input_data)
        print("ğŸ”� Data Type:", type(input_data))

        # Ensure input_data is a string
        if not isinstance(input_data, str):
            print("â�Œ Error: input_data is not a string!")
            return "<svg></svg>"  # Return empty SVG as fallback

        # Generate a simple random shape based on input
        svg_output = self.generate_svg(input_data)

        print("âœ… Returning SVG:", svg_output)
        return svg_output

    def generate_svg(self, description):
        """Generates a simple SVG drawing based on input description."""
        width, height = 256, 256  # Fixed canvas size

        # Randomly select a shape to draw
        shape = random.choice(["circle", "rectangle", "ellipse", "path"])
        
        if shape == "circle":
            cx, cy, r = random.randint(50, 200), random.randint(50, 200), random.randint(20, 80)
            return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><circle cx="{cx}" cy="{cy}" r="{r}" fill="blue" /></svg>'
        
        elif shape == "rectangle":
            x, y, w, h = random.randint(20, 150), random.randint(20, 150), random.randint(50, 100), random.randint(50, 100)
            return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="red" /></svg>'
        
        elif shape == "ellipse":
            cx, cy, rx, ry = random.randint(50, 200), random.randint(50, 200), random.randint(30, 80), random.randint(20, 60)
            return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="green" /></svg>'
        
        else:  # Random path
            d = f'M {random.randint(20, 50)},{random.randint(20, 50)} L {random.randint(100, 200)},{random.randint(100, 200)}'
            return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><path d="{d}" stroke="black" stroke-width="3" fill="none"/></svg>'



# Load test dataset
test_df = pd.read_csv("/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv")

# Evaluate using Kaggle's system
test(Model)


import random

class Model:
    def __init__(self):
        print("Model initialized!")

    def predict(self, input_data):
        """Generate an SVG representation based on the input text description."""
        shapes = [
            '<rect x="{}" y="{}" width="{}" height="{}" fill="red" />',
            '<circle cx="{}" cy="{}" r="{}" fill="blue" />',
            '<ellipse cx="{}" cy="{}" rx="{}" ry="{}" fill="green" />',
            '<path d="M {},{} L {},{}" stroke="black" stroke-width="3" fill="none"/>'
        ]

        shape_template = random.choice(shapes)

        # Generate random numbers to place the shape
        values = [random.randint(20, 200) for _ in range(shape_template.count("{}"))]
        
        svg = f'<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">{shape_template.format(*values)}</svg>'
        return svg



# Create an instance of the model
model = Model()

# Example test cases
test_cases = [
    "an emerald lake beneath an overcast sky",
    "a beacon tower facing the sea",
    "chestnut ribbed pants with cargo pockets and pewter clasps"
]

# Run predictions
for test in test_cases:
    print(f"\nğŸ“Œ Input: {test}")
    svg_output = model.predict(test)
    print(f"âœ… Output SVG: {svg_output}")


from IPython.core.display import display, SVG

svg_list = [
    '<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg"><path d="M 156,84 L 73,191" stroke="black" stroke-width="3" fill="none"/></svg>',
    '<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg"><path d="M 182,62 L 198,177" stroke="black" stroke-width="3" fill="none"/></svg>',
    '<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg"><circle cx="93" cy="67" r="151" fill="blue" /></svg>'
]

for svg in svg_list:
    display(SVG(svg))


submission = []
for idx, row in test_df.iterrows():
    text_input = row['description']  # Assuming the column name is 'description'
    svg_output = model.predict(text_input)
    submission.append([row['id'], svg_output])


submission_df = pd.DataFrame(submission, columns=["id", "svg"])


submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")




