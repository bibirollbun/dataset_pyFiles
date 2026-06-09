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


!pip install svgwrite
import svgwrite
import random


class Model:
    def __init__(self):
        pass
    
    def predict(self, description: str) -> str:
        """
        Generates SVG code based on a given text description.
        """
        svg_code = """
        <svg width="500" height="500" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="white" />
        """
        
        # Simple shape generator based on keywords
        if "circle" in description.lower():
            svg_code += """<circle cx="250" cy="250" r="100" fill="blue" stroke="black" stroke-width="2" />"""
        if "square" in description.lower():
            svg_code += """<rect x="150" y="150" width="200" height="200" fill="red" stroke="black" stroke-width="2" />"""
        if "triangle" in description.lower():
            svg_code += """<polygon points="250,100 150,350 350,350" fill="green" stroke="black" stroke-width="2" />"""
        
        # Random extra decorations
        if random.random() > 0.5:
            svg_code += """<line x1="50" y1="50" x2="450" y2="450" stroke="black" stroke-width="3" />"""
        if random.random() > 0.5:
            svg_code += """<ellipse cx="250" cy="400" rx="60" ry="30" fill="purple" stroke="black" stroke-width="2" />"""
        
        svg_code += "</svg>"
        
        output_path = "/kaggle/working/generated_image.svg"
        with open(output_path, "w") as file:
            file.write(svg_code)
        
        return output_path





