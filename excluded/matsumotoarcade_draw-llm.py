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
"""
This is the core module of the submission package.
"""

#| export
import pandas as pd
import re
import os
from defusedxml import ElementTree as etree
from dataclasses import dataclass, field
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from IPython.display import HTML, display

# Add any other necessary imports here
from kaggle_evaluation.svg_constraints import SVGConstraints as SVGConstraintsLib

#| export
@dataclass(frozen=True)
class SVGConstraints:
    allowed_shapes: set[str] = field(
        default_factory=lambda: {
            "polygon", "arc", "circle", "square", "triangle", "rectangle", "line", "ellipse", "prism", "parallelogram", "shape", "rhombus", "oval", "cube", "sphere", "pyramid", "cone", "cylinder", "spiral", "curve", "dot", "cross", "crescent", "abstract", "geometric", "organic", "linear", "clothing", "food", "color"
        }
    )

    def validate_shape(self, shape: str) -> bool:
        return shape.lower() in self.allowed_shapes

#| export
class Model:  # IMPORTANT: Class name must be "Model"
    def __init__(self):
        # Define paths based on Kaggle environment
        self.MODEL_PATH = './bart-large-mnli'  # Relative path for local use
        if not os.path.exists(self.MODEL_PATH): #Download model is not available in Kaggle
            try:
                model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
                os.makedirs(self.MODEL_PATH, exist_ok=True) # Ensure directory exists
                model.save_pretrained(self.MODEL_PATH)
                tokenizer.save_pretrained(self.MODEL_PATH)
                print("Model downloaded and saved!")

            except Exception as e:
                print(f"ERROR: Failed to download the model: {e}")
                self.classifier = None
                return # Exit if we cannot load the model
        else:
            print('Model Already Exists')

        try:
            self.classifier = pipeline("zero-shot-classification", model=self.MODEL_PATH, tokenizer=self.MODEL_PATH, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"ERROR: Failed to load pipeline from local model: {e}")
            self.classifier = None # Handle this classifier being None in the predict function

    def extract_shape(self, description):
        shape_keywords = ["polygon", "arc", "circle", "square", "triangle", "rectangle", "line", "ellipse", "prism", "parallelogram", "shape", "rhombus", "oval", "cube", "sphere", "pyramid", "cone", "cylinder", "spiral", "curve", "dot", "cross", "crescent"]
        clothing_keywords = ["dress", "shirt", "pants", "skirt", "coat", "sweater", "blouse"]
        food_keywords = ["apple", "banana", "bread", "cake", "pizza", "sandwich", "cookie"]
        color_keywords = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "gray", "brown"]

        for keyword in shape_keywords:
            if keyword in description.lower():
                return keyword

        for keyword in clothing_keywords:
            if keyword in description.lower():
                return "clothing"

        for keyword in food_keywords:
            if keyword in description.lower():
                return "food"

        for keyword in color_keywords:
            if keyword in description.lower():
                return "color"

        match = re.search(r"(\d+)-sided", description.lower())
        if match:
            return f"{match.group(1)}-sided shape"

        if "abstract" in description.lower():
            return "abstract"
        if "geometric" in description.lower():
            return "geometric"
        if "organic" in description.lower():
            return "organic"
        if "linear" in description.lower():
            return "linear"

        candidate_labels = ["polygon", "arc", "circle", "square", "triangle", "rectangle", "line", "ellipse", "clothing", "food", "color", "abstract", "geometric", "organic", "linear", "no shape"]

        if self.classifier:  # Only try to classify if the classifier was loaded successfully
            try:
                result = self.classifier(description, candidate_labels=candidate_labels)
                gemini_shape = result['labels'][0]
                if gemini_shape != "no shape":
                    return gemini_shape
            except Exception as e:
                print(f"Local LLM error: {e}")
                return "no shape"
        else:
            print("Classifier not loaded, returning default 'No shape identified'")
            return "no shape" #Or another appropriate default if the LLM is unavailable

        return "no shape"

    #| export
    def predict(self, descriptions: pd.Series) -> pd.Series:  # IMPORTANT: Method name must be "predict"
        """
        Generates SVG predictions for the given descriptions.

        Parameters
        ----------
        descriptions : pd.Series
            A pandas Series containing the 'description' strings.

        Returns
        -------
        pd.Series
            A pandas Series with the generated SVGs.
        """

        extracted_shapes = descriptions.apply(self.extract_shape)

        # Apply Validation against allowed shape words using new Dataclass
        svg_constraints = SVGConstraints()
        is_valid_shape = extracted_shapes.apply(svg_constraints.validate_shape)


        # Create SVG code based on the shape
        def create_svg(shape):
            shape = shape.lower()
            if shape == "circle":
                return f'<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red" /></svg>'
            elif shape == "square":
                return f'<svg width="100" height="100"><rect width="80" height="80" x="10" y="10" fill="blue" /></svg>'
            elif shape == "rectangle":
                return f'<svg width="150" height="100"><rect width="130" height="80" x="10" y="10" fill="green" /></svg>'
            elif shape == "triangle":
                return f'<svg width="100" height="100"><polygon points="50,10 90,90 10,90" fill="yellow" /></svg>'
            elif shape == "clothing":
                return f'<svg width="100" height="100"><rect width="100" height="100" fill="pink" /></svg>'
            elif shape == "food":
                return f'<svg width="100" height="100"><rect width="100" height="100" fill="brown" /></svg>'
            elif shape == "color":
                return f'<svg width="100" height="100"><rect width="100" height="100" fill="currentColor" /></svg>'
            elif shape == "abstract":
                return f'<svg width="100" height="100"><polygon points="20,20 80,20 50,80" fill="purple" /></svg>'
            elif shape == "geometric":
                return f'<svg width="100" height="100"><rect x="20" y="20" width="60" height="60" fill="orange" /></svg>'
            elif shape == "organic":
                 return f'<svg width="100" height="100"><ellipse cx="50" cy="50" rx="40" ry="20" fill="lime" /></svg>'
            elif shape == "linear":
                 return f'<svg width="100" height="100"><line x1="10" y1="10" x2="90" y2="90" stroke="black" stroke-width="3" /></svg>'
            elif shape == "polygon":
                 return f'<svg width="100" height="100"><polygon points="20,20 80,20 90,80 10,80" fill="skyblue" /></svg>'
            elif shape == "arc":
                return f'<svg width="100" height="100"><path d="M10 90 A 40 40 0 0 1 90 10" stroke="red" fill="none" stroke-width="3"/></svg>'
            elif shape == "shape":
                return f'<svg width="100" height="100"><rect x="10" y="10" width="80" height="80" fill="gray" /></svg>'
            elif shape == "prism":
                return f'<svg width="100" height="100"><polygon points="30,10 70,10 90,50 50,90 10,50" fill="teal" /></svg>'
            elif shape == "parallelogram":
                return f'<svg width="150" height="100"><polygon points="20,20 120,20 150,80 50,80" fill="coral" /></svg>'
            else:
                return f'<svg width="100" height="100"><rect x="0" y="0" width="100" height="100" fill="white" /></svg>'

        svgs = extracted_shapes.apply(create_svg)  # 'drawing' is the column name the test function expects
        return pd.DataFrame({'shape': extracted_shapes, 'svg': svgs})

#| export
def create_submission(test_file: str, submission_file: str):
    """
    Generates a submission file given a test file.

    Args:
        test_file (str): Path to the test CSV file.
        submission_file (str): Path to save the submission CSV file.
    """
    test_df = pd.read_csv(test_file)
    model = Model()
    results_df = model.predict(test_df['description'])
    test_df['shape'] = results_df['shape']
    test_df['drawing'] = results_df['svg']
    submission = test_df[['id', 'drawing']]
    submission.to_csv(submission_file, index=False)


# Example usage (assuming you have a test_df loaded):
if __name__ == '__main__':
    # Load test.csv, handling FileNotFoundError
    try:
        #test_df = pd.read_csv("test.csv")  # Adjust path if necessary for local testing
        test_df = pd.read_csv("/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv") #Kaggle Path
    except FileNotFoundError:
        print("Error: test.csv not found.  Make sure it is in the same directory as this script, or adjust the path.")
        test_df = None

    if test_df is not None:
        model = Model()
        results_df = model.predict(test_df['description'])

        def create_svg_from_predicted_shape(shape):
            """Creates an SVG string based directly on the predicted shape."""
            shape = shape.lower()
            if shape == "circle":
                svg_code = f'<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red" /></svg>'
            elif shape == "square":
                svg_code = f'<svg width="100" height="100"><rect width="80" height="80" x="10" y="10" fill="blue" /></svg>'
            elif shape == "rectangle":
                svg_code = f'<svg width="150" height="100"><rect width="130" height="80" x="10" y="10" fill="green" /></svg>'
            elif shape == "triangle":
                svg_code = f'<svg width="100" height="100"><polygon points="50,10 90,90 10,90" fill="yellow" /></svg>'
            elif shape == "clothing":
                svg_code = f'<svg width="100" height="100"><rect width="100" height="100" fill="pink" /></svg>'
            elif shape == "food":
                svg_code = f'<svg width="100" height="100"><rect width="100" height="100" fill="brown" /></svg>'
            elif shape == "color":
                svg_code = f'<svg width="100" height="100"><rect width="100" height="100" fill="currentColor" /></svg>'
            elif shape == "abstract":
                svg_code = f'<svg width="100" height="100"><polygon points="20,20 80,20 50,80" fill="purple" /></svg>'
            elif shape == "geometric":
                svg_code = f'<svg width="100" height="100"><rect x="20" y="20" width="60" height="60" fill="orange" /></svg>'
            elif shape == "organic":
                svg_code = f'<svg width="100" height="100"><ellipse cx="50" cy="50" rx="40" ry="20" fill="lime" /></svg>'
            elif shape == "linear":
                svg_code = f'<svg width="100" height="100"><line x1="10" y1="10" x2="90" y2="90" stroke="black" stroke-width="3" /></svg>'
            elif shape == "polygon":
                svg_code = f'<svg width="100" height="100"><polygon points="20,20 80,20 90,80 10,80" fill="skyblue" /></svg>'
            elif shape == "arc":
                svg_code = f'<svg width="100" height="100"><path d="M10 90 A 40 40 0 0 1 90 10" stroke="red" fill="none" stroke-width="3"/></svg>'
            elif shape == "shape":
                svg_code = f'<svg width="100" height="100"><rect x="10" y="10" width="80" height="80" fill="gray" /></svg>'
            elif shape == "prism":
                svg_code = f'<svg width="100" height="100"><polygon points="30,10 70,10 90,50 50,90 10,50" fill="teal" /></svg>'
            elif shape == "parallelogram":
                svg_code = f'<svg width="150" height="100"><polygon points="20,20 120,20 150,80 50,80" fill="coral" /></svg>'
            else:
                svg_code = f'<svg width="100" height="100"><rect x="0" y="0" width="100" height="100" fill="white" /></svg>'

            # Validate the generated SVG using SVGConstraintsLib
            svg_constraints = SVGConstraintsLib()
            try:
                svg_constraints.validate_svg(svg_code)
                return svg_code  # Return the SVG if it's valid
            except ValueError as e:
                print(f"SVG Validation Error for shape {shape}: {e}")
                return f'<svg width="100" height="100"><rect x="0" y="0" width="100" height="100" fill="white" /></svg>' # Return a default invalid SVG

        # Modify results_df to use create_svg_from_predicted_shape
        results_df['svg'] = results_df['shape'].apply(create_svg_from_predicted_shape)

        # Display the results in a dynamic table with SVG rendering
        def display_svg(svg_code):
             return svg_code

        def generate_html_table(df):
            html = "<table style='border-collapse: collapse; width: 100%;'>"
            # Table header
            html += "<thead><tr>"
            for column in df.columns:
                html += f"<th style='border: 1px solid black; padding: 8px; text-align: left;'>{column}</th>"
            html += "</tr></thead>"
            # Table body
            html += "<tbody>"
            for index, row in df.iterrows():
                html += "<tr>"
                for column in df.columns:
                    cell_value = row[column]
                    if column == 'svg':
                        html += f"<td style='border: 1px solid black; padding: 8px; text-align: left;'>{cell_value}</td>"
                    else:
                        html += f"<td style='border: 1px solid black; padding: 8px; text-align: left;'>{cell_value}</td>"
                html += "</tr>"
            html += "</tbody></table>"
            return html

        # Create a DataFrame for display
        display_df = pd.DataFrame({
            'id': test_df['id'],
            'description': test_df['description'],
            'predicted_shape': results_df['shape'],
            'svg': results_df['svg']
        })

        html_table = generate_html_table(display_df)
        display(HTML(html_table))  # Print the HTML for the table and render it.
        # print(display_df)
    else:
        print("Could not load test_df.  Exiting.")

