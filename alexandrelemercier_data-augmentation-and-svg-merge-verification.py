import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from IPython.display import SVG, display
import os


# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

# Set the path to the file you'd like to load
file_path = "train.json"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "alexandrelemercier/visual-scene-instructions-for-generative-llms",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:")
df.head()


df2 = df.description
df2.shape


df2.to_csv("instructions.csv")


directory = "/kaggle/working/"
files = os.listdir(directory)
files


file_path = "combined_train.json"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "alexandrelemercier/visual-scene-instructions-for-generative-llms",
  file_path
)

df.head()


def add_background_if_white(svg_str, bg_color="lightgray", opacity=0.5):
    # Check for white fill attributes
    if "fill='white'" in svg_str or 'fill="white"' in svg_str:
        # Find the end of the opening <svg> tag
        idx = svg_str.find('>')
        if idx != -1:
            background_rect = f"<rect x='0' y='0' width='500' height='300' fill='{bg_color}' opacity='{opacity}'/>"
            # Insert the background rectangle immediately after the <svg> tag
            svg_str = svg_str[:idx+1] + background_rect + svg_str[idx+1:]
    return svg_str

# Loop through the first 10 SVG images
for k in range(df.shape[0]):
    print(f"N°{k}")
    print(df.concept[k].upper())
    print(df.description[k])
    # Modify the SVG code if white elements are present
    svg_modified = add_background_if_white(df.svg[k])
    display(SVG(svg_modified))


