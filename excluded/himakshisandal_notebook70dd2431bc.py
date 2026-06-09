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


import pandas as pd
from IPython.display import SVG, display


train_df = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

train_df


descriptions = train_df['description']

print(descriptions)


def process_descriptions(descriptions):
    # Example: Just print out each description for now
    for idx, description in enumerate(descriptions):
        print(f"Description {idx + 1}: {description}")


process_descriptions(descriptions)


def generate_svg(description):
    if description == "a starlit night over snow-covered peaks":
        return """
        <svg width="200" height="200">
            <rect width="200" height="200" fill="black"/>
            <circle cx="50" cy="50" r="5" fill="yellow"/>
            <circle cx="60" cy="60" r="5" fill="yellow"/>
            <polygon points="50,150 100,50 150,150" fill="white"/>
        </svg>
        """
    elif description == "black and white checkered pants":
        return """
        <svg width="200" height="200">
            <rect width="200" height="200" fill="white"/>
            <rect width="100" height="100" fill="black"/>
            <rect x="100" width="100" height="100" fill="white"/>
            <rect y="100" width="100" height="100" fill="black"/>
            <rect x="100" y="100" width="100" height="100" fill="white"/>
        </svg>
        """
    elif description == "crimson rectangles forming a chaotic grid":
        return """
        <svg width="200" height="200">
            """ + ''.join([f"<rect x='{i*30}' y='{j*30}' width='30' height='30' fill='crimson' />" for i in range(7) for j in range(7)]) + """
        </svg>
        """
    elif description == "burgundy corduroy pants with patch pockets and silver buttons":
        return """
        <svg width="200" height="200">
            <rect width="100" height="200" fill="burgundy"/>
            <circle cx="80" cy="50" r="5" fill="silver"/>
            <circle cx="80" cy="70" r="5" fill="silver"/>
        </svg>
        """
    elif description == "orange corduroy overalls":
        return """
        <svg width="200" height="200">
            <rect x="50" y="50" width="100" height="150" fill="orange"/>
            <line x1="50" y1="50" x2="100" y2="20" stroke="orange" stroke-width="10"/>
            <line x1="150" y1="50" x2="100" y2="20" stroke="orange" stroke-width="10"/>
        </svg>
        """
    elif description == "a lighthouse overlooking the ocean":
        return """
        <svg width="200" height="200">
            <rect x="80" y="50" width="40" height="100" fill="white"/>
            <circle cx="100" cy="50" r="20" fill="white"/>
            <polygon points="50,150 150,150 100,200" fill="blue"/>
        </svg>
        """
    elif description == "a green lagoon under a cloudy sky":
        return """
        <svg width="200" height="200">
            <rect width="200" height="100" fill="lightblue"/>
            <circle cx="50" cy="50" r="30" fill="green"/>
            <circle cx="100" cy="60" r="40" fill="green"/>
            <circle cx="150" cy="50" r="30" fill="green"/>
        </svg>
        """
    elif description == "a snowy plain":
        return """
        <svg width="200" height="200">
            <rect width="200" height="200" fill="white"/>
            <polygon points="0,100 200,100 200,200 0,200" fill="lightgray"/>
        </svg>
        """
    elif description == "a maroon dodecahedron interwoven with teal threads":
        return """
        <svg width="200" height="200">
            <polygon points="100,50 120,30 140,50 120,70" fill="maroon"/>
            <polygon points="100,50 80,30 60,50 80,70" fill="teal"/>
        </svg>
        """
    elif description == "a purple silk scarf with tassel trim":
        return """
        <svg width="200" height="200">
            <rect x="50" y="50" width="100" height="20" fill="purple"/>
            <rect x="50" y="70" width="20" height="60" fill="purple"/>
            <rect x="130" y="70" width="20" height="60" fill="purple"/>
        </svg>
        """
    elif description == "magenta trapezoids layered on a translucent silver sheet":
        return """
        <svg width="200" height="200">
            <polygon points="50,100 150,100 180,150 20,150" fill="magenta" opacity="0.5"/>
            <rect x="50" y="50" width="100" height="100" fill="silver" opacity="0.5"/>
        </svg>
        """
    elif description == "gray wool coat with a faux fur collar":
        return """
        <svg width="200" height="200">
            <rect x="60" y="50" width="80" height="100" fill="gray"/>
            <circle cx="100" cy="60" r="5" fill="white"/>
            <circle cx="100" cy="70" r="5" fill="white"/>
        </svg>
        """
    elif description == "a purple forest at dusk":
        return """
        <svg width="200" height="200">
            <rect width="200" height="200" fill="purple"/>
            <circle cx="40" cy="60" r="20" fill="green"/>
            <circle cx="60" cy="80" r="20" fill="green"/>
        </svg>
        """
    elif description == "purple pyramids spiraling around a bronze cone":
        return """
        <svg width="200" height="200">
            <polygon points="50,150 100,50 150,150" fill="purple"/>
            <polygon points="60,150 100,60 140,150" fill="brown"/>
        </svg>
        """
    elif description == "khaki triangles and azure crescents":
        return """
        <svg width="200" height="200">
            <polygon points="50,150 100,50 150,150" fill="khaki"/>
            <path d="M 100 50 A 50 50 0 0 1 150 50" fill="azure"/>
        </svg>
        """
    else:
        return """
        <svg width="200" height="200">
            <rect width="200" height="200" fill="gray"/>
        </svg>
        """


# Step 3: Function to display all SVGs based on descriptions
def process_descriptions(descriptions):
    for idx, description in enumerate(descriptions):
        print(f"Description {idx + 1}: {description}")
        
        # Generate SVG based on the description
        svg_code = generate_svg(description)
        
        # Display the generated SVG image in the notebook
        display(SVG(svg_code))


# Step 4: Process and display all descriptions from the CSV
process_descriptions(train_df['description'])

