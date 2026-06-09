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


import requests
import pathlib
from PIL import Image


image = Image.open("/kaggle/input/imagine-test/palton.png")
image.thumbnail([512,512])
display(image)





!pip3 install google_genai

from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_ID = "models/gemini-2.0-flash-exp"

# Generate content
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?"
)

print(response.text)






from IPython.display import display, Markdown
image = Image.open("/kaggle/input/imagine-test/__results___35_3.png")
image.thumbnail([512,512])

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "Write a short and engaging blog post based on this picture."
    ]
)

display(image)
Markdown(response.text)


response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "describe the image."
    ]
)
display(image)
Markdown(response.text)


response.text


response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "ddescribe the image as follow: decompose the image in geometrical shapes. approximate shapes if there are too complex shape, describe the image in cartesian coordinates, for a complex closed grey poligon 0,0; 0,800;800,800 ...etc"
    ]
)
display(image)
Markdown(response.text)


response.text




