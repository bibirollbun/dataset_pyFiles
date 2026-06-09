
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import plotly
plotly.offline.init_notebook_mode(connected=True)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#https://www.kaggle.com/competitions/drawing-with-llms/data

"""Gateway notebook for SVG Image Generation"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType, IS_RERUN
import kaggle_evaluation.core.templates
from kaggle_evaluation.svg_constraints import SVGConstraints


## Installing Dependencies I still need to install CairoSVG.

!pip install cairosvg


#| export

#`#| export` tag above which means all code in this cell will be exported to your Package.
# Make sure to `import` any python packages required by your Model, either here or in other exported cells.

import concurrent
import io
import logging
import re

import cairosvg
import kagglehub
import torch
from lxml import etree
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


#| default_exp core


#| export

#D. J Sterling https://www.kaggle.com/code/dster/drawing-with-llms-starter-notebook

#The vertical bar + export should be the 1st line

# NOTE the special `#| export` tag above which means all code in this cell will be exported to your Package.
# Make sure to `import` any python packages required by your Model, either here or in other exported cells.

class Model:
    def __init__(self):
        '''Optional constructor, performs any setup logic, model instantiation, etc.'''
        pass
      
    def predict(self, prompt: str) -> str:
        '''Generates SVG which produces an image described by the prompt.

        Args:
            prompt (str): A prompt describing an image
        Returns:
            String of valid SVG code.
        '''
        # Renders a simple circle regardless of input
        return '<svg width="200" height="200" viewBox="0 0 100 100"><circle cx="100" cy="100" r="50" fill="LightPink" /></svg>'


#D J. Sterling https://www.kaggle.com/code/dster/drawing-with-llms-starter-notebook

# We can play with our Model and render its SVG output (don't export!)

from IPython.display import SVG

model = Model()
svg = model.predict('A Unicorn Goose Horn')

print(svg)
display(SVG(svg))


train = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')
train.head(15)


test = pd.read_csv('/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv')
test.head(15)

