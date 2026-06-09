# Importing Libraries
import warnings
warnings.filterwarnings("ignore")
import sys
import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from glob import glob 
from tqdm import tqdm
import IPython
from IPython.display import display


print(sys.version)
modules = [
    ("numpy", np),
    ("pandas", pd),
    ("seaborn", sns),
    ("matplotlib", matplotlib),
    ("IPython", IPython),
]

for name, module in modules:
    print(f"{name}: {module.__version__}")




