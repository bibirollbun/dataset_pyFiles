!pip install --upgrade protobuf==4.25.6 tensorflow==2.17.0


import os
import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

