# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#download and run independently, not even space to run on kaggle locally

from PIL import Image, ImageShow
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from moviepy.editor import VideoClip
#import kagglehub
#from moviepy.video.io.bindings import mplfig_to_npimage
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from IPython.display import HTML
import base64
def gif_view(gif_path):
    def gif_to_html(gif_path):
        with open(gif_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'<img src="data:image/gif;base64,{encoded}">'
    return HTML(gif_to_html(gif_path))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

