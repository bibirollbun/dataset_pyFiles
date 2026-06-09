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


ðŸ“Œ Project Code & Dataset:

The complete source code, model weights, and README for the RealityFix AI â€“ Live App are available here:

Kaggle Dataset: https://www.kaggle.com/datasets/aasthasukhadiya/realityfix-ai-code

Contents:
- main.py â†’ Core app logic and AI integration
- utils.py â†’ Helper functions
- model/realityfix_model_weights.h5 â†’ Trained AI model weights
- README.md â†’ Instructions to run the app and project details

This dataset allows judges to explore the appâ€™s source code, understand the AI model, and reproduce the project.

