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


git clone https://github.com/zdx3578/popper-arc
git checkout backup/singlework-before-paral  ##branch version
## version:  678e066861fc4ec613b4832a3127ccbcc5a4eef4
#678e066861fc4ec613b4832a3127ccbcc5a4eef4


(py31022) ➜  popper-arcoldtest git:(backup/singlework-before-paral) ✗ python mainpopperarc.py  --task-id  e3721c99
当前目录: /home/zdx/github/VSAHDC/popper-arcoldtest
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Processing e3721c99 (1/1)
Determining background color with threshold: 40%
Color distribution across training data:
  color 0: 70.86%
  color 5: 24.89%
  color 2: 5.89%
  color 1: 4.97%
  color 3: 4.14%
  color 4: 3.39%
确定全局背景色: 0 (占比: 70.86%)

Train e3721c99 input 0
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟩🟩🟩⬛🟦🟦🟦🟦🟦⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟩⬛🟩⬛🟦⬛🟦⬛🟦⬛🟥⬛🟥⬛🟥⬛🟥⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟩🟩🟩⬛🟦🟦🟦🟦🟦⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟫🟫🟫⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟫🟫⬛🟫🟫⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟫🟫⬛🟫🟫⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛
⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛🟫🟫⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛🟫🟫⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟫⬛⬛🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛
⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫⬛🟫🟫⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫⬛🟫⬛🟫⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛🟫⬛⬛
⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛🟫🟫⬛🟫🟫🟫🟫⬛⬛
⬛⬛⬛⬛⬛🟫🟫🟫⬛🟫⬛⬛🟫⬛🟫🟫🟫⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫🟫⬛⬛
⬛⬛⬛🟫🟫🟫🟫🟫⬛🟫⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛
⬛⬛🟫🟫🟫🟫⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟫🟫⬛🟫⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟫🟫⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛

Train e3721c99 output 0
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟩🟩🟩⬛🟦🟦🟦🟦🟦⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟩⬛🟩⬛🟦⬛🟦⬛🟦⬛🟥⬛🟥⬛🟥⬛🟥⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟩🟩🟩⬛🟦🟦🟦🟦🟦⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟩🟩🟩⬛⬛⬛🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟩🟩⬛🟩🟩⬛🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟩🟩⬛🟩🟩⬛🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛
⬛⬛⬛🟩🟩🟩🟩⬛⬛⬛⬛⬛🟨🟨⬛⬛⬛⬛⬛🟦🟦⬛⬛⬛🟦🟦⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨⬛⬛⬛⬛🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨⬛⬛⬛🟦🟦⬛⬛⬛🟦🟦⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛⬛🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟩🟩🟩🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟩⬛⬛🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟩🟩🟩🟩⬛⬛⬛⬛⬛⬛⬛🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛🟦🟦🟦🟦⬛⬛
⬛⬛⬛🟩🟩🟩⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛🟦🟦🟦⬛🟦🟦⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥⬛🟥⬛🟥⬛⬛⬛⬛🟦🟦🟦🟦⬛⬛🟦⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥🟥⬛⬛⬛🟦🟦⬛🟦🟦🟦🟦⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥⬛🟥🟥🟥⬛⬛⬛⬛🟦🟦⬛⬛🟦🟦🟦⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛🟦🟦🟦🟦🟦⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛

Train e3721c99 input 1
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟩🟩🟩⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟩⬛🟩⬛🟦⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟩🟩🟩⬛🟦⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛🟫🟫🟫⬛⬛
⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫⬛
⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫⬛
⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛🟫🟫🟫🟫⬛
⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛
⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫⬛🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛🟫⬛⬛🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫⬛⬛🟫⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟫⬛⬛🟫🟫⬛⬛⬛🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛🟫🟫⬛⬛🟫⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟫🟫⬛🟫🟫⬛⬛⬛🟫⬛🟫⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛🟫⬛🟫⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛🟫⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛

Train e3721c99 output 1
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟩🟩🟩⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟩⬛🟩⬛🟦⬛⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟩🟩🟩⬛🟦⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛
🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩🟩⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛🟩🟩🟩🟩🟩🟩🟩🟩⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥⬛⬛⬛⬛⬛⬛⬛🟩🟩🟩⬛⬛🟩🟩🟩⬛⬛
⬛⬛⬛⬛🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩🟩🟩⬛⬛⬛🟩🟩🟩🟩⬛
⬛⬛🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩🟩🟩⬛⬛⬛🟩🟩🟩🟩⬛
⬛⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩🟩⬛⬛⬛🟩🟩🟩🟩⬛
⬛⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛🟥🟥🟥🟥⬛⬛⬛⬛🟩🟩🟩🟩🟩🟩🟩🟩⬛⬛
⬛🟥🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛🟥🟥🟥⬛⬛⬛⬛⬛⬛🟩🟩⬛⬛⬛⬛⬛⬛⬛
⬛⬛🟥🟥🟥🟥🟥🟥⬛⬛⬛⬛🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟥🟥⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥⬛⬛⬛⬛⬛🟩🟩🟩🟩🟩⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥⬛⬛⬛⬛⬛🟩🟩⬛🟩🟩⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥⬛⬛⬛⬛⬛🟩⬛⬛🟩🟩⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟩🟩🟩🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩⬛⬛🟩⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟩⬛⬛🟩🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩🟩🟩🟩🟩⬛⬛⬛⬛⬛
⬛⬛⬛🟩🟩⬛⬛🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟩🟩🟩🟩⬛⬛⬛⬛⬛⬛
⬛⬛⬛🟩🟩⬛🟩🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟩🟩🟩🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛🟩⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
/home/zdx/github/VSAHDC/popper-arc/popper/popper/tester.py:3: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
Max rules: 2
Max vars: 8
Max body: 3
Loading recalls
Loading bkcons
Load exact solver: rc2
Load anytime solver:nuwls
Program 1:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
Generating programs of size: 3
tp:19 fn:471 tn:1465 fp:5 mdl:479
********************
New best hypothesis:
tp:19 fn:471 tn:1465 fp:5 size:3 mdl:479
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
********************
Program 2:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_gray_obj(V0,V5,V3).
tp:0 fn:490 tn:1389 fp:81 mdl:574
Program 3:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_color_obj(V0,V5,V3).
tp:221 fn:269 tn:1261 fp:209 mdl:481
Program 4:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_color_obj(V0,V4,V3).
tp:20 fn:470 tn:1462 fp:8 mdl:481
Program 5:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:490 fn:0 tn:1022 fp:448 mdl:451
********************
New best hypothesis:
tp:490 fn:0 tn:1022 fp:448 size:3 mdl:451
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
********************
Program 6:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V1,V2),is_color_obj(V0,V4,V3).
tp:111 fn:379 tn:1470 fp:0 mdl:382
********************
New best hypothesis:
tp:111 fn:379 tn:1470 fp:0 size:3 mdl:382
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V1,V2),is_color_obj(V0,V4,V3).
********************
Program 7:
outpix(V0,V1,V2,V3):- same_hole_but_diff_obj(V0,V5,V6,V4,V3),inbelongs(V0,V5,V1,V2).
tp:81 fn:409 tn:1470 fp:0 mdl:412
Program 8:
outpix(V0,V1,V2,V3):- is_color_obj(V0,V5,V3),objholes(V0,V5,V4),inbelongs(V0,V5,V2,V1).
Generating programs of size: 4
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 9:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_color_obj(V0,V5,V3),is_color_obj(V0,V4,V3).
tp:20 fn:470 tn:1462 fp:8 mdl:482
Program 10:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:20 fn:470 tn:1462 fp:8 mdl:482
Program 11:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:62 fn:428 tn:1416 fp:54 mdl:486
Program 12:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),is_color_obj(V0,V5,V6),is_color_obj(V0,V4,V3).
tp:47 fn:443 tn:1419 fp:51 mdl:498
Program 13:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),is_color_obj(V0,V4,V3),is_gray_obj(V0,V5,V6).
tp:174 fn:316 tn:1312 fp:158 mdl:478
Program 14:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),is_color_obj(V0,V6,V3),is_color_obj(V0,V4,V3).
tp:221 fn:269 tn:1261 fp:209 mdl:482
Program 15:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),is_color_obj(V0,V4,V3),is_gray_obj(V0,V6,V7).
tp:221 fn:269 tn:1261 fp:209 mdl:482
Program 16:
outpix(V0,V1,V2,V3):- is_color_obj(V0,V6,V7),is_color_obj(V0,V4,V3),inbelongs(V0,V5,V2,V1).
tp:221 fn:269 tn:1261 fp:209 mdl:482
Program 17:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),is_color_obj(V0,V4,V3),inbelongs(V0,V6,V1,V2).
tp:221 fn:269 tn:1261 fp:209 mdl:482
Program 18:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),same_hole_but_diff_obj(V0,V4,V5,V6,V7),is_color_obj(V0,V4,V3).
tp:82 fn:408 tn:1420 fp:50 mdl:462
Program 19:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),same_hole_but_diff_obj(V0,V5,V7,V6,V3),is_color_obj(V0,V4,V3).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 20:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),objholes(V0,V4,V6),is_color_obj(V0,V4,V3).
tp:221 fn:269 tn:1261 fp:209 mdl:482
Program 21:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),objholes(V0,V5,V6),is_color_obj(V0,V4,V3).
tp:209 fn:281 tn:1276 fp:194 mdl:479
Program 22:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V1,V2),is_color_obj(V0,V6,V3),is_color_obj(V0,V4,V3).
tp:490 fn:0 tn:1022 fp:448 mdl:452
Program 23:
outpix(V0,V1,V2,V3):- is_gray_obj(V0,V5,V7),is_color_obj(V0,V4,V3),inbelongs(V0,V6,V1,V2).
tp:490 fn:0 tn:1022 fp:448 mdl:452
Program 24:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3),is_gray_obj(V0,V5,V6).
tp:379 fn:111 tn:1135 fp:335 mdl:450
Program 25:
outpix(V0,V1,V2,V3):- same_hole_but_diff_obj(V0,V4,V5,V6,V7),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:374 fn:116 tn:1423 fp:47 mdl:167
********************
New best hypothesis:
tp:374 fn:116 tn:1423 fp:47 size:4 mdl:167
outpix(V0,V1,V2,V3):- same_hole_but_diff_obj(V0,V4,V5,V6,V7),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
********************
Program 26:
outpix(V0,V1,V2,V3):- is_color_obj(V0,V5,V6),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:111 fn:379 tn:1357 fp:113 mdl:496
Program 27:
outpix(V0,V1,V2,V3):- is_color_obj(V0,V7,V6),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:490 fn:0 tn:1022 fp:448 mdl:452
Program 28:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V1,V2),objholes(V0,V4,V6),is_color_obj(V0,V4,V3).
tp:490 fn:0 tn:1022 fp:448 mdl:452
Program 29:
outpix(V0,V1,V2,V3):- objholes(V0,V5,V6),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
tp:455 fn:35 tn:1057 fp:413 mdl:452
Program 30:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V5,V6,V7),is_color_obj(V0,V4,V3).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 31:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_gray_obj(V0,V6,V5),is_color_obj(V0,V4,V3).
tp:20 fn:470 tn:1462 fp:8 mdl:482
Program 32:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_color_obj(V0,V4,V3),is_color_obj(V0,V6,V5).
tp:20 fn:470 tn:1462 fp:8 mdl:482
Program 33:
outpix(V0,V1,V2,V3):- inbelongs(V0,V5,V2,V1),same_hole_but_diff_obj(V0,V5,V6,V4,V3),is_color_obj(V0,V5,V7).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 34:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V5,V4,V6,V3),is_color_obj(V0,V5,V7).
tp:82 fn:408 tn:1420 fp:50 mdl:462
Program 35:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V1,V2),same_hole_but_diff_obj(V0,V5,V4,V6,V3),is_color_obj(V0,V5,V7).
tp:374 fn:116 tn:1423 fp:47 mdl:167
Program 36:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V6,V5,V3),inbelongs(V0,V6,V1,V2).
tp:0 fn:490 tn:1470 fp:0 mdl:494
Program 37:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),is_gray_obj(V0,V6,V7),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 38:
outpix(V0,V1,V2,V3):- same_hole_but_diff_obj(V0,V4,V7,V5,V3),inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 39:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),inbelongs(V0,V7,V1,V2),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 40:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),int_0(V5),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:19 fn:471 tn:1470 fp:0 mdl:475
Program 41:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V6,V5,V3),int_3(V5).
tp:0 fn:490 tn:1469 fp:1 mdl:495
Program 42:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),int_2(V5),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:0 fn:490 tn:1468 fp:2 mdl:496
Program 43:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),int_1(V5),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:0 fn:490 tn:1468 fp:2 mdl:496
Program 44:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),same_hole_but_diff_obj(V0,V4,V6,V5,V3),objholes(V0,V6,V7).
tp:19 fn:471 tn:1465 fp:5 mdl:480
Program 45:
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V2,V1),objholes(V0,V4,V7),same_hole_but_diff_obj(V0,V4,V6,V5,V3).
tp:19 fn:471 tn:1465 fp:5 mdl:480
********************
New best hypothesis:
tp:485 fn:5 tn:1423 fp:47 size:7 mdl:59
outpix(V0,V1,V2,V3):- inbelongs(V0,V4,V1,V2),is_color_obj(V0,V4,V3).
outpix(V0,V1,V2,V3):- same_hole_but_diff_obj(V0,V4,V5,V6,V7),inbelongs(V0,V5,V1,V2),is_color_obj(V0,V4,V3).
********************
！！！！！！！！！！！！！！！！！！！！！！Solved e3721c99 with score (485, 5, 1423, 47, 7)
当前成功记录数: 1

Test 0 input
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛🟫🟫🟫🟫🟫🟫🟫⬛🟫🟫🟫⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟦⬛🔹⬛🔹⬛
⬛🟫⬛🟫🟫⬛⬛🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛🟫⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛🟫⬛⬛🟫⬛⬛🟫⬛⬛⬛⬛⬛🟫⬛⬛⬛⬛🟫⬛🟫🟫🟫⬛🟦⬛🔹⬛🔹⬛
⬛🟫🟫🟫🟫⬛⬛🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫⬛🟫⬛🟫⬛🟦⬛🔹🔹🔹⬛
⬛⬛🟫🟫🟫⬛🟫🟫⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛🟫🟫🟫⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛🟫🟫🟫⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟪🟪🟪⬛
🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟪⬛🟪⬛
🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛🟦⬛🟪🟪🟪⬛
🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫🟫⬛⬛🟫🟫🟫🟫⬛🟦⬛🟪⬛🟪⬛
🟫🟫🟫⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛🟫🟫⬛⬛🟫⬛🟦⬛🟪🟪🟪⬛
🟫🟫🟫⬛🟫🟫⬛🟫🟫🟫🟫⬛⬛🟫⬛⬛🟫🟫🟫⬛⬛⬛🟫⬛🟦⬛🟪⬛🟪⬛
🟫🟫⬛⬛🟫⬛⬛⬛🟫⬛🟫⬛⬛🟫⬛⬛⬛🟫🟫🟫⬛⬛🟫⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫🟫⬛⬛🟫🟫⬛⬛🟫⬛🟫🟫🟫🟫⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛🟫🟫🟫🟫⬛⬛⬛🟫🟫⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫⬛🟫⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛🟦⬛🟨🟨🟨⬛
⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟨⬛🟨⬛
🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛🟫🟫🟫🟫🟫🟫⬛⬛🟫🟫🟫🟫⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫⬛⬛🟫⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫⬛⬛⬛⬛🟫🟫⬛🟫🟫⬛⬛🟦⬛🟨⬛🟨⬛
⬛🟫🟫⬛⬛⬛⬛🟫🟫⬛⬛⬛🟫⬛⬛⬛🟫🟫⬛⬛🟫⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛🟫🟫🟫⬛⬛🟫🟫🟫⬛⬛⬛🟫⬛⬛🟫🟫⬛⬛⬛🟫⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛🟫🟫🟫⬛🟫🟫⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛🟫⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛🟫🟫⬛⬛🟫⬛⬛⬛🟫🟫🟫⬛🟫🟫⬛⬛⬛⬛⬛🟫⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛🟫⬛⬛⬛⬛🟫⬛⬛⬛🟫⬛⬛⬛⬛🟫🟫⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛🟫⬛⬛⬛🟫⬛⬛⬛🟫🟫⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛

Test 0 expected
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛🔹🔹🔹🔹🔹🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹⬛🔹⬛
⬛🔹⬛🔹🔹⬛⬛🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛🔹⬛⬛🔹⬛⬛🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹⬛🔹⬛
⬛🔹🔹🔹🔹⬛⬛🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛⬛🔹🔹🔹⬛🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛🔹🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟪🟪🟪🟪🟪🟪⬛⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟪🟪🟪⬛⬛🟪🟪🟪🟪⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛🔹🔹🔹⬛⬛⬛⬛⬛🟪🟪🟪🟪⬛🟪🟪⬛⬛🟪⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛🔹🔹⬛🔹🔹🔹🔹⬛⬛🟪⬛⬛🟪🟪🟪⬛⬛⬛🟪⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛🔹⬛⬛⬛🔹⬛🔹⬛⬛🟪⬛⬛⬛🟪🟪🟪⬛⬛🟪⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛🔹🔹⬛⬛🔹🔹🔹⬛⬛🟪🟪⬛⬛🟪⬛🟪🟪🟪🟪⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛🔹🔹🔹🔹🔹🔹🔹⬛⬛🟪🟪🟪🟪⬛⬛⬛🟪🟪⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🔹🔹🔹⬛⬛⬛🟪🟪🟪🟪⬛🟪⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🔹🔹⬛⬛⬛⬛⬛⬛🟪🟪🟪⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨⬛⬛🟨⬛⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨⬛⬛🟨⬛⬛⬛⬛🟨🟨⬛🟨🟨⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛🟨🟨⬛⬛⬛🟨⬛⬛⬛🟨🟨⬛⬛🟨⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛🟨🟨🟨⬛⬛⬛🟨⬛⬛🟨🟨⬛⬛⬛🟨⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛🟨🟨⬛🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛🟨⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛🟨⬛⬛⬛🟨🟨🟨⬛🟨🟨⬛⬛⬛⬛⬛🟨⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛🟨⬛⬛⬛⬛🟨⬛⬛⬛🟨⬛⬛⬛⬛🟨🟨⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛🟨🟨⬛⬛⬛🟨⬛⬛⬛🟨⬛⬛⬛🟨🟨⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/hyp.pl:2:
Warning:    Singleton variables: [V6,V7]
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test0/testbk.pl:3:
Warning:    Redefined static procedure col_0/1
Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/bk.pl:3
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test0/testbk.pl:5:
Warning:    Redefined static procedure col_1/1
Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/bk.pl:5


Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/bk.pl:690
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test0/testbk.pl:506:
Warning:    Redefined static procedure objholes/3
Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/bk.pl:708
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test0/testbk.pl:521:
Warning:    Redefined static procedure same_hole_but_diff_obj/5
Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/bk.pl:732

Test 0 predicted
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛🔹🔹🔹🔹🔹🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹⬛🔹⬛
⬛🔹⬛🔹🔹⬛⬛🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛🔹⬛⬛🔹⬛⬛🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹⬛🔹⬛
⬛🔹🔹🔹🔹⬛⬛🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🔹🔹🔹⬛
⬛⬛🔹🔹🔹⬛🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛🔹🔹🔹⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟪🟪🟪🟪🟪🟪⬛⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟪🟪🟪⬛⬛🟪🟪🟪🟪⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛🔹🔹🔹⬛⬛⬛⬛⬛🟪🟪🟪🟪⬛🟪🟪⬛⬛🟪⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛🔹🔹⬛🔹🔹🔹🔹⬛⬛🟪⬛⬛🟪🟪🟪⬛⬛⬛🟪⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛🔹⬛⬛⬛🔹⬛🔹⬛⬛🟪⬛⬛⬛🟪🟪🟪⬛⬛🟪⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛🔹🔹⬛⬛🔹🔹🔹⬛⬛🟪🟪⬛⬛🟪⬛🟪🟪🟪🟪⬛🟦⬛🟪⬛🟪⬛
⬛⬛⬛⬛⬛🔹🔹🔹🔹🔹🔹🔹⬛⬛🟪🟪🟪🟪⬛⬛⬛🟪🟪⬛🟦⬛🟪🟪🟪⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🔹🔹🔹⬛⬛⬛🟪🟪🟪🟪⬛🟪⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🔹🔹⬛⬛⬛⬛⬛⬛🟪🟪🟪⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛⬛🟨⬛⬛🟨⬛⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛⬛⬛🟨🟨⬛⬛🟨⬛⬛⬛⬛🟨🟨⬛🟨🟨⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛⬛⬛🟨🟨⬛⬛⬛🟨⬛⬛⬛🟨🟨⬛⬛🟨⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛⬛🟨🟨🟨⬛⬛⬛🟨⬛⬛🟨🟨⬛⬛⬛🟨⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛🟨🟨⬛🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛🟨⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛🟨⬛⬛⬛🟨🟨🟨⬛🟨🟨⬛⬛⬛⬛⬛🟨⬛⬛⬛🟦⬛🟨⬛🟨⬛
⬛⬛⬛⬛⬛🟨⬛⬛⬛⬛🟨⬛⬛⬛🟨⬛⬛⬛⬛🟨🟨⬛⬛⬛🟦⬛🟨🟨🟨⬛
⬛⬛⬛⬛⬛🟨🟨⬛⬛⬛🟨⬛⬛⬛🟨⬛⬛⬛🟨🟨⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛⬛🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛
Test 0 - Exact match? True
Test 0 - Pixel accuracy: 1.0

Test 1 input
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟦⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨⬛🟨⬛🟦⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟦⬛⬛🟫🟫🟫🟫🟫⬛⬛🟫⬛⬛🟫🟫🟫🟫⬛🟫🟫⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛🟫⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛
⬛🟠⬛🟠⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛🟠⬛🟠⬛🟦⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛⬛⬛🟫⬛⬛🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛🟫🟫🟫🟫
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛🟫🟫🟫⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛🟫🟫🟫⬛🟫🟫🟫⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫🟫🟫⬛🟫🟫🟫⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛
⬛🟦🟦🟦⬛🟦⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟫🟫⬛⬛🟫⬛🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫🟫🟫⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫🟫⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫⬛⬛🟫🟫⬛⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟫🟫🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟫🟫🟫🟫⬛⬛🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟫🟫⬛🟫🟫🟫🟫🟫⬛⬛⬛🟫⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛
⬛🟦🟦🟦⬛🟦⬛⬛🟫🟫🟫🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛🟫🟫🟫⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛🟫🟫⬛⬛⬛⬛🟫🟫🟫🟫🟫⬛⬛⬛⬛🟫🟫🟫⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟫🟫🟫⬛⬛⬛⬛⬛⬛⬛⬛⬛

Test 1 expected
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟦⬛⬛⬛🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨⬛🟨⬛🟦⬛⬛🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟦⬛⬛🟨🟨🟨🟨🟨⬛⬛🟨⬛⬛🟠🟠🟠🟠⬛🟠🟠⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛🟠🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛⬛🟠⬛🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛
⬛🟠⬛🟠⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛
⬛🟠⬛🟠⬛🟦⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠⬛⬛🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛⬛⬛🟠⬛⬛🟠⬛⬛⬛⬛⬛🟠🟠⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛🟠🟠🟠⬛🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥⬛🟥🟥🟥⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥⬛🟥🟥🟥⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛
⬛🟦🟦🟦⬛🟦⬛⬛⬛🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦⬛⬛🟦⬛🟦⬛⬛⬛⬛⬛🟥🟥⬛⬛🟥🟥🟥🟥⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥⬛⬛🟥🟥🟥⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥⬛⬛🟥🟥⬛⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟦🟦🟦🟦⬛⬛🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦⬛🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛⬛🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/hyp.pl:2:
Warning:    Singleton variables: [V6,V7]
Warning: /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test1/testbk.pl:3:
Warning:    Redefined static procedure col_0/1
Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test0/testbk.pl:3

Warning:    Redefined static procedure same_hole_but_diff_obj/5
Warning:    Previously defined at /home/zdx/github/VSAHDC/popper-arcoldtest/popper_kb/e3721c99/test0/testbk.pl:521

Test 1 predicted
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟦⬛⬛⬛🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟨⬛🟨⬛🟦⬛⬛🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛⬛⬛🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛
⬛🟨🟨🟨⬛🟦⬛⬛🟨🟨🟨🟨🟨⬛⬛🟨⬛⬛🟠🟠🟠🟠⬛🟠🟠⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛🟨🟨🟨🟨🟨🟨🟨🟨⬛⬛🟠🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛🟨🟨🟨🟨⬛⬛⬛⬛🟠⬛🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛
⬛🟠⬛🟠⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛
⬛🟠⬛🟠⬛🟦⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠⬛⬛🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛
⬛🟠🟠🟠⬛🟦⬛⬛⬛⬛⬛⬛🟠⬛⬛🟠⬛⬛⬛⬛⬛🟠🟠⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛🟠🟠🟠⬛🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛🟠🟠🟠🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛🟠🟠🟠🟠⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥⬛⬛⬛⬛
⬛🟥⬛🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥⬛🟥🟥🟥⬛⬛
⬛🟥🟥🟥⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥🟥🟥🟥⬛🟥🟥🟥⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛🟥⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛
⬛🟦🟦🟦⬛🟦⬛⬛⬛🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦⬛⬛🟦⬛🟦⬛⬛⬛⬛⬛🟥🟥⬛⬛🟥🟥🟥🟥⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥🟥🟥⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥⬛⬛🟥🟥🟥⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥⬛⬛🟥🟥⬛⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛🟥🟥🟥🟥🟥⬛⬛⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛🟦🟦🟦🟦⬛⬛🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟦⬛🟦⬛🟦⬛🟦🟦⬛🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛🟦🟦🟦⬛🟦⬛⬛🟦🟦🟦🟦🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛🟦🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
⬛⬛⬛⬛⬛🟦⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
Test 1 - Exact match? True
Test 1 - Pixel accuracy: 1.0




Finished e3721c99







