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


# ✅ Checkpoint 1 GPT nalysisA
dataset_id = "OpenTopography: OT-BR-SC-001"
model_version = "gpt-4o"

gpt_response = """
Garopaba 지역의 LiDAR 데이터는 평탄하고 규칙적인 지형 패턴을 보여줍니다. 
이 중 일부는 자연 형상이 아닌 인공적 개입의 가능성을 시사합니다.
예: 직선형 구역, 대칭 분포, 일정한 간격의 구조 등.
이는 과거 토지 개간이나 의식 장소의 흔적일 수 있습니다.
"""

print("Dataset ID:", dataset_id)
print("Model Version:", model_version)
print("GPT Response:", gpt_response)

