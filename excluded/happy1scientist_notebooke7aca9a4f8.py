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


from kaggle_secrets import UserSecretsClient


!git clone https://github.com/epang080516/arc_agi.git


%cd arc_agi


!pip install -U -r requirements_py311.txt


!pip install xai-sdk


!pip install "protobuf<6,>=5.26.1"


import os

# os.environ['LOGFIRE_TOKEN'] = UserSecretsClient().get_secret('LOGFIRE_TOKEN')
os.environ['XAI_API_KEY'] = UserSecretsClient().get_secret('XAI_API_KEY')


#%env LOGFIRE_TOKEN={UserSecretsClient().get_secret('LOGFIRE_TOKEN')}
#%env XAI_API_KEY={UserSecretsClient().get_secret('XAI_API_KEY')}


!python -m src.submission -e -p /kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json





"""
import numpy as np

def score_submission_dict(
    submission: dict[str, list[dict[str, list[list[int]]]]],
    correct_outputs: dict[str, list[list[list[int]]]],
    allowed_attempts: int = 2,
):
    corr_by_name: dict[str, bool] = {}
    for name, expected in correct_outputs.items():
        sub = submission[name]
        assert len(sub) == len(expected)

        def get_warn(x, attempt, default):
            if attempt not in x:
                print("Missing!", name, attempt)
            return x.get(attempt, default)

        convert_subs_to_lists = [
            [get_warn(sub[i], f"attempt_{j+1}", None) for i in range(len(expected))]
            for j in range(allowed_attempts)
        ]

        corr = any(x == expected for x in convert_subs_to_lists)
        corr_by_name[name] = corr

    return corr_by_name, np.mean(list(corr_by_name.values()))
"""


"""
import json

with open("/kaggle/input/notebooke7aca9a4f8/arc_agi/submission.json", "r") as file:
    loaded_sub = json.load(file)
"""


"""
with open("/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json", "r") as file:
    expected_sub = json.load(file)
"""


# score_submission_dict(loaded_sub, expected_sub)




