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
import random

# Step 1: Load the data
# Replace with your file paths if running locally
test_file_path = '/kaggle/input/llms-you-cant-please-them-all/test.csv'
sample_submission_file_path = '/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv'

# Load the test and sample submission files
test_df = pd.read_csv(test_file_path)
sample_submission_df = pd.read_csv(sample_submission_file_path)

# Step 2: Define essay generation logic
def generate_essay(topic):
    """
    Generates an essay given a topic. The essay is approximately 100 words long
    and is crafted to maximize disagreement between LLM judges.
    """
    essay_template = (
        f"The topic of {topic} presents a myriad of viewpoints." 
        "While some argue passionately for its merits others highlight critical challenges that cannot be ignored. "
        "Considerations include ethical dilemmas, socio-economic impacts, and historical context. "
        "Personal perspectives may diverge significantly, adding layers of complexity to the discourse. "
        "Ultimately, the debate underscores the importance of nuanced understanding and open dialogue in approaching such multifaceted subjects."
    )
    return essay_template

# Step 3: Generate essays for each ID in the test set
output = []
for index, row in test_df.iterrows():
    topic = row.get("topic", "")  # Assuming 'topic' column exists in the test set
    essay = generate_essay(topic)
    output.append((row['id'], essay))

# Step 4: Prepare the submission DataFrame
submission_df = pd.DataFrame(output, columns=['id', 'essay'])

# Step 5: Save the submission file
output_file_path = '/kaggle/working/submission.csv'
submission_df.to_csv(output_file_path, index=False)

print(f"Submission file saved to {output_file_path}")


