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


# Competition Notebook: Exploiting LLM-Judge Disagreement
import pandas as pd
import numpy as np
import random


# File paths
test_file_path = '/kaggle/input/llms-you-cant-please-them-all/test.csv'
sample_submission_path = '/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv'
submission_output_path = '/kaggle/working/submission.csv'


# Load data
test_df = pd.read_csv(test_file_path)
sample_submission_df = pd.read_csv(sample_submission_path)


print("Test data overview:")
test_df.head()


print("Sample submission overview:")
sample_submission_df.head()


# Function to generate essays
def generate_essay(topic):
    """Generates an essay designed to maximize disagreement between LLM judges."""
    # Strategy: Use controversial and ambiguous language, vary sentence structures, and include conflicting viewpoints.
    essay = (
        f"The topic '{topic}' is both fascinating and divisive. On one hand, some argue that it represents "
        "progress and innovation, while others see it as a step backward fraught with risks. This duality "
        "underscores the complexity of the issue. Historical examples provide lessons, yet predictions remain uncertain. "
        "As the discussion evolves, new perspectives emerge, challenging conventional wisdom and sparking debate."
    )
    return essay


# Generate essays for submission
submission_df = test_df.copy()
submission_df['essay'] = submission_df['topic'].apply(generate_essay)


print("Generated essays for submission:")
submission_df.head()


# Save submission file
submission_df[['id', 'essay']].to_csv(submission_output_path, index=False)
print(f"Submission file saved to {submission_output_path}")

