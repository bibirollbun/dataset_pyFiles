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


train = pd.read_csv("/kaggle/input/drawing-with-llms/train.csv")
train


from transformers import AutoTokenizer, AutoModelForCausalLM

class Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, description):
        inputs = self.tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        output = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)



from kaggle_evaluation import test

model = Model()
output = model.predict("A sunset over mountains")
print(output)



# Load the test data (assuming a 'description' column exists)
test_csv_path = '/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv'
test_df = pd.read_csv(test_csv_path)
print("Test Data Overview:")
print(test_df.head())

# Generate SVG predictions for the test data.
test_df['svg'] = test_df['description'].apply(lambda x: model.predict(x))
print("\nSample predictions:")
print(test_df[['description', 'svg']].head())


# Prepare the submission DataFrame.
if 'id' in test_df.columns:
    submission_df = test_df[['id', 'svg']]
else:
    submission_df = test_df.reset_index()[['index', 'svg']].rename(columns={'index': 'id'})

# Save the submission file.
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"Submission file saved as {submission_filename}")


import kagglehub
package = kagglehub.package_import('dster/drawing-with-llms-starter-notebook/versions/3')
print("Kaggle Package imported successfully.")

