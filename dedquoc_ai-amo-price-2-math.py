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


import os
import time
import pandas as pd
import polars as pl
import torch

import kaggle_evaluation.aimo_2_inference_server

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
cutoff_time = time.time() + (4 * 60 + 30) * 60


# Load datasets

path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/'
reference_df  = pd.read_csv(path + 'reference.csv')
test_df =  pd.read_csv(path + 'test.csv')


reference_df.head()


test_df.head()


import sympy
import pandas as pd
import matplotlib.pyplot as plt

def find_three_digit_divisors(exp):
    """Finds all three-digit divisors of 10^exp - 1 using sympy."""
    num = pow(10, exp) - 1  # Compute 10^exp - 1
    all_divisors = sympy.divisors(num)  # Get all divisors
    three_digit_divs = [d for d in all_divisors if 100 <= d <= 999 and d % 2 == 1]  # Filter three-digit odd divisors
    return three_digit_divs


# Ensure numeric values in test_df
test_df = test_df.map(lambda x: pd.to_numeric(x, errors='coerce'))

# Drop NaN values after conversion
test_df = test_df.dropna()

# Set exponent value for local testing
exp = 100  # Change back to 2024 for full Kaggle run

# Compute three-digit divisors of 10^exp - 1
divisors_2024 = find_three_digit_divisors(exp)

# Display first 10 results
print("First 10 three-digit divisors of 10^", exp, "- 1:", divisors_2024[:10])

# Verify all divisors are odd
assert all(d % 2 == 1 for d in divisors_2024), "Error: Found an even divisor!"

# Cross-check with test dataset
test_values = set(test_df.stack().dropna().astype(int))  # Flatten test_df and convert to set of integers
matching_divisors = [d for d in divisors_2024 if d in test_values]

# Display matches
print("Matching divisors found in test dataset:", matching_divisors)

# Plot histogram of three-digit divisors
plt.figure(figsize=(10, 5))
plt.hist(divisors_2024, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Three-digit Divisors")
plt.ylabel("Frequency")
plt.title("Distribution of Three-Digit Divisors of 10^" + str(exp) + " - 1")
plt.grid(True)
plt.show()


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3

import os

os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

import keras
import keras_nlp


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")


datasets = []
    
for index, row in reference_df.iterrows():
    question, answer = row['problem'], row['answer']
    template = (f"problem:\n{question}\n\nanswer:\n{answer}")
    datasets.append(template)


# Enable LoRA for the model and set the LoRA rank to 64.
gemma_lm.backbone.enable_lora(rank=64)


# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 512
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


from keras_nlp.models import GemmaTokenizer

print(GemmaTokenizer.presets.keys())


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset("gemma_2b_en")


%%time

gemma_lm.fit(datasets, epochs=1, batch_size=1)


for i in range(0, 4):
    print(reference_df['problem'][i])


gemma_lm.summary()


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    # Make a prediction
    # prediction = 0  # model.predict(question)
    #answer = predict_for_question(question)
    #print(question)
    print("------\n\n")
    return pl.DataFrame({'id': id_, 'answer': 0})


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            #'/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            'reference.csv',
        )
    )

