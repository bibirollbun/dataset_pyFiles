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


# Environment Setup
!pip install -q flax optax dm-haiku jax jaxlib --upgrade
!pip install -q polars cairosvg


# Standard imports
import jax
import jax.numpy as jnp
import flax.linen as nn
import haiku as hk
import optax
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import cairosvg
from pathlib import Path
from tqdm import tqdm
import random
import os


# Load paths
DATA_PATH = "/kaggle/input/drawing-with-llms/"
TRAIN_PATH = "/kaggle/input/drawing-with-llms/train.csv"
TEST_PATH = "/kaggle/input/drawing-with-llms/kaggle_evaluation/test.csv"
QUESTIONS_PATH = "/kaggle/input/drawing-with-llms/questions.parquet"

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
questions_df = pd.read_parquet(QUESTIONS_PATH)

print(train_df)


print(test_df)


print(questions_df)


# Tokenizer - basic character level
def simple_tokenize(text):
    return np.array([ord(c) for c in text])

train_df["tokens"] = train_df["description"].apply(simple_tokenize)
test_df["tokens"] = test_df["description"].apply(simple_tokenize)

# Padding/Truncation to fixed length
MAX_LEN = 64
def pad_or_truncate(arr):
    if len(arr) > MAX_LEN:
        return arr[:MAX_LEN]
    return np.pad(arr, (0, MAX_LEN - len(arr)))

train_df["tokens"] = train_df["tokens"].apply(pad_or_truncate)
test_df["tokens"] = test_df["tokens"].apply(pad_or_truncate)


# Define Transformer decoder with Haiku
def model_fn(x):
    mlp = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(512), jax.nn.relu,
        hk.Linear(MAX_LEN)  # Predict SVG token logits
    ])
    return mlp(x)

# Transform into Haiku model
model = hk.transform(model_fn)


import jax

print("JAX devices:", jax.devices())  # Confirm it's not using TPU


# Define optimizer
learning_rate = 1e-3
optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)

rng = jax.random.PRNGKey(42)

sample_input = jnp.ones((MAX_LEN,), dtype=jnp.float32)

params = model.init(rng, sample_input)

opt_state = optimizer.init(params)


@jax.jit
def loss_fn(params, x, y):
    preds = model.apply(params, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(params, opt_state, x, y, rng):
    def loss_fn(params):
        preds = model.apply(params, rng, x)
        return jnp.mean((preds - y) ** 2)

    grads = jax.grad(loss_fn)(params)

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)

    return params, opt_state


# Simulated training
num_epochs = 3

for epoch in range(num_epochs):
    for i, row in train_df.iterrows():  
        x = jnp.array(row["tokens"], dtype=jnp.float32)  
        y = jnp.sin(x)

        rng, subkey = jax.random.split(rng)
        params, opt_state = train_step(params, opt_state, x, y, subkey)

    print(f"âœ… Epoch {epoch + 1} complete.")


# Convert model output to SVG-like placeholder
def decode_svg(preds):
    return f"<svg width='100' height='100'><circle cx='{int(preds[0]*50)}' cy='{int(preds[1]*50)}' r='30' fill='blue'/></svg>"

def predict_svg(text):
    tokens = pad_or_truncate(simple_tokenize(text))
    x = jnp.array(tokens)
    rng = jax.random.PRNGKey(0)
    y = model.apply(params, rng, x)

    return decode_svg(y)

# Try a test prompt
print(predict_svg("a red triangle with green background"))


def render_svg_to_png(svg_code, output_path="sample.png"):
    cairosvg.svg2png(bytestring=svg_code.encode('utf-8'), write_to=output_path)

# Try rendering
svg = predict_svg("a goose flying through a rainbow")
render_svg_to_png(svg, "goose.png")


def clip_similarity(text, image_path):
    return random.uniform(0.5, 0.9)  # Placeholder

score = clip_similarity("a goose flying through a rainbow", "goose.png")
print("Simulated CLIP Score:", score)

