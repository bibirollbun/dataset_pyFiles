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


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp datasets --quiet
!pip install -q -U keras --quiet
!pip install -q -U datasets --quiet


import os

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"


model_id = "gemma2_instruct_2b_en"
token_limit = 256
num_data_limit = 200
batch_size= 1
lora_rank = 16
learning_rate = 1e-4
epochs = 20
lora_name = "translator"


import pandas as pd
df = pd.read_csv("/kaggle/input/shahnameh/Rostam.csv")
df.head()


from datasets import Dataset, DatasetDict

train_df = df.reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
train_dataset


import keras
import keras_nlp

import time

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()

tick_start = 0

def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def text_gen(prompt):
    tick()
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma_lm.generate(input, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()


# inference before fine-tuning
text_gen("translate:\n\"دگر باره اسپان ببستند سخت\"")


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

data = train_dataset.with_format("np", columns=['old_text',	'new_text',	'translated_text'], output_all_columns=False)
train = []
for x in data:
  text1 = x['old_text']
  text2 = x['new_text']
  text3 = x['translated_text']
  item = f"<start_of_turn>user\ntranslate:\n\"{text1}\"<end_of_turn>\n<start_of_turn>model\n{text3}<end_of_turn>"
  length = len(tokenizer(item))
  # skip data if the token length is longer than our limit
  if length < token_limit:
    train.append(item)
    if(len(train)>=num_data_limit):
      break

print(len(train))
print(train[0])
print(train[1])
print(train[2])


# Enable LoRA for the model and set the LoRA rank to 16.
gemma_lm.backbone.enable_lora(rank=lora_rank)

# print a summary of the model's architecture, including the LoRA layers, 
# allowing us to see the impact of applying LoRA to the model.
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


import torch
import gc

def reset_memory():
    # Releases GPU VRAM held by the model. Useful for avoiding CUDA memory issues.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("VRAM cleared.")
    else:
        print("CUDA is not available. No VRAM to clear.")

reset_memory()


history = gemma_lm.fit(train, epochs=epochs, batch_size=1)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()


text_gen("translate:\n\"دگر باره اسپان ببستند سخت\"")


tmp_model_dir = "/kaggle/tmp/gemma2_instruct_2b_en"  # Use /kaggle/tmp
preset_dir = "gemma2_2b_instruct_OldPersianText"
os.makedirs(tmp_model_dir, exist_ok=True)
gemma_lm.save_to_preset(tmp_model_dir)

print(f"Model saved to: {tmp_model_dir}")


preset_dir = "gemma2_2b_instruct_OldPersianText"

import kagglehub
import keras_hub
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    kagglehub.login()

model_version = 1
kaggle_username = kagglehub.whoami()["username"]
kaggle_uri = f"kaggle://{kaggle_username}/gemma2_keras_translate_Shahname/keras/{preset_dir}"
keras_hub.upload_preset(kaggle_uri, tmp_model_dir)
print("Pushed the model to kaggle_uri= ", kaggle_uri)
print("tmp_model_dir= ", tmp_model_dir)


!pip install sacrebleu --quiet
!pip install evaluate --quiet


import evaluate

metric = evaluate.load("sacrebleu")


predictions = [
    "They tied her tightly to the horse.."
]
references = [
    [
        "They tied the horses tightly."
    ]
]
metric.compute(predictions=predictions, references=references)

