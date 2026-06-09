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
import keras_nlp
import keras
import pandas as pd
import time
from tqdm.notebook import tqdm


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


os.environ["KERAS_BACKEND"]="jax"

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

class Config:
    token_limit=256
    lora_name="English_Russian_Translation"
    lora_rank=4
    lr_value=1e-4
    train_epoch=4
    model_id = "gemma2_instruct_2b_en"


df=pd.read_csv("/kaggle/input/translations/random_slice_desc_translation.csv")
df


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(Config.model_id)
gemma = keras_nlp.models.GemmaCausalLM.from_preset(Config.model_id)
gemma.summary()


train = []

for i,x in tqdm(df.iterrows(), desc='tokenize dataset'):
    item = f"<start_of_turn>user\n{x['english_description']}<end_of_turn>\n<start_of_turn>system\n{x['russian_description']}<end_of_turn>"
    length = len(tokenizer(item))
    if length < Config.token_limit:
        train.append(item)

print(len(train))
print(train[0])
print(train[1])
print(train[2])
print(train[3])


tick_start = 0


def tick():
    global tick_start
    tick_start = time.time()


def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")


def text_gen(prompt):
    tick()
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma.generate(input, max_length=Config.token_limit)
    print("\nGemma output:")
    print(output)
    tock()


gemma.backbone.enable_lora(rank=Config.lora_rank)
gemma.summary()


# Limit the input sequence length (to control memory usage).
gemma.preprocessor.sequence_length = Config.token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=Config.lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


print(df.tail(2))


class CustomCallback(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        model_name = f"/kaggle/working/{Config.lora_name}_{Config.lora_rank}_last.lora.h5"
        gemma.backbone.save_lora_weights(model_name)

        # Evaluate
        text_gen("Dry milk formula Nutrilon Sour-milk 2")
        text_gen(
          "Selling a quilted coat-down jacket made of gen."
        )

history = gemma.fit(train, epochs=Config.train_epoch, batch_size=1, callbacks=[CustomCallback()])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()

