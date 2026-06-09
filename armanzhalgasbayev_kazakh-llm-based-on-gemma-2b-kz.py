import warnings
warnings.filterwarnings('ignore')

from IPython.display import display, Markdown


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U "keras>=3"

import os

os.environ["KERAS_BACKEND"] = "jax" 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


import pandas as pd
import numpy as np

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/saillab/alpaca_kazakh_taco/" + splits["train"])


df = df[['instruction', 'text']]
df['text'] = df['text'].str[:3000]
df = df[:30000]

# 30000 datarows with length <= 3000


import keras
import keras_nlp

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")


display(Markdown(gemma_lm.generate("Қазақша бірдеңе айтшы?")))


display(Markdown(gemma_lm.generate("Пернетақта дегеніміз не?")))


display(Markdown(gemma_lm.generate("Сен Google компаниясының деректер орталығының жұмысымен қанағаттандырылып тұрсың ба?")))


dataset = []
for index, row in df.iterrows():
    instruction, response = row['instruction'], row['text']
    template = (f"Instruction:\n{instruction}\n\nResponse:\n{response}")
    dataset.append(template)


# Enable LoRA for the model and set the LoRA rank to 64.
gemma_lm.backbone.enable_lora(rank=64)


gemma_lm.preprocessor.sequence_length = 314

# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    beta_1=0.9,          # Adjust beta_1 parameter
    beta_2=0.999         # Adjust beta_2 parameter
)

# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],)


gemma_lm.fit(dataset, epochs=1, batch_size=2)


gemma_lm.save("Gemma2_2B_QazPerry.keras")


prompt = "Instruction:\n{}\n\nResponse:\n"


display(Markdown(gemma_lm.generate(prompt.format("Қазақша бірдеңе айтшы?"))))


display(Markdown(gemma_lm.generate(prompt.format("Пернетақта дегеніміз не?"))))


display(Markdown(gemma_lm.generate(prompt.format("Сен Google компаниясының деректер орталығының жұмысымен қанағаттандырылып тұрсың ба?"))))

