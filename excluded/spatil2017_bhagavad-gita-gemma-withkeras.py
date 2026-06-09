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


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook
# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3
!pip install -q -U kagglehub --upgrade


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00" # avoid memory fragmentation on JAX backend.
os.environ["JAX_PLATFORMS"] = ""
import keras
import keras_nlp
import kagglehub


# #Make yours and Add copy to clipboard
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("hf_licorne")

#Gabriel's line
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
#os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("kaggle_username")
#os.environ["KAGGLE_KEY"] = user_secrets.get_secret("kaggle_key")

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas() # progress bar for pandas

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

class Config:
    seed = 42
    dataset_path = "/kaggle/input/bhagwad-gita-dataset"
    preset = "/kaggle/input/gemma2/keras/gemma2_instruct_2b_en/2" # name of pretrained Gemma 2
    sequence_length = 512 # max size of input sequence for training
    batch_size = 1 # size of the input batch in training
    lora_rank = 4 # rank for LoRA, higher means more trainable parameters
    learning_rate=8e-5 # learning rate used in train
    epochs = 10 # number of epochs to train


keras.utils.set_random_seed(Config.seed)


df = pd.read_csv(f"{Config.dataset_path}/Bhagwad_Gita.csv", nrows= 52) #Original has 674 rows
df.tail()


#Don't change anything on Template line. Just the rows (in blue)
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

template = "\n\nCategory:\nkaggle-{Category}\n\nQuestion:\n{Question}\n\nAnswer:\n{Answer}"
df["prompt"] = df.apply(lambda row: template.format(Category=row.Verse,
                                                             Question=row.WordMeaning,
                                                             Answer=row.HinMeaning), axis=1)
data = df.prompt.tolist()


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

def colorize_text(text):
    for word, color in zip(["Category", "Question", "Answer"], ["blue", "red", "green"]):
        text = text.replace(f"\n\n{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

gemma_causal_lm = keras_nlp.models.GemmaCausalLM.from_preset(Config.preset)
gemma_causal_lm.summary()


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

class GemmaQA:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.prompt = template
        self.gemma_causal_lm = gemma_causal_lm
        
    def query(self, category, question):
        response = self.gemma_causal_lm.generate(
            self.prompt.format(
                Category=category,
                Question=question,
                Answer=""), 
            max_length=self.max_length)
        display(Markdown(colorize_text(response)))


x, y, sample_weight = gemma_causal_lm.preprocessor(data[0:2])


print(x, y)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

# Enable LoRA for the model and set the LoRA rank to the lora_rank as set in Config (4).
gemma_causal_lm.backbone.enable_lora(rank=Config.lora_rank)
gemma_causal_lm.summary()


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

#set sequence length cf. config (512)
gemma_causal_lm.preprocessor.sequence_length = Config.sequence_length 

# Compile the model with loss, optimizer, and metric
gemma_causal_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=Config.learning_rate),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train model
gemma_causal_lm.fit(data, epochs=Config.epochs, batch_size=Config.batch_size)


gemma_qa = GemmaQA()


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[13]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[49]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[1]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[10]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[30]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[35]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[20]
gemma_qa.query(row.Verse,row.WordMeaning)


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

row = df.iloc[48]
gemma_qa.query(row.Verse,row.WordMeaning)


#Who was Arjuna?
#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

category = "HinMeaning"
question = "अर्जुन कौन थे??"
gemma_qa.query(category,question)


#What was the role of Dhritarashtra?

category = "HinMeaning"
question = "धृतराष्ट्र की भूमिका क्या थी?"
gemma_qa.query(category,question)


preset_dir = ".\gemma2_2b_en_kaggle_docs"
gemma_causal_lm.save_to_preset(preset_dir)

