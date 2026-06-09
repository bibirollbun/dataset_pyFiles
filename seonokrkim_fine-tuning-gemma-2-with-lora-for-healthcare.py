!pip install -q -U keras-nlp
!pip install -q -U keras>=3
!pip install -q -U kagglehub --upgrade
!pip install -q keras_nlp


import os
os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00" # avoid memory fragmentation on JAX backend.
os.environ["JAX_PLATFORMS"] = ""
import keras
import keras_nlp
import kagglehub


#Make yours and Add copy to clipboard
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("seonokrkim")

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas() # progress bar for pandas

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


class Config:
    seed = 42
    dataset_path = "/kaggle/input/medqa-4k"
    preset = "gemma2_2b_en" # name of pretrained Gemma 2
    sequence_length = 256 #512 # max size of input sequence for training
    batch_size = 1  # size of the input batch in training
    lora_rank = 4 # rank for LoRA, higher means more trainable parameters
    learning_rate=8e-5 # learning rate used in train
    epochs = 1 #10 # number of epochs to train


keras.utils.set_random_seed(Config.seed)


df = pd.read_csv(f"{Config.dataset_path}/train.csv") 
df.tail()


# Define a function to count rows per language in the dataset
def count_language_rows(file_path):
    df = pd.read_csv(file_path)  # Load the dataset
    language_counts = df['language'].value_counts()  # Count rows per language
    return language_counts

# File paths for train and test datasets
train_file = f"{Config.dataset_path}/train.csv"
test_file = f"{Config.dataset_path}/test.csv"

# Print language counts for train and test datasets
print("Train dataset language counts:")
print(count_language_rows(train_file))

print("\nTest dataset language counts:")
print(count_language_rows(test_file))


template = "\n\nCategory:\nkaggle-{Category}\n\nQuestion:\n{Question}\n\nAnswer:\n{Answer}"

df["prompt"] = df.apply(
    lambda row: template.format(
        Category=row.language,  
        Question=row.question,  
        Answer=row.answer       
    ),
    axis=1
)

data = df.prompt.tolist()

data[:5]  


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


data[0:2]


# Preprocess the first two samples from the data using the model's preprocessor.
# Returns:
# - x: The processed input features (e.g., tokenized text).
# - y: The target labels corresponding to the input.
# - sample_weight: Weights for each sample, used for training or evaluation.

x, y, sample_weight = gemma_causal_lm.preprocessor(data[0:2])



print(x, y)



#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

# Enable LoRA for the model and set the LoRA rank to the lora_rank as set in Config (4).
gemma_causal_lm.backbone.enable_lora(rank=Config.lora_rank)
gemma_causal_lm.summary()


#By Gabriel Preda https://www.kaggle.com/code/gpreda/fine-tuning-gemma-2-model-using-lora-and-keras/notebook

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


category = "english_american"
question = "A 6-month-old baby boy presents to his pediatrician for the evaluation of recurrent bacterial infections. He is currently well but has already been hospitalized multiple times due to his bacterial infections. His blood pressure is 103/67 mm Hg and heart rate is 74/min. Physical examination reveals light-colored skin and silver hair. On examination of a peripheral blood smear, large cytoplasmic vacuoles containing microbes are found within the neutrophils. What diagnosis do these findings suggest?"
gemma_qa.query(category,question)


# Save the trained LoRA weights
lora_weights_path = f"/kaggle/working/{Config.preset}_lora_rank{Config.lora_rank}_epoch{Config.epochs}.lora.h5"
gemma_causal_lm.backbone.save_lora_weights(lora_weights_path)
print(f"LoRA weights saved to: {lora_weights_path}")

# Save the entire model if needed
model_save_path = f"/kaggle/working/{Config.preset}_trained_model_epoch{Config.epochs}.h5"
gemma_causal_lm.save(model_save_path)
print(f"Full model saved to: {model_save_path}")

