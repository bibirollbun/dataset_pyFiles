import gc
from tensorflow.keras import backend as K


# Clear GPU memory if using TensorFlow
K.clear_session()

# Force garbage collection
gc.collect()

# Print memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp datasets
!pip install -q -U keras


import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

## IMPORTANT: DO NOT ENABLE THIS LINE
# Run at half precision.
#keras.config.set_floatx("bfloat16")


import subprocess

# List of libraries to check
libraries = ['keras', 'keras_nlp','numpy','pandas','matplotlib','seaborn']

# Open a file to write the output
with open("libraries_versions.txt", "w") as file:
    for library in libraries:
        # Run the pip show command
        result = subprocess.run(['pip', 'show', library], capture_output=True, text=True)
        
        # Write the result to the file
        file.write(f"--- {library} ---\n")
        file.write(result.stdout)
        file.write("\n\n")

print("Library versions have been written to 'libraries_versions.txt'")


# model to be fine-tuned
model_id = "gemma2_instruct_2b_en" #model that will be fine-tuned

# Original training dataset source
dataset_path = '/kaggle/input/input-ds-en-es-translations-only/Language_DS (11).xlsx' 
# List of sources to filter 
#sources_to_filter = ['Anki:https://www.manythings.org/anki/', 'Source2', 'Source3']
sources_to_filter = ['Anki:https://www.manythings.org/anki/']
# Nbr of source records to filter 
sample_size = 100000 # nbr of records from original dataset to include in training

# Training Configurations
token_limit_train = 77 #50 # 400 # not enough memory for 512 affects memory
token_limit_infer = 250 # token limit ot infer

batch_size_train = 4 # 4 for 30K rows
lora_name = "translator"
lora_rank = 16
lr_value = 1e-4
train_epoch =  1

# Kaggle fine-tuned model
fine_tuned_model = "Gemma2b_instruct_SpanishEU_TranslationOnly_LoS_C" 
preset_dir = ".\gemma_2b_instruct_2b_es"

# for Huggingface
# from kaggle site
kaggle_model_path = "asunsada/gemma_2b_instruct_2b_es/keras/gemma2b_instruct_spanisheu_translationonly_los_c"
repo_name = "Translation_SPA3" # HF folder - update this for every new model
username = "asunsada"  # Your Hugging Face username
# HF path to fine-tuned model
#path='hf://asunsada/Translation_SPA3'
#repo_id = f"{username}/{repo_name}"




import pandas as pd

df = pd.read_excel(dataset_path, sheet_name='DS es_eu 3')
data_row = len(df['original'])

print("Number of rows:", data_row)

#df = pd.read_excel(dataset_path)
df.head()


df.shape, df.columns


# Check data types of all columns
print(df.dtypes)


# Remove \n\n or any leading newlines at the beginning of each string in the column
df["original"] = df["original"].str.replace(r'^\n+', '', regex=True)
df["final"] = df["final"].str.replace(r'^\n+', '', regex=True)


#### Use Anki's entire DS and only Anki
#Anki:https://www.manythings.org/anki/# Keep only rows where 'source ' is Anki,
# human curated 
#df = df[df['source'] == 'Anki:https://www.manythings.org/anki/']

# Filter the DataFrame to keep only rows where 'source' matches any value in the list
df = df[df['source'].isin(sources_to_filter)]


data_row = len(df['original'])
print("Number of rows:", data_row)


### Take only source Anki and see how the model performs:
# 30K rows randomlmy taken from the larger Anki DS but balancing based on the sentence length. 
#Anki:https://www.manythings.org/anki/# Keep only rows where 'source ' is Anki,
# human curated 

# Filter the DataFrame to keep only rows where 'source' matches any value in the list
df = df[df['source'].isin(sources_to_filter)]

# Add a new column 'length' to the DataFrame
df['length'] = df['original'].apply(len)

# Number of rows in the filtered DataFrame
total_rows = len(df)

# Number of rows needed in the final DataFrame
#sample_size = 100000

# Group the DataFrame by the 'length' column
grouped = df.groupby('length')

# Calculate the number of rows to sample from each group
# Here we calculate a balanced proportion based on group size
group_sample_sizes = grouped.size() * (sample_size / total_rows)

# Initialize an empty list to hold the sampled DataFrames
sampled_df_list = []

# Sample from each group based on the calculated size
for length, group in grouped:
    # Calculate the number of rows to sample for this group
    group_size = group_sample_sizes[length]
    
    # Ensure that we round to a whole number of rows
    n_samples = int(group_size)
    
    # If the group has enough rows, sample randomly, otherwise take all rows
    sampled_group = group.sample(n=n_samples, random_state=42) if len(group) > n_samples else group
    
    # Append the sampled group to the list
    sampled_df_list.append(sampled_group)

# Concatenate all sampled groups into a single DataFrame
sampled_df = pd.concat(sampled_df_list)

# If there are more than 15,000 rows, truncate to the exact size
sampled_df = sampled_df.head(sample_size).reset_index(drop=True)

df= sampled_df

# Training set used
df.to_excel("/kaggle/working/TrainingSetUsed_DF.xlsx", index=False)


# Shuffle the dataset
df = df.sample(frac=1, random_state=0)
df.iloc[1000:1005]


data_row = len(df['original'])

print("Number of rows:", data_row)


# Add special tokens to training dataset for Gemma formatting
train = []
train_id = []

arr = []

# for i in range(data_row):
for i in range(data_row):
    item = f"<start_of_turn>user\n{df['original'][i]}<end_of_turn>\n<start_of_turn>model\n{df['final'][i]}<end_of_turn>"
    train.append(item)
    train_id.append(i)

print(len(train))
print("-" * 25)
print(train[0])
print("-" * 25)
print(train[1])


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)


gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma.summary()


# Util function to format the output of the Gemma response
import time

tick_start = 0


def tick():
    global tick_start
    tick_start = time.time()


def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")


def text_gen(model, prompt, token_limit):
    tick()

    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = model.generate(input, max_length=token_limit)
    
    output = output.replace(input, "") 
    # Remove unwanted characters
    characters_to_remove = ["#", "*", "[", "]", "{", "}", "<end_of_turn>", '"', "«", "»", "Traducción:", "'"]
    for char in characters_to_remove:
        output = output.replace(char, "")
    
    #print(output)
    tock()
    return output


# Enable LoRA for the model and set the LoRA rank (4, 8 or 16).
gemma.backbone.enable_lora(rank=lora_rank)
gemma.summary()


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
    gemma.backbone.save_lora_weights(model_name)

    # Evaluate
    text_gen(gemma, df['original'][train_id[0]], token_limit_train)


# epoch nbr
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0, 
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)


# epoch nbr
callback_epoch = keras.callbacks.EarlyStopping(monitor='loss',
                                               patience=3)
#optimizer = keras.optimizers.SGD(learning_rate=1e-4)


# Limit the input sequence length (to control memory usage).
gemma.preprocessor.sequence_length = token_limit_train
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


num_rows = len(df)
print(f"Number of rows for training in train df: {num_rows}")



#history = gemma.fit(train, epochs=train_epoch, batch_size=4, callbacks=[CustomCallback(), callback_epoch])
history = gemma.fit(train, epochs=train_epoch, batch_size=batch_size_train, callbacks=[ callback_epoch])

# with batch size of 4, the training runs faster, 30K divided by 4.
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
# Save the plot to a file (e.g., 'training_plot.png')
plt.savefig('/kaggle/working/training_plot.png')
plt.show()


print(history.history) # Training results



### SAVE IT

gemma.save_to_preset(preset_dir)
kaggle_uri = f"kaggle://asunsada/gemma_2b_instruct_2b_es/keras/{fine_tuned_model}"
keras_nlp.upload_preset(kaggle_uri, preset_dir)

