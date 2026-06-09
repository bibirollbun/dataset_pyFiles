!pip install -q -U datasets

!pip install -q -U keras-nlp

!pip install -q -U tensorflow-hub

!pip install -q -U tensorflow-cpu tensorflow-text

!pip install -q -U keras

!pip install -q -U langdetect


from datasets import load_dataset

dataset = load_dataset("jojo-ai-mst/Roleplay-Tamil")
dataset


import pandas as pd 

df = dataset["train"].to_pandas()
df.head()


#Aligning null values in English and Tamil

df.replace('#VALUE!',None, inplace=True)


df.info()


# i want every 6th row to be english prompt
# I want every 5th row to have an input

# LCM = 30 -> every 30th row will be english input


import numpy as np

# Step 1: Split non-null and null rows based on 'input'
non_null_input_df = df[df['input'].notna()].copy()
null_input_df = df[df['input'].isna()].copy()

# Step 2: Inject non-null rows into every 5th position
def inject_rows(null_df, non_null_df, interval):
    result = []
    non_null_idx = 0
    total_non_null = len(non_null_df)

    for i in range(len(null_df)):
        # Add the row from the null DataFrame
        result.append(null_df.iloc[i])

        # Every 'interval' rows, inject a non-null row if available
        if (len(result) % interval == 0) and non_null_idx < total_non_null:
            result.append(non_null_df.iloc[non_null_idx])
            non_null_idx += 1

    # Convert the result back to a DataFrame
    return pd.DataFrame(result)

# Inject non-null input rows into every 5th position
modified_df = inject_rows(null_input_df, non_null_input_df, interval=5)

# Step 3: Add the 'input_language' column
modified_df.reset_index(drop=True, inplace=True)  # Reset index for consistency
modified_df['input_language'] = np.where(modified_df.index % 6 == 0, 'en', 'ta')

# Display the resulting DataFrame
modified_df.head(10)


#manually removing bad quality input
modified_df = modified_df.drop(95)

#splitting test and train dataset
train_df = modified_df.head(1800)
test_df = modified_df.tail(10)


test_df.head(10)


import jax

jax.devices()


import os

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation overhead
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


import keras
import keras_nlp

# Uncomment the line below if you want to enable mixed precision training on GPUs
keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Create a device mesh with (1, 8) shape so that the weights are sharded across all 8 TPUs.
device_mesh = keras.distribution.DeviceMesh(
    (1, 8),
    ["batch", "model"],
    devices=keras.distribution.list_devices())


model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh) 

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in the decoder attention layers
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (None, model_dim)

layout_map["decoder_block.*attention_output.*kernel"] = (model_dim, None)
layout_map["decoder_block.*ffw_gating.*kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear.*kernel"] = (model_dim, None)


model_parallel = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)


# Training Configurations
token_limit = 1024
lora_name = "translator"
lora_rank = 4 #start from 4 and move up based on performance
lr_value = 5e-5 
train_epoch = 20 #model performs better with more epochs
model_id = "gemma2_instruct_2b_en"


model = keras_nlp.models.GemmaCausalLM.from_preset(model_id)

# print(model.backbone.layers)

decoder_block_1 = model.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
  print(f'{variable.path:<58}  {str(variable.shape):<16}  {str(variable.value.sharding.spec)}')


import time

tick_start = 0


def tick():
    global tick_start
    tick_start = time.time()


def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def prompt_selection(x):
    
    input = None
    
    tamil_prompt_template_with_input = f'''

    <start_of_turn>user\n
    First translate and write the given instruction to English. Then write the response in English Then translate the generated English response to Tamil.
    
    Your response should look like:
    Instruction in English: <English Translated Instruction> 
    Response in English: <Generated response in English> 
    Response in Tamil: <Translated response from English to Tamil> 
    
    ###Instruction: {x['அறிவுறுத்தல்']}{x['உள்ளீடு']}
    <end_of_turn>\n
    <start_of_turn>model\n
    '''
    
    tamil_prompt_template_without_input = f'''
    
     <start_of_turn>user\n
    First translate and write the given instruction to English. Then write the response in English Then translate the generated English response to Tamil.
    
    Your response should look like:
    Instruction in English: <English Translated Instruction> 
    Response in English: <Generated response in English> 
    Response in Tamil: <Translated generated response from English to Tamil> 
    
    ###Instruction: {x['அறிவுறுத்தல்']}
    <end_of_turn>\n
    <start_of_turn>model\n
    '''
    
    english_prompt_template_with_input = f'''
    
    <start_of_turn>user\n
    ###Instruction: {x['instruction']}{x['input']}
    <end_of_turn>\n
    <start_of_turn>model\n
    '''
    
    english_prompt_template_without_input = f'''
    
    <start_of_turn>user\n
    ###Instruction: {x['instruction']}
    <end_of_turn>\n
    <start_of_turn>model\n
    '''
    
    if x['input_language'] == 'en':
        if x['input']:
            input = english_prompt_template_with_input
        else:
            input = english_prompt_template_without_input

    else:
        if x['input']:
            input = tamil_prompt_template_with_input
        else:
            input = tamil_prompt_template_without_input


    return input

def text_gen(x):
    
    tick()

    input_prompt = prompt_selection(x)
    
    output = model.generate(input_prompt, max_length=512)

    print('Gemma output before ft:', output)
    tock()

# Inferencing with the testing dataset
for i,x in test_df[4:7].iterrows():
    text_gen(x)


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

train = []

def training_prompt_selection(x):
    
    input = None
    
    tamil_prompt_template_with_input = f'''

    <start_of_turn>user\n
    First translate and write the given instruction to English. 
    Then write the response in English.
    Then translate the English response to Tamil.
    
    Your response should look like:
    Instruction in English: <English Translated Instruction> 
    Response in English: <Generated response in English> 
    Response in Tamil: <Translated response from English to Tamil> 
    
    ###Instruction: {x['அறிவுறுத்தல்']}{x['உள்ளீடு']}
    <end_of_turn>\n
    <start_of_turn>model\n
    Instruction in English: {x['instruction']}{x['input']}
    Response in English: {x['response']}
    Response in Tamil: {x['பதில்']}
    <end_of_turn>
    '''
    
    tamil_prompt_template_without_input = f'''
    
     <start_of_turn>user\n
    First translate and write the given instruction to English. 
    Then write the response in English.
    Then translate the English response to Tamil.
    
    Your response should look like:
    Instruction in English: <English Translated Instruction> 
    Response in English: <Generated response in English> 
    Response in Tamil: <Translated response from English to Tamil> 
    
    ###Instruction: {x['அறிவுறுத்தல்']}
    <end_of_turn>\n
    <start_of_turn>model\n
    Instruction in English: {x['instruction']}
    Response in English: {x['response']}
    Response in Tamil: {x['பதில்']}
    <end_of_turn>
    '''
    
    english_prompt_template_with_input = f'''
    
    <start_of_turn>user\n
    ###Instruction: {x['instruction']}{x['input']}
    <end_of_turn>\n
    <start_of_turn>model\n
    {x['response']}
    <end_of_turn>
    '''
    
    english_prompt_template_without_input = f'''
    
    <start_of_turn>user\n
    ###Instruction: {x['instruction']}
    <end_of_turn>\n
    <start_of_turn>model\n
    {x['response']}
    <end_of_turn>
    '''
    
    if x['input_language'] == 'en':
        if x['input']:
            input = english_prompt_template_with_input
        else:
            input = english_prompt_template_without_input

    else:
        if x['input']:
            input = tamil_prompt_template_with_input
        else:
            input = tamil_prompt_template_without_input


    return input


for i,x in train_df.iterrows():
    
    item = training_prompt_selection(x)
    
    length = len(tokenizer(item))
    
    # skip data if the token length is longer than our limit
    if length < token_limit:
        train.append(item)

print(len(train))
print(train[0])
print(train[5])
print(train[6])


# Enable LoRA for the model and set the LoRA rank (4, 8 or 16).
model.backbone.enable_lora(rank=lora_rank)
model.summary()

# Limit the input sequence length (to control memory usage).
model.preprocessor.sequence_length = token_limit


# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(

    learning_rate=5e-5,
    weight_decay=0.01,
    
)

# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


def text_gen_after_ft(x):
    
    tick()

    input_prompt = prompt_selection(x)
    
    output = model.generate(input_prompt, max_length=1024)
    
    print('Gemma output after ft:',output)
    tock()


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/{lora_name}_{lora_rank}_last.lora.h5"
    model.backbone.save_lora_weights(model_name)

    
    # Inferencing with the testing dataset
    for i,x in test_df[4:7].iterrows():
        text_gen_after_ft(x)


history = model.fit(train, epochs=train_epoch, callbacks=[CustomCallback()])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()


# Save the finetuned model as a KerasNLP preset.
preset_dir = "./finetuned_gemma2_2b_tamil_toca_1"
model.save_to_preset(preset_dir)


# Upload the preset as a new model variant on Kaggle
kaggle_uri = f"kaggle://vikrantnarayan/gemma2/keras/finetuned_gemma2_2b_tamil_toca_1"
keras_nlp.upload_preset(kaggle_uri, preset_dir)




