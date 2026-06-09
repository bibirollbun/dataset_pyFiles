import numpy as np 
import pandas as pd 
import os
import keras_nlp
import tensorflow as tf
import keras

from tensorflow.keras.optimizers import AdamW


idioms = pd.read_csv("/kaggle/input/chinese-proverbs/Chinese_proverbs.csv")
idioms = idioms[['in_chinese', 'text']]
idioms.head()


data = pd.DataFrame({
    "input" : [f"<start_of_turn>user\n What is this idiom's meaning in English：{idiom}?<end_of_turn>" for idiom in idioms['in_chinese']],
    "output" : [f"\n<start_of_turn>model\n This idiom in English means: {meaning}.<end_of_turn>" for meaning in idioms['text']]
})

inputs = data['input'].tolist()
outputs = data['output'].tolist()

# Create a TensorFlow dataset from raw strings
train_dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(1)


model_id = "gemma2_instruct_2b_en"
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)


# Enable LoRA
gemma_lm.backbone.enable_lora(rank=4)


keras.config.set_floatx("bfloat16")


optimizer = AdamW(learning_rate = 1e-4)

optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gemma_lm.backbone.load_lora_weights("gemma2_lora_weights.lora.h5")
    input = f"<start_of_turn>user\n What is this idiom's meaning in English：一笑解千愁?<end_of_turn>\n<start_of_turn>model\n"
    output = gemma_lm.generate(input, max_length=64)
    print("\nGemma output:")
    print(output)
      
gemma_lm.preprocessor.sequence_length = 128
gemma_lm.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

#Train
history = gemma_lm.fit(train_dataset, epochs=6, verbose=1, callbacks=[CustomCallback()])

#Save LoRa Weights
gemma_lm.backbone.save_lora_weights("gemma2_lora_weights.lora.h5")


gemma_lm.backbone.load_lora_weights("gemma2_lora_weights.lora.h5")
#Test on a unseen example
# Tokenize the input
tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)
test_input = "<start_of_turn>user\n What is this idiom's meaning in English：举一反三?<end_of_turn>\n <start_of_turn>model"
#Roughly means to infer three things from one

output_text = gemma_lm.generate(test_input, max_length=50)
print("Model Output:", output_text)


import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.show()


"""
from keras_nlp.models import GemmaCausalLM, GemmaTokenizer

model_id = "gemma2_instruct_2b_en"
gemma_lm = GemmaCausalLM.from_preset(model_id)

# Enable LoRA and load weights
gemma_lm.backbone.enable_lora()
gemma_lm.backbone.load_lora_weights("./models/gemma2_lora_weights.lora.h5")

# Prepare input
test_input = "<start_of_turn>user\n What is this idiom's meaning in English：一笑解千愁?<end_of_turn>"

# Generate output
output_tokens = gemma_lm.generate(test_input, max_length=50)
tokenizer = GemmaTokenizer.from_preset(model_id)
output_text = tokenizer.detokenize(output_tokens).numpy()[0].decode("utf-8")

print("Model Output:", output_text)
"""

