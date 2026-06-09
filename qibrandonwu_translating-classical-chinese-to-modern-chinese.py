import json
import os

#enter the API for your kaggle's gemma2 model into the following code
kaggle_api = {"username": your_username, "key": your_key}
with open("/content/kaggle.json", "w") as f:
    json.dump(kaggle_api, f)

os.environ['KAGGLE_CONFIG_DIR'] = "/content"



!pip install -q -U keras-nlp datasets
!pip install -q -U keras



data=[]

names = ["dataset1.json","dataset2.json","dataset3.json","dataset4.json","dataset5.json","dataset6.json"]
#we choose using these four datasets. If you want to use other dataset or more dataset, please modify the above code.
for name in names:
  with open(name) as f:
      temp = json.load(f)
      for i in temp:

        data.append("Instruction:\n"+i['古文']+"\n\nResponse:\n"+i["现代文"])



import keras_nlp
import keras


# Set JAX backend for Keras before importing any Keras modules
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"  # Full memory use for JAX backend

# Run with half precision if needed; uncomment if desired
# keras.config.set_floatx("bfloat16")

# Training Configurations
token_limit = 256  # Maximum token limit for each training instance
lora_name = "translator"  # Name for LoRA fine-tuning
lora_rank = 4  # Rank for LoRA tuning
lr_value = 1e-4  # Learning rate for fine-tuning
train_epoch = 5  # Number of epochs for training
model_id = "gemma2_instruct_2b_en"  # Model ID


print(keras_nlp.models.GemmaCausalLM.presets.keys()) # optional keras model we can use, here we use "gemma2_instruct_2b_en"



gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma.summary()


# Example usage of the model for generation before we fit the model
prompt = "Instruction:\n「但责己，不责人，此远怨之道也；但信己，不信人，此取败之由也。」翻译成现代汉语\n\nResponse:\n"
output = gemma.generate(prompt, max_length=100) # Adjust max_length as needed
print(output)



# Enable LoRA for the model and set the LoRA rank (4, 8 or 16).
gemma.backbone.enable_lora(rank=lora_rank)
gemma.summary()

# Limit the input sequence length (to control memory usage).
gemma.preprocessor.sequence_length = token_limit
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


gemma.fit(data,epochs=10,batch_size=8)


# Example usage of the model for generation after fitting
prompt = "Instruction:\n「正而过则迂，直而过则拙，故迂拙之人，犹不失为正直。」\n\nResponse:\n"
output = gemma.generate(prompt, max_length=100) # Adjust max_length as needed
print(output)


!pip install git+https://github.com/huggingface/trl.git
!pip install -U bitsandbytes
!pip install ipywidgets
!pip install python-dotenv
!pip install -U peft
!pip install matplotlib
!pip install seaborn
!pip install wordcloud
!pip install nltk
!pip install rouge_score
!pip install evaluate


test=[]
names = ["test.json"]
for name in names:
  with open(name) as f:
      temp = json.load(f)
      for i in temp:

        test.append("Instruction:\n"+i['古文']+"\n\nResponse:\n"+i["现代文"])


def get_response_text(text):
  # Find the Chinese content after "Response:"
  start_marker = "Response:"
  start_idx = text.find(start_marker) + len(start_marker)

  #The end location is located by a newline
  end_idx = len(text)

  response_text = text[start_idx:end_idx].strip()
  return response_text


def get_instruction_text(text):
    # Find the location of "Instruction:" and "Response:"
    start_marker = "Instruction:"
    end_marker = "Response:"
    start_idx = text.find(start_marker) + len(start_marker)
    end_idx = text.find(end_marker)

    # Extract the contents of the Instruction section
    instruction_text = text[start_idx:end_idx].strip()

    # Construct a new format
    response_text = f"Instruction:\n「{instruction_text}」翻译为现代汉语\n\nResponse:\n"
    return response_text



refer_data=[]
test_data=[]
for i in test:
  test_data.append(get_response_text(gemma.generate(get_instruction_text(i), max_length=1000)))
  refer_data.append(get_response_text(i))



import evaluate
import jieba
from nltk.translate.bleu_score import sentence_bleu

score_data=[]

for i in range(len(refer_data)):
  references_t = " ".join(jieba.cut(refer_data[i])).split(" ")
  predictions_t = " ".join(jieba.cut(test_data[i])).split(" ")
  references = [references_t]
  predictions = predictions_t
  score = sentence_bleu(references, predictions)
  score_data.append(score)


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
bleu_scores = score_data
# Plotting BLEU score
plt.figure(figsize=(10, 6))
plt.bar(range(len(bleu_scores)), bleu_scores, color='royalblue')
average_bleu_score=sum(score_data)/len(score_data)
plt.axhline(y=average_bleu_score, color='r', linestyle='--', label=f"Average BLEU: {average_bleu_score:.4f}")
plt.xlabel("Sample Index")
plt.ylabel("BLEU Score")
plt.title("BLEU Scores for Each Sample")
plt.legend(loc="best")
plt.show()


import random

random.seed(14)
random_numbers = random.sample(range(1, len(test)), k=10)
a=1
for i in random_numbers:
  print(a)
  print(test[i]+"\n")
  print(test_data[i])
  print(score_data[i])
  print("\n")
  a=a+1

