!pip install datasets


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U "keras>=3"


!pip install transformers peft sentencepiece


import os
import keras
import keras_nlp
import pandas as pd


os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"


spanish_dataset_path = '/kaggle/input/spanish-dataset/Spanish_Dataset.csv'
df = pd.read_csv(spanish_dataset_path)[:3000]
df


data = []

for _, row in df.iterrows():
    instruction = f"Translate to Spanish: {row['english']}"
    response = row['spanish']

    # Format the English and Spanish phrases using the template
    template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
    data.append(template.format(instruction=instruction, response=response))





gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
gemma_lm.summary()


prompt1 = template.format(
    instruction="Translate to Spanish: Hello, how are you?",
    response="",
)

print(gemma_lm.generate(prompt1, max_length=256))


prompt2 = template.format(
    instruction="Translate to Spanish: What is the weather like in Madrid?",
    response="",
)

print(gemma_lm.generate(prompt2, max_length=256))


prompt3 = template.format(
    instruction="Translate to Spanish: I would like a cup of coffee, please.",
    response="",
)

print(gemma_lm.generate(prompt3, max_length=256))


prompt4 = template.format(
    instruction="Translate to Spanish. Explain the cultural significance of DÃ­a de los Muertos.",
    response="",
)

print(gemma_lm.generate(prompt4, max_length=256))





# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()


gemma_lm.preprocessor.sequence_length = 256

optimizer = keras.optimizers.AdamW(
    learning_rate=1e-5,
    weight_decay=0.005,
)

optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(data, epochs=5, batch_size=1)


prompt1 = template.format(
    instruction="Translate to Spanish: hello",
    response="",
)

print(gemma_lm.generate(prompt1, max_length=256))


prompt2 = template.format(
    instruction="Translate to Spanish: how are you?",
    response="",
)

print(gemma_lm.generate(prompt2, max_length=256))


prompt3 = template.format(
    instruction="Translate to Spanish: I like to read books",
    response="",
)

print(gemma_lm.generate(prompt3, max_length=256))


prompt4 = template.format(
    instruction="Translate to Spanish: one two three",
    response="",
)

print(gemma_lm.generate(prompt4, max_length=256))


prompt5 = template.format(
    instruction="Translate to Spanish: The weather is very nice today",
    response="",
)

print(gemma_lm.generate(prompt5, max_length=256))


prompt6 = template.format(
    instruction="Translate to Spanish. Explain the cultural significance of DÃ­a de los Muertos.",
    response="",
)

print(gemma_lm.generate(prompt6, max_length=256))


tmp_model_dir = "/kaggle/tmp/gemma2_spa" 
preset_dir = "gemma2_spa"
os.makedirs(tmp_model_dir, exist_ok=True)
gemma_lm.save_to_preset(tmp_model_dir)

print(f"Model saved to: {tmp_model_dir}")


import kagglehub
import keras_hub

model_version = 1
kaggle_username = kagglehub.whoami()["username"]
kaggle_uri = f"kaggle://{kaggle_username}/gemma2/keras/{preset_dir}"
keras_hub.upload_preset(kaggle_uri, tmp_model_dir)
print("Done!")




