DEMO = False
TRAINING = False


!pip install -q -U keras-nlp tensorflow-text
# Install tensorflow-cpu so tensorflow does not attempt to access the TPU.
!pip install -q -U tensorflow-cpu


from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'
os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = user_secrets.get_secret("KAGGLE_KEY")
# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"


import jax
jax.devices()


import keras
import keras_nlp


def set_distribution(mesh):
    model_dim = "model"
    layout_map = keras.distribution.LayoutMap(mesh)
    
    # Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
    layout_map["token_embedding/embeddings"] = (model_dim, None)
    # Regex to match against the query, key and value matrices in attention layers
    layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (model_dim, None, None)
    layout_map["decoder_block.*attention_output/kernel"] = (model_dim, None, None)
    layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, model_dim)
    layout_map["decoder_block.*ffw_linear/kernel"] = (model_dim, None)

    model_parallel = keras.distribution.ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch"
    )
    
    keras.distribution.set_distribution(model_parallel)


def test_gemma(model_id, user_prompt):
    prompt = f"user\n{user_prompt}\nmodel\n"
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
    output = gemma_lm.generate(prompt, max_length=512)
    print("\nGemma output:")
    print(output)
    del gemma_lm


TEST_PROMPT = """
Tu tarea es analizar y corregir textos escritos en Spanglish para convertirlos en español estándar. Sigue estos pasos:

Identifica las palabras en inglés dentro del texto
Para cada palabra en inglés:

Verifica si existe un término equivalente aceptado por la Real Academia Española (RAE)
Si existe un término aceptado por la RAE, reemplaza la palabra
Si no existe un equivalente aceptado, mantén la palabra original en inglés


Mantén el significado y contexto original del texto
Conserva el tono y registro del texto original

Responde con:

El texto corregido
Lista de cambios realizados, indicando: palabra original → reemplazo (si aplica)

TEXTO A CORREGIR:
Necesito que hagas un deploy del nuevo feature en el server de staging. El frontend está ready, pero todavía hay algunos bugs en el backend que necesitan fixing. También hay que updatear las dependencies porque el framework está outdated. Pushea los changes al repo cuando termines, pero antes asegúrate de que todos los unit tests estén running correctamente. El performance del API está slow, así que también deberíamos implementar más caching. Ya hice un backup de la database por si acaso crashea durante el update.
"""


device_mesh = keras.distribution.DeviceMesh(
    (1, 4),
    ["batch", "model"],
    devices=keras.distribution.list_devices()[4:]
)
set_distribution(device_mesh)
test_gemma('gemma2_instruct_2b_en', TEST_PROMPT)


device_mesh = keras.distribution.DeviceMesh(
    (1, 8),
    ["batch", "model"],
    devices=keras.distribution.list_devices()
)
set_distribution(device_mesh)

test_gemma('gemma2_instruct_9b_en', TEST_PROMPT)


!cat /kaggle/input/rae-spanglish/dpd.txt /kaggle/input/rae-spanglish/twitter.txt > reference.txt


with open('reference.txt') as f:
    text = f.read()
    chunks = text.split(20 * '-')
    chunks = [c.strip() for c in chunks if c.strip()]


# Training Configurations
token_limit = 4096
num_data_limit = 100
lora_name = "my_lora"
lora_rank = 4
lr_value = 1e-3
train_epoch = 5
model_id = "gemma2_instruct_2b_en"


def search_chunks(search_word):
    search_word = search_word.lower()
    matching_chunks = [chunk for chunk in chunks if search_word in chunk.lower()]
    return matching_chunks[:2]


import json
with open('/kaggle/input/rae-spanglish/data.json', encoding='utf-8') as file:
    raw_data = json.load(file)


def prepare_prompt(text, words, std):
    
    prompt = f"user\nTu tarea es analizar y corregir textos escritos en Spanglish para convertirlos en español estándar. Identifica primero las palabras en inglés de este texto:\n{text}\n"
    prompt += f"model\nPalabras en inglés encontradas: {','.join(words)}<end_of_turn><eos>\n"

    context = [chunk for word in words for chunk in search_chunks(word)]

    if context:
        context = ('\n' + 20 * '-' + '\n').join(context)
        prompt += f"<start_of_turn>user\nUsa el siguiente contexto basado en la informacion recopilada de la Real Academia Española para convertir el texto entregado a español estándar:\n{context}\n"
    else:
        prompt += f"<start_of_turn>user\nAdvertencia: No se pudo encontrar contexto relevante para esta consulta. Intenta responder usando tus conocimientos propios.\n"
    
    prompt += f"model\n{std}<end_of_turn><eos>"
    
    return prompt


data = [prepare_prompt(**i) for i in raw_data]


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)


train = []

for x in data:
    length = len(tokenizer(x))
    # skip data if the token length is longer than our limit
    if length < token_limit:
        train.append(x)
        if (len(train) >= num_data_limit):
            break


train[5]


device_mesh = keras.distribution.DeviceMesh(
    (1, 4),
    ["batch", "model"],
    devices=keras.distribution.list_devices()[4:]
)
set_distribution(device_mesh)
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)


# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


!mkdir gemma_finetuning


import re

def find_words(text):# Regular expression to capture text between the two phrases
    pattern = r"Palabras en inglés:(.*?)<end_of_turn>"
    match = re.search(pattern, text)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text.split(',')
    else:
        return []


def convert_to_std_spanish(text):
    prompt = f"user\nTu tarea es analizar y corregir textos escritos en Spanglish para convertirlos en español estándar. Identifica primero las palabras en inglés de este texto:\n{text}\nmodel\n"
    response = gemma_lm.generate(prompt, max_length=token_limit)
    
    words = find_words(response)#[w for w in response.split('Palabras en inglés encontradas:')[-1].strip('<end_of_turn>').strip().split(',') if w]
    
    context = [chunk for word in words for chunk in search_chunks(word)]

    if context:
        context = ('\n' + 20 * '-' + '\n').join(context)
        response += f"\n<start_of_turn>user\nUsa el siguiente contexto basado en la informacion recopilada de la Real Academia Española para convertir el texto entregado a español estándar:\n{context}\nmodel\n"
    else:
        response += f"\n<start_of_turn>user\nAdvertencia: No se pudo encontrar contexto relevante para esta consulta. Intenta responder usando tus conocimientos propios.\nmodel\n"

    response = gemma_lm.generate(response, max_length=token_limit)
    
    return response


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/gemma_finetuning/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
    gemma_lm.backbone.save_lora_weights(model_name)

    # Evaluate
    # convert_to_std_spanish("Se ha producido un mailing de 2 millones de cartas. Las autoridades ya estan investigando el hecho.")

history = gemma_lm.fit(train, epochs=10, batch_size=2, callbacks=[CustomCallback()])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()


print(convert_to_std_spanish("Necesito que me ayudes con mi e-mail."))

