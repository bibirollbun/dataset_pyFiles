import pandas as pd

df_path = '/kaggle/input/italian-enlightenment-q-and-a/gemma_enlightened_dataset_def.csv'
df = pd.read_csv(df_path)
len(df)


df.head()


data = []

SYSTEM_INSTRUCTION = '''Sei un intellettuale e filosofo illuminista italiano del XVIII secolo.
Rispondi alle domande utilizzando lo stile, la lingua, l'approccio critico e il sistema di pensiero dell'illuminismo italiano.'''

df["formatted"] = df.apply(
    lambda row: f"""<system>
    {SYSTEM_INSTRUCTION}
    </system>
    <instruction>
    {row['generated_prompt']}
    </instruction>
    <response>
    {row['completion']}
    </response>
    """,
    axis=1
)

data = df['formatted'].tolist()
data[0]


!pip install -q -U keras-nlp
!pip install -q -U keras

import os

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras


gemma_lm = keras_nlp.models.CausalLM.from_preset('/kaggle/input/lumigemma/keras/gemma2-enlightened/2')


template = "System:\n{SYSTEM_INSTRUCTION}\n\nInstruction:\n{instruction}\n\nResponse:\n{response}"


SYSTEM_INSTRUCTION = """Sei un intellettuale e filosofo illuminista italiano del XVIII secolo, con una profonda conoscenza di architettura, urbanistica, economia, politica e filosofia.
Il tuo compito Ã¨ rispondere in modo dettagliato e approfondito alle domande che ti vengono poste, utilizzando lo stile, la lingua, l'approccio critico e il sistema di pensiero dell'Illuminismo italiano.
Le tue risposte devono essere ben argomentate, strutturate in paragrafi, e supportate da esempi concreti o riferimenti a teorie illuministe.
Utilizza un linguaggio ricco e colto, tipico dell'epoca illuminista.
"""


prompt = template.format(
    SYSTEM_INSTRUCTION=SYSTEM_INSTRUCTION,
    instruction = """
    Quali sono le principali cause della disuguaglianza sociale e come si possono affrontare attraverso riforme illuministe?
""",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=10, seed=2) 

gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=2048))


prompt = template.format(
    SYSTEM_INSTRUCTION=SYSTEM_INSTRUCTION,
    instruction='''Qual Ã¨ il ruolo dell'educazione nella formazione di un cittadino illuminato e responsabile?''',
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=10, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=1024))


prompt = template.format(
    SYSTEM_INSTRUCTION=SYSTEM_INSTRUCTION,
    instruction='''Come dovrebbe essere strutturato un sistema educativo che promuova la ragione, il pensiero critico e la cittadinanza attiva?''',
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=10, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=1024))

