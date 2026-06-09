# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('../input/map-charting-student-math-misunderstandings/train.csv')
train.tail(3)


test = pd.read_csv('../input/map-charting-student-math-misunderstandings/test.csv')
test.tail()


test['QuestionText'][0]


test['StudentExplanation'][0]


test['StudentExplanation'][1]


test['QuestionText'][2]


sub = pd.read_csv('../input/map-charting-student-math-misunderstandings/sample_submission.csv')
sub.tail()


train.info()


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Category', order=train['Category'].value_counts().index, color='g')
plt.xticks(rotation=45)
plt.title("Misconception Categories Distribution")
plt.show()


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='StudentExplanation', order=train['StudentExplanation'].value_counts().head().index, color='r')
plt.xticks(rotation=45)
plt.title("Student Explanations Distribution")
plt.show()


train['QuestionText'][36694]


train['StudentExplanation'][36694]


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Misconception', order=train['Misconception'].value_counts().head(20).index, color='purple')
plt.xticks(rotation=45)
plt.title("Misconceptions top 20 Distribution")
plt.show()


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Misconception', order=train['Misconception'].value_counts().tail(15).index, color='orange')
plt.xticks(rotation=60)
plt.title("Misconceptions (15 bottom) Distribution")
plt.show()


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3

import os

os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

import keras
import keras_nlp


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
#gemma_lm.summary()


#By York Yong https://www.kaggle.com/code/yorkyong/gemma-trial-ace-a-data-science-interview

MAP_dataset = []
    
for index, row in test.iterrows():
    question, answer = row['QuestionText'], row['MC_Answer']
    template = (f"QuestionText:\n{question}\n\nMC_Answer:\n{answer}")
    MAP_dataset.append(template)


# Enable LoRA for the model and set the LoRA rank to 64.
gemma_lm.backbone.enable_lora(rank=64)


# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 512
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


%%time

gemma_lm.fit(MAP_dataset, epochs=1, batch_size=1)


%%time
print(gemma_lm.generate("Which number is the greatest 6 or 6.2?", max_length=256))


%%time
print(gemma_lm.generate("A triangle split into nine equal smaller triangles. Six of them are shaded. What fraction of the shape is not shaded?", max_length=256))

