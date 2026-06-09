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


import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


# Below code will extract all links from the Wiki pages going upto 5 level depth.

# def extract_links_from_webpage(url):
#     """Extracts all links from a given webpage URL."""

#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for bad status codes

#         soup = BeautifulSoup(response.content, 'html.parser')
#         body = soup.find("div", {"id": "bodyContent"})
#         if body:
#             links = [a['href'] for a in body.find_all('a', href=True)]
#         else:
#             links = []

#         return links

#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")
#         return []



# maxDepth = 5 # Set the maximum depth to traverse child pages


# baseURL = 'https://en.wikipedia.org/wiki/Culture_of_Bengal'

# allLinks = []
# visitedLinks = set()  # To keep track of visited links

# def crawl_and_extract_links(url, depth):
#     if depth > maxDepth or url in visitedLinks:
#         return
#     visitedLinks.add(url)

#     try:
#         links = extract_links_from_webpage(url)
#         for link in links:
#             if link.endswith('.jpg') or link.endswith('.png') or link.endswith('.gif') or link.endswith('.svg'):
#                 continue
#             if link.startswith('#'):
#                 continue
#             if '/wiki/Category:' in link:
#                 continue
#             if '/Help:Category' in link:
#                 continue
#             if link.startswith('/'):
#                 link = 'https://en.wikipedia.org' + link
#             allLinks.append(link)
#             visitedLinks.add(link)
#             crawl_and_extract_links(link, depth + 1)
#     except Exception as e:
#         print(f"Error processing {url}: {e}")


# crawl_and_extract_links(baseURL, 0)
# allLinks = list(set(allLinks))
# allLinks.append(baseURL)
# print("Extracted Links size:" + str(len(allLinks)))


## This code will extract the text from each of the links from above websites.

# def extract_text_from_webpage(url):
#     """Extracts text content from a given webpage URL and saves it to a text file."""

#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for bad status codes

#         soup = BeautifulSoup(response.content, 'html.parser')
#         title = soup.title.string if soup.title else "Webpage_NO_Title" + random.randint(10, 99)

#         print(f"Title of the webpage: {title}")

#         # Extract all text from <p> tags
#         text_content = " ".join([p.get_text() for p in soup.find_all('p')])

#         # Save the extracted text to a file
#         with open('all_bengali_culture_texts/' + title + '.txt', 'w', encoding='utf-8') as file:
#             file.write(text_content)

#         print(f"Text extracted and saved for {title}")

#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")


# # Import the genAI library
# import google.generativeai as genai

# # for Kaggle secrects
# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()
# GOOGLE_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

# #configure
# genai.configure(api_key=GOOGLE_API_KEY)




# def translate_text(text, target_language="bangla"):
#     try:
#         model = genai.GenerativeModel("gemini-1.5-pro-001")
#         response = model.generate_content(
#             f"Translate the following text to {target_language}:\n\n{text}",
#             generation_config = genai.GenerationConfig(
#                                     max_output_tokens=1000,
#                                     temperature=0.1,
#                                 )
#         )
#         return response.text
#     except Exception as e:
#         print(f"Error during translation: {e}")
#         return None


# translated_files = []
# for file_info in files_data:
#     translated_text = translate_text(file_info['content'])
#     print(f"translation done for file {file_info['filename']}")
#     if translated_text:
#         translated_files.append({
#             'filename': file_info['filename'],
#             'original_text': file_info['content'],
#             'translated_text': translated_text
#         })
#     else:
#         print(f"Skipping translation for {file_info['filename']} due to error.")

# print(f'translated file sizes {len(translated_files)}')


# def generate_alpaca_json(text : str, model_name = 'gemini-1.5-flash'):
    # """
    #   This function uses Google's powerful LLM Gemini, to create structured data.
    #   It takes input a text string, combines it with specific instructions,
    #   and asks Gemini to create JSON in the "Alpaca" format,
    #   which is often used for tasks like instruction following and question answering.
    # """
    # try:
    #     model = genai.GenerativeModel(model_name)
    #     response = model.generate_content(
    #         prompt_text + text,
    #         generation_config = genai.GenerationConfig(
    #                                 temperature=0.7
    #                             )
    #     )
    #     return response.text
    # except Exception as e:
    #     print(f"Error during generation: {e}")
    #     return None


!pip install -q -U keras-nlp
!pip install -q -U "keras>=3"
!pip install -q -U pynvml


os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"


import keras
import keras_nlp


# Training Configurations
token_limit = 1024
seq_length = 512 # max size of input sequence for training
batch_size = 1
num_data_limit = 100
lora_name = "arnab_gemma_bengali"
lora_rank = 4
lr_value = 1e-4
train_epoch = 10
model_id = "gemma2_instruct_2b_en"


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()


## A helper method to add colors
def colorize_text(text):
    for word, color in zip(["Instruction", "Response"], ["blue", "green"]):
        text = text.replace(f"\n\n{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text


# Helper method to return Q&A
class GemmaInstruct:
    def __init__(self, max_length=1024):
        self.max_length = max_length
        self.prompt = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        self.gemma_lm = gemma_lm

    def query(self, question):
        response = self.gemma_lm.generate(
            self.prompt.format(instruction=question, response=""), max_length=self.max_length)
        display(Markdown(colorize_text(response)))


gemma_inst = GemmaInstruct()
query = "দুর্গাপূজার ইতিহাসের সংক্ষিপ্তসার লেখো বাংলা ভাষায়।"
gemma_inst.query(query)



import json
file_input_path = "/kaggle/input/bengali-culture-history-wiki/bengali_culture_wiki.txt"
file_input_path = arnabkarsarkar_bengali_culture_history_wiki_path + '/bengali_culture_wiki.txt'
json_data = []
# Read the text file
with open(file_input_path, 'r', encoding='utf-8') as file:
    text_data = file.read()


# Convert the text data into JSON list
json_data = json.loads(text_data)


input_data = []
for item in json_data:
    template = "Instruction:\n{instruction}\n\nResponse:\n{output}"
    input_data.append(template.format(instruction=item['instruction'], output=item['output']))

input_data = input_data[-20:]
print(input_data[-1])


gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()



gemma_lm.preprocessor.sequence_length = 256
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
history = gemma_lm.fit(input_data, epochs=train_epoch, batch_size=batch_size)



plt.plot(history.history['loss'])
plt.show()


# We will first try with the query above.
gemma_inst.query(query)


#
gemma_inst.query("বাংলার নামকরণের পিছনে কী কী ধারণা রয়েছে?")


MODEL_NAME = "gemma_bengali_history"
gemma_lm.save_to_preset(MODEL_NAME)


import  kagglehub
from datetime import datetime


kaggle_username = kagglehub.whoami()['username']
date_today = datetime.today().strftime('%Y-%m-%d')

print(kaggle_username)


MODEL_SLUG = 'bengali_history'

kagglehub.model_upload(
  handle = f"{kaggle_username}/{MODEL_NAME}/keras/{MODEL_SLUG}",
  local_model_dir = MODEL_NAME,
  version_notes = f'Update {date_today}')

