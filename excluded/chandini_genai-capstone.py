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


print("Hello AgriCopilotLite")


!pip uninstall -qqy jupyterlab  # Remove unused packages from Kaggle's base image that conflict
!pip install -U -q "google-genai==1.7.0"


from google import genai
from google.genai import types

from IPython.display import HTML, Markdown, display


from google.api_core import retry


is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

genai.models.Models.generate_content = retry.Retry(
    predicate=is_retriable)(genai.models.Models.generate_content)


from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")


client = genai.Client(api_key=GOOGLE_API_KEY)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents='Explain farming to a beginner who has 5 acres of Rice paddy at this point'
)

print(response.text)



# multi-turn chat structure
chat = client.chats.create(model='gemini-2.0-flash', history=[])
response = chat.send_message("Hello! My name is Trisha")
print(response.text) 


response = chat.send_message("Can you tell me what crops can I do in the next year insted of rice? ")
print(response.text)


model_config = types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
    #max_output_tokens=5,
)

#story_prompt="you are an expert in farming, give a summary on what crops are good in a season"
zero_shot_prompt = """Suggest if a crop is a good choice for given conditions.
Question: I am planning to grow quinoa in my backyard in Austin,Texas along with other vegetables 
Answer: """

response = client.models.generate_content(
    model='gemini-2.0-flash', 
    config=model_config,
    contents= zero_shot_prompt
    #story_prompt
)

print(response.text)


few_shot_prompt = """Parse the question into valid JSON:
EXAMPLE: 
I want to grow okra in my backyard in Austin. 
JSON Response: 
```
{
"Crop": "okra",
"Soil Type": "well-drained",
"Optimal Planting Time":"Late Feb/Early March or Late Aug/Early Sep, avoid hottest time"
"Water Management" : "Consistent moisture during germination and early growth, reduce once plants are established"
"Pests and Diseases": "",
"Varieties" : "Heat tolerant - Cherry Vanilla, Red Head",
"Other Vegetables": "sweet potatoes, tomatoes"
}
```

EXAMPLE: 
I want to grow quinoa in my backyard in Seattle. 
JSON Response: 
```
{
"Crop": "okra",
"Soil Type": "well-drained",
"Optimal Planting Time":"Late Feb/Early March or Late Aug/Early Sep, avoid hottest time"
"Water Management" : "Consistent moisture during germination and early growth, reduce once plants are established"
"Pests and Diseases": "",
"Varieties" : "Heat tolerant - Cherry Vanilla, Red Head"
}
```
"""

question = "Can I grow indian vegetable bottle gourd and roses in my backyard, austin, texas"

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=1,
        max_output_tokens=500,
    ),
    contents=[few_shot_prompt, question]
)

print(response.text)


!pip install chromadb


import chromadb
import pandas as pd


! pip install PyMuPDF


# read premature-scene cotton pdf
import fitz


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


doc = extract_text_from_pdf("/kaggle/input/premature-scene/Cotton_1.pdf")


!pip install chromaDB


from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

from google.genai import types


# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]


import chromadb

DB_NAME = "cottonPrematureDB"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

#db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

db.add(documents=doc, ids=str(1))


db.count()


embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "Tell me about prematuring?"

result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

Markdown(all_passages[0])


embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "What are some reasons for premature?"

result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

Markdown(all_passages[0])


query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

print(prompt)


answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)

Markdown(answer.text)

