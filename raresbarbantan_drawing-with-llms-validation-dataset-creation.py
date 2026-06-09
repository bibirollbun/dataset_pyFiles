!pip install -U -q "google-genai==1.7.0"


from google import genai
from google.genai import types
import kagglehub
import pandas as pd
from pydantic import BaseModel


from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)


drawing_with_llms_path = kagglehub.competition_download("drawing-with-llms")

df = pd.read_csv(f"{drawing_with_llms_path}/train.csv")
questions = pd.read_parquet(f"{drawing_with_llms_path}/questions.parquet")
df = pd.merge(df, questions, how="left", on="id")
df.to_csv('example.csv', index=False)
df.head()


document_file = client.files.upload(file='example.csv')


import random
import string

def generate_hex_id() -> str:
    """Generate a random hexadecimal ID of specified length."""
    hex_chars = string.digits + 'abcdef'  # 0-9 and a-f
    return ''.join(random.choice(hex_chars) for _ in range(6))

generate_hex_id()


class ValidationSample(BaseModel):
    id: str
    description: str
    question: str
    choices: list[str]
    answer: str

prompt = """You are given some examples from a validation dataset, containing descriptions for simple SVG images, and several questions with choices and a correct answer for each.
Your job is to create a new dataset by generating {} image descriptions, and 4 questions each. Use generate_hex_id to make sure each image has a unique id.
"""


# Define a retry policy. The model might make multiple consecutive calls automatically
# for a complex query, this ensures the client retries if it hits quota limits.
from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)


def create_examples(n=2):
    chat = client.chats.create(model='gemini-2.0-flash')

    response = chat.send_message(
        [prompt.format(n), document_file],
        config={
            "tools": [generate_hex_id],
            "temperature": 0.9,
            "top_p": 0.99,
            "top_k": 40
        }
    )

    response = chat.send_message(
        "Convert to json please",
        config={
            "response_schema": list[ValidationSample],
            "response_mime_type": "application/json"
        }
    )
    return response.parsed


from tqdm.auto import tqdm

total_images = 100
batch_size = 10

examples = []
for _ in tqdm(range(total_images // batch_size)):
    examples.extend(create_examples(batch_size))


extended_df = pd.DataFrame([dict(ex) for ex in examples])
extended_df.to_csv("validation.csv",index=False)
print(f"Created a dataset with {len(extended_df)} rows\n")

extended_df.head()


import kagglehub

handle = 'raresbarbantan/draw-svg-validation'
local_dataset_file = '/kaggle/working/validation.csv'

kagglehub.dataset_upload(handle, local_dataset_file)

