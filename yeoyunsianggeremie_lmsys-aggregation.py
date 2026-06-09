import numpy as np
import pandas as pd


df = pd.read_parquet("/kaggle/input/extra-dataset-process/all_extra_154k.parquet")
wsdm = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet")


df = pd.concat([wsdm, df.drop(columns=['turn'])])


df = df.drop_duplicates(subset=['prompt', 'response_a', 'response_b']).sample(frac=1, random_state=42).reset_index(drop=True)


df['winner'].value_counts()


from tqdm import tqdm
tqdm.pandas()

def preprocess(row):

    prompt = {'role': 'user', 'content': row['prompt']}
    if row['winner'] == 'model_b':
        rejected = [prompt] + [{'role': 'assistant', 'content': row['response_a']}]
        chosen = [prompt] + [{'role': 'assistant', 'content': row['response_b']}]
    else:
        rejected = [prompt] + [{'role': 'assistant', 'content': row['response_b']}]
        chosen = [prompt] + [{'role': 'assistant', 'content': row['response_a']}]

    return {'chosen': chosen, 'rejected': rejected}

df[['chosen', 'rejected']] = df.progress_apply(preprocess, axis=1, result_type='expand')


from datasets import Dataset

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_WRITE")

from huggingface_hub import login
login(secret_value_0)

ds = Dataset.from_pandas(df[['id', 'chosen', 'rejected']])
ds.push_to_hub(
    "bogoconic1/lmsys_wsdm_external_data",
    private=True
)

