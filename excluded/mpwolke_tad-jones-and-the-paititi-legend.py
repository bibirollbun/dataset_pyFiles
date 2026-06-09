#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook

import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook

def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-use-openai-models-on-kaggle/notebook

openai_key = load_secret('tadjones')

client = OpenAI(
  api_key=openai_key
)

prompt = "Produce a plan for Tad Jones an archaeologist explorer about how he could use GPT-4o to analyze imagery with the goal of finding evidences of the existence of Paititi city."

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);

