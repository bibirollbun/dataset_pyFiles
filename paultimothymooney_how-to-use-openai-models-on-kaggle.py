import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


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


openai_key = load_secret('openai_key_2025')

client = OpenAI(
  api_key=openai_key
)

prompt = "Produce a detailed plan for a research scientist and provide recommendations about how and where they could use GPT-4o to analyze multi-spectral satellite imagery with the goal of discovering evidence of ancient civilizations in Brazil."

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);






