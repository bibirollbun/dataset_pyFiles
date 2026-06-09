import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# Load OpenAI API key securely from Kaggle secrets
user_secrets = UserSecretsClient()
openai_key = user_secrets.get_secret("openai_key_2025")

client = OpenAI(api_key=openai_key)

prompt = """
Design a detailed research plan for using GPT-4o to analyze multi-spectral satellite imagery for archaeological discoveries in Brazil, including:
- Key research goals
- Data sources and preprocessing
- Analysis and GPT-4o integration steps
- Collaboration and ethical considerations
- Field validation and visualization
- Funding and future research directions
"""

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(completion.choices[0].message.content)





