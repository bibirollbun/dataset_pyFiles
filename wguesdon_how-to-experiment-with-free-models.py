!pip install --upgrade openai  > /dev/null 2>&1


import os
from openai import OpenAI
from IPython.display import display, Markdown
from kaggle_secrets import UserSecretsClient

# Load API keys from secrets and set environment variables
def _set_api_keys():
    user_secrets = UserSecretsClient()
    os.environ["OPENAI_API_KEY"] = user_secrets.get_secret("openai_key")
    os.environ["OPEN_ROUTER_KEY"] = user_secrets.get_secret("openrouter_key")


# Choose the correct client based on the model identifier
def _get_client_for_model(model: str) -> OpenAI:
    _set_api_keys()
    if model.startswith("gpt-") or model.startswith("text-"):
        # Official OpenAI API
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        # Free models via OpenRouter
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY")
        )

# Unified function to create chat completions
def create_chat_completion(
    model: str,
    messages: list[dict],
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
) -> dict:
    client = _get_client_for_model(model)
    params = {"model": model, "messages": messages}
    if extra_headers:
        params.setdefault("extra_headers", {}).update(extra_headers)
    if extra_body:
        params.setdefault("extra_body", {}).update(extra_body)
    return client.chat.completions.create(**params)


messages = [
    {"role": "user", "content": (
        "How have recent methods and technologies been used to uncover new archaeological sitesin the Amazon forest? return your answer in a mardown format"
    )}
]


# 1️⃣ Use an official OpenAI GPT model
response = create_chat_completion(
    model="gpt-4o",
    messages=messages
)
display(Markdown(response.choices[0].message.content))


# 1️⃣ Use an official OpenAI GPT model
response = create_chat_completion(
    model="o3-mini",
    messages=messages
)
display(Markdown(response.choices[0].message.content))


# 2️⃣ Use a free OpenRouter model
response = create_chat_completion(
    model="microsoft/phi-4-reasoning-plus:free",
    messages=messages,
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>"
    }
)
display(Markdown(response.choices[0].message.content))


# 2️⃣ Use a free OpenRouter model
response = create_chat_completion(
    model="meta-llama/llama-3.3-8b-instruct:free",
    messages=messages,
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>"
    }
)
display(Markdown(response.choices[0].message.content))


# 2️⃣ Use a free OpenRouter model
response = create_chat_completion(
    model="qwen/qwen3-4b:free",
    messages=messages,
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>"
    }
)
display(Markdown(response.choices[0].message.content))


# 2️⃣ Use a free OpenRouter model
response = create_chat_completion(
    model="deepseek/deepseek-prover-v2:free",
    messages=messages,
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>"
    }
)
display(Markdown(response.choices[0].message.content))

