!pip install openai
!pip install google-generativeai
!pip install anthropic
!pip install xai-sdk


import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["OPENAI_API_KEY"] = user_secrets.get_secret("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = user_secrets.get_secret("GEMINI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = user_secrets.get_secret("ANTHROPIC_API_KEY")
os.environ["XAI_API_KEY"] = user_secrets.get_secret("XAI_API_KEY")


prompt = """
あなたはコードゴルフのエキスパートです。
下記のアルゴリズムを実現する、できるだけ短いコードを書いてください。

def p(g):
    print("今日も元気にコードゴルフ！⛳️")
    return g
"""





from openai import OpenAI
openai_client = OpenAI()


def call_openai_api(prompt: str, model: str="gpt-5.1") -> str:
    response = openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.output_text


print(call_openai_api(prompt))


from google.genai import Client
gemini_client = Client()


def call_gemini_api(prompt: str, model: str="gemini-2.5-flash") -> str:
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


print(call_gemini_api(prompt))


from anthropic import Anthropic
anthropic_client = Anthropic()


def call_anthropic_api(prompt: str, model: str="claude-opus-4-5", max_tokens: int=4096) -> str:
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return message.content[0].text


print(call_anthropic_api(prompt))


from xai_sdk import Client 
from xai_sdk.chat import user, system
xai_client = Client()


def call_xai_api(prompt: str, model: str="grok-4-1-fast-non-reasoning") -> str:
    chat = xai_client.chat.create(model=model)
    chat.append(user(prompt))
    response = chat.sample()
    return response.content


print(call_xai_api(prompt))




