import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
import requests
import base64


image_url = "https://raw.githubusercontent.com/Purvanshsingh/Amazonia/master/Manuscrito_512_01.jpg"
image_data = requests.get(image_url).content
image_base64 = base64.b64encode(image_data).decode("utf-8")

response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "user", "content": [
            {"type": "text",
             "text": """Please analyze the content of this manuscript. 
            Extract and present the readable text, provide a clear explanation of what is written,
            and identify any references or evidence that may point to archaeological sites or ancient human activity.
            If any such sites are described or mentioned, please highlight them along with the supporting details from the manuscript."""},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]}
    ],
    max_tokens=1000
)

print(response['choices'][0]['message']['content'])




