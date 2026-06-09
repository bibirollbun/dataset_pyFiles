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


import os
import time
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient


#By Paul Mooney https://www.kaggle.com/code/paultimothymooney/how-to-upload-large-files-to-gemini-1-5/notebook

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="#System Prompt:You are an AI Research Assistance designed to analyse psychiatry concepts of some body-focused repetitive behaviors (BFRBs). Answer briefly, some questions referring only to the context.\"\n",
)


ex_d = genai.upload_file("/kaggle/input/anxiety-disorders/anxiety_disorders.txt")


chat_session = model.start_chat(
history =[
    {
        'role':'user',
        'parts': [
            ex_d,
        ]
    }
])

def chatI(prompt):
    response = chat_session.send_message(prompt)
    print(response.text)


# Anxiety disorders: body-focused repetitive behaviors (BFRBs)
bfrb = open("../input/anxiety-disorders/anxiety_disorders.txt", "r").read()
print (bfrb[:])


Prompt="Cite some body-focused repetitive behaviors (BFRBs)?"


chatI(Prompt)


chatI("What are the symptoms of excoriation disorders?")


chatI("When skin picking is considered a disorder?")


chatI("What are the types of skin picking?")


chatI("What are the treatments for excoriation disorder?")


chatI("What is Air Writing?")


chatI("Why do hyperlexic learners do air writing?")


chatI("What is hypernumeracy?")


#chatI("Should Air writing be discouraged?")


chatI("How Air writing can be used?")


chatI("What is trichotillomania?")


chatI("What are the types of hair pulling?")


chatI("What people do with their pulled hair?")


#chatI("Why is important to know what they do with the pulled hair?")


chatI("How is trichotillomania treatment?")


#chatI("What are the related disorders of people with trichotillomania?")


#chatI("How can we distinguish BFRB-like and non-BFRB-like activity?")

