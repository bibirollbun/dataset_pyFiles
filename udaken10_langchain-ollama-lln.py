!pip install langchain_ollama

import subprocess
import torch
from langchain_ollama import ChatOllama
from IPython.display import clear_output
import os
clear_output()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# let's download and host/open ollama server on kaggle
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen("ollama serve", shell=True)

clear_output()


# now we need to download the model we want to use, here it is gemma3n latest
subprocess.Popen("ollama pull gemma3n", shell=True)

clear_output()


llm = ChatOllama(
    model='gemma3n',
    temperature=0)


import time

t = time.time()
result = llm.invoke("make three funny jokes")
print(result.content)

print(time.time() - t)


import time

t = time.time()
result = llm.invoke("ほんまや！で終わるボケツッコミのジョークを言ってください")
print(result.content)

print(time.time() - t)


import time

t = time.time()
result = llm.invoke("ほんまや！で終わるノリツッコミのジョークを言ってください")
print(result.content)

print(time.time() - t)


import time

t = time.time()
result = llm.invoke("面白いジョークを言って笑わせてください")
print(result.content)

print(time.time() - t)


print("LLM is not good at Joking")




