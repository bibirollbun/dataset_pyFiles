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


from IPython.display import clear_output


!pip install langchain
clear_output()


!pip install langchain-community
clear_output()


!pip install langchain-ollama
clear_output()


#!pip install peft -q
#!pip install accelerate -q
#!pip install transformers -q


from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)


test = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
test.head()


submission = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv")
submission.head()


!pip install ipython-autotime

%load_ext autotime
clear_output()


!curl https://ollama.ai/install.sh | sh
!sudo apt install -y neofetch
clear_output()


import subprocess
import time

# Start ollama as a backrgound process
command = "nohup ollama serve&"

# Use subprocess.Popen to start the process in the background
process = subprocess.Popen(command,
                            shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
print("Process ID:", process.pid)
# Let's use fly.io resources
#!OLLAMA_HOST=https://ollama-demo.fly.dev:443
time.sleep(5)  # Makes Python wait for 5 seconds


!ollama -v


OLLAMA_MODEL='llama3'
#OLLAMA_MODEL='phi3:mini'
#OLLAMA_MODEL='mistral'
#OLLAMA_MODEL='gemma:2b'
import os
os.environ['OLLAMA_MODEL'] = OLLAMA_MODEL
!echo $OLLAMA_MODEL


!ollama run $OLLAMA_MODEL "Who wrote Supplier Selection: An MCDA-Based Approach "
clear_output()


OLLAMA_SERVER = 'http://127.0.0.1:11434'


%%time
MODEL='llama3'
model = Ollama(model=OLLAMA_MODEL,base_url=OLLAMA_SERVER,keep_alive=True)



parser = StrOutputParser()
#response_from_model = model.invoke(test["topic"].iloc[0])
#parsed_response = parser.parse(response_from_model)
#parsed_response


#Modify the above output with prompt
template = '''Write an essay no more than 100 words to the {topic}'''
prompt = PromptTemplate.from_template(template)
prompt.format(topic=test['topic'].iloc[0])


formatted_prompt = prompt.format(topic=test['topic'].iloc[0])
response_from_model = model.invoke(formatted_prompt)
parsed_response = parser.parse(response_from_model)
print(parsed_response)


#submission
for i,row in test.iterrows():
    idx = submission[submission['id']==row['id']].index
    formatted_prompt = prompt.format(topic=test['topic'].iloc[i])
    response_from_model = model.invoke(formatted_prompt)
    parsed_response = parser.parse(response_from_model)
    submission.at[idx[0],'essay']= parsed_response.replace("##","")


submission['essay'].iloc[0]


submission['essay'].iloc[1]


submission['essay'].iloc[2]


submission.to_csv("/kaggle/working/submission.csv",sep=',',index=False)

