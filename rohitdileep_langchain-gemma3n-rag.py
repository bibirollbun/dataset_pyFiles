


## Installing Libraries ##

!pip install langchain langchain_community  unstructured langchain_chroma langchain_chroma chromadb  langchain_ollama langchain_huggingface -q





## Importing Libraries for ChatOllama 
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


### Creating LLM ###
llm = ChatOllama(
    model='gemma3n',
    temperature=0)


### Extracting data for Rag ##

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

urls = ["https://www.ready.gov/evacuation" , "https://www.ready.gov/wildfires" , "https://www.ready.gov/people-disabilities" , 
       "https://www.ready.gov/avalanche" , "https://www.ready.gov/tsunamis" , "https://www.ready.gov/tornadoes" , "https://www.ready.gov/volcanoes" ,
       "https://www.ready.gov/landslides-debris-flow" , "https://www.ready.gov/floods" , "https://www.ready.gov/avalanche" , 
        "https://www.wikihow.com/Survive-an-Avalanche" , "https://www.mountaineering.scot/safety-and-skills/essential-skills/weather-conditions/avalanches/avalanche-rescue" , 
       "https://emsaok.gov/resource-library/summer-safety-tips/creating-a-tornado-survival-kit/" , "https://weready.org/tsunami/index.php?option=com_content&view=article&id=20&Itemid=15"]

loader = UnstructuredURLLoader(urls)

docs = loader.load()

docs_splitter = RecursiveCharacterTextSplitter(chunk_size = 600 , chunk_overlap = 50)
docs = docs_splitter.split_documents(docs)


!mkdir retriever


### Creating Chroma Rag 

from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore  = Chroma.from_documents(docs , embeddings , persist_directory='/kaggle/working/retriever')

retriever  = vectorstore.as_retriever()


###Loading it offline 

vectorstore = Chroma(
    persist_directory="/kaggle/working/retriever",
    embedding_function=embeddings
)

# Get retriever
retriever = vectorstore.as_retriever()


### Creating ChatPrompt template ##
from langchain.prompts import ChatPromptTemplate 


system_message = """

You are a disaster response assistant. Use ONLY the provided context to help victims during natural disasters.
Be factual, precise, and detailed in your response.

If the answer to the question is NOT in the context, respond with:
"I don't have information about that. Please contact emergency services."

NEVER make up answers. NEVER repeat the fallback message if the answer is present in the context.

Context:
{context}

Question: {input}

Answer:
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message)
])


### Creating Rag chain ##
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

output_parser = StrOutputParser()

document_chain  = create_stuff_documents_chain(llm  =  llm , prompt = prompt , output_parser = output_parser )

rag_chain = create_retrieval_chain(retriever = retriever , combine_docs_chain = document_chain )



### Inferencing ##
input_que = "We got an Tsunami warning, so what to do next?"
response  = rag_chain.invoke({'input' : input_que})


print(response['answer'])




