


!pip install langchain langchain_community langgraph unstructured langchain_ollama langchain_huggingface chromadb -q





from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    temperature=0.5)


## Creating Documents for RAG  ###
urls = ["https://www.ready.gov/evacuation" , "https://www.ready.gov/wildfires" , "https://www.ready.gov/people-disabilities" , 
       "https://www.ready.gov/avalanche" , "https://www.ready.gov/tsunamis" , "https://www.ready.gov/tornadoes" , "https://www.ready.gov/volcanoes" ,
       "https://www.ready.gov/landslides-debris-flow" , "https://www.ready.gov/floods" , "https://www.ready.gov/avalanche" , 
        "https://www.wikihow.com/Survive-an-Avalanche" , "https://www.mountaineering.scot/safety-and-skills/essential-skills/weather-conditions/avalanches/avalanche-rescue" , 
       "https://emsaok.gov/resource-library/summer-safety-tips/creating-a-tornado-survival-kit/" , 
        "https://weready.org/tsunami/index.php?option=com_content&view=article&id=20&Itemid=15" , 
       "https://www.survivorfilter.com/blogs/home/surviving-a-flood-7-safety-tips?srsltid=AfmBOopXJ5lx92O51bRAjkIY3h19H_PdY7qNrkpL0JWPuNa2ldlsWEvl" ,
       "https://www.homes247.in/blogs/causes-of-cloudburst-1570" , 
       "https://byjus.com/free-ias-prep/cloudburst/" , 
       "https://english.jaf.or.jp/safe-driving/disaster/protect-in-an-earthquake"]

loader = UnstructuredURLLoader(urls)
docs  = loader.load()

text_splitter  = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50 )
docs = text_splitter.split_documents(docs)





!mkdir retriever



## Creating Chroma-RAG

from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name =  "all-MiniLM-L6-v2")
db  = Chroma.from_documents(docs , embeddings , persist_directory='/kaggle/working/retriever')
retriever = db.as_retriever()


###Loading it offline 

vectorstore = Chroma(
    persist_directory="/kaggle/working/retriever",  # your offline directory
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()


### Defining Template  for Chatbot ##

from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder


sys_prompt =  """ You are a disaster response assistant. Use ONLY the provided context to help victims during natural disasters.
Be factual, precise, and detailed in your response.

If the answer to the question is NOT in the context, respond with:
"I don't have information about that. Please contact emergency services."

NEVER make up answers. NEVER repeat the fallback message if the answer is present in the context.

Context:
{context}

"""



prompt  = ChatPromptTemplate.from_messages([
    ('system' , sys_prompt) ,
    MessagesPlaceholder('chat_history') ,
    ('user' , '{question}' )
])




## Langgraph for creating State

from typing_extensions import TypedDict
from typing import Annotated 
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage , HumanMessage
from langgraph.graph import StateGraph , START , END 


class State(TypedDict):
    question : str
    context : str
    chat_history : Annotated[list[BaseMessage] , add_messages]



def rag(state: State):
    docs = retriever.invoke(state['question'])
    content = "\n\n".join(doc.page_content for doc in docs)
    return {'context': content}

def chatbot(state: State):
    message = prompt.invoke({
        'context': state['context'],
        'question': state['question'],
        'chat_history': state['chat_history']
    })

    response = llm.invoke(message)
    return {
        'chat_history': state['chat_history'] + [
            HumanMessage(content=state['question']),
            response
        ]
    }





## State-Machine
graph = StateGraph(State)
graph.add_node("retriever" , rag)
graph.add_node("chatbot" , chatbot)
graph.add_edge(START , "retriever")
graph.add_edge("retriever" , "chatbot" )
graph.add_edge("chatbot" , END)


rag_graph  = graph.compile()


## Displaying Graph ###
from IPython.display import Image , display

display(Image(rag_graph.get_graph().draw_mermaid_png()))


## history Aware Chat Assistant ##
chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() in {"quit", "exit", "q"}:
        print("Goodbye!")
        break

    state = {
        "question": user_input,
        "chat_history": chat_history
    }

    result = rag_graph.invoke(state)
    chat_history = result["chat_history"]
    print("Assistant:", chat_history[-1].content)




