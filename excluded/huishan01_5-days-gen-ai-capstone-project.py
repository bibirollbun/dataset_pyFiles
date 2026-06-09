!pip uninstall -y langchain -qqy jupyterlab kfp
!pip install -qU "google-genai==1.7.0"
!pip install -qU "chromadb==0.6.3"


#install necessary library packages
!pip install chromadb langchain pypdf2
!pip install langchain-community


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings



from google import genai
from google.genai import types

from IPython.display import Markdown

genai.__version__


from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")


#Explore available models
client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)


#Load pdf document to be used 
data_path = "/kaggle/input/paper1/CDSR_141.pdf"
def load_documents():
    document_loader = PyPDFLoader(data_path)
    return document_loader.load()

documents = load_documents()


#Split the text into appropriate length chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300,
    length_function = len,
    add_start_index = True,
)

chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

document = chunks[0]
print(document.page_content)
print(document.metadata)



from google.api_core import retry

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)


from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from google.genai import types

# Create a custom function to generate Embeddings with the Google's Gemini API

class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    @retry.Retry(predicate = is_retriable)
    def __call__(self, input: document) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model = "models/text-embedding-004",
            contents = input,
            config = types.EmbedContentConfig(
                task_type = embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]


#Create a Chroma Database Client to populate the dataase with the documents defined above

import chromadb

documents = load_documents()

DB_NAME = "googlecardb"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents = [doc.page_content for doc in documents],
       metadatas = [doc.metadata for doc in documents],
       ids = [str(i) for i in range(len(documents))])



#Find relevant documents using Retrieval method
embed_fn.document_mode = False

query = "Summarize this paper as an academic abstract "

result = db.query(query_texts=[query], n_results=1)
[all_passages] = result["documents"]

Markdown(all_passages[0])


#Answer the question using Augmented Generation

query_oneline = query.replace("\n", " ")

prompt = \
f"""You are a helpful and informative bot that answers 
questions using text from the reference passage included below. 
Be sure to respond in a complete sentence, being in details, 
including all relevant background information. 
If the passage is irrelevant to the answer, you may ignore it. 

QUESTION: {query_oneline}
"""

for passage in all_passages: 
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

print(prompt)

answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)

response = answer.text


#Use embeddings to calculate similarity scores by using cosine similarity 
import requests

def cosine_similarity_manual(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
    
embed_fn = GeminiEmbeddingFunction(EmbeddingFunction)
embed_fn.document_mode = True

my_paper = {
    "title": "Curve Tracking of Nonlinear Dynamic System Using Linear State-Space Model",
    "abstract": str(response)
}
my_text = my_paper["title"] + ": " + my_paper["abstract"]

response = client.models.embed_content(
    model = 'models/text-embedding-004',
    contents = my_text,
    config = types.EmbedContentConfig(task_type = 'semantic_similarity')
)

my_embedding = response.embeddings[0].values


#Retrieve related papers from Semantic Scholar
import json

url = "https://api.semanticscholar.org/graph/v1/paper/search"
params = {
    "query": "nonlinear state-space control",
    "fields": "title,abstract,url",
    "limit": 10
}
semantic_scholar_response = requests.get(url, params=params)
papers = semantic_scholar_response.json().get("data", [])

print(papers)


##Mock data to replace Semantic Scholar API
papers = [
    {
        "title": "Linear State-Space Approaches for Nonlinear Control Systems",
        "abstract": "This paper presents a method of using linear state-space models to approximate and track the behavior of nonlinear systems.",
        "url": "https://example.com/paper1"
    },
    {
        "title": "Curve Tracking Control Using Adaptive Estimation",
        "abstract": "An adaptive control strategy is proposed to handle the curve tracking problem in a nonlinear dynamic environment using state estimation.",
        "url": "https://example.com/paper2"
    },
    {
        "title": "Nonlinear Dynamic Systems Identification via Linear Embeddings",
        "abstract": "We investigate how linear embeddings can be used to model and control nonlinear dynamic systems, with application to robotics.",
        "url": "https://example.com/paper3"
    },
    {
        "title": "State-Space Modeling for Robotic Arm Tracking",
        "abstract": "A comprehensive modeling technique using linearized state-space equations is developed for trajectory tracking in robotic systems.",
        "url": "https://example.com/paper4"
    },
    {
        "title": "Using Linear Systems to Approximate Nonlinear Behaviors",
        "abstract": "We propose a system identification approach that fits linear state-space models to data generated by nonlinear dynamic processes.",
        "url": "https://example.com/paper5"
    }
]



#Generate similar papers with title, abstract, url and similarity scores
similarities = []
for paper in papers:
    if not paper.get("abstract"):
        continue
    paper_text = paper["title"] + ". " + paper["abstract"]
    emb = embed_fn([paper_text])[0]
    score = cosine_similarity_manual(my_embedding, emb)
    similarities.append({
        "title": paper["title"],
        "abstract": paper["abstract"],
        "url": paper.get("url",""),
        "score": score
    })

top_5 = sorted(similarities, key=lambda x: x["score"], reverse=True)[:5]

for i, p in enumerate(top_5):
    print(f"\n#{i+1}: {p['title']}")
    print(f"Similarity: {p['score']:.4f}")
    print(f"Link: {p['url']}")
    print(f"Abstract: {p['abstract'][:1000]}...")

