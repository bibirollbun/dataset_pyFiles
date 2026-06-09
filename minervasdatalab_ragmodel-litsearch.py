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


# Install only once per environment
!pip install -U langchain faiss-cpu openai langchain-openai pymupdf langchain_community langchain_core






from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
!git clone https://github.com/Jojo666/openai-to-z-challenge.git






pdf_dir = Path("openai-to-z-challenge/literature")
docs = []

# Go through all PDF files
for pdf_path in pdf_dir.rglob("*.pdf"):
    loader = PyMuPDFLoader(str(pdf_path))  # Load the PDF
    for doc in loader.load():
        doc.metadata["source"] = pdf_path.name  # Save the filename as metadata
        docs.append(doc)


docs





from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = splitter.split_documents(docs)


docs_split





import os
os.environ["OPENAI_API_KEY"] = "sk-"  # Replace with your actual key






from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs_split, embedding_model)





from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Get top 5 relevant chunks

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)






response = qa_chain.invoke("Summarize findings about Amazonian archeological sites.")
print(response)


response['source_documents']





for doc in response["source_documents"]:
    print(f"- Source: {doc.metadata.get('source')}")


print(response['result'])







