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


!pip install pdfplumber
!pip install chromadb
!pip install ollama
!pip install pandas
!pip install cleantext
!pip install scikit-learn



import pdfplumber
import ollama
import json 
import chromadb
import pandas as pd
import cleantext

from sklearn.feature_extraction.text import TfidfVectorizer


SECTION_JSON_PATH = '/kaggle/input/section/sections.json'
QUERRY_JSON_PATH = '/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json'
BOOK_PDF_PATH = '/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/book.pdf'
MODEL_NAME = 'llama3.2:3b'


#Download ollama
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess
process = subprocess.Popen("ollama serve", shell=True) #runs on a different thread
#Download model
!ollama pull llama3.2:3b
!ollama pull mxbai-embed-large
output = ollama.generate(
        model= MODEL_NAME,
        prompt="say hellow"
    )


print(output['response'])


def get_answer(question,reference_data):
    output = ollama.generate(
        model=MODEL_NAME,
        prompt=f"Read this data:\n {reference_data} \n and answer the question {question}"
    )
    return output['response']
def get_ans_withoutKNO(question):
    output = ollama.generate(
        model=MODEL_NAME,
        prompt=question
    )
    return output['response']
def extract_pdf_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            if raw_text:
                book_text = cleantext.clean(raw_text, extra_spaces=True, numbers=True, punct=True).replace('\n','')
                pages.append(book_text)
            else :
                pages.append('None')
    return pages
def pageNum2SectionPath(page_num,section_dict):
    data = section_dict
    page_num = int(page_num)
    parrent_name = ''
    for k,v in data.items():
        if page_num >= int(v['page_start']) \
            and page_num<= int(v['page_end']):
            parrent_name = k
            break
    if parrent_name == '':
        return 'Untitled'
    elif 'subsections' in data[parrent_name]:
        chiled_name = ''
        for k,v in data[parrent_name]['subsections'].items():
            if page_num >= int(v['page_start']) \
                and page_num<= int(v['page_end']):
                chiled_name = k
                break
        if chiled_name == '':
            return parrent_name
        else:
            return f"{parrent_name}/{chiled_name}"
    else :
        return parrent_name


# step2
book_data= extract_pdf_text(BOOK_PDF_PATH)[12:]
with open(QUERRY_JSON_PATH) as f:
    question_data = json.load(f)

with open(SECTION_JSON_PATH) as f:
    sections_data = json.load(f)
    
vectorizer = TfidfVectorizer(max_features=1000,stop_words='english')
tfidf_matrix = vectorizer.fit_transform(book_data).toarray()


# step3 
client = chromadb.Client()
collection_embd = client.create_collection(name="embd",metadata={"hnsw:space": "cosine"})
collection_tfidf = client.create_collection(name="TFIDF",metadata={"hnsw:space": "ip"})

for idx, book_text in enumerate(book_data):
    # embeddings = get_embd(book_text)
    # collection_embd.add(
    #     ids=[str(idx+1)],
    #     embeddings=embeddings,
    #     documents=[book_text]
    # )
    tfidf = tfidf_matrix[idx]
    collection_tfidf.add(
        ids=[str(idx+1)],
        embeddings=tfidf,
        documents=[book_text]
    )



my_output = {'ID':[],'context':[],'answer':[],'references':[]}
for querry in question_data:
    tfidf = vectorizer.transform([querry['question']]).toarray()
    result = collection_tfidf.query(
        query_embeddings=tfidf[0],
        n_results=2
    )
    # result = {'ids':[[a,b,c]],'documents'[[d1,d2,d3]]}
    # print(result)
    knowledge = ''
    for x in result['documents'][0]:
        knowledge += x
    ans_with_k = get_answer(querry['question'],knowledge)
    for x in ['**','\n']:
        ans_with_k = ans_with_k.replace(x,'')
        
    my_output['ID'].append(querry['query_id'])
    my_output['context'].append(knowledge)
    my_output['answer'].append(ans_with_k)
    pages = result['ids'][0]
    sec = []
    for num in pages:
        sec.append(pageNum2SectionPath(num,sections_data))
    my_output['references'].append(json.dumps({'sections':sec[:],
                                        'pages':pages}))
    
pd.DataFrame(my_output).to_csv('submition.csv',index_label=False)

