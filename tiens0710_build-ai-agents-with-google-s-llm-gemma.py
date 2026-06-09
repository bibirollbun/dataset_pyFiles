!pip install -U \
transformers \
datasets \
trl \
peft \
accelerate \
bitsandbytes \
ipywidgets


from transformers import set_seed

# Random number generators seed for reproducibility (same code == same results)
# Set the seed in random, numpy, torch and/or tensorflow if installed
set_seed(22)


# Utility Code, You can ignore
import utilty_scripts_for_build_ai_agents_with_gemma as utils

utils.plot_memory_usage()


from kaggle_secrets import UserSecretsClient
import os

hogging_face_token = UserSecretsClient().get_secret("hf_token")

os.environ['HF_TOKEN'] = hogging_face_token


# TODO: Change to the new gemma 1.1 versions once they have been tested enough by the community
gemma_2b = '/kaggle/input/gemma/transformers/2b/2'
gemma_7b_it = '/kaggle/input/gemma/transformers/7b-it/3'
gemma_2b_it = '/kaggle/input/gemma/transformers/2b-it/3'

# Model IDs on Hugging Face Hub
#gemma_7b = 'google/gemma-7b'
#gemma_7b_it = 'google/gemma-7b-it'
#gemma_2b_it = 'google/gemma-2b-it'

# Test prompts
prompt_1 = "What is Data Science?"
prompt_2 = "Explain 3 important Data Science concepts, and tell why each concept is important"
prompt_3 = "I'm a marketing specialist, I know nothing about Data Science. Explain to me what Data Science is and simplifie it as much as you can. When possible, use analogies that I can understand better as a marketing specialist"
# These prompts are just a way to test our models and see how they perform depending on the complexity of the task.


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc


def run_inference(model_id, prompts):
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
    
    # Tokenize the prompts
    input_ids = tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
    
    # Generate ouputs and decode them
    outputs = model.generate(**input_ids, max_new_tokens=1024)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Clear GPU memeory to enable the next model to be loaded (We need this because we are testing multiple models)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_texts


output_2b_it = run_inference(gemma_2b_it, [prompt_1,prompt_2,prompt_3])
output_7b_it = run_inference(gemma_7b_it, [prompt_1,prompt_2,prompt_3])
output_2b = run_inference(gemma_2b, [prompt_1,prompt_2,prompt_3])


# Utility Code, You can ignore
outputs = [
    ('Gemma 7B Instriction Tuned Outputs', output_7b_it),
    ('Gemma 2B Instriction Tuned Outputs', output_2b_it), 
    ('Gemma 2B Pretrained Outputs', output_2b),
]
utils.display_formatted_models_comparaison_outputs(outputs)


# Utility Code, You can ignore
utils.plot_memory_usage()


!pip install bitsandbytes


from transformers import BitsAndBytesConfig
import torch

gemma_7b_it = '/kaggle/input/gemma/transformers/7b-it/3'

def load_model(model_id=gemma_7b_it, load_in_4bit=True, quant_compute_dtype=torch.float16):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=quant_compute_dtype,
    ) if load_in_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    return tokenizer, model

# Táº£i mÃ´ hÃ¬nh
gemma_7b_tokenizer, gemma_7b_model = load_model()


from kaggle_secrets import UserSecretsClient
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = UserSecretsClient().get_secret("LANGCHAIN_API_KEY") 


!pip install -U langchain langchain-community chromadb sentence_transformers wikipedia faiss-gpu


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from datasets import load_dataset

# Táº£i dá»¯ liá»‡u tá»« letmecook.json
dataset = load_dataset('json', data_files="/kaggle/input/letmecook/letmecook.json", split='train')

# Chuyá»ƒn Ä‘á»•i dá»¯ liá»‡u thÃ nh danh sÃ¡ch Document, giá»¯ metadata
docs = [
    Document(
        page_content=f"{ex['messages'][0]['content']}\n{ex['messages'][1]['content']}",
        metadata={"source": "letmecook", "lang": ex['lang'], "question": ex['messages'][0]['content']}
    ) for ex in dataset
]

# Chia nhá»� tÃ i liá»‡u
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

# Sá»­ dá»¥ng embedding Ä‘a ngÃ´n ngá»¯
embedding_model = HuggingFaceBgeEmbeddings(model_name='intfloat/multilingual-e5-small')

child_docs_store = Chroma(
    collection_name="split_parents",
    embedding_function=embedding_model
)
parent_docs_store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=child_docs_store,
    docstore=parent_docs_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3},  # TÄƒng k Ä‘á»ƒ láº¥y nhiá»�u tÃ i liá»‡u hÆ¡n vÃ¬ cÃ³ cÃ¢u tráº£ lá»�i giá»‘ng nhau
)

retriever.add_documents(docs)


queries = [
    'LÃ m sao Ä‘á»ƒ xem thÃ´ng tin sinh viÃªn?',
    'Ä�iá»�u kiá»‡n há»�c bá»•ng lÃ  gÃ¬?',
    'ThÃ´ng bÃ¡o má»›i á»Ÿ Ä‘Ã¢u?',
    'QuÃªn máº­t kháº©u thÃ¬ pháº£i lÃ m sao?',
    'LÃ m sao biáº¿t lá»‹ch cÃ´ng tÃ¡c cá»§a cÃ¡n bá»™?'
]


# Utility Code, You can ignore
from IPython.display import display_html, clear_output
import utilty_scripts_for_build_ai_agents_with_gemma as utils

def retrieve_and_display(queries, retriever):
    comparaison = ''

    for query in queries:
        retrieved_doc = retriever.get_relevant_documents(query)[0].page_content
        # The ".replace('\n', '<br>')" below is just for formatting, if you don't understand, ignore it.
        retrieved_doc = retrieved_doc.replace('\n', '<br>')
        comparaison += utils.expandable_section(title=query, content=retrieved_doc)

    clear_output()
    display_html(comparaison, raw=True)


retrieve_and_display(queries, retriever)


queries = [
    'tuyá»ƒn sinh',
    'báº£o hiá»ƒm y táº¿?',
    'Tell me what "missing values" means in data science',
    'Teach me something about missing values',
]

retrieve_and_display(queries, retriever)


import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_id, tokenizer_id=None, load_in_4bit=True, quant_compute_dtype=torch.float16):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=quant_compute_dtype,
    ) if load_in_4bit else None

    # Sá»­ dá»¥ng tokenizer_id náº¿u Ä‘Æ°á»£c cung cáº¥p, náº¿u khÃ´ng dÃ¹ng model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id if tokenizer_id else model_id)
        print(f"Loaded tokenizer from: {tokenizer_id if tokenizer_id else model_id}")
    except Exception as e:
        print(f"Error loading tokenizer for {model_id}: {e}")
        raise

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
        print(f"Loaded model from: {model_id}")
    except Exception as e:
        print(f"Error loading model for {model_id}: {e}")
        raise

    return tokenizer, model

# XÃ³a bá»™ nhá»› GPU trÆ°á»›c khi táº£i
gc.collect()
torch.cuda.empty_cache()

# Táº£i Gemma 7B-IT
gemma_7b_it = '/kaggle/input/gemma/transformers/7b-it/3'
try:
    gemma_7b_tokenizer, gemma_7b_model = load_model(gemma_7b_it)
except Exception as e:
    print(f"Failed to load Gemma 7B-IT: {e}")
    raise

# Táº£i coder model vá»›i tokenizer tá»« Gemma 2B-IT
coder_model_id = '/kaggle/input/dsi-coder/transformers/2b/2'
try:
    coder_tokenizer, coder_model = load_model(coder_model_id, tokenizer_id='/kaggle/input/gemma/transformers/2b-it/3')
except Exception as e:
    print(f"Failed to load coder model: {e}")
    raise


# from pydantic import Field
# from langchain_core.language_models.llms import LLM
# from typing import Any

# class GemmaLangChain(LLM):
#     # Khai bÃ¡o cÃ¡c trÆ°á»�ng model vÃ  tokenizer
#     model: Any = Field(default=None)
#     tokenizer: Any = Field(default=None)

#     def __init__(self, model, tokenizer):
#         super().__init__(model=model, tokenizer=tokenizer)

#     @property
#     def _llm_type(self) -> str:
#         return "gemma"

#     def _call(self, prompt, stop=None, **kwargs) -> str:
#         inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=512,
#             stop_strings=stop,
#             tokenizer=self.tokenizer  # ThÃªm tokenizer vÃ o generate
#         )
#         output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
#         return output_text

# # Khá»Ÿi táº¡o GemmaLangChain
# gemma_langchain = GemmaLangChain(model=gemma_7b_model, tokenizer=gemma_7b_tokenizer)


from pydantic import Field
from langchain_core.language_models.llms import LLM
from typing import Any

class GemmaLangChain(LLM):
    # Khai bÃ¡o cÃ¡c trÆ°á»�ng model vÃ  tokenizer
    model: Any = Field(default=None)
    tokenizer: Any = Field(default=None)

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)

    @property
    def _llm_type(self) -> str:
        return "gemma"

    def _call(self, prompt, stop=None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            stop_strings=stop,
            tokenizer=self.tokenizer  # ThÃªm tokenizer vÃ o generate
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        return output_text

# Khá»Ÿi táº¡o GemmaLangChain
gemma_langchain = GemmaLangChain(model=gemma_7b_model, tokenizer=gemma_7b_tokenizer)


!pip install langdetect


from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langdetect import detect

# Táº¡o cÃ´ng cá»¥ tÆ° váº¥n sinh viÃªn
@tool(return_direct=True)
def student_advisor_tool(query: str) -> str:
    """CÃ´ng cá»¥ tráº£ lá»�i cÃ¢u há»�i tÆ° váº¥n sinh viÃªn báº±ng tiáº¿ng Viá»‡t vá»� quáº£n lÃ½ sinh viÃªn, há»�c bá»•ng, thÃ´ng bÃ¡o, v.v."""
    context = retriever.get_relevant_documents(query)[0].page_content if retriever.get_relevant_documents(query) else ""
    prompt = f"""
    Báº¡n lÃ  trá»£ lÃ½ tÆ° váº¥n sinh viÃªn. Tráº£ lá»�i cÃ¢u há»�i báº±ng tiáº¿ng Viá»‡t, rÃµ rÃ ng, chi tiáº¿t vÃ  chuyÃªn nghiá»‡p.

    Ngá»¯ cáº£nh: {context}

    CÃ¢u há»�i: {query}

    Tráº£ lá»�i:
    """
    input_ids = coder_tokenizer(prompt, return_tensors='pt').to('cuda')
    output = coder_model.generate(**input_ids, max_new_tokens=512)
    return coder_tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]


import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
from langchain_core.tools import tool

# Táº¡o database vÃ  tables
def create_ctuet_database():
    """Táº¡o database SQLite vá»›i Ä‘áº§y Ä‘á»§ tables cho há»‡ thá»‘ng sinh viÃªn"""
    
    # Káº¿t ná»‘i Ä‘áº¿n database (táº¡o file náº¿u chÆ°a cÃ³)
    conn = sqlite3.connect('/tmp/ctuet_database.db')
    cursor = conn.cursor()
    
    # 1. Báº£ng sinh viÃªn
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code VARCHAR(20) UNIQUE NOT NULL,
        username VARCHAR(50) UNIQUE NOT NULL,
        full_name VARCHAR(100) NOT NULL,
        email VARCHAR(100),
        phone VARCHAR(15),
        class_name VARCHAR(50),
        major VARCHAR(100),
        year_of_study INTEGER,
        gpa FLOAT DEFAULT 0.0,
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 2. Báº£ng giáº£ng viÃªn
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS teachers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_code VARCHAR(20) UNIQUE NOT NULL,
        full_name VARCHAR(100) NOT NULL,
        email VARCHAR(100),
        phone VARCHAR(15),
        department VARCHAR(100),
        position VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 3. Báº£ng mÃ´n há»�c
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_code VARCHAR(20) UNIQUE NOT NULL,
        subject_name VARCHAR(200) NOT NULL,
        credits INTEGER NOT NULL,
        department VARCHAR(100),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 4. Báº£ng há»�c ká»³
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS semesters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        semester_name VARCHAR(50) NOT NULL,
        academic_year VARCHAR(20) NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        is_current BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 5. Báº£ng lá»‹ch há»�c
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER NOT NULL,
        teacher_id INTEGER NOT NULL,
        semester_id INTEGER NOT NULL,
        day_of_week INTEGER NOT NULL, -- 1=CN, 2=T2, 3=T3, ..., 7=T7
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        room VARCHAR(20),
        week_start INTEGER DEFAULT 1,
        week_end INTEGER DEFAULT 20,
        class_type VARCHAR(20) DEFAULT 'theory', -- theory, practice, lab
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        FOREIGN KEY (teacher_id) REFERENCES teachers (id),
        FOREIGN KEY (semester_id) REFERENCES semesters (id)
    )
    ''')
    
    # 6. Báº£ng Ä‘Äƒng kÃ½ mÃ´n há»�c
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS enrollments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_id INTEGER NOT NULL,
        semester_id INTEGER NOT NULL,
        enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'enrolled', -- enrolled, dropped, completed
        FOREIGN KEY (student_id) REFERENCES students (id),
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        FOREIGN KEY (semester_id) REFERENCES semesters (id),
        UNIQUE(student_id, subject_id, semester_id)
    )
    ''')
    
    # 7. Báº£ng Ä‘iá»ƒm sá»‘
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS grades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_id INTEGER NOT NULL,
        semester_id INTEGER NOT NULL,
        midterm_score FLOAT,
        final_score FLOAT,
        assignment_score FLOAT,
        attendance_score FLOAT,
        total_score FLOAT,
        letter_grade VARCHAR(5),
        grade_points FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students (id),
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        FOREIGN KEY (semester_id) REFERENCES semesters (id),
        UNIQUE(student_id, subject_id, semester_id)
    )
    ''')
    
    # 8. Báº£ng thÃ´ng bÃ¡o
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS announcements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title VARCHAR(200) NOT NULL,
        content TEXT NOT NULL,
        category VARCHAR(50), -- academic, administrative, event, etc.
        priority VARCHAR(20) DEFAULT 'normal', -- low, normal, high, urgent
        target_audience VARCHAR(50) DEFAULT 'all', -- all, students, teachers, class_specific
        target_class VARCHAR(50),
        is_published BOOLEAN DEFAULT 1,
        publish_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expire_date TIMESTAMP,
        created_by VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 9. Báº£ng há»�c bá»•ng
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scholarships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scholarship_name VARCHAR(200) NOT NULL,
        description TEXT,
        amount DECIMAL(10,2),
        requirements TEXT,
        application_deadline DATE,
        semester_id INTEGER,
        status VARCHAR(20) DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (semester_id) REFERENCES semesters (id)
    )
    ''')
    
    # 10. Báº£ng Ä‘Äƒng kÃ½ há»�c bá»•ng
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scholarship_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        scholarship_id INTEGER NOT NULL,
        application_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'pending', -- pending, approved, rejected
        documents TEXT, -- JSON string of submitted documents
        notes TEXT,
        reviewed_by VARCHAR(100),
        reviewed_at TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students (id),
        FOREIGN KEY (scholarship_id) REFERENCES scholarships (id)
    )
    ''')
    
    conn.commit()
    return conn

# Táº¡o database
print("ğŸ”¨ Ä�ang táº¡o database...")
conn = create_ctuet_database()
print("âœ… Database Ä‘Ã£ Ä‘Æ°á»£c táº¡o thÃ nh cÃ´ng!")


def insert_sample_data(conn):
    """ChÃ¨n dá»¯ liá»‡u máº«u vÃ o database"""
    cursor = conn.cursor()
    
    print("ğŸ“� Ä�ang chÃ¨n dá»¯ liá»‡u máº«u...")
    
    # 1. ThÃªm há»�c ká»³ hiá»‡n táº¡i
    semesters_data = [
        ('Há»�c ká»³ 1', '2024-2025', '2024-09-01', '2025-01-15', 1),
        ('Há»�c ká»³ 2', '2024-2025', '2025-02-01', '2025-06-30', 0),
        ('Há»�c ká»³ hÃ¨', '2024-2025', '2025-07-01', '2025-08-31', 0)
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO semesters (semester_name, academic_year, start_date, end_date, is_current)
    VALUES (?, ?, ?, ?, ?)
    ''', semesters_data)
    
    # 2. ThÃªm sinh viÃªn (bao gá»“m user hiá»‡n táº¡i)
    students_data = [
        ('SV2021001', 'tieens1802', 'Nguyá»…n VÄƒn Tiáº¿n', 'tieens1802@student.ctuet.edu.vn', '0123456789', 'CT2021A', 'CÃ´ng nghá»‡ thÃ´ng tin', 4, 3.2, 'active'),
        ('SV2021002', 'student002', 'Tráº§n Thá»‹ Mai', 'mai.tran@student.ctuet.edu.vn', '0987654321', 'CT2021A', 'CÃ´ng nghá»‡ thÃ´ng tin', 4, 3.5, 'active'),
        ('SV2021003', 'student003', 'LÃª HoÃ ng Nam', 'nam.le@student.ctuet.edu.vn', '0912345678', 'CT2021B', 'CÃ´ng nghá»‡ thÃ´ng tin', 4, 3.8, 'active'),
        ('SV2021004', 'student004', 'Pháº¡m Thá»‹ Lan', 'lan.pham@student.ctuet.edu.vn', '0934567890', 'KT2021A', 'Kinh táº¿', 4, 3.1, 'active'),
        ('SV2021005', 'student005', 'VÃµ Minh Khoa', 'khoa.vo@student.ctuet.edu.vn', '0945678901', 'CT2021A', 'CÃ´ng nghá»‡ thÃ´ng tin', 4, 3.6, 'active')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO students (student_code, username, full_name, email, phone, class_name, major, year_of_study, gpa, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', students_data)
    
    # 3. ThÃªm giáº£ng viÃªn
    teachers_data = [
        ('GV001', 'Tiáº¿n sÄ© Nguyá»…n VÄƒn HÃ¹ng', 'hung.nguyen@ctuet.edu.vn', '0901234567', 'CÃ´ng nghá»‡ thÃ´ng tin', 'PhÃ³ GiÃ¡o sÆ°'),
        ('GV002', 'Tháº¡c sÄ© Tráº§n Thá»‹ Lan', 'lan.tran@ctuet.edu.vn', '0902345678', 'CÃ´ng nghá»‡ thÃ´ng tin', 'Giáº£ng viÃªn'),
        ('GV003', 'Tiáº¿n sÄ© LÃª Minh Tuáº¥n', 'tuan.le@ctuet.edu.vn', '0903456789', 'ToÃ¡n há»�c', 'GiÃ¡o sÆ°'),
        ('GV004', 'Tháº¡c sÄ© Pháº¡m Thá»‹ Hoa', 'hoa.pham@ctuet.edu.vn', '0904567890', 'Ngoáº¡i ngá»¯', 'Giáº£ng viÃªn'),
        ('GV005', 'Tiáº¿n sÄ© VÃµ VÄƒn Nam', 'nam.vo@ctuet.edu.vn', '0905678901', 'CÃ´ng nghá»‡ thÃ´ng tin', 'PhÃ³ GiÃ¡o sÆ°')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO teachers (teacher_code, full_name, email, phone, department, position)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', teachers_data)
    
    # 4. ThÃªm mÃ´n há»�c
    subjects_data = [
        ('CS101', 'Nháº­p mÃ´n Láº­p trÃ¬nh', 3, 'CÃ´ng nghá»‡ thÃ´ng tin', 'MÃ´n há»�c cÆ¡ sá»Ÿ vá»� láº­p trÃ¬nh'),
        ('CS201', 'Cáº¥u trÃºc dá»¯ liá»‡u vÃ  Giáº£i thuáº­t', 4, 'CÃ´ng nghá»‡ thÃ´ng tin', 'MÃ´n há»�c vá»� cáº¥u trÃºc dá»¯ liá»‡u'),
        ('CS301', 'CÆ¡ sá»Ÿ dá»¯ liá»‡u', 3, 'CÃ´ng nghá»‡ thÃ´ng tin', 'Thiáº¿t káº¿ vÃ  quáº£n lÃ½ cÆ¡ sá»Ÿ dá»¯ liá»‡u'),
        ('CS401', 'TrÃ­ tuá»‡ nhÃ¢n táº¡o', 4, 'CÃ´ng nghá»‡ thÃ´ng tin', 'CÃ¡c thuáº­t toÃ¡n AI vÃ  Machine Learning'),
        ('MT101', 'ToÃ¡n cao cáº¥p A1', 3, 'ToÃ¡n há»�c', 'Giáº£i tÃ­ch má»™t biáº¿n'),
        ('MT201', 'ToÃ¡n rá»�i ráº¡c', 3, 'ToÃ¡n há»�c', 'ToÃ¡n há»�c cho khoa há»�c mÃ¡y tÃ­nh'),
        ('EN101', 'Tiáº¿ng Anh 1', 2, 'Ngoáº¡i ngá»¯', 'Tiáº¿ng Anh cÆ¡ báº£n'),
        ('EN201', 'Tiáº¿ng Anh 2', 2, 'Ngoáº¡i ngá»¯', 'Tiáº¿ng Anh trung cáº¥p')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO subjects (subject_code, subject_name, credits, department, description)
    VALUES (?, ?, ?, ?, ?)
    ''', subjects_data)
    
    # 5. Táº¡o lá»‹ch há»�c cho sinh viÃªn tieens1802
    current_semester_id = 1  # Há»�c ká»³ 1 2024-2025
    
    # Láº¥y IDs
    cursor.execute("SELECT id FROM subjects WHERE subject_code IN ('CS401', 'CS301', 'MT201', 'EN201')")
    subject_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT id FROM teachers LIMIT 4")
    teacher_ids = [row[0] for row in cursor.fetchall()]
    
    # Táº¡o lá»‹ch há»�c (tuáº§n nÃ y)
    schedules_data = [
        # Thá»© 2
        (subject_ids[0], teacher_ids[0], current_semester_id, 2, '07:30', '09:30', 'A101', 1, 20, 'theory'),  # AI
        (subject_ids[1], teacher_ids[1], current_semester_id, 2, '13:30', '15:30', 'B201', 1, 20, 'lab'),     # Database
        
        # Thá»© 3  
        (subject_ids[2], teacher_ids[2], current_semester_id, 3, '09:30', '11:30', 'C301', 1, 20, 'theory'),  # ToÃ¡n rá»�i ráº¡c
        
        # Thá»© 4
        (subject_ids[0], teacher_ids[0], current_semester_id, 4, '07:30', '09:30', 'A102', 1, 20, 'practice'), # AI thá»±c hÃ nh
        (subject_ids[3], teacher_ids[3], current_semester_id, 4, '15:30', '17:30', 'D101', 1, 20, 'theory'),  # Tiáº¿ng Anh
        
        # Thá»© 5
        (subject_ids[1], teacher_ids[1], current_semester_id, 5, '07:30', '10:30', 'B202', 1, 20, 'theory'),  # Database lÃ½ thuyáº¿t
        
        # Thá»© 6
        (subject_ids[2], teacher_ids[2], current_semester_id, 6, '13:30', '15:30', 'C302', 1, 20, 'practice'), # ToÃ¡n thá»±c hÃ nh
        (subject_ids[3], teacher_ids[3], current_semester_id, 6, '15:30', '17:30', 'D102', 1, 20, 'practice')  # Tiáº¿ng Anh thá»±c hÃ nh
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO schedules (subject_id, teacher_id, semester_id, day_of_week, start_time, end_time, room, week_start, week_end, class_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', schedules_data)
    
    # 6. Ä�Äƒng kÃ½ mÃ´n há»�c cho tieens1802
    student_id = 1  # tieens1802
    
    enrollments_data = [(student_id, sid, current_semester_id, 'enrolled') for sid in subject_ids]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO enrollments (student_id, subject_id, semester_id, status)
    VALUES (?, ?, ?, ?)
    ''', enrollments_data)
    
    # 7. ThÃªm Ä‘iá»ƒm sá»‘
    grades_data = [
        (student_id, subject_ids[0], current_semester_id, 8.5, 9.0, 8.0, 9.5, 8.7, 'A', 4.0),  # AI
        (student_id, subject_ids[1], current_semester_id, 7.5, 8.0, 7.8, 9.0, 8.1, 'B+', 3.5), # Database  
        (student_id, subject_ids[2], current_semester_id, 6.5, 7.0, 7.2, 8.5, 7.3, 'B', 3.0),  # ToÃ¡n rá»�i ráº¡c
        (student_id, subject_ids[3], current_semester_id, 9.0, 8.5, 9.2, 9.8, 9.1, 'A+', 4.0)  # Tiáº¿ng Anh
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO grades (student_id, subject_id, semester_id, midterm_score, final_score, assignment_score, attendance_score, total_score, letter_grade, grade_points)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', grades_data)
    
    # 8. ThÃªm thÃ´ng bÃ¡o
    announcements_data = [
        ('ThÃ´ng bÃ¡o nghá»‰ Táº¿t NguyÃªn Ä‘Ã¡n 2025', 'TrÆ°á»�ng thÃ´ng bÃ¡o lá»‹ch nghá»‰ Táº¿t NguyÃªn Ä‘Ã¡n tá»« ngÃ y 25/1 Ä‘áº¿n 2/2/2025', 'academic', 'high', 'all', None, 1, '2025-01-15 09:00:00', '2025-02-03 23:59:59', 'Ban GiÃ¡m hiá»‡u'),
        ('Ä�Äƒng kÃ½ mÃ´n há»�c ká»³ 2', 'Sinh viÃªn Ä‘Äƒng kÃ½ mÃ´n há»�c ká»³ 2 tá»« ngÃ y 20/1 Ä‘áº¿n 25/1/2025', 'academic', 'urgent', 'students', None, 1, '2025-01-10 08:00:00', '2025-01-26 23:59:59', 'PhÃ²ng Ä�Ã o táº¡o'),
        ('ThÃ´ng bÃ¡o há»�c bá»•ng khuyáº¿n khÃ­ch há»�c táº­p', 'Má»Ÿ Ä‘Æ¡n Ä‘Äƒng kÃ½ há»�c bá»•ng cho sinh viÃªn cÃ³ GPA >= 3.2', 'administrative', 'normal', 'students', None, 1, '2025-01-01 08:00:00', '2025-02-28 23:59:59', 'PhÃ²ng CTSV'),
        ('Lá»‹ch thi cuá»‘i ká»³ há»�c ká»³ 1', 'Lá»‹ch thi chi tiáº¿t Ä‘Ã£ Ä‘Æ°á»£c cáº­p nháº­t trÃªn website', 'academic', 'high', 'students', None, 1, '2024-12-01 08:00:00', '2025-01-20 23:59:59', 'PhÃ²ng Ä�Ã o táº¡o'),
        ('Há»�p phá»¥ huynh cuá»‘i nÄƒm', 'Má»�i phá»¥ huynh sinh viÃªn tham dá»± há»�p tá»•ng káº¿t nÄƒm há»�c', 'event', 'normal', 'all', None, 1, '2024-12-15 08:00:00', '2025-01-31 23:59:59', 'PhÃ²ng CTSV')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO announcements (title, content, category, priority, target_audience, target_class, is_published, publish_date, expire_date, created_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', announcements_data)
    
    # 9. ThÃªm há»�c bá»•ng
    scholarships_data = [
        ('Há»�c bá»•ng khuyáº¿n khÃ­ch há»�c táº­p', 'DÃ nh cho sinh viÃªn cÃ³ GPA tá»« 3.2 trá»Ÿ lÃªn', 2000000, 'GPA >= 3.2, khÃ´ng cÃ³ mÃ´n nÃ o dÆ°á»›i Ä‘iá»ƒm C', '2025-02-28', current_semester_id, 'active'),
        ('Há»�c bá»•ng tÃ i nÄƒng', 'DÃ nh cho sinh viÃªn xuáº¥t sáº¯c trong nghiÃªn cá»©u khoa há»�c', 5000000, 'CÃ³ cÃ´ng trÃ¬nh nghiÃªn cá»©u Ä‘Æ°á»£c cÃ´ng bá»‘', '2025-03-15', current_semester_id, 'active'),
        ('Há»�c bá»•ng há»— trá»£ sinh viÃªn khÃ³ khÄƒn', 'Há»— trá»£ sinh viÃªn cÃ³ hoÃ n cáº£nh khÃ³ khÄƒn', 1500000, 'CÃ³ giáº¥y xÃ¡c nháº­n hoÃ n cáº£nh tá»« Ä‘á»‹a phÆ°Æ¡ng', '2025-02-28', current_semester_id, 'active')
    ]
    
    cursor.executemany('''
    INSERT OR REPLACE INTO scholarships (scholarship_name, description, amount, requirements, application_deadline, semester_id, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', scholarships_data)
    
    conn.commit()
    print("âœ… Dá»¯ liá»‡u máº«u Ä‘Ã£ Ä‘Æ°á»£c chÃ¨n thÃ nh cÃ´ng!")

# ChÃ¨n dá»¯ liá»‡u máº«u
insert_sample_data(conn)


@tool(return_direct=True)
def ctuet_schedule_db_tool(query: str) -> str:
    """
    Láº¥y lá»‹ch há»�c tá»« database SQLite ná»™i bá»™
    Há»— trá»£: 'hÃ´m nay', 'ngÃ y mai', 'tuáº§n nÃ y', 'thá»© X'
    """
    try:
        conn = sqlite3.connect('/tmp/ctuet_database.db')
        cursor = conn.cursor()
        
        # Láº¥y student_id cá»§a user hiá»‡n táº¡i
        cursor.execute("SELECT id FROM students WHERE username = ?", ('tieens1802',))
        result = cursor.fetchone()
        if not result:
            return "â�Œ KhÃ´ng tÃ¬m tháº¥y thÃ´ng tin sinh viÃªn trong há»‡ thá»‘ng."
        
        student_id = result[0]
        
        # XÃ¡c Ä‘á»‹nh ngÃ y hiá»‡n táº¡i vÃ  tuáº§n
        from datetime import datetime, timedelta
        now = datetime.now()
        current_weekday = now.weekday() + 2  # Convert to DB format (1=CN, 2=T2, ...)
        if current_weekday == 8:  # Sunday
            current_weekday = 1
        
        # PhÃ¢n tÃ­ch query
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['hÃ´m nay', 'today']):
            # Lá»‹ch hÃ´m nay
            sql = """
            SELECT s.subject_name, sc.start_time, sc.end_time, sc.room, t.full_name, sc.class_type
            FROM schedules sc
            JOIN subjects s ON sc.subject_id = s.id
            JOIN teachers t ON sc.teacher_id = t.id
            JOIN enrollments e ON e.subject_id = sc.subject_id AND e.student_id = ?
            WHERE sc.day_of_week = ? AND sc.semester_id = 1
            ORDER BY sc.start_time
            """
            cursor.execute(sql, (student_id, current_weekday))
            title = f"ğŸ“… **Lá»ŠCH Há»ŒC HÃ”M NAY** ({now.strftime('%d/%m/%Y')})"
            
        elif any(keyword in query_lower for keyword in ['ngÃ y mai', 'tomorrow']):
            # Lá»‹ch ngÃ y mai
            tomorrow_weekday = current_weekday + 1
            if tomorrow_weekday > 7:
                tomorrow_weekday = 1
                
            sql = """
            SELECT s.subject_name, sc.start_time, sc.end_time, sc.room, t.full_name, sc.class_type
            FROM schedules sc
            JOIN subjects s ON sc.subject_id = s.id  
            JOIN teachers t ON sc.teacher_id = t.id
            JOIN enrollments e ON e.subject_id = sc.subject_id AND e.student_id = ?
            WHERE sc.day_of_week = ? AND sc.semester_id = 1
            ORDER BY sc.start_time
            """
            cursor.execute(sql, (student_id, tomorrow_weekday))
            tomorrow = now + timedelta(days=1)
            title = f"ğŸ“… **Lá»ŠCH Há»ŒC NGÃ€Y MAI** ({tomorrow.strftime('%d/%m/%Y')})"
            
        else:
            # Lá»‹ch cáº£ tuáº§n
            sql = """
            SELECT s.subject_name, sc.day_of_week, sc.start_time, sc.end_time, sc.room, t.full_name, sc.class_type
            FROM schedules sc
            JOIN subjects s ON sc.subject_id = s.id
            JOIN teachers t ON sc.teacher_id = t.id  
            JOIN enrollments e ON e.subject_id = sc.subject_id AND e.student_id = ?
            WHERE sc.semester_id = 1
            ORDER BY sc.day_of_week, sc.start_time
            """
            cursor.execute(sql, (student_id,))
            title = f"ğŸ“… **Lá»ŠCH Há»ŒC TUáº¦N NÃ€Y** (Tuáº§n {now.strftime('%W')})"
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"{title}\n\nğŸ“­ KhÃ´ng cÃ³ lá»‹ch há»�c nÃ o trong thá»�i gian nÃ y."
        
        # Format output
        return format_schedule_results(results, title, len(results[0]) > 6)
        
    except Exception as e:
        return f"â�Œ Lá»—i truy váº¥n database: {str(e)}"

@tool(return_direct=True)
def ctuet_grades_db_tool(query: str) -> str:
    """
    Láº¥y Ä‘iá»ƒm sá»‘ tá»« database SQLite ná»™i bá»™
    Há»— trá»£: 'Ä‘iá»ƒm sá»‘', 'báº£ng Ä‘iá»ƒm', 'káº¿t quáº£ há»�c táº­p'
    """
    try:
        conn = sqlite3.connect('/tmp/ctuet_database.db')
        cursor = conn.cursor()
        
        # Láº¥y Ä‘iá»ƒm sá»‘ cá»§a sinh viÃªn
        sql = """
        SELECT s.subject_name, s.credits, g.midterm_score, g.final_score, 
               g.assignment_score, g.attendance_score, g.total_score, 
               g.letter_grade, g.grade_points
        FROM grades g
        JOIN subjects s ON g.subject_id = s.id
        JOIN students st ON g.student_id = st.id
        WHERE st.username = ? AND g.semester_id = 1
        ORDER BY s.subject_name
        """
        
        cursor.execute(sql, ('tieens1802',))
        results = cursor.fetchall()
        
        if not results:
            return "ğŸ“Š ChÆ°a cÃ³ Ä‘iá»ƒm sá»‘ nÃ o Ä‘Æ°á»£c cáº­p nháº­t."
        
        # TÃ­nh GPA
        total_credits = sum(row[1] for row in results)
        total_grade_points = sum(row[1] * row[8] for row in results)
        gpa = total_grade_points / total_credits if total_credits > 0 else 0
        
        # Format output
        output = f"ğŸ“Š **Báº¢NG Ä�Iá»‚M Há»ŒC Ká»² 1 - 2024-2025**\n"
        output += f"ğŸ‘¤ **Sinh viÃªn**: Nguyá»…n VÄƒn Tiáº¿n (tieens1802)\n"
        output += f"ğŸ“… **Cáº­p nháº­t**: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        
        output += "| MÃ´n há»�c | TC | Giá»¯a ká»³ | Cuá»‘i ká»³ | BT | ChuyÃªn cáº§n | Tá»•ng | Xáº¿p loáº¡i |\n"
        output += "|---------|----|---------|---------|----|------------|------|----------|\n"
        
        for row in results:
            subject, credits, mid, final, assign, attend, total, letter, points = row
            output += f"| {subject[:20]:<20} | {credits} | {mid:>6.1f} | {final:>7.1f} | {assign:>2.1f} | {attend:>10.1f} | {total:>4.1f} | {letter:>8} |\n"
        
        output += f"\nğŸ“ˆ **Thá»‘ng kÃª:**\n"
        output += f"â€¢ **Tá»•ng sá»‘ tÃ­n chá»‰**: {total_credits}\n"
        output += f"â€¢ **GPA há»�c ká»³**: {gpa:.2f}/4.0\n"
        output += f"â€¢ **Xáº¿p loáº¡i**: {'Xuáº¥t sáº¯c' if gpa >= 3.6 else 'Giá»�i' if gpa >= 3.2 else 'KhÃ¡' if gpa >= 2.5 else 'Trung bÃ¬nh'}\n"
        
        return output
        
    except Exception as e:
        return f"â�Œ Lá»—i truy váº¥n Ä‘iá»ƒm sá»‘: {str(e)}"

@tool(return_direct=True)
def ctuet_announcements_db_tool(query: str) -> str:
    """
    Láº¥y thÃ´ng bÃ¡o tá»« database SQLite ná»™i bá»™
    Há»— trá»£: 'thÃ´ng bÃ¡o', 'tin tá»©c', 'announcements'
    """
    try:
        conn = sqlite3.connect('/tmp/ctuet_database.db')
        cursor = conn.cursor()
        
        # Láº¥y thÃ´ng bÃ¡o cÃ²n hiá»‡u lá»±c
        sql = """
        SELECT title, content, category, priority, publish_date, expire_date, created_by
        FROM announcements
        WHERE is_published = 1 
        AND (expire_date IS NULL OR expire_date > datetime('now'))
        AND (target_audience = 'all' OR target_audience = 'students')
        ORDER BY 
            CASE priority 
                WHEN 'urgent' THEN 1
                WHEN 'high' THEN 2  
                WHEN 'normal' THEN 3
                WHEN 'low' THEN 4
            END,
            publish_date DESC
        LIMIT 10
        """
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        if not results:
            return "ğŸ“¢ Hiá»‡n táº¡i khÃ´ng cÃ³ thÃ´ng bÃ¡o má»›i."
        
        # Format output
        output = f"ğŸ“¢ **THÃ”NG BÃ�O Má»šI NHáº¤T**\n"
        output += f"ğŸ“… Cáº­p nháº­t: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        
        priority_icons = {
            'urgent': 'ğŸš¨',
            'high': 'â�—',
            'normal': 'ğŸ“Œ',
            'low': 'ğŸ’¬'
        }
        
        category_icons = {
            'academic': 'ğŸ“š',
            'administrative': 'ğŸ“‹', 
            'event': 'ğŸ�‰',
            'scholarship': 'ğŸ’°'
        }
        
        for i, (title, content, category, priority, pub_date, exp_date, created_by) in enumerate(results, 1):
            priority_icon = priority_icons.get(priority, 'ğŸ“Œ')
            category_icon = category_icons.get(category, 'ğŸ“„')
            
            output += f"## {priority_icon} **{i}. {title}**\n"
            output += f"{category_icon} *{category.title()}* | ğŸ‘¤ {created_by}\n"
            output += f"ğŸ“… {pub_date[:10]}"
            
            if exp_date:
                output += f" â�° Háº¿t háº¡n: {exp_date[:10]}"
            output += f"\n\n{content}\n\n"
            output += "---\n"
        
        return output
        
    except Exception as e:
        return f"â�Œ Lá»—i truy váº¥n thÃ´ng bÃ¡o: {str(e)}"

@tool(return_direct=True)
def ctuet_scholarships_db_tool(query: str) -> str:
    """
    Láº¥y thÃ´ng tin há»�c bá»•ng tá»« database SQLite ná»™i bá»™
    """
    try:
        conn = sqlite3.connect('/tmp/ctuet_database.db')
        cursor = conn.cursor()
        
        # Láº¥y thÃ´ng tin há»�c bá»•ng
        sql = """
        SELECT scholarship_name, description, amount, requirements, application_deadline, status
        FROM scholarships
        WHERE status = 'active' AND application_deadline >= date('now')
        ORDER BY application_deadline ASC
        """
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        if not results:
            return "ğŸ’° Hiá»‡n táº¡i khÃ´ng cÃ³ há»�c bá»•ng nÃ o Ä‘ang má»Ÿ Ä‘Äƒng kÃ½."
        
        # Kiá»ƒm tra Ä‘iá»�u kiá»‡n sinh viÃªn hiá»‡n táº¡i
        cursor.execute("SELECT gpa FROM students WHERE username = ?", ('tieens1802',))
        student_gpa = cursor.fetchone()[0]
        
        output = f"ğŸ’° **Há»ŒC Bá»”NG Ä�ANG Má»� Ä�Ä‚NG KÃ�**\n"
        output += f"ğŸ‘¤ **GPA hiá»‡n táº¡i cá»§a báº¡n**: {student_gpa:.2f}/4.0\n\n"
        
        for i, (name, desc, amount, requirements, deadline, status) in enumerate(results, 1):
            output += f"## ğŸ�“ **{i}. {name}**\n"
            output += f"ğŸ’µ **Sá»‘ tiá»�n**: {amount:,.0f} VNÄ�\n"
            output += f"ğŸ“� **MÃ´ táº£**: {desc}\n"
            output += f"ğŸ“‹ **Ä�iá»�u kiá»‡n**: {requirements}\n"
            output += f"â�° **Háº¡n ná»™p Ä‘Æ¡n**: {deadline}\n"
            
            # Kiá»ƒm tra Ä‘iá»�u kiá»‡n
            if "GPA >= 3.2" in requirements and student_gpa >= 3.2:
                output += f"âœ… **Báº¡n Ä‘á»§ Ä‘iá»�u kiá»‡n GPA Ä‘á»ƒ Ä‘Äƒng kÃ½!**\n"
            elif "GPA >= 3.2" in requirements:
                output += f"â�Œ **Báº¡n chÆ°a Ä‘á»§ Ä‘iá»�u kiá»‡n GPA (cáº§n >= 3.2)**\n"
            else:
                output += f"ğŸ’¡ **Vui lÃ²ng kiá»ƒm tra Ä‘iá»�u kiá»‡n chi tiáº¿t**\n"
                
            output += "\n---\n"
        
        return output
        
    except Exception as e:
        return f"â�Œ Lá»—i truy váº¥n há»�c bá»•ng: {str(e)}"

def format_schedule_results(results, title, is_weekly=False):
    """Format káº¿t quáº£ lá»‹ch há»�c"""
    output = f"{title}\n"
    output += f"ğŸ•� Cáº­p nháº­t: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
    
    if is_weekly:
        # Group by day for weekly view
        days_map = {1: "ğŸ—“ï¸� Chá»§ nháº­t", 2: "ğŸ“… Thá»© 2", 3: "ğŸ“… Thá»© 3", 4: "ğŸ“… Thá»© 4", 5: "ğŸ“… Thá»© 5", 6: "ğŸ“… Thá»© 6", 7: "ğŸ“… Thá»© 7"}
        daily_schedule = {}
        
        for row in results:
            subject, day, start_time, end_time, room, teacher, class_type = row
            day_name = days_map.get(day, f"NgÃ y {day}")
            
            if day_name not in daily_schedule:
                daily_schedule[day_name] = []
            
            daily_schedule[day_name].append({
                'subject': subject,
                'time': f"{start_time[:5]} - {end_time[:5]}", 
                'room': room,
                'teacher': teacher,
                'type': class_type
            })
        
        for day, classes in daily_schedule.items():
            output += f"### {day}\n"
            for class_info in classes:
                type_icon = "ğŸ“–" if class_info['type'] == 'theory' else "ğŸ’»" if class_info['type'] == 'lab' else "âœ�ï¸�"
                output += f"â€¢ {type_icon} **{class_info['subject']}**\n"
                output += f"  â�° {class_info['time']}\n"
                output += f"  ğŸ‘¨â€�ğŸ�« {class_info['teacher']}\n"
                output += f"  ğŸ�« {class_info['room']}\n\n"
            output += "---\n"
    else:
        # Single day view
        for row in results:
            subject, start_time, end_time, room, teacher, class_type = row
            type_icon = "ğŸ“–" if class_type == 'theory' else "ğŸ’»" if class_type == 'lab' else "âœ�ï¸�"
            output += f"â€¢ {type_icon} **{subject}**\n"
            output += f"  â�° {start_time[:5]} - {end_time[:5]}\n"
            output += f"  ğŸ‘¨â€�ğŸ�« {teacher}\n"
            output += f"  ğŸ�« {room}\n\n"
    
    return output

print("âœ… Database tools Ä‘Ã£ Ä‘Æ°á»£c táº¡o thÃ nh cÃ´ng!")


# CÃ i Ä‘áº·t thÆ° viá»‡n
!pip install -q -U google-genai


import requests
import json
from kaggle_secrets import UserSecretsClient
import time

# Setup Gemini API
try:
    gemini_api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")
    if not gemini_api_key:
        print("â�Œ Vui lÃ²ng thÃªm GEMINI_API_KEY vÃ o Kaggle Secrets")
    else:
        print("âœ… Gemini API Key found!")
        
except Exception as e:
    print(f"â�Œ Error getting Gemini API key: {e}")

# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini_api(prompt: str, system_instruction: str = None) -> str:
    """Call Gemini 2.0 Flash API"""
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    # Táº¡o content cho API
    parts = []
    
    if system_instruction:
        parts.append({"text": f"System: {system_instruction}\n\nUser: {prompt}"})
    else:
        parts.append({"text": prompt})
    
    data = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1500,
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "â�Œ KhÃ´ng nháº­n Ä‘Æ°á»£c response tá»« Gemini API"
            
    except Exception as e:
        return f"â�Œ Lá»—i Gemini API: {str(e)}"

print("ğŸ”§ Gemini API functions ready!")


@tool(return_direct=True)
def gemini_research_assistant_tool(query: str) -> str:
    """
    Gemini 2.0 Flash Research Assistant - Tráº£ lá»�i cÃ¢u há»�i lÃ½ thuyáº¿t, khÃ¡i niá»‡m 
    Há»— trá»£: Ä‘á»‹nh nghÄ©a, giáº£i thÃ­ch, lÃ½ thuyáº¿t, kiáº¿n thá»©c chung
    """
    try:
        # ThÃªm instruction ngáº¯n gá»�n Ä‘á»ƒ Gemini tráº£ lá»�i sÃºc tÃ­ch
        prompt = f"Tráº£ lá»�i ngáº¯n gá»�n vÃ  Ä‘Ãºng trá»�ng tÃ¢m cÃ¢u há»�i sau: {query}"
        
        gemini_response = call_gemini_api(prompt)
        return gemini_response
        
    except Exception as e:
        return f"â�Œ Lá»—i: {str(e)}"

print("ğŸ¤– Gemini Research Assistant Ä‘Ã£ Ä‘Æ°á»£c Ä‘Æ¡n giáº£n hÃ³a!")


# OPTIMIZED ROUTER PROMPT cho Há»‡ thá»‘ng Há»— trá»£ Sinh viÃªn CTUET

from langchain.prompts import PromptTemplate

# Cáº­p nháº­t tools list
tools = [
    student_advisor_tool,           # Tool chÃ­nh - tÆ° váº¥n sinh viÃªn CTUET
    ctuet_schedule_db_tool,        # Lá»‹ch há»�c tá»« DB
    ctuet_grades_db_tool,          # Ä�iá»ƒm sá»‘ tá»« DB  
    ctuet_announcements_db_tool,   # ThÃ´ng bÃ¡o tá»« DB
    ctuet_scholarships_db_tool,    # Há»�c bá»•ng tá»« DB
    gemini_research_assistant_tool  # Tool Gemini - fallback cho cÃ¢u há»�i khÃ¡c
]

# ENHANCED ROUTER PROMPT - Fixed ReAct Format
enhanced_router_prompt = PromptTemplate.from_template("""
You are an AI routing agent for CTUET student support system.

ROUTING LOGIC:

TIER 1 - DATABASE TOOLS (Highest priority):
- ctuet_schedule_db_tool: schedule, timetable, class schedule keywords
- ctuet_grades_db_tool: grades, scores, academic results keywords  
- ctuet_announcements_db_tool: announcements, news from school keywords
- ctuet_scholarships_db_tool: scholarships, financial aid keywords

TIER 2 - STUDENT ADVISOR (Medium priority):
- student_advisor_tool: CTUET support issues (passwords, student info, forms, regulations, etc.)

TIER 3 - GEMINI FALLBACK (All other questions):
- gemini_research_assistant_tool: General knowledge, theory, greetings, non-CTUET topics

EXAMPLES:
- "Xem báº£ng Ä‘iá»ƒm" â†’ ctuet_grades_db_tool
- "Lá»‹ch há»�c ngÃ y mai" â†’ ctuet_schedule_db_tool  
- "Ä�á»•i máº­t kháº©u CTUET" â†’ student_advisor_tool
- "Python lÃ  gÃ¬" â†’ gemini_research_assistant_tool

IMPORTANT: 
- Action must be EXACTLY one of the tool names: {tool_names}
- Action Input should be the user's question or relevant part of it
- Do NOT add parentheses or function call syntax in Action field

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# Táº¡o agent vá»›i prompt Ä‘Æ°á»£c tá»‘i Æ°u
agent = create_react_agent(
    llm=gemma_langchain, 
    tools=tools, 
    prompt=enhanced_router_prompt,
    stop_sequence=['\nObservation:']
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    max_iterations=3,
    handle_parsing_errors=True
)

print("ğŸ�¯ Router Agent Ä‘Ã£ Ä‘Æ°á»£c tá»‘i Æ°u vá»›i logic phÃ¢n táº§ng!")
print("ğŸ“Š Cáº¥u trÃºc routing:")
print("  ğŸ¥‡ Táº§ng 1 - Database Tools (Real-time data)")
print("  ğŸ¥ˆ Táº§ng 2 - Student Advisor (CTUET support)")  
print("  ğŸ¥‰ Táº§ng 3 - Gemini (General knowledge + fallback)")

# Test cases Ä‘á»ƒ verify routing logic
test_cases = [
    "Lá»‹ch há»�c ngÃ y mai cá»§a tÃ´i",                    # â†’ ctuet_schedule_db_tool
    "Xem báº£ng Ä‘iá»ƒm há»�c ká»³ nÃ y",                     # â†’ ctuet_grades_db_tool
    "CÃ³ thÃ´ng bÃ¡o má»›i nÃ o khÃ´ng",                   # â†’ ctuet_announcements_db_tool
    "Ä�iá»�u kiá»‡n há»�c bá»•ng khuyáº¿n khÃ­ch",              # â†’ ctuet_scholarships_db_tool
    "LÃ m sao Ä‘á»•i máº­t kháº©u tÃ i khoáº£n",              # â†’ student_advisor_tool
    "Máº«u Ä‘Æ¡n xin nghá»‰ há»�c á»Ÿ Ä‘Ã¢u",                  # â†’ student_advisor_tool
    "Python lÃ  gÃ¬",                                 # â†’ gemini_research_assistant_tool
    "CÃ¡ch há»�c machine learning hiá»‡u quáº£",          # â†’ gemini_research_assistant_tool
    "HÃ´m nay thá»© máº¥y",                             # â†’ gemini_research_assistant_tool
]

print("\nğŸ§ª Test cases máº«u:")
for i, case in enumerate(test_cases, 1):
    print(f"  {i}. '{case}'")


result = agent_executor.invoke({"input": "mÃ¡y há»�c lÃ  gÃ¬"})


# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_community.vectorstores import FAISS

# # Point to the directory containing .txt files, not the JSON file
# loader = DirectoryLoader(
#     '/kaggle/input/letmecook/',  # Directory path, not file path
#     loader_cls=TextLoader,
#     glob='**/*.txt',
#     show_progress=True,
# )
# docs = loader.load()
# print(f"{len(docs)} Documents loaded")


# queries = [
#     'Báº£o hiá»ƒm y táº¿',
#     'bao hiem y te',
#     'tuyen sinh',
#     'tuyá»ƒn sinh',
#     'Missing Values',
#     'What are missing values?',
#     'Tell me what "missing values" means in data science',
#     'Teach me something about missing values',
#     'Naive Bayes',
#     'What are the applications of naive bayes',
#     'Explain to me what naive bayes is',
#     'What is PCA?',
#     'Teach me what Feature engineering is',
#     'How to build a Linear Regression model?'
# ]
    
# retrieve_and_display(queries, retriever)


# import gc

# # Free some memory before loading the new model
# gc.collect()
# torch.cuda.empty_cache()

# librarian_tokenizer, librarian_model = load_model() # If this step fails check the resource consumption of the notebook
# lc_librarian_model = GemmaLangChain(model=librarian_model, tokenizer=librarian_tokenizer)


# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough


# rag_prompt = PromptTemplate.from_template("""You are a helpful assistant, the Librarian from the DSI Crew (Data Science Instructor Crew). Your role is to answer Data Science questions asked by a student.

# Make sure to give clear, detailed, and relevant answers, and be professional with a teacher tone. 

# If the following pieces of context are relevant to the question, use them to respond to the question. 

# Context: {context} 

# If the Context is not relevant to the question, ignore the Context and provide your own relevant, detailed, explanatory, and high-quality response. You must answer the question!

# Question: {question}

# Answer:
# """)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # More about this syntax below
# librarian = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | rag_prompt
#     | lc_librarian_model
#     | StrOutputParser()
# )


# print(librarian.invoke("What is PCA?"))


# @tool(return_direct=True) # Return answer directly to the user
# def data_science_concept_definitions(query:str) -> str:
#     """This tool contains definitions of data science concepts. Use this tool for theoretical questions. Don\'t rely on this tool for programming/code implementation tasks. The input is a query"""
    
#     query = query.removesuffix(stop_word) # The agent was including the stop word
#     return librarian.invoke(query)

# data_science_concept_definitions.description = data_science_concept_definitions.description.removeprefix('data_science_concept_definitions(query:str) -> str - ')

# # programming_tool.return_direct = True # Return answer directly to the user

# tools = [data_science_concept_definitions]


# router_agent_prompt = PromptTemplate.from_template(""" 
# You have access to the following tools that you must forward the question to:

# {tools}

# Your role is to choose the most appropriate tool for the given question. 

# Use the following format:

# Question: the input question
# Thought: you must always think about what tool is best suited to answer the question
# Action: the name of the tool to use, it has to be exactly one of [{tool_names}]. Only the name!
# Action Input: the input question
# Action Output: only the output of the tool


# Begin!

# Given the following question choose the tool that is most likely to provide a relevant good answer.

# Question: {input}
# Thought: {agent_scratchpad}""")

# agent = create_react_agent(gemma_langchain, tools, router_agent_prompt, stop_sequence=[stop_word])

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     max_iterations=2,
#     return_intermediate_steps=True,
# )


result = agent_executor.invoke({"input": "xin chÃ o"})


result = agent_executor.invoke({"input":  "cÃ³ dá»‹ch vá»¥ tÆ° váº¥n tÃ¢m lÃ½ khÃ´ng"})


# def invoke_agent(input):
#     return agent_executor.invoke({"input": input})['intermediate_steps'][0][1]
    
# messages = []


def invoke_agent(input):
    return agent_executor.invoke({"input": input})['output']


# Utility Code, You can ignore
import utilty_scripts_for_build_ai_agents_with_gemma as utils
utils.start_demo(messages, invoke_agent)

