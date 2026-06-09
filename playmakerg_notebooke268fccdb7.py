!pip install tqdm


import numpy as np
import pandas as pd
import os
import string
import re
from tqdm import tqdm

pd.set_option('display.max_columns', None)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# df_sample = pd.read_csv("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv")
df_train =pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet")
df_test = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet")
df=df_train
dft = df_test



def contains_code(text):
    code_patterns = [
        r'```[\s\S]+?```',          
        r'`[^`]+`',          
        r'(?m)^(?: {4}|\t).+',
        r'\bdef\b|\bclass\b|\bimport\b|\breturn\b|\bprint\b|\bfor\b|\bwhile\b|\bif\b|\belse\b',
        r'<[a-zA-Z/][^>]*?>',          
    ]

    text = text.strip()

    for pattern in code_patterns:
        if re.search(pattern, text):
            if re.search(r'`[^`]+`', text):
                if not re.search(r'`[a-zA-Z0-9\s]+`', text):
                    return True
            else:
                return True
    return False


def first_text_categorization(row):
    prompt = row['prompt']
    response_a = row['response_a']
    response_b = row['response_b']
    
    prompt_bool = contains_code(prompt.lower())
    response_a_bool = contains_code(response_a.lower())
    response_b_bool = contains_code(response_b.lower())
    
    text_bool = prompt_bool or response_a_bool or response_b_bool
    return prompt_bool,response_a_bool,response_b_bool,text_bool


tqdm.pandas()
cs_data = list(df.progress_apply(first_text_categorization, axis=1))


tqdm.pandas()
test_cs_data = list(dft.progress_apply(first_text_categorization, axis=1))


df['cs_data'] = cs_data
df['code_in_prompt']= df['cs_data'].apply(lambda x: x[0])
df['code_in_response_a']= df['cs_data'].apply(lambda x: x[1])
df['code_in_response_b']= df['cs_data'].apply(lambda x: x[2])
df['code']= df['cs_data'].apply(lambda x: x[3])
del(df['cs_data'])


dft['cs_data'] = test_cs_data
dft['code_in_prompt']= dft['cs_data'].apply(lambda x: x[0])
dft['code_in_response_a']= dft['cs_data'].apply(lambda x: x[1])
dft['code_in_response_b']= dft['cs_data'].apply(lambda x: x[2])
dft['code']= dft['cs_data'].apply(lambda x: x[3])
del(dft['cs_data'])


def contains_code_xtreme(text):
    code_patterns = [
        r'[{}();=\[\]:<>]',
        r'#include\s*<.+>',        
        r'\bdef\s+\w+\s*\(.*\):',
        r'\bclass\s+\w+\s*\(?.*?\)?:',        
        r'```[a-zA-Z]*\n.*?```',
        r'^[ \t]*\w+.*\(.*\).*{',
    ]

    matches = sum(bool(re.search(pattern, text, re.MULTILINE)) for pattern in code_patterns)
    
    if matches >= 2:
        return True

    if re.search(r'```.*?```|`[^`]+`', text, re.MULTILINE):
        return True

    lines = text.split('\n')
    symbol_count_threshold = 5
    for line in lines:
        symbols = re.findall(r'[{}();=\[\]:<>]', line)
        if len(symbols) >= symbol_count_threshold:
            return True

    return False


def main_contains_code_xtreme(row):
    prompt = row['prompt']
    response_a = row['response_a']
    response_b = row['response_b']

    prompt_bool = row['code_in_prompt']
    response_a_bool = row['code_in_response_a']
    response_b_bool = row['code_in_response_b']
    text_bool = row['code']
    
    if text_bool==False:
        return False,False,False,False

    if prompt_bool:
        prompt_bool = contains_code_xtreme(prompt)
    if response_a_bool:
        response_a_bool = contains_code_xtreme(response_a)
    if response_b_bool:
        response_b_bool = contains_code_xtreme(response_b)
    
    text_bool = prompt_bool or response_a_bool or response_b_bool
    return prompt_bool,response_a_bool,response_b_bool,text_bool


tqdm.pandas()
x_cs_data = list(df.progress_apply(main_contains_code_xtreme, axis=1))


tqdm.pandas()
test_x_cs_data = list(dft.progress_apply(main_contains_code_xtreme, axis=1))


df['x_cs_data'] = x_cs_data
df['code_in_prompt']= df['x_cs_data'].apply(lambda x: x[0])
df['code_in_response_a']= df['x_cs_data'].apply(lambda x: x[1])
df['code_in_response_b']= df['x_cs_data'].apply(lambda x: x[2])
df['code']= df['x_cs_data'].apply(lambda x: x[3])
del(df['x_cs_data'])


dft['x_cs_data'] = test_x_cs_data
dft['code_in_prompt']= dft['x_cs_data'].apply(lambda x: x[0])
dft['code_in_response_a']= dft['x_cs_data'].apply(lambda x: x[1])
dft['code_in_response_b']= dft['x_cs_data'].apply(lambda x: x[2])
dft['code']= dft['x_cs_data'].apply(lambda x: x[3])
del(dft['x_cs_data'])


print(len(df[df['code']==True]) , len(df[df['code_in_prompt']==True]) , len(df[df['code_in_response_a']==True]) , len(df[df['code_in_response_b']==True]))
len(dft[dft['code']==True]) , len(dft[dft['code_in_prompt']==True]) , len(dft[dft['code_in_response_a']==True]) , len(dft[dft['code_in_response_b']==True])


import re

def detect_code_language_details(text):

    web_dev_pattern = re.compile(
    r'(?i)(?:'
    r'\{[^{}]*?\b[a-zA-Z-]+\s*:\s*[^;{}]+;\s*\}|'  
    r'<style\b[^>]*>[^<]*?\{[^{}]*?\b[a-zA-Z-]+\s*:\s*[^;{}]+;\s*\}[^<]*<\/style>|'
    r'\b(function|var|let|const|=>|return|if|else|for|while|class)\b|'  
    r'<[a-zA-Z!\/][^>]*>'
    r')'
) 
    python_pattern = re.compile(r"\b(def|class|import|from|try|except|with|open|input|print|len|map|filter|zip|lambda|enumerate|range|dict|list|set|tuple|pandas|numpy|matplotlib|seaborn|scikit-learn|tensorflow|keras|torch|DataFrame|Series|read_csv|read_excel|head|tail|describe|groupby|merge|concat|plot|scatter|hist|bar|boxplot|mean|median|std|min|max|sum|apply|transform|pivot_table|correlation|train_test_split|fit|predict|accuracy_score|confusion_matrix|classification_report|roc_curve|auc)\b")
    java_pattern = re.compile(r"\b(public|private|protected|class|void|String|new|try|catch)\b")
    cpp_pattern = re.compile(r"\b(int|float|double|char|class|#include)\b")
    comprehensive_pattern = re.compile(r"\b(lambda|print|System\.out\.println|cout|cin|alert)\b")

    web_dev_matches = web_dev_pattern.findall(text)
    python_matches = python_pattern.findall(text)
    java_matches = java_pattern.findall(text)
    cpp_matches = cpp_pattern.findall(text)
    comprehensive_matches = comprehensive_pattern.findall(text)

    filtered_python_matches = [m for m in python_matches if m not in {"if", "return", "for", "while"}]
    filtered_java_matches = [m for m in java_matches if m not in {"if", "return", "for", "while"}]
    filtered_cpp_matches = [m for m in cpp_matches if m not in {"if", "return", "for", "while"}]

    lang_list = []
    if len(list(set(web_dev_matches)))>0:
        lang_list.append(1)
    else:
        lang_list.append(0)
        
    if len(list(set(python_matches)))>0 or len(list(set(java_matches)))>0 or len(list(set(cpp_matches)))>0:
        lang_list.append(1)
    else:
        lang_list.append(0)
        
    if len(list(set(comprehensive_matches)))>0:
        lang_list.append(1)
    else:
        lang_list.append(0)

    return lang_list


# "Web-dev", ["Python" | "Java" | "C++"], "Comprehensive"

def detect_code_language(row):
    if row['code']==True:
        code_languages_in_prompt = detect_code_language_details(row['prompt'])
        code_languages_in_response_a =  detect_code_language_details(row['response_a'])
        code_languages_in_response_b =  detect_code_language_details(row['response_b'])
        return [code_languages_in_prompt,code_languages_in_response_a,code_languages_in_response_b]
    else:
        code_languages_in_prompt=[0,0,0]
        code_languages_in_response_a=[0,0,0]
        code_languages_in_response_b=[0,0,0]
        return [code_languages_in_prompt,code_languages_in_response_a,code_languages_in_response_b]


df[["code_languages_in_prompt", "code_languages_in_response_a", "code_languages_in_response_b"]] = df.apply(detect_code_language, axis=1, result_type="expand")


dft[["code_languages_in_prompt", "code_languages_in_response_a", "code_languages_in_response_b"]] = dft.apply(detect_code_language, axis=1, result_type="expand")


def clean_for_segmentation(row):    
    text = row['prompt']
    characters_to_remove = r"[-.,\n'\":!?]"
    cleaned_text = re.sub(characters_to_remove, "", text)
    final_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return final_text


df['ft_prompt'] = df.apply(clean_for_segmentation,axis=1)


dft['ft_prompt'] = dft.apply(clean_for_segmentation,axis=1)


df["prompt_len"] = df['ft_prompt'].apply(lambda x: len(x.split(' ')))


dft["prompt_len"] = dft['ft_prompt'].apply(lambda x: len(x.split(' ')))


def segmentation(row):
    sentences=[]
    prompt_len = row['prompt_len']
    prompt = row['ft_prompt']
    prompt_arr = prompt.split(' ')
    segment = 6
    if prompt_len<6:
        text = " ".join(prompt_arr[0:prompt_len])
        sentences.append(text)
    else:
        for i in range(0,prompt_len,segment):
            if i+(2*segment)>prompt_len:
                text = " ".join(prompt_arr[i:prompt_len])
                sentences.append(text)
                break
            else:    
                text = " ".join(prompt_arr[i:i+segment])
                sentences.append(text)
    return sentences


df['prompt_segements'] = df.apply(segmentation,axis=1)


dft['prompt_segements'] = dft.apply(segmentation,axis=1)


!pip install langcodes
!pip install langdetect 


from nltk.corpus import stopwords
import nltk 

nltk.download('punkt')
nltk.download('stopwords')


import langcodes
from langdetect import detect, detect_langs


def lang_detection_prompt(row):
    sentences = row['prompt_segements']
    extra_stopwords = []
    language_dict = {}
    language_frequency = {}
    language_weight = {}
    total= 0
    filtered_prompt = ''
    
    for sentence in sentences:
        try:
            detected_code = detect(sentence) 
            detected_language = langcodes.Language.get(detected_code).display_name()

            if detected_language in language_frequency:
                language_frequency[detected_language] += 1
            else:
                language_frequency[detected_language] = 1

            if (detected_language is None) or (detected_language ==''):
                detected_language = 'english'
            
        except Exception as e:
            break
    
    for key in list(language_frequency.keys()):
        total = total+ language_frequency[key]

    for key in list(language_frequency.keys()):
        language_weight[key]= language_frequency[key]/total

    # return [filtered_prompt,language_frequency,language_weight]
    return [language_frequency,language_weight]


tqdm.pandas()
ls_prompt = list(df.progress_apply(lang_detection_prompt, axis=1))


tqdm.pandas()
test_ls_prompt = list(dft.progress_apply(lang_detection_prompt, axis=1))


df['prompt_meta'] = ls_prompt
df['prompt_counter']= df['prompt_meta'].apply(lambda x: x[0])
df['prompt_weight']= df['prompt_meta'].apply(lambda x: x[1])
del(df['prompt_meta'])


dft['prompt_meta'] = test_ls_prompt
dft['prompt_counter']= dft['prompt_meta'].apply(lambda x: x[0])
dft['prompt_weight']= dft['prompt_meta'].apply(lambda x: x[1])
del(dft['prompt_meta'])


# later : just check when dictionary have languages with equal ratio then care 

def get_dom_language(row):
    counter = row['prompt_counter']
    if counter:
        max_pair = max(counter.items(), key=lambda x: (x[1], x[0] == "english"))
        return max_pair[0]
    else:
        print(counter,' : ',row['id'],' : ',row.index[0])


tqdm.pandas()
df['dom_lang'] = df.progress_apply(get_dom_language, axis=1)


tqdm.pandas()
dft['dom_lang'] = dft.progress_apply(get_dom_language, axis=1)


def contains_mathematical_operations(text, threshold=0.3):
    numbers = re.findall(r'\d+(\.\d+)?', text)
    operators = re.findall(r'[+\-*/=]', text)
    equation_pattern = r'(\d+=)+\d+'
    equations = re.findall(equation_pattern, text)
    math_expression_pattern = r'(\d+[+\-*/=]\d+)'
    math_expressions = re.findall(math_expression_pattern, text)
    total_chars = len(text)
    try:
        num_chars = sum(len(num) for num in numbers)
        if total_chars<=0:
            return False
        proportion_of_numbers = num_chars / total_chars
        if proportion_of_numbers > threshold or len(equations) > 0 or len(math_expressions) > 0:
            return True
        else:
            return False
    except:
        print(text)


def text_code_unknown_categorization(row):
    if row['code']:
        return 'code'
    if contains_mathematical_operations(row['ft_prompt']) or contains_mathematical_operations(row['response_a']) or contains_mathematical_operations(row['response_b']):
        return 'logical_operation'
    counter = row['prompt_counter']
    if counter:
        return 'text'
    else:
        return 'unknown'


tqdm.pandas()
df['type'] = df.progress_apply(text_code_unknown_categorization, axis=1)


tqdm.pandas()
dft['type'] = dft.progress_apply(text_code_unknown_categorization, axis=1)


!pip install googletrans==4.0.0-rc1


from googletrans import Translator, LANGUAGES

translator = Translator()


def translation(row):
        
    source_language = "auto"
    target_language = row['dom_lang']  
    if target_language=='None' or target_language==None:
        target_language='english'
        
    response_a = row['response_a']
    response_b = row['response_b']
    prompt = row['ft_prompt']
    
    if row['type']=='unknown' or row['type']=='code' or row['type']=='logical_operation':
        return prompt,response_a,response_b
  
    src_code = next((code for code, name in LANGUAGES.items() if name.lower() == source_language.lower()), None)
    dest_code = next((code for code, name in LANGUAGES.items() if name.lower() == target_language.lower()), None)
    
    if src_code and dest_code:
        tr_prompt = translator.translate(prompt, src=src_code, dest=dest_code)
        tr_response_a = translator.translate(response_a, src=src_code, dest=dest_code)
        tr_response_b= translator.translate(response_b, src=src_code, dest=dest_code)
        return tr_prompt.text,tr_response_a.text,tr_response_b.text
    else:
        return prompt,response_a,response_b


tqdm.pandas()
tr_list = list(df.progress_apply(translation, axis=1))


tqdm.pandas()
test_tr_list = list(dft.progress_apply(translation, axis=1))


df['tr_meta'] = tr_list
df['tr_prompt']= df['tr_meta'].apply(lambda x: x[0])
df['tr_response_a']= df['tr_meta'].apply(lambda x: x[1])
df['tr_response_b']= df['tr_meta'].apply(lambda x: x[2])
del(df['tr_meta'])


dft['tr_meta'] = test_tr_list
dft['tr_prompt']= dft['tr_meta'].apply(lambda x: x[0])
dft['tr_response_a']= dft['tr_meta'].apply(lambda x: x[1])
dft['tr_response_b']= dft['tr_meta'].apply(lambda x: x[2])
del(dft['tr_meta'])


def eng_translation(row):
        
    source_language = "auto"
    target_language = 'english' 
        
    response_a = row['response_a']
    response_b = row['response_b']
    prompt = row['ft_prompt']
    
    if row['type']=='unknown' or row['type']=='code' or row['type']=='logical_operation':
        return prompt,response_a,response_b
  
    src_code = next((code for code, name in LANGUAGES.items() if name.lower() == source_language.lower()), None)
    dest_code = next((code for code, name in LANGUAGES.items() if name.lower() == target_language.lower()), None)
    
    if src_code and dest_code:
        tr_prompt = translator.translate(prompt, src=src_code, dest=dest_code)
        tr_response_a = translator.translate(response_a, src=src_code, dest=dest_code)
        tr_response_b= translator.translate(response_b, src=src_code, dest=dest_code)
        return tr_prompt.text,tr_response_a.text,tr_response_b.text
    else:
        return prompt,response_a,response_b


tqdm.pandas()
eng_tr_list = list(df.progress_apply(eng_translation, axis=1))


tqdm.pandas()
test_eng_tr_list = list(dft.progress_apply(eng_translation, axis=1))


df['eng_tr_meta'] = eng_tr_list
df['eng_tr_prompt']= df['eng_tr_meta'].apply(lambda x: x[0])
df['eng_tr_response_a']= df['eng_tr_meta'].apply(lambda x: x[1])
df['eng_tr_response_b']= df['eng_tr_meta'].apply(lambda x: x[2])
del(df['eng_tr_meta'])


dft['eng_tr_meta'] = test_eng_tr_list
dft['eng_tr_prompt']= dft['eng_tr_meta'].apply(lambda x: x[0])
dft['eng_tr_response_a']= dft['eng_tr_meta'].apply(lambda x: x[1])
dft['eng_tr_response_b']= dft['eng_tr_meta'].apply(lambda x: x[2])
del(dft['eng_tr_meta'])


!pip install polyglot
!pip install PyICU
!pip install pycld2


def clean_before_wt_sum(text):    
    characters_to_remove = r"[-.,\n'\":!?]"
    cleaned_text = re.sub(characters_to_remove, "", text)
    final_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return final_text


from polyglot.detect import Detector
from charset_normalizer import from_bytes

def club_detection(row):
    
    if row['type']=='unknown' or row['type']=='logical_operation':
        return -1,-1,{},{}
    
    response_a_language_frequency={}
    response_b_language_frequency={}
    response_a_weighted_sum=0
    response_b_weighted_sum=0
      
    set_weight = row['prompt_weight']
    response_a = clean_before_wt_sum(row['response_a'])
    response_b = clean_before_wt_sum(row['response_b'])
    
    try:
        response_a_detector = Detector(response_a, quiet=True)
        response_b_detector = Detector(response_b, quiet=True)
        
        for language in response_a_detector.languages:
            if language.read_bytes==0 or language.name=='un':
                proportion=0
            else:
                proportion = language.read_bytes / sum(lang.read_bytes for lang in response_a_detector.languages)
            if (language.name not in response_a_language_frequency) and language.name!='un':
                response_a_language_frequency[language.name]=proportion
    
        for language in response_b_detector.languages:
            if language.read_bytes==0 or language.name=='un':
                proportion=0
            else:
                proportion = language.read_bytes / sum(lang.read_bytes for lang in response_b_detector.languages)
            if (language.name not in response_b_language_frequency) and language.name!='un':
                response_b_language_frequency[language.name]=proportion
    
        for key in list(response_a_language_frequency.keys()):
            if key in list(set_weight.keys()):
                response_a_weighted_sum = response_a_weighted_sum + set_weight[key]*response_a_language_frequency[key]
            else:
                response_a_weighted_sum = response_a_weighted_sum + 0*response_a_language_frequency[key]
    
        for key in list(response_b_language_frequency.keys()):
            if key in list(set_weight.keys()):
                response_b_weighted_sum = response_b_weighted_sum + set_weight[key]*response_b_language_frequency[key]
            else:
                response_b_weighted_sum = response_b_weighted_sum + 0*response_b_language_frequency[key]

    except:
        print(row['id'])
        # print(response_b_detector)

    return response_a_weighted_sum,response_b_weighted_sum,response_a_language_frequency,response_b_language_frequency


tqdm.pandas()
wt_list = list(df.progress_apply(club_detection, axis=1))


tqdm.pandas()
test_wt_list = list(dft.progress_apply(club_detection, axis=1))


df['wt_meta'] = wt_list
df['wt_sum_response_a']= df['wt_meta'].apply(lambda x: x[0])
df['wt_sum_response_b']= df['wt_meta'].apply(lambda x: x[1])
df['counter_response_a']= df['wt_meta'].apply(lambda x: x[2])
df['counter_response_b']= df['wt_meta'].apply(lambda x: x[3])
del(df['wt_meta'])


dft['wt_meta'] = test_wt_list
dft['wt_sum_response_a']= dft['wt_meta'].apply(lambda x: x[0])
dft['wt_sum_response_b']= dft['wt_meta'].apply(lambda x: x[1])
dft['counter_response_a']= dft['wt_meta'].apply(lambda x: x[2])
dft['counter_response_b']= dft['wt_meta'].apply(lambda x: x[3])
del(dft['wt_meta'])


print(len(df[df['type']=='text']) , len(df[df['type']=='code']) , len(df[df['type']=='logical_operation']) , len(df[df['type']=='unknown']))
len(dft[dft['type']=='text']) , len(dft[dft['type']=='code']) , len(dft[dft['type']=='logical_operation']) , len(dft[dft['type']=='unknown'])


!pip install sentence-transformers


import logging
from sentence_transformers import SentenceTransformer, util

# Suppress unnecessary output
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Load a pre-trained multilingual model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)


def compute_similarity(row):

    if row['type']=='logical_operations' or row['type']=='unknown':
        return -1,-1
        
    prompt = row['tr_prompt']
    response_a = row['tr_response_a']
    response_b = row['tr_response_b']

    prompt_embeddings = model.encode(prompt, convert_to_tensor=True)
    response_a_embeddings = model.encode(response_a, convert_to_tensor=True)
    response_b_embeddings = model.encode(response_b, convert_to_tensor=True)
    
    response_a_similarity_score = util.pytorch_cos_sim(prompt_embeddings, response_a_embeddings)
    response_b_similarity_score = util.pytorch_cos_sim(prompt_embeddings, response_b_embeddings)
    
    return response_a_similarity_score.item(), response_b_similarity_score.item()


tqdm.pandas()
sm_list = list(df.progress_apply(compute_similarity, axis=1))


tqdm.pandas()
test_sm_list = list(dft.progress_apply(compute_similarity, axis=1))


df['sm_list'] = sm_list
df['similarity_score_response_a']= df['sm_list'].apply(lambda x: x[0])
df['similarity_score_response_b']= df['sm_list'].apply(lambda x: x[1])
del(df['sm_list'])


dft['sm_list'] = test_sm_list
dft['similarity_score_response_a']= dft['sm_list'].apply(lambda x: x[0])
dft['similarity_score_response_b']= dft['sm_list'].apply(lambda x: x[1])
del(dft['sm_list'])


import re

def calculate_vocabulary(text):

    tokens = re.findall(r'\w+', text, re.UNICODE)
    normalized_tokens = [token.lower() for token in tokens]
    unique_words = list(set(normalized_tokens))
    return len(unique_words)


def vocab_complexity(row):
    if row['type']=='logical_operation' or row['type']=='unknown':
        return -1,-1,-1
        
    tr_prompt = row['tr_prompt']
    tr_response_a = row['tr_response_a']
    tr_response_b = row['tr_response_b']

    prompt_score=0
    response_a_score=0
    response_b_score=0
    
    if len(tr_prompt) != 0:
        prompt_score = round(calculate_vocabulary(tr_prompt)/len(tr_prompt),4)
    if len(tr_response_a)!=0:
        response_a_score = round(calculate_vocabulary(tr_response_a)/len(tr_response_a),4)
    if len(tr_response_b) != 0:
        response_b_score = round(calculate_vocabulary(tr_response_b)/len(tr_response_b),4)
    
    return prompt_score, response_a_score, response_b_score



tqdm.pandas()
vcs_list = list(df.progress_apply(vocab_complexity, axis=1))


tqdm.pandas()
test_vcs_list = list(dft.progress_apply(vocab_complexity, axis=1))


df['vcs_list'] = vcs_list
df['vocabulary_complexity_prompt']= df['vcs_list'].apply(lambda x: x[0])
df['vocabulary_complexity_response_a']= df['vcs_list'].apply(lambda x: x[1])
df['vocabulary_complexity_response_b']= df['vcs_list'].apply(lambda x: x[2])
del(df['vcs_list'])


dft['vcs_list'] = test_vcs_list
dft['vocabulary_complexity_prompt']= dft['vcs_list'].apply(lambda x: x[0])
dft['vocabulary_complexity_response_a']= dft['vcs_list'].apply(lambda x: x[1])
dft['vocabulary_complexity_response_b']= dft['vcs_list'].apply(lambda x: x[2])
del(dft['vcs_list'])


import pickle

path = '/kaggle/input/newera/newera_dataset'

try:
    with open(path, 'rb') as f:
        df = pickle.load(f)
    print("Pickle file loaded successfully!")
except Exception as e:
    print(f"Error loading pickle file: {e}")


df.head()


columns_to_transform = ['code_in_prompt','code_in_response_a','code_in_response_b']
df[columns_to_transform] = df[columns_to_transform].astype(int)


columns_to_transform = ['code_in_prompt','code_in_response_a','code_in_response_b']
dft[columns_to_transform] = dft[columns_to_transform].astype(int)


df[["web_dev_prompt", "dev_prompt", "comprehensive_prompt"]] = pd.DataFrame(df["code_languages_in_prompt"].tolist(), index=df.index)
df[["web_dev_response_a", "dev_response_a", "comprehensive_response_a"]]= pd.DataFrame(df["code_languages_in_response_a"].tolist(), index=df.index)
df[["web_dev_response_b", "dev_response_b", "comprehensive_response_b"]] = pd.DataFrame(df["code_languages_in_response_b"].tolist(), index=df.index)


dft[["web_dev_prompt", "dev_prompt", "comprehensive_prompt"]] = pd.DataFrame(dft["code_languages_in_prompt"].tolist(), index=dft.index)
dft[["web_dev_response_a", "dev_response_a", "comprehensive_response_a"]]= pd.DataFrame(dft["code_languages_in_response_a"].tolist(), index=dft.index)
dft[["web_dev_response_b", "dev_response_b", "comprehensive_response_b"]] = pd.DataFrame(dft["code_languages_in_response_b"].tolist(), index=dft.index)


# label is winner column 

def target_conversion_to_categorical(model):
    if model == 'model_a':
        return 0
    else:
        return 1
        
df['label'] = df['winner'].apply(target_conversion_to_categorical)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split


dt = df[(df['type']=='code') | (df['type']=='text')]


dtt = dft[(dft['type']=='code') | (dft['type']=='text')]


print(len(dt[dt['type']=='unknown']) , len(dt[dt['type']=='logical_operations']))
len(dtt[dtt['type']=='unknown']) , len(dtt[dtt['type']=='logical_operations'])


y = dt['label']
X = dt.drop(columns=["label"])


# Split data for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from matplotlib import pyplot as plt
import seaborn as sns
import eli5


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=150000)


X_train_text = vectorizer.fit_transform(X_train['eng_tr_prompt']+X_train['eng_tr_response_a']+X_train['eng_tr_response_b'])


X_test_text = vectorizer.transform(X_test['eng_tr_prompt']+X_test['eng_tr_response_a']+X_test['eng_tr_response_b'])


logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=17, n_jobs=4)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)



%%time
cv_results = cross_val_score(logit, X_train_text, y_train, cv=skf, scoring='f1_micro')


cv_results, cv_results.mean()





%%time
logit.fit(X_train_text, y_train)


test_preds = logit.predict(X_test_text)

