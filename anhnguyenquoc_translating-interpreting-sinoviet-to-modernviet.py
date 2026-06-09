import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datasets import load_dataset
import matplotlib.pyplot as plt
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
    



#Load raw full dataset
raw_data = load_dataset("GooglyEyeSuperman/HanScript2HanViet2ModernViet_Literature", 'full_raw')
full_df = pd.DataFrame(raw_data['train'])



def plot_count(array, title = '', stats = False):
    names, counts = np.unique(array, return_counts = True)
    if stats:
        for i in range(len(names)):
            print(names[i], ': ', counts[i])
    
    plt.xticks(rotation= 90)
    plt.bar(names, counts)
    plt.title(title)
    plt.show()

plot_count(full_df.language.tolist(), 'Language')
poem_by_lang = [f'{x}_{y}' for x, y in zip(full_df.language.tolist(), full_df.poem_type.tolist())]
plot_count(poem_by_lang, 'Poem_type by language')



hanviet_poems = [x for x in full_df['poem_content@HanVietText'] if not pd.isna(x)]
# Ensure all poems at least have hanviet_poem
assert len(hanviet_poems) == len(full_df)
modernviet_poems = [x for x in full_df['poem_content@modernVietText'] if not pd.isna(x)]
hanscript_poems = [x for x in full_df['poem_content@HanScript'] if not pd.isna(x)]
translation_posts = [x for x in full_df['post_content'] if not pd.isna(x)]

absolute_content_count = ['HanVietText' for x in range(len(hanviet_poems))] + \
    ['modernVietText' for x in range(len(modernviet_poems))] + \
    ['HanScript' for x in range(len(hanscript_poems))] + \
    ['user-uploaded-translation' for x in range(len(translation_posts))]
plot_count(absolute_content_count, 'Absolute content count', stats = True)



no_post_no_modernViet = full_df[pd.isna(full_df['poem_content@modernVietText']) & pd.isna(full_df['post_content'])]
no_post_yes_modernViet = full_df[ ~pd.isna(full_df['poem_content@modernVietText']) & pd.isna(full_df['post_content'])]
yes_post_no_modernViet = full_df[ pd.isna(full_df['poem_content@modernVietText']) & ~pd.isna(full_df['post_content'])]
yes_post_yes_modernViet = full_df[~pd.isna(full_df['poem_content@modernVietText']) & ~pd.isna(full_df['post_content'])]
y = np.array([len(no_post_no_modernViet), 
              len(no_post_yes_modernViet), 
              len(yes_post_no_modernViet), 
              len(yes_post_yes_modernViet)
              ])
mylabels = ["without user or official translation", 
            "with only official translation", 
            "with only user translation", 
            "has both user and official translation"]
plt.pie(y, labels = mylabels,  startangle = 90, autopct=lambda p: f'{int(p * sum(y) / 100)} ({p:.1f}%)'
)
# plt.legend()
plt.title('Distribution of poems with and without translation') 
plt.show() 


train_official = load_dataset("GooglyEyeSuperman/HanScript2HanViet2ModernViet_Literature", 'train_official')
train_ai = load_dataset("GooglyEyeSuperman/HanScript2HanViet2ModernViet_Literature", 'train_ai')
train_comment = load_dataset("GooglyEyeSuperman/HanScript2HanViet2ModernViet_Literature", 'train_comment')
test = load_dataset("GooglyEyeSuperman/HanScript2HanViet2ModernViet_Literature", 'test_official')



# Training Configurations
token_limit = 256
lora_rank = 4 # or 4
lr_value = 1e-4
train_epoch = 3 	# up to
weight_decay=0.01
model_id = "gemma2_instruct_2b_en"


#Import
import os
import keras
import keras_nlp

import time
# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"



tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)



# Prompt generation
import tqdm
import random
def generate_random_list_summing_to_n(n):
    if n <= 0 or type(n) == float:
        raise ValueError("A must be a positive integer.")

    result = []
    while n > 0:
        # Generate a random integer between 1 and A
        rand_int = random.randint(1, n)
        result.append(rand_int)
        n -= rand_int

    return result

def split_list_by_partition(partition, target_list):
    '''
    partition: list of integers, summing up to len(target_list)
    target_list: list of elements
    '''
    new_list = []
    cum_partition = [partition[i] + sum(partition[0:i]) for i in range(len(partition))]

    for i in range(len(cum_partition)):
        if i == 0:
            new_list.append(target_list[0:cum_partition[i]])
        else:
            new_list.append(target_list[cum_partition[i-1]:cum_partition[i]])

    assert [len(x) for x in new_list] == partition

    return new_list

def join_couple_str_randomly(str_listA, str_listB, join_choices = [' ', '\n']):
    '''
    str_list: list of strings
    '''
    if len(str_listA) == 1:
        return str_listA[0].strip(), str_listB[0].strip()

    resultA = ""
    resultB = ""
    for i in range(len(str_listA)):
        resultA += str_listA[i].strip()
        resultB += str_listB[i].strip()

        if i < len(str_listA) - 1:  # Avoid appending at the end
            chosen_connector = random.choice(join_choices)
            resultA += chosen_connector
            resultB += chosen_connector

    return resultA, resultB

def preproc_str(string):
    if not string:
        return string

    if string[0] == '"' and string[-1] == '"':
        string = string[1:-1]

    return string


# print('Testing split generation')
# n = 8
# a = generate_random_list_summing_to_n(n)
# print(n, a)

# print('Testing split')
# b = [x for x in range(n)]
# print(split_list_by_partition(a, b))

# print('Testing join random')
# linesA = [
#     "Nháº­t nguyá»‡t xuáº¥t nham Ä‘áº§u,",
#     "NhÃ¢n nhÃ¢n táº­n tháº¥t chÃ¢u.",
#     "PhÃº nhÃ¢n há»¯u cÃ¢u tá»­,",
#     "Bá»™ hÃ nh báº¥t ká»µ cÃ¢u?"
# ]

# linesB = [
#     "Máº·t trá»�i rá»“i máº·t trÄƒng káº¿ nhau má»�c á»Ÿ Ä‘áº§u nÃºi, ",
#     "[TrÃªn cÃµi Ä‘á»�i nÃ y], ngÆ°á»�i nÃ o ngÆ°á»�i áº¥y Ä‘á»�u Ä‘Ã¡nh máº¥t háº¡t ngá»�c cá»§a mÃ¬nh.",
#     "NhÆ° anh nhÃ  giÃ u cÃ³ con ngá»±a quÃ½, ",
#     "Láº¡i Ä‘i bá»™ mÃ  khÃ´ng cÆ°á»¡i ngá»±a"
# ]

import pandas as pd


class DataGenerator():
    def __init__(self,
                # csv_paths,
                df,
                source_col = 'poem_content@HanVietText',
                target_col = 'poem_content@modernVietText',
                tokenizer = tokenizer
                 ):
        # Read and concat all csv_paths into 1 df
        # df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
        self.df = df

        self.tokenizer = tokenizer
        self.source_col = source_col
        self.target_col = target_col


    def split(self, source, target, token_limit = 256):
        '''
        Split both source and target (poems) into sub-sentences by linebreaks, joined again by linebreaks or punctuation
        The newly joined sentence would be joined of a random number of sub-sentences, such that it doesn't overshoot the token_limit
        '''


        # Split into sentences
        source_sents = source.split('\n')
        target_sents = target.split('\n')


        # Check if the number of sentences is the same
        if len(source_sents) != len(target_sents):
            return None, None


        # Try splitting the original poem

        # Trial and error to find a good split that creates no more than half the token_limit
        trial = 0
        max_trial = 10
        split_success = False
        while not split_success:
            split_cnt = generate_random_list_summing_to_n(len(target_sents))
            # print(split_cnt, len(source_sents))
            split_list = split_list_by_partition(split_cnt, target_sents) #Target is usually longer
            split_list = [''.join(el) for el in split_list]

            len_split_list = [len(self.tokenizer(sent)) for sent in split_list]
            # (tokenizer took too much time, consider each 3 character is 1 token)
            # print(split_list)
            # len_split_list = [len(sent)/3 for sent in split_list]
            len_split_list = [x <= token_limit//2-20 for x in len_split_list]

            if False not in len_split_list:
                split_success = True
            else:
                trial += 1
                if trial >= max_trial:
                    split_cnt = [1 for x in range(len(source_sents))]
                    split_success = True

        # Split by split_cnt and join randomly
        split_source = split_list_by_partition(split_cnt, source_sents)
        split_target = split_list_by_partition(split_cnt, target_sents)

        join_source, join_target = [], []
        for s_el, t_el in zip(split_source, split_target):
            s_joined, t_joined = join_couple_str_randomly(str_listA=s_el, str_listB=t_el)
            join_source.append(s_joined)
            join_target.append(t_joined)

        return join_source, join_target

    def split_into_ones(self, source, target, token_limit):
        # Split into sentences
        source_sents = source.split('\n')
        target_sents = target.split('\n')


        # Check if the number of sentences is the same
        if len(source_sents) != len(target_sents):
            return None, None

        source_sents = [x.strip() for x in source_sents if x != '']
        target_sents = [x.strip() for x in target_sents if x != '']

        return source_sents, target_sents

    def generate(self,instruction, split_type, token_limit = 256):
        processed_data = []

        for i in tqdm.tqdm(range(len(self.df))):

            cur_series = self.df.iloc[i]
            # print(cur_series)
            # raise KeyboardInterrupt
            source = cur_series[self.source_col]
            target = cur_series[self.target_col]
            if pd.isna(source) or pd.isna(target):
              continue

            if split_type == 'mixed':
              sources, targets = self.split(source, target, token_limit)
            elif split_type == 'single':
              sources, targets = self.split_into_ones(source, target, token_limit)
            else:
              raise ValueError(f'{split_type} not implemented')

            if source is None or target is None or sources is None or targets is None:
                continue
            
            for source_sent, target_sent in zip(sources, targets):
                item = f"<start_of_turn>user\n{instruction}{source_sent}<end_of_turn>\n<start_of_turn>model\n{target_sent}<end_of_turn>"

                length = len(tokenizer(item))
                # skip data if the token length is longer than our limit
                if length < token_limit:
                    processed_data.append(item)


        return processed_data



df_test = pd.DataFrame(test['train'])
datagenerator = DataGenerator(df_test)
instruction = 'HÃ£y giáº£i thÃ­ch Ä‘oáº¡n vÄƒn báº£n tiáº¿ng HÃ¡n Viá»‡t nÃ y sang tiáº¿ng Viá»‡t hiá»‡n Ä‘áº¡i:\n'
# instruction = ""

train = datagenerator.generate(instruction = instruction, split_type = 'single', token_limit=token_limit)
print(f'Generated {len(train)} samples')


import random
print('Sample of a "single" pair: Only having 1 sentence \n-----')
print(train[random.randint(0, len(train))])


train = datagenerator.generate(instruction = instruction, split_type = 'mixed', token_limit=token_limit)
print(f'Generated {len(train)} samples')


print('Sample of a "joint" pair: Having 1+ sentences concat together \n-----')
print('Note that since the joining process is random, the result of this cell might be singular.\n You can keep rerun this, should be able to find many sample with multiple sents \n-----')
print(train[random.randint(0, len(train))])


#Eval
import pandas as pd
import numpy as np

# Input dictionary
data = {
    'pred_rank4_run_ai_20250106_075927_E2.lora.csv':            {'metric': np.float64(0.6220494658325040),'epoch': 2, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'gemini', 'unmatched': 1},
    'pred_rank4_run_ai_20250106_110846_E3.lora.csv':            {'metric': np.float64(0.6360887833961599),'epoch': 3, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'gemini', 'unmatched': 1},
    'pred_rank4_run_ai_20250106_121635_E6.lora.csv':            {'metric': np.float64(0.5989576871882079),'epoch': 6, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'gemini', 'unmatched': 2},
    'pred_rank4_run_official_20250108_124815.lora.csv':         {'metric': np.float64(0.5872032625264716),'epoch': 2, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'official', 'unmatched': 2},
    'pred_rank4_run_official_20250108_131638.lora.csv':         {'metric': np.float64(0.5488609985840792),'epoch': 3, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'official', 'unmatched': 3},
    'pred_rank4_run_official+ai_20250108_150250.lora.csv':      {'metric': np.float64(0.5449349726923254),'epoch': 1, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'gemini+official', 'unmatched': 3},
    'pred_rank8_run_official_20250111_112148.lora.csv':         {'metric': np.float64(0.5119647664962061),'epoch': 1, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'official', 'unmatched': 4},
    'pred_rank8_run_official_20250111_115126.lora.csv':         {'metric': np.float64(0.5103413504024817),'epoch': 2, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'official', 'unmatched': 4},
    'pred_rank8_run_official_20250111_122026.lora.csv':         {'metric': np.float64(0.5432960879250189),'epoch': 3, 'train_config': 'joint', 'test_config': 'joint', 'train_mix': 'official', 'unmatched': 3},
    'pred_rank8_run_ai_single_20250112_074328.lora_seg1.csv':   {'metric': np.float64(0.6152980717513602),'epoch': 1, 'train_config': 'single', 'test_config': 'single', 'train_mix': 'gemini', 'unmatched': 1},
    'pred_rank8_run_ai_single_20250112_074328.lora_seg3.csv':   {'metric': np.float64(0.0000000000000000),'epoch': 1, 'train_config': 'single', 'test_config': 'joint', 'train_mix': 'gemini', 'unmatched': 20},
    'pred_rank8_run_gpt4_single_20250113_110637.lora_seg1.csv': {'metric': np.float64(0.6293031537048365),'epoch': 1, 'train_config': 'single', 'test_config': 'single', 'train_mix': 'gpt4o', 'unmatched': 1},
    'pred_rank8_run_gpt4_single_20250113_110637.lora_seg3.csv': {'metric': np.float64(0.0362990141809972),'epoch': 1, 'train_config': 'single', 'test_config': 'joint', 'train_mix': 'gpt4o', 'unmatched': 19},
    'pred_rank8_run_gpt4_single_20250113_124420.lora_seg1.csv': {'metric': np.float64(0.6324505382180076),'epoch': 2, 'train_config': 'single', 'test_config': 'single', 'train_mix': 'gpt4o', 'unmatched': 1},
    'pred_rank8_run_gpt4_single_20250113_124420.lora_seg3.csv': {'metric': np.float64(0.0342670245288717),'epoch': 2, 'train_config': 'single', 'test_config': 'joint', 'train_mix': 'gpt4o', 'unmatched': 19},
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(data, orient='index')

# Reset index to have file names as a column
df.reset_index(inplace=True)
df.columns = ['file_name', 'metric', 'epoch','train_config','test_config', 'train_mix','unmatched']

print('Evaluation')
# Display the DataFrame in Jupyter Notebook
df



dictionary = load_dataset("GooglyEyeSuperman/HanScript2HanViet2ModernViet_Literature", 'dictionary')
vocab = pd.DataFrame(dictionary['train'])


print('Some sample translation:')
print('----')
print(vocab[vocab['HanViet'] ==  'bá»‰'])
print('----')
print(vocab[vocab['HanViet'] ==  'á»•i'])
print('----')
print('Searching for bá»‰ á»•i: ')
print(vocab[vocab['HanViet'] ==  'bá»‰ á»•i'])


import re
import string
import pandas as pd

def remove_punctuation(input_string):
    """
    Removes all punctuation from the given string.

    Args:
        input_string (str): The input string.

    Returns:
        str: The string with all punctuation removed.
    """
    # Create a translation table to replace punctuation with None
    translator = str.maketrans('', '', string.punctuation)

    # Use the translation table to remove punctuation
    cleaned_string = input_string.translate(translator)

    cleaned_string = cleaned_string.replace('â€œ', '').replace('â€�', '')
    return cleaned_string

def split_by_punctuation(text):


  pattern =   r'([^.,!?;:\-\n]+|[.,!?;:\-\n])'

  # r'[.,!?;:\-\n]'

  # Split the text
  split_text = re.split(pattern, text)

  # Remove any empty strings caused by consecutive matches or leading/trailing punctuation
  split_text = [segment.strip() for segment in split_text if segment.strip()]

  # Merge the punctuation to the previous string
  i = 0
  while i < len(split_text):
    if i != 0 and len(split_text[i]) == 1:
      split_text[i-1] = split_text[i-1] + split_text[i]
      del split_text[i]
    else:
      i += 1

  return split_text

def flatten(xss):
    return [x for xs in xss for x in xs]

def extract_ngrams(text, n=2):
    text = remove_punctuation(text)
    words = [x.strip() for x in text.split()]
    ngrams =  [' '.join(words[i:i+n]).lower() for i in range(len(words) - n + 1)]
    if n> len(words):
        return [text.lower().strip()]
    else:
        return ngrams


def extract_poem_ngrams(poem, dictionary):
    poem_lines = [x for x in poem.strip().split('\n') if x.strip() != '']
    poem_lines = flatten([split_by_punctuation(x) for x in poem_lines])
    result_ngrams = []
    not_found = []
    for poem_line in poem_lines:
        poem_line = remove_punctuation(poem_line)
        cur_result_ngrams = []
        for i in range(4, 0, -1):
            ngrams = extract_ngrams(poem_line, i)
            for ngram in ngrams:
                if ngram in dictionary.keys():
                    cur_result_ngrams.append(ngram)
                else:
                    if i == 1:
                        not_found.append(ngram)

        result_ngrams.extend(cur_result_ngrams)

    return result_ngrams, not_found

def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

INSTRUCTION = 'Báº¡n lÃ  nhÃ  vÄƒn/ há»�c giáº£/ nhÃ  ngÃ´n ngá»¯ há»�c tiáº¿ng Viá»‡t. Dá»±a vÃ o tá»« Ä‘iá»ƒn dá»‹ch tá»« HÃ¡n Viá»‡t sang tiáº¿ng Viá»‡t hiá»‡n Ä‘áº¡i nhÆ° trÃªn (lÆ°u Ã½ lÃ  tá»« Ä‘iá»ƒn cÃ³ thá»ƒ thiá»ƒu hay sai sÃ³t), hÃ£y giáº£i thÃ­ch Ä‘oáº¡n vÄƒn báº£n tiáº¿ng HÃ¡n Viá»‡t nÃ y sang tiáº¿ng Viá»‡t hiá»‡n Ä‘áº¡i:'

class RAGPrompter:
    def __init__(self, vocab_df):
        self.vocab_df = vocab_df #pd.read_csv(vocab_path)
        self.init_vocab()

    def init_vocab(self):
        self.vocab_dict = {}
        for i in range(len(self.vocab_df)):
            mv = self.vocab_df.iloc[i]['modernViet']
            hv = self.vocab_df.iloc[i]['HanViet']

            if hv not in self.vocab_dict:
                self.vocab_dict[hv] = mv
            else:
                self.vocab_dict[hv] += '/' + mv

    def retrieve_relevant_vocab(self, input_txt):
        result_ngrams, _ = extract_poem_ngrams(input_txt, self.vocab_dict)
        result_ngrams = remove_duplicates_preserve_order(result_ngrams)
        output_str = 'Tá»« Ä‘iá»ƒn HÃ¡n Viá»‡t-Viá»‡t:\n'
        for ngram in result_ngrams:
            output_str = output_str + ngram + ': ' + self.vocab_dict[ngram] + '\n'

        return output_str

    def generate_prompt(self, input_txt, instruction = INSTRUCTION):
        retrieved_data = self.retrieve_relevant_vocab(input_txt)
        prompt = f'{retrieved_data}\n{instruction}\n{input_txt}'
        return prompt




prompter = RAGPrompter(vocab)
prompt = prompter.generate_prompt('Cáº£nh bá»©c tÃ¢y sÆ¡n má»™')
print(prompt)


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)



import re

def extract_output(string):
    if not string:
      return ''
    output = string.split('<start_of_turn>model')[-1].split('<end_of_turn>')[0]
    if not output:
        return ''
    return output.strip()
    
def infer_list(model, poem_list, instruction, num_seg_per_list = 3):
    new_list = []
    for poem in tqdm.tqdm(poem_list):

      if pd.isna(poem):
        new_list.append('')
        continue

      #Split
      individual_poem_sents = [x.strip() for x in poem.split('\n') if x.strip() != '']
      poem_sents = []
      cur_poem_sents = []

      if num_seg_per_list == 1:
        poem_sents = individual_poem_sents
      else:
        for i in range(len(individual_poem_sents)):
          if i % num_seg_per_list != num_seg_per_list - 1:
            cur_poem_sents.append(individual_poem_sents[i])
          else:
            cur_poem_sents.append(individual_poem_sents[i])
            poem_sents.append('\n'.join(cur_poem_sents))
            cur_poem_sents = []
        else:
          poem_sents.append('\n'.join(cur_poem_sents))

      new_poem_sents = []
      for poem_sent in poem_sents:
        prompt = f"{instruction}{poem_sent}"
        print(prompt)
        translated = text_gen(model, prompt, 256)
        translated = extract_output(translated)
        new_poem_sents.append(translated)

      new_poem = '\n'.join(new_poem_sents)
      new_list.append(new_poem)

    return new_list

def text_gen(model, prompt, token_limit = token_limit):
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = model.generate(input, max_length=token_limit)
    return output
    



sample_input = ' Kiá»�u thá»§ váº¡n lÃ½ thiÃªn. '
prompt = prompter.generate_prompt(sample_input)
print(text_gen(gemma_lm, prompt, 500))


sample_input = 'Tá»±a danh quá»‘c lÃ£o Ä‘á»‘i hiá»ƒn hoÃ ,'


#LORA
lora_rank = 8
lora_path= '/kaggle/input/gemma-2b_e3_lora/keras/default/3/anhnguyen_rank8_epoch1_rungpt4_single_20250113_124420.lora.h5'
# lora_path = '/kaggle/input/gemma-2b_e3_lora/keras/default/2/anhnguyen_rank4_epoch3_runai_20250106_110846.lora.h5'
inference_mode = 'single'
if inference_mode == 'joint':
    num_segment_per_inference = 3
elif inference_mode == 'single':
    num_segment_per_inference = 1
trained_instruction = 'HÃ£y giáº£i thÃ­ch Ä‘oáº¡n vÄƒn báº£n tiáº¿ng HÃ¡n Viá»‡t nÃ y sang tiáº¿ng Viá»‡t hiá»‡n Ä‘áº¡i:\n'



new_translation = infer_list(gemma_lm, instruction =trained_instruction,  poem_list=[sample_input], num_seg_per_list=num_segment_per_inference)



print('Pre finetuning:')
print('----')
print(new_translation[0])


gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.backbone.load_lora_weights(lora_path)



new_translation = infer_list(model = gemma_lm, instruction = trained_instruction, poem_list=[sample_input], num_seg_per_list=1)



print('After finetuning:')
print('----')
print(new_translation[0])


prompt = prompter.generate_prompt(sample_input, instruction = trained_instruction)
print(prompt)
print(text_gen(gemma_lm, prompt, 500))




