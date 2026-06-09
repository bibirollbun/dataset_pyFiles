# Let's print out the GPU info to make sure we're using the right GPU

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


import os
from google.colab import userdata

# Note: `userdata.get` is a Colab API. If you're not using Colab, set the env
# vars as appropriate for your system.

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
# !pip install -q -U keras-nlp
!pip install --upgrade keras
!pip install --upgrade keras-hub


# For this tutorial, configure the backend for JAX.
os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

# Import Packages
import keras
import keras_hub


!git clone https://github.com/Peng-Peng/highquality-chinese-couplet-dataset.git


import os
import json
import glob

files = glob.glob('./**/dataset/*.json', recursive=True)
assert (len(files))== 90

data = []
for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:  # Handle potential encoding issues
            json_data = json.load(f)
            for line in json_data:
               data.append((line['0'],line['1']))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file}: {e}")
    except Exception as e:
        print(f"An error occurred while processing file {file}: {e}")

data.sort()
print(len(data))


#let's spot check a few data
for d in data[:5]:
    print(f"===\n{d}")


import random
# Randomly shuffle the sorted data with a seed for reproducibility
random.seed(42)  # You can change the seed value if you want a different shuffle
random.shuffle(data)
train_data = data[:2000]
test_data = data[2000:]


#check the distribution of dataset on # of characters per couplet

import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(data):
  # Calculate # of character in couplets, e.g. "å¤©"VSâ€œåœ°" is considered as 1-character couplets, a.k.a ä¸€å­—å¯¹
  # Calculate the character counts for each couplet
    char_counts = [len(couplet[0]) for couplet in data]

    # Define the bins for the histogram
    bins = range(1, 16)  # 1 to 15 characters
    plt.hist(char_counts, bins=bins, edgecolor='black')


    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.title('Distribution of Character Counts in Couplets')

    plt.show()

    # Print exact value per bucket
    for i in range(len(bins)-1):
      count = sum(1 for x in char_counts if bins[i] <= x < bins[i+1])
      print(f"{bins[i]} charcters couplet: {count}")

plot_distribution(data)


plot_distribution(train_data)


plot_distribution(test_data)


import json
import requests

def get_json_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON data: {e}")
        return None


# Using an online dataset, let's create a dictionary that can help us look up a charcter's tune.
github_url = "https://raw.githubusercontent.com/charlesix59/chinese_word_rhyme/main/data/Word_Tune.json"
CHAR_TO_TUNE = get_json_from_github(github_url)

if CHAR_TO_TUNE:
    print(len(CHAR_TO_TUNE.keys()))
    #print(CHAR_TO_TUNE)
else:
    print("Failed to retrieve or parse JSON from GitHub.")


import pandas as pd
import re

def sentence_to_tune_pattern(sentence):
  if not sentence or not isinstance(sentence, str):
    return ''
  tune_pattern =  [CHAR_TO_TUNE[c] if c in CHAR_TO_TUNE else 'X' for c in sentence if c not in ['ï¼Œ',' ']]
  return ''.join(tune_pattern)

#unit test
#print(sentence_to_tune_pattern('ç�‰æ–�', char_to_tune))
assert (sentence_to_tune_pattern('ç§‹é›¨æ½‡æ½‡ï¼Œæ¼«çƒ‚é»„èŠ±éƒ½æ»¡å¾„'))== "å¹³ä»„å¹³å¹³å¤šä»„å¹³å¤šå¹³ä»„ä»„"


def eval_couplet_basic_rules(first_line, second_line):
  if not first_line or not second_line:
    return 0
  if not isinstance(first_line, str) or not isinstance(second_line, str):
    return 0
  if len(first_line) != len(second_line):
    return 0

  score = 0.0
  assert len(first_line) == len(second_line), f"couplet sentences' lengths are different {first_line} {second_line}"
  len_first_line = len(first_line)
  for i in range(len_first_line):
    char_a = first_line[i]
    char_b = second_line[i]
    if (char_a not in CHAR_TO_TUNE) or (char_b not in CHAR_TO_TUNE):
      continue
    char_a_tune = CHAR_TO_TUNE[char_a]
    char_b_tune = CHAR_TO_TUNE[char_b]
    if char_a_tune == 'å¤š' or char_b_tune == 'å¤š': # the tune of the character could be either ping or ze
      score += 1.0
    elif char_a_tune != char_b_tune:
      score += 1.0
    #print(f"{char_a} {char_a_tune} {char_b} {char_b_tune} {score}")

  assert score <= len_first_line, f"total score should not be more than # of charcters {first_line} {second_line} {score} {len_first_line} "
  score /= len_first_line
  return score

#unit tests
assert eval_couplet_basic_rules("ä»°é«˜çº¢æ—¥è¿‘","æœ›è¿œç™½äº‘å­¤") == 1.0, "matched # of chars, perfectly reversed tune pattern, score should be 1.0"
assert eval_couplet_basic_rules("ä»°é«˜çº¢æ—¥è¿‘","æœ›è¿œç™½å­¤")== 0.0,"unmatched # of chars, score should be 0.0"
assert eval_couplet_basic_rules("ä»°é«˜çƒˆæ—¥è¿‘","æœ›è¿œç™½äº‘å­¤") == 0.8, "matched # of chars, 1 out of tune, showing the proportion of chars' whose tunes are perfectly reversed."



def eval_model_basic_rules(model_name="gemma2_lora", read_prev_eval_result=False):
    if read_prev_eval_result:
      test_result_df = pd.read_csv(f"{model_name}_basic_rules_eval_result.csv")
    else:
      test_result_df = pd.read_csv(f"{model_name}_test_output.csv")

      test_result_df['first_line_tune_pattern']= test_result_df.apply(lambda row: sentence_to_tune_pattern(row['input']), axis=1)
      test_result_df['second_line_tune_pattern']= test_result_df.apply(lambda row: sentence_to_tune_pattern(row['output']), axis=1)
      test_result_df['basic_rules_score'] = test_result_df.apply(lambda row: eval_couplet_basic_rules(row['input'], row['output']), axis=1)
      test_result_df.to_csv(f"{model_name}_basic_rules_eval_result.csv", index=False)

    num_of_zeros = (test_result_df['basic_rules_score'] == 0).sum()
    avg_tune_pattern_score = test_result_df[test_result_df['basic_rules_score'] != 0]['basic_rules_score'].mean()
    ratio_matched_examples = (len(test_result_df) - num_of_zeros) / len(test_result_df)
    final_score = test_result_df['basic_rules_score'].mean()

    print(f"=== Basic Rules Eval Result for {model_name} ===\n")
    print(f"Number of examples: {len(test_result_df)}")
    print(f"Number of invalid couplets (a.k.a where # of charcters of first line and second line are not matched): {num_of_zeros}")
    print(f"Ratio of valid couplets (range 0-1): {ratio_matched_examples:.3f}")
    print(f"Average tune pattern score for valid couplets (range 0-1): {avg_tune_pattern_score:.3f}")
    print(f"\nBasic Rules Eval Final Score (range 0-1): {final_score:.3f}")
    #return final_score

# Test run
# eval_model_basic_rules(model_name="gemma2_few_shot")#, read_prev_eval_result=True)


import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], # Your open ai API key
)

def clean_up_json_string(response_string):
    # Escape newlines in the response string
    cleaned_string = response_string.replace("\n", "").replace("    ","")
    return cleaned_string

def eval_couplet_pairwise_selection(input, output, label):
    if not input or not output or not label:
        return None
    if isinstance(input, str) and isinstance(output, str) and isinstance(label, str):
        pass
    else:
        return None
    if len(input) != len(output):
        return None

    eval_prompt = f"""
    ä½ çš„ä»»åŠ¡æ˜¯æŒ‘é€‰æ›´å¥½çš„ä¸‹å�¥ï¼Œè¯·ä»¥Valid Jsonçš„æ–¹å¼�å›�å¤�ã€‚
    ç�°åœ¨æˆ‘ä¼šç»™ä½ ä¸€ä¸ªä¸Šå�¥ï¼Œå’Œä¸¤ä¸ªä¸‹å�¥çš„å¤‡é€‰ï¼Œè¯·æ ¹æ�®ä»¥ä¸‹æ ‡å‡†é€‰å‡ºæ›´å¥½çš„ä¸‹å�¥å¹¶å‘Šè¯‰æˆ‘ä¸ºä»€ä¹ˆ,

    é€‰æ‹©æ ‡å‡†ï¼š
    1. ç»“æ�„å’Œè¯�æ€§ç›¸å�Œï¼šä¸Šå�¥å’Œä¸‹å�¥åœ¨è¯�æ€§å’Œç»“æ�„ä¸Šå¿…é¡»ä¿�æŒ�ä¸€è‡´ã€‚ä¾‹å¦‚ï¼Œè‹¥ä¸Šå�¥æ˜¯â€œå��è¯�+åŠ¨è¯�+å��è¯�â€�ï¼Œåˆ™ä¸‹å�¥ä¹Ÿåº”ä¸ºâ€œå��è¯�+åŠ¨è¯�+å��è¯�â€�ã€‚
    2. å†…å®¹ç›¸å…³æˆ–ç›¸å��ï¼šä¸Šå�¥å’Œä¸‹å�¥çš„å†…å®¹è¦�ä¹ˆæ˜¯ç›¸ä¼¼çš„æ��å†™ï¼ˆç›¸å…³ï¼‰ï¼Œè¦�ä¹ˆæ˜¯å½¢æˆ�é²œæ˜�å¯¹æ¯”çš„æƒ…å¢ƒï¼ˆç›¸å��ï¼‰ã€‚ç›¸å��æ¯”ç›¸å…³æ›´å¥½ï¼�
    3. æ•´ä½“æ–‡å­¦æ€§ï¼šä¸‹å�¥çš„æ–‡å­¦æ€§æ˜¯å¯¹ä¸Šå�¥å†…å®¹çš„å»¶ä¼¸ï¼Œæ„�å¢ƒæ˜¯å�¦ä¼˜ç¾�ã€�è¡¨è¾¾æ˜¯å�¦å…·æœ‰åˆ›æ–°æ€§ã€‚é™¤äº†å¯¹ä»—å¤–ï¼Œè¿˜éœ€è¦�è€ƒè™‘è¯­è¨€çš„ç²¾ç»ƒã€�æ„�è±¡çš„ç‹¬ç‰¹ä»¥å�Šæ•´ä½“çš„è‰ºæœ¯æ„Ÿ

    ====ä¾‹å­�===
    ä¸Šå�¥: è¿”å“ºæ…ˆä¹Œï¼Œå¤œæœˆæ��å¤´å•¼å“‘å“‘
    ä¸‹å�¥_1: å½’æ�¥æ‚²ç‡•ï¼Œæ˜¥é£�æŸ³æ¢¢å•¼å“‘å“‘
    ä¸‹å�¥_2: è¿�ä¹”å¥½é¸Ÿï¼Œæ˜¥é£�èŠ±åº•è¯­å…³å…³

    {{'é€‰æ‹©': 2,
      'è§£é‡Š': 'ç�†ç”±å¦‚ä¸‹ï¼š
        ç»“æ�„å’Œè¯�æ€§ç›¸å�Œï¼š
        ä¸Šå�¥: â€œè¿”å“ºæ…ˆä¹Œï¼Œå¤œæœˆæ��å¤´å•¼å“‘å“‘â€�æ˜¯ä¸€ä¸ªå¤�å�¥ç»“æ�„ï¼ŒåŒ…å�«äº†ä¸¤ä¸ªéƒ¨åˆ†ï¼šç¬¬ä¸€éƒ¨åˆ†â€œè¿”å“ºæ…ˆä¹Œâ€�æ˜¯â€œåŠ¨è¯�+å��è¯�â€�çš„ç»“æ�„ï¼›ç¬¬äºŒéƒ¨åˆ†â€œå¤œæœˆæ��å¤´å•¼å“‘å“‘â€�æ˜¯â€œå��è¯�+å��è¯�+åŠ¨è¯�+å½¢å®¹è¯�â€�çš„ç»“æ�„ã€‚
        ä¸‹å�¥_2: â€œè¿�ä¹”å¥½é¸Ÿï¼Œæ˜¥é£�èŠ±åº•è¯­å…³å…³â€�ä¹Ÿä½¿ç”¨äº†å¤�å�¥ç»“æ�„ï¼Œç¬¬ä¸€éƒ¨åˆ†â€œè¿�ä¹”å¥½é¸Ÿâ€�æ˜¯â€œåŠ¨è¯�+å��è¯�â€�çš„ç»“æ�„ï¼›ç¬¬äºŒéƒ¨åˆ†â€œæ˜¥é£�èŠ±åº•è¯­å…³å…³â€�æ˜¯â€œå��è¯�+å��è¯�+åŠ¨è¯�+å½¢å®¹è¯�â€�çš„ç»“æ�„ã€‚
        ä¸¤è€…åœ¨ç»“æ�„å’Œè¯�æ€§ä¸Šå®Œå…¨ä¸€è‡´ã€‚

        å†…å®¹ç›¸å…³æˆ–ç›¸å��ï¼š
        ä¸Šå�¥æ��å†™çš„æ˜¯â€œæ…ˆä¹Œâ€�å›�å½’æ¯�äº²å–‚é£Ÿï¼Œå¹¶åœ¨å¤œæœˆä¸‹å•¼é¸£çš„æƒ…æ™¯ï¼Œæ„�å¢ƒæ¸©æƒ…è€Œå®‰é�™ã€‚
        ä¸‹å�¥_1: â€œå½’æ�¥æ‚²ç‡•ï¼Œæ˜¥é£�æŸ³æ¢¢å•¼å“‘å“‘â€�å»¶ç»­äº†é¸Ÿç±»å½’æ�¥çš„æƒ…èŠ‚ï¼Œä½†ä»�â€œæ‚²ç‡•â€�æ�¥çœ‹ï¼Œè¯­æ°”å��å�‘å“€ä¼¤ï¼Œè€Œä¸Šå�¥çš„â€œè¿”å“ºæ…ˆä¹Œâ€�æ˜¯æ¸©é¦¨çš„ï¼Œæƒ…æ„Ÿä¸Šæœ‰ä¸€å®šä¸�å��è°ƒï¼Œä¸”â€œå“‘å“‘â€�ä¸�â€œå•¼å“‘å“‘â€�é‡�å¤�ï¼Œä¸�å¤Ÿç²¾ç‚¼ã€‚
        ä¸‹å�¥_2: â€œè¿�ä¹”å¥½é¸Ÿï¼Œæ˜¥é£�èŠ±åº•è¯­å…³å…³â€�åˆ™ä¸�ä¸Šå�¥å½¢æˆ�é²œæ˜�çš„å¯¹æ¯”ã€‚ä¸Šå�¥çš„ä¹Œé¸Ÿæ˜¯å½’æ�¥å¹¶å•¼å�«ï¼Œè€Œä¸‹å�¥çš„â€œè¿�ä¹”â€�åˆ™æ˜¯é¸Ÿå„¿è¿�å¾™åˆ°æ›´é«˜çš„åœ°æ–¹ï¼Œå½¢æˆ�äº†å¯¹æ¯”ã€‚å¹¶ä¸”â€œæ˜¥é£�èŠ±åº•è¯­å…³å…³â€�ç»™äººä¸€ç§�æ¸©æŸ”ã€�å’Œè°�çš„ç”»é�¢ï¼Œä¸�ä¸Šå�¥çš„â€œå¤œæœˆâ€�å½¢æˆ�äº†å¯¹æ¯”ï¼Œå¤œä¸�æ˜¥ï¼Œé�™ä¸�åŠ¨ï¼Œå½¢æˆ�äº†æœ‰è¶£çš„å��å·®ã€‚

        æ•´ä½“æ–‡å­¦æ€§ï¼š
        ä¸‹å�¥_2çš„â€œè¿�ä¹”å¥½é¸Ÿï¼Œæ˜¥é£�èŠ±åº•è¯­å…³å…³â€�åœ¨æ„�å¢ƒä¸Šå»¶ä¼¸äº†ä¸Šå�¥çš„ç¾�å­¦ï¼Œæ—¢ä¿�æŒ�äº†é¸Ÿç±»çš„ä¸»é¢˜ï¼Œå�ˆå¼•å…¥äº†æ˜¥é£�å’ŒèŠ±åº•çš„è‡ªç„¶æ™¯è±¡ï¼Œå……æ»¡äº†ç”Ÿæœºä¸�å’Œè°�ï¼Œä½“ç�°å‡ºè¾ƒé«˜çš„æ–‡å­¦æ€§å’Œè‰ºæœ¯æ€§ã€‚
        ä¸‹å�¥_1è™½ç„¶åœ¨ç»“æ�„ä¸Šæ²¡æœ‰é—®é¢˜ï¼Œä½†â€œæ‚²ç‡•â€�æ˜¾å¾—è¿‡äº�çª�å…€ï¼Œä¸”é‡�å¤�çš„â€œå“‘å“‘â€�å½±å“�äº†æµ�ç•…æ„Ÿï¼Œæ–‡å­¦æ€§è¾ƒå¼±ï¼Œè¡¨ç�°çš„æƒ…æ„Ÿä¹Ÿå��å�‘å�•ä¸€çš„å“€ä¼¤ã€‚'}}


    ä¸Šå�¥: {input}
    ä¸‹å�¥_1: {output}
    ä¸‹å�¥_2: {label}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-4 for better quality
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.0,  # Low temperature for deterministic output
        )
        response_msg = response.choices[0].message.content
        # DEBUG
        # print(response_msg)
        # print(clean_up_json_string(response_msg))
        response_json = json.loads(clean_up_json_string(response_msg))

        return response_json

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def evaluate_and_add_result_pairwise_selection(result_df):
    for index, row in tqdm(result_df.iterrows(), total=result_df.shape[0]):
        eval_json = eval_couplet_pairwise_selection(row['input'], row['output'],row['label'])
        #print(eval_json)

        if not eval_json:  # If the eval_json is empty or None
            result_df.loc[index, 'choice']  = None
            result_df.loc[index, 'reason'] = None
            continue

        # Assuming eval_json contains keys 'é€‰æ‹©' and 'è§£é‡Š'
        result_df.loc[index, 'choice']  = eval_json['é€‰æ‹©']
        result_df.loc[index, 'reason'] = eval_json['è§£é‡Š']

    return result_df

def eval_model_pairwise_selection(model_name="gemma2_lora", read_prev_eval_result=False):
    if read_prev_eval_result:
      eval_result_df = pd.read_csv(f"{model_name}_pairwise_selection_eval_result.csv")
    else:
      test_result_df = pd.read_csv(f"{model_name}_test_output.csv")
      eval_result_df = evaluate_and_add_result_pairwise_selection(test_result_df)
      eval_result_df.to_csv(f"{model_name}_pairwise_selection_eval_result.csv", index=False)


    valid_eval_count = len(eval_result_df[eval_result_df['choice'].notnull()])
    num_choice_one = len(eval_result_df[eval_result_df['choice'] == 1])
    ratio_choice_one = 0.0 if valid_eval_count ==0 else (num_choice_one / valid_eval_count)

    print(f"=== Pairwise Selection Eval Result for {model_name} w/o Human Adjustment: ===\n")
    print(f"Valid evaluation count: {valid_eval_count}")
    print(f"Number of {model_name}'s answer won: {num_choice_one}")
    print(f"\nPairwise Selection Final Score (range 0-1): {ratio_choice_one:.3f}")
    #return ratio_choice_one


# eval_couplet_pairwise_selection("æ°´æµ�åˆ†ç‡•å°¾","é£�å�¹æˆ�é›�ç¿…","å±±ç§€æ‹¥è�ºé¬Ÿ") #unit test a single example
# eval_model_pairwise_selection("gemma2_lora", read_prev_eval_result=True)



import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore')

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # This is the default and can be omitted
)


def eval_couplet_single_answer(first_line, second_line):
    if not first_line or not second_line:
        return None
    if isinstance(first_line, str) and isinstance(second_line, str):
        pass
    else:
        return None
    if len(first_line) != len(second_line):
        return None

    eval_prompt = f"""
    ä½ çš„ä»»åŠ¡æ˜¯ä¸€ä¸ªå¯¹ä»—çš„ä¸‹å�¥æ‰“åˆ†, è¯·ä»¥Jsonçš„æ–¹å¼�å›�å¤�ã€‚æˆ‘ä¼šç»™ä½ ä¸Šå�¥, é¢�å¤–ä¿¡æ�¯, å’Œä¸‹å�¥ã€‚
    è¯·ä½ æ ¹æ�®ç»™å®šçš„ä¸Šå�¥å’Œé¢�å¤–ä¿¡æ�¯ï¼Œå¯¹ä»¥ä¸‹çš„3ä¸ªç»´åº¦æ‰“1åˆ°5åˆ†ã€‚
    1. ç»“æ�„å’Œè¯�æ€§ç›¸å�Œï¼šä¸Šå�¥å’Œä¸‹å�¥åœ¨è¯�æ€§å’Œç»“æ�„ä¸Šå¿…é¡»ä¿�æŒ�ä¸€è‡´ã€‚ä¾‹å¦‚ï¼Œè‹¥ä¸Šå�¥æ˜¯â€œå��è¯�+åŠ¨è¯�+å��è¯�â€�ï¼Œåˆ™ä¸‹å�¥ä¹Ÿåº”ä¸ºâ€œå��è¯�+åŠ¨è¯�+å��è¯�â€�ã€‚
    2. å†…å®¹ç›¸å…³æˆ–ç›¸å��ï¼šä¸Šå�¥å’Œä¸‹å�¥çš„å†…å®¹è¦�ä¹ˆæ˜¯ç›¸ä¼¼çš„æ��å†™ï¼ˆç›¸å…³ï¼‰ï¼Œè¦�ä¹ˆæ˜¯å½¢æˆ�é²œæ˜�å¯¹æ¯”çš„æƒ…å¢ƒï¼ˆç›¸å��ï¼‰ã€‚ç›¸å��æ¯”ç›¸å…³æ›´å¥½ï¼�
    3. æ•´ä½“æ–‡å­¦æ€§ï¼šä¸‹å�¥çš„æ–‡å­¦æ€§æ˜¯å¯¹ä¸Šå�¥å†…å®¹çš„å»¶ä¼¸ï¼Œæ„�å¢ƒæ˜¯å�¦ä¼˜ç¾�ã€�è¡¨è¾¾æ˜¯å�¦å…·æœ‰åˆ›æ–°æ€§ã€‚é™¤äº†å¯¹ä»—å¤–ï¼Œè¿˜éœ€è¦�è€ƒè™‘è¯­è¨€çš„ç²¾ç»ƒã€�æ„�è±¡çš„ç‹¬ç‰¹ä»¥å�Šæ•´ä½“çš„è‰ºæœ¯æ„Ÿã€‚
    å…·ä½“è¯„åˆ†è§„åˆ™è§£é‡Šï¼š
    1åˆ†ï¼šå®Œå…¨ä¸�ç¬¦å�ˆæ ‡å‡†ã€‚
    2åˆ†ï¼šéƒ¨åˆ†ç¬¦å�ˆæ ‡å‡†ï¼Œä½†æœ‰æ˜�æ˜¾çš„ç¼ºé™·æˆ–å��å·®ã€‚
    3åˆ†ï¼šç¬¦å�ˆè¦�æ±‚ï¼Œä½†æ˜¯ä»�æœ‰è¾ƒé«˜çš„æ��å�‡ç©ºé—´ã€‚
    4åˆ†ï¼šå¤§éƒ¨åˆ†ç¬¦å�ˆè¦�æ±‚ï¼Œç»†èŠ‚ä¸Šå�¯ä»¥è¿›ä¸€æ­¥ä¼˜åŒ–ã€‚
    5åˆ†ï¼šå®Œå…¨ç¬¦å�ˆæ ‡å‡†ã€‚
    ====ä¾‹å­�===
    ä¸Šå�¥ï¼šä»°é«˜çº¢æ—¥è¿‘
    é¢�å¤–ä¿¡æ�¯ï¼š
    ä¸‹å�¥ï¼šæœ›è¿œç™½äº‘å­¤


     {{
      "ç»“æ�„å’Œè¯�æ€§ç›¸å�Œ": {{
        "è¯„åˆ†": 5,
        "è§£é‡Š": "ä¸Šå�¥ç»“æ�„ä¸ºâ€œåŠ¨è¯�+å½¢å®¹è¯�+å��è¯�â€�ï¼Œä¸‹å�¥ç»“æ�„ä¸ºâ€œåŠ¨è¯�+å½¢å®¹è¯�+å��è¯�â€�ï¼Œä¸¤è€…ç»“æ�„å®Œå…¨ç›¸å�Œï¼Œç¬¦å�ˆè¦�æ±‚ã€‚"
      }},
      "å†…å®¹ç›¸å…³": {{
        "è¯„åˆ†": 4,
        "è§£é‡Š": "ä¸Šå�¥æ��è¿°çš„æ˜¯ä¸€ä¸ªâ€œçº¢æ—¥â€�çš„æ�¥è¿‘ã€�ä»°æœ›çš„çŠ¶æ€�ï¼Œå…·æœ‰åŠ¨æ€�çš„æ„ŸçŸ¥ï¼›ä¸‹å�¥æ��è¿°çš„æ˜¯â€œè¿œâ€�ä¸�â€œå­¤â€�çš„é�™æ€�åœºæ™¯ï¼Œæ„Ÿè§‰æœ‰ä¸€äº›å¯¹ç«‹ï¼Œä½†ä¾�ç„¶æ˜¯æ��è¿°è‡ªç„¶æ™¯è§‚ã€‚â€œçº¢æ—¥â€�ä¸�â€œç™½äº‘â€�éƒ½å±�äº�è‡ªç„¶æ™¯ç‰©ï¼Œä¸”æœ‰ä¸€å®šçš„ç©ºé—´ç»´åº¦è�”ç³»ï¼ˆè¿œè¿‘ä¹‹æ„Ÿï¼‰ï¼Œå› æ­¤å†…å®¹æ˜¯ç›¸å…³çš„ï¼Œè™½ç„¶æ�„æˆ�äº†ä¸¤ç§�ä¸�å�Œçš„æ™¯è±¡ï¼Œä½†æ²¡æœ‰è¿�èƒŒå¸¸è¯†ã€‚"
      }},
      "æ•´ä½“æ–‡å­¦æ€§": {{
        "è¯„åˆ†": 4,
        "è§£é‡Š": "ä¸Šå�¥â€œä»°é«˜çº¢æ—¥è¿‘â€�å¯Œæœ‰è¯—æ„�ï¼Œå±•ç�°äº†ä¸€ç§�ä»°æœ›çš„æƒ…æ„Ÿå’Œè‡ªç„¶æ™¯è±¡ï¼Œæ��å…·ç”»é�¢æ„Ÿï¼Œä¸”â€œçº¢æ—¥â€�å¸¦æœ‰æ¸©æš–çš„è±¡å¾�ã€‚ä¸‹å�¥â€œæœ›è¿œç™½äº‘å­¤â€�ä¹Ÿå¯Œæœ‰è¯—æ„�ï¼Œç»™äººä¸€ç§�å­¤ç‹¬ã€�è¿œç¦»çš„æ„�è±¡ï¼Œæ�„æˆ�é²œæ˜�çš„å¯¹æ¯”ã€‚æ•´ä½“ä¸Šï¼Œè¯­è¨€ç®€æ´�ã€�æ„�è±¡é²œæ˜�ï¼Œå¯Œæœ‰ç¾�æ„Ÿå’Œå±‚æ¬¡æ„Ÿï¼Œå…·å¤‡ä¸€å®šçš„æ–‡å­¦æ€§ã€‚"
      }},
    }}



    ä¸Šå�¥: {first_line}
    é¢�å¤–ä¿¡æ�¯:
    ä¸‹å�¥: {second_line}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-4 for better quality
            messages=[{"role": "user", "content": eval_prompt}],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.0,  # Low temperature for deterministic output
        )
        #print(response.choices[0].message.content)
        response_json = json.loads(response.choices[0].message.content)

        return response_json

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def evaluate_and_add_result_single_answer(result_df):
    for index, row in tqdm(result_df.iterrows(),total=result_df.shape[0]):
        first_line = row[0]
        second_line = row[1]

        eval_json = eval_couplet_single_answer(first_line, second_line)
        #print(eval_json)

        if not eval_json:  # If the eval_json is empty or None
            result_df.loc[index, 'structure_score']  = None
            result_df.loc[index, 'structure_score_reason'] = None
            result_df.loc[index, 'content_score']  = None
            result_df.loc[index, 'content_score_reason']  = None
            result_df.loc[index, 'expression_score']  = None
            result_df.loc[index, 'expression_score_reason']  = None
            continue

        # Assuming eval_json contains these keys
        result_df.loc[index, 'structure_score']  = eval_json['ç»“æ�„å’Œè¯�æ€§ç›¸å�Œ']['è¯„åˆ†']
        result_df.loc[index, 'structure_score_reason'] = eval_json['ç»“æ�„å’Œè¯�æ€§ç›¸å�Œ']['è§£é‡Š']
        result_df.loc[index, 'content_score']  = eval_json['å†…å®¹ç›¸å…³']['è¯„åˆ†']
        result_df.loc[index, 'content_score_reason']  = eval_json['å†…å®¹ç›¸å…³']['è§£é‡Š']
        result_df.loc[index, 'expression_score']  = eval_json['æ•´ä½“æ–‡å­¦æ€§']['è¯„åˆ†']
        result_df.loc[index, 'expression_score_reason']  = eval_json['æ•´ä½“æ–‡å­¦æ€§']['è§£é‡Š']

    result_df['final_score'] = result_df['structure_score'] + result_df['content_score'] + result_df['expression_score']
    return result_df

# To evaluate a single pair (for testing):
# evaluation = eval_couplet(extract_first_line(test_result_df.iloc[0,0]), test_result_df.iloc[0,1])
# if evaluation:
#    print(evaluation)

def eval_model_single_answer(model_name="gemma2_lora",read_prev_eval_result=False):
    if read_prev_eval_result:
       eval_result_df =pd.read_csv(f"{model_name}_single_answer_eval_result.csv")
    else:
      test_result_df = pd.read_csv(f"{model_name}_test_output.csv")
      eval_result_df = evaluate_and_add_result_single_answer(test_result_df)
      eval_result_df.to_csv(f"{model_name}_single_answer_eval_result.csv", index=False)


    valid_eval_count = len(eval_result_df[eval_result_df['structure_score'].notnull()])
    avg_structure_score = eval_result_df['structure_score'].mean()
    avg_content_score = eval_result_df['content_score'].mean()
    avg_expression_score = eval_result_df['expression_score'].mean()
    avg_final_score = eval_result_df['final_score'].mean()/15 #15 is the max avg final score possible

    print(f"=== Single Answer Grading Eval Result for {model_name} w/o Human Adjustment: ===\n")
    print(f"Valid Evaluation Count: {valid_eval_count}")
    print(f"Average Structure Score (range 1-5): {avg_structure_score:.3f}")
    print(f"Average Content Score (range 1-5): {avg_content_score:.3f}")
    print(f"Average Expression Score (range 1-5): {avg_expression_score:.3f}")
    print(f"\nSingle Answer Grading Eval Final Score (range 0-1): {avg_final_score:.3f}")
    #return avg_final_score


#eval_model_single_answer(model_name="golden_label")#, read_prev_eval_result=True)



gemma2 = keras_hub.models.GemmaCausalLM.from_preset("gemma2_2b_en")
gemma2.summary()


# A zero shot template where we give the model first line, and ask it to generate the second line.
ZERO_SHOT_TEMPLATE = "ä¸Šå�¥:{first_line}\nä¸‹å�¥:{second_line}"
# From the data distribution, we know test data's max first line length is 13 Chinese characters.
# Based on the same # of characters rule, we know model's output ideally should not exceed 13 characters.
MAX_LENGTH = 48

# spot checking a few on test data before fine-tuning
def spot_check_model(model, template):

  for example in test_data[0:5]:
      test_prompt = template.format(first_line=example[0], second_line="")
      response = model.generate(test_prompt, max_length=MAX_LENGTH)
      output = response[len(test_prompt):]
      print(f"===\nprompt:{test_prompt!r}\nfirst_line:{example[0]!r}\nmodel_output:{output!r}\nreference_answer:{example[1]!r}")

spot_check_model(gemma2,ZERO_SHOT_TEMPLATE)


for example in train_data[0:20]:
    print(example)


# Grab a few training example as example in the prompt
# for example in train_data[0:20]:
#     print(example)


# One-shot prompting template
one_shot_template = """
ä¸Šå�¥:æ­Œæ—§æ›²
ä¸‹å�¥:é…¿æ–°é†…

ä¸Šå�¥:{first_line}
ä¸‹å�¥:"""

# Few-shot prompting template with short couplets only
few_shot_short_template = """
ä¸Šå�¥:æ™¦
ä¸‹å�¥:æ˜�

ä¸Šå�¥:ç‰§å­�
ä¸‹å�¥:æ¸”ç¿�

ä¸Šå�¥:æ­Œæ—§æ›²
ä¸‹å�¥:é…¿æ–°é†…

ä¸Šå�¥:{first_line}
ä¸‹å�¥:"""

# Few-shot prompting template with couplets of various lengths
few_shot_varied_template = """
ä¸Šå�¥:ç‰§å­�
ä¸‹å�¥:æ¸”ç¿�

ä¸Šå�¥:å²­å¤–äº‘éœ�èŠ±ä¸‹æœˆ
ä¸‹å�¥:æ¹–è¾¹çƒŸé›¨æŸ³æ¢¢æ™´

ä¸Šå�¥:è‰²è‰³åŒ—å ‚ï¼Œè�‰å�·å¿˜å¿§å¿§ç”šäº‹
ä¸‹å�¥:é¦™æµ“å�—å›½ï¼ŒèŠ±å��å�«ç¬‘ç¬‘ä½•äºº

ä¸Šå�¥:{first_line}
ä¸‹å�¥:"""


# Observe that now the model's output becomes much shorter, but still much longer than reference answer
spot_check_model(gemma2, one_shot_template)


# Observe that all model's output now has length closer to reference answer, despite some empty response
spot_check_model(gemma2, few_shot_short_template)


# Observe that all model's ouput becomes empty.
spot_check_model(gemma2, few_shot_varied_template)


# First, let's do a batch inference on test data and save model output to a csv file
import pandas as pd
from tqdm import tqdm


def inference_and_save_model_output(model, model_name, template, test_data=test_data):
  results = []

  for example in tqdm(test_data):
      test_prompt = template.format(first_line=example[0], second_line="")
      response = model.generate(test_prompt, max_length=MAX_LENGTH)
      output = response[len(test_prompt):]
      results.append([example[0], output, example[1],test_prompt])

  df = pd.DataFrame(results, columns=['input', 'output', 'label','test_prompt'])

  df.to_csv(f'{model_name}_test_output.csv', index=False)
  return df


inference_and_save_model_output(gemma2, model_name="gemma2_few_shot_short", template=few_shot_short_template)


eval_model_basic_rules("gemma2_few_shot_short")


eval_model_pairwise_selection("gemma2_few_shot_short")


eval_model_single_answer("gemma2_few_shot_short")


# Enable LoRA for the model and set the LoRA rank to 4.
gemma2.backbone.enable_lora(rank=4)
gemma2.summary()


# Uncomment this line if you want to enable mixed precision training on GPUs
keras.mixed_precision.set_global_policy('mixed_bfloat16')


# keras.config.set_floatx("float32") # to update inference precisiion


train_examples = [
    ZERO_SHOT_TEMPLATE.format(first_line=example[0], second_line=example[1])
    for example in train_data
]

# spot checking a few training examples
print(train_examples[0:5])


from dataclasses import dataclass

@dataclass
class FineTuneConfig:
  sequence_length: int = 256
  learning_rate: float = 5e-5
  weight_decay: float = 0.01
  epochs: int = 1
  batch_size: int = 1


import pandas as pd

def find_min_required_preprocessor_length(train_data, template):
    max_length = 0
    for couplet in train_data:
      input_lenth = 2 * len(couplet[0]) + len(template)
      max_length = max(max_length, input_lenth)

    # Add some buffer to account for special tokens, etc.  Adjust buffer as needed.
    suggested_max_length = max_length + 50
    print(f"preprocessor sequence_length should be larger than: {suggested_max_length}")
    return suggested_max_length

# Example usage (replace with your actual train_data)
# Assuming train_data is defined in your original code
best_max_length = find_min_required_preprocessor_length(train_data, ZERO_SHOT_TEMPLATE)


# MAX_LENGTH = best_max_length # Update MAX_LENGTH


def finetune(base_model, finetune_config, train_examples):
  # Limit the input sequence length to 512 (to control memory usage)... same as before
  base_model.preprocessor.sequence_length = finetune_config.sequence_length

  optimizer = keras.optimizers.AdamW(
      learning_rate=finetune_config.learning_rate,
      weight_decay=finetune_config.weight_decay
  )
  # Exclude layernorm and bias terms from decay.
  optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

  base_model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  history = base_model.fit(
      train_examples,
      epochs=finetune_config.epochs,
      batch_size=finetune_config.batch_size
  )
  return base_model, history


gemma2_lora, history = finetune(gemma2, FineTuneConfig(), train_examples)


# gemma2_lora.backbone.save_lora_weights("gemma2_2b_lora.lora.h5")

## # Load only the lora weights in...
# model.backbone.load_lora_weights(MODEL_LORA_WEIGHTS)


# Now let's see how model is doing by spot checking a few examples.
spot_check_model(gemma2_lora, ZERO_SHOT_TEMPLATE)


inference_and_save_model_output(gemma2_lora, model_name="gemma2_lora", template=ZERO_SHOT_TEMPLATE)


eval_model_basic_rules("gemma2_lora")


# Tip: do this part with CPU. Because this function does not require a model, only saved model test result

eval_model_pairwise_selection("gemma2_lora")


# Tip: do this part with CPU. Because this function does not require a model, only saved model test result

eval_model_single_answer("gemma2_lora")


# Add additional context to explicitly tell model what the tune pattern of the first line is
RAG_PROMPT_TEMPLATE  = "ä¸Šå�¥:{first_line}\nå¹³ä»„:{first_line_hint}\nä¸‹å�¥:{second_line}"


rag_train_examples = [
    RAG_PROMPT_TEMPLATE.format(first_line=example[0],
                               first_line_hint= sentence_to_tune_pattern(example[0]),
                               second_line=example[1])
    for example in train_data
]

# spot checking a few training examples
print(rag_train_examples[0:5])
print(rag_train_examples[0])


gemma2 = keras_hub.models.GemmaCausalLM.from_preset("gemma2_2b_en")
gemma2.backbone.enable_lora(rank=4)
gemma2.summary()


gemma2_rag, history = finetune(gemma2, FineTuneConfig(), rag_train_examples)


# Now let's see how model is doing by spot checking a few examples.

for example in test_data[:10]:
    test_prompt=RAG_PROMPT_TEMPLATE.format(first_line=example[0],
                               first_line_hint= sentence_to_tune_pattern(example[0]),
                               second_line="")
    response = gemma2_rag.generate(test_prompt, max_length=MAX_LENGTH)
    output = response[len(test_prompt):]
    print(f"===\ninput:{test_prompt!r}\noutput:{output!r}\nlabel:{example[1]}")


def inference_and_save_rag_model_output(model, model_name, template, test_data=test_data):
  results = []

  for example in tqdm(test_data):
      test_prompt = template.format(first_line=example[0],
                               first_line_hint= sentence_to_tune_pattern(example[0]),
                               second_line="")
      response = model.generate(test_prompt, max_length=MAX_LENGTH)
      output = response[len(test_prompt):]
      results.append([example[0], output, example[1],test_prompt])

  df = pd.DataFrame(results, columns=['input', 'output', 'label','test_prompt'])

  df.to_csv(f'{model_name}_test_output.csv', index=False)
  return df


inference_and_save_rag_model_output(gemma2_rag,model_name="gemma2_lora_rag",template=RAG_PROMPT_TEMPLATE)


eval_model_basic_rules("gemma2_lora_rag")


eval_model_pairwise_selection("gemma2_lora_rag")


eval_model_single_answer("gemma2_lora_rag")


# Uncomment the following lines to also compute evaluation results for golden label.
# Note: It's meaningless to run pairwise selection here because the golden_label = reference answer.

# golden_label_df = pd.DataFrame(test_data, columns=['input', 'output'])
# golden_label_df.to_csv('golden_label_test_output.csv', index=False)
# eval_model_basic_rules("golden_label")
# eval_model_single_answer("golden_label")


import kagglehub

model_family = "gemma2_chinese_couplet"

best_model = gemma2_lora
model_variation = "2b_lora"

# Save the best finetuned model as a KerasHub preset.
preset_dir = f'./{model_family}'
best_model.save_to_preset(preset_dir)


kaggle_username = os.environ["KAGGLE_USERNAME"]

# Kaggle model handle follows this pattern: owner_slug/model_slug/framework/variation_slug/version_number
kaggle_uri = f"kaggle://{kaggle_username}/{model_family}/keras/{model_variation}"
keras_hub.upload_preset(kaggle_uri, preset_dir)


gemma2_chinese_couplets = keras_hub.models.GemmaCausalLM.from_preset("kaggle://rachmanilove/gemma2_chinese_couplet/keras/2b_lora/1")


gemma2_chinese_couplets.generate("ä¸Šå�¥:æ­Œæ—§æ›²\nä¸‹å�¥:")

