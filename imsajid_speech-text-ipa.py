import os
import pandas as pd
import librosa
from tqdm import tqdm
import numpy as np
import random
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt
import os
import librosa
from multiprocessing import Pool
import time
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


!ls /kaggle/input/ben10/ben10


df = pd.read_csv("/kaggle/input/ben10/ben10/16_kHz_train_audio/train.csv")
train_dir = "/kaggle/input/ben10/ben10/16_kHz_train_audio/"
print("Train dataframe : ")
display(df.head())


test_dir = "/kaggle/input/ben10/ben10/16_kHz_valid_audio/"
test_paths = [test_dir+path for path in os.listdir(test_dir)]
test = pd.DataFrame(test_paths,columns=['region'])
test


def extract_regions(path):
    unwanted_strs = ["train_","valid_",".wav","1","2","3","4","5","6","7","8","9","0","(",")"," ","/kaggle/input/ben/ben/_kHz_audio/"]
    for i in unwanted_strs:
        path = path.replace(i,"")
    return path
    

df["region"] = df["file_name"].apply(lambda x:extract_regions(x))
test["region"] = test["region"].apply(lambda x:extract_regions(x))

#Plot the distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

df.region.value_counts().sort_values().plot(kind='barh', ax=axes[0])
axes[0].set_title('Distribution of Regions in training set')

test.region.value_counts().sort_values().plot(kind='barh', ax=axes[1])
axes[1].set_title('Distribution of Regions in test')

plt.tight_layout()
plt.show()



list(df["region"].unique())


test_paths


display(AudioSegment.from_file('/kaggle/input/ben10/ben10/16_kHz_valid_audio/valid_habiganj (66).wav'))


regions = ['barishal', 'chittagong', 'habiganj', 'kishoreganj', 'narail',
           'narsingdi', 'rangpur', 'sandwip', 'sylhet', 'tangail']

for region in regions:
    sample = df[df["region"] == region]
    
    if sample.empty:
        print(f"No data found for region: {region}")
        continue

    idx = random.randint(0, len(sample) - 1)

    file = sample['file_name'].iloc[idx]
    path = train_dir + file

    print("Region :", region)
    display(AudioSegment.from_file(path))
    print("Original transcription :", sample['transcriptions'].iloc[idx])
    print("=" * 40)



!cp /kaggle/input/bengali-eval-data/predict.py .

!cp -r ../input/python-packages2 ./
!tar xvfz ./python-packages2/jiwer.tgz
!pip install ./jiwer/python-Levenshtein-0.12.2.tar.gz -f ./ --no-index
!pip install ./jiwer/jiwer-2.3.0-py3-none-any.whl -f ./ --no-index


import os
import csv
import time
import glob

MODEL = '/kaggle/input/bengali-ai-asr-submission/bengali-whisper-medium/'

CHUNK_LENGTH_S = 20.1
ENABLE_BEAM = True

if ENABLE_BEAM:
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 8
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


pipe = pipeline(task="automatic-speech-recognition",
                model=MODEL,
                tokenizer=MODEL,
                chunk_length_s=CHUNK_LENGTH_S,device=0, batch_size=BATCH_SIZE)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="bn", task="transcribe")

print("model loaded!")


if ENABLE_BEAM:
    texts = pipe(test_paths, generate_kwargs={"max_length": 260, "num_beams": 4})
else:
    texts = pipe(test_paths)
preds = []
for i in texts:
    preds.append(i['text'])
len(preds)


sub = pd.DataFrame({"id":test_paths,"sentence":preds})
sub.head()


sub['id'] = sub['id'].apply(lambda x: x.replace(test_dir,""))
sub.head()


sub.to_csv("submission1.csv",index=False)


path = '/kaggle/input/bangla-text2ipa-transformer-model/text2ipa-transformer-model'
new_model=tf.saved_model.load(path)

vb = ['', '[UNK]', '[start]', '[end]', 'া', 'র', '্', 'ে', 'ি', 'ন', 'ক', 'ব', 'স', 'ল', 'ত', 'ম', 'প', 'ু', 'দ', 'ট', 'য়', 'জ', '।', 'ো', 'গ', 'হ', 'য', 'শ', 'ী', 'ই', 'চ', 'ভ', 'আ', 'ও', 'ছ', 'ষ', 'ড', 'ফ', 'অ', 'ধ', 'খ', 'ড়', 'উ', 'ণ', 'এ', 'থ', 'ং', 'ঁ', 'ূ', 'ৃ', 'ঠ', 'ঘ', 'ঞ', 'ঙ', 'ৌ', '‘', 'ৎ', 'ঝ', 'ৈ', '়', 'ঢ', 'ঃ', 'ঈ', '\u200c', 'ৗ', 'a', 'ঐ', 'd', 'w', 'ঋ', 'i', 'e', 't', 's', 'n', 'm', 'b', '“', 'u', 'r', 'œ', 'o', '–', 'ঊ', 'ঢ়', 'Í', 'g', 'p', '\xad', 'h', 'c', 'l', 'ঔ', 'ƒ', '”', 'Ñ', '¡', 'y', 'j', 'f', '→', '—', 'ø', 'è', '¦', '¥', 'x', 'v', 'k']
vipa = ['', '[UNK]', '[start]', '[end]', 'ɐ', 'ɾ', 'i', 'o', 'e', '̪', 't', 'n', 'k', 'ɔ', 'ʃ', 'b', 'd', 'l', 'u', 'p', 'm', 'ʰ', 'ɟ', '͡', '̯', 'g', 'ʱ', '।', 'c', 'ʲ', 'h', 's', 'ŋ', 'ɛ', 'ɽ', '̃', 'ʷ', '‘', '“', '–', '”', '—', 'w', 'j']
v = vb + vipa
s = set()
for ch in v:
  s.add(ch)

vocab = sorted(list(s))
print("Length of vocab:", len(s))
print(vocab)
vocab_size = len(vocab)

sequence_length = 64 # 20
batch_size = 64

eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
    vocabulary=vocab
)

spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    vocabulary=vocab
)

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 64 #20

def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = '[start]'

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = new_model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == '[UNK]':
            break
    return decoded_sentence

def sentence_word(sentence):
  trg=''
  for ch in sentence:
      if ch != " ":
        trg += ch
  return trg

def word_sentence(word):
  sentence = ""
  for ch in word:
    sentence += (ch + " ")
  return sentence



def bangla_vocabulary():
  Vowels = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'ঌ', 'এ', 'ঐ', 'ও', 'ঔ']
  Vowel_signs = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ে', 'ৈ', 'ো', 'ৌ']
  Consonants = ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ', 'ং', 'ঃ', 'ঁ']
  Operators = ['=', '+', '-', '*', '/', '%', '<', '>', '×', '÷']
  Punctuation_marks = ['।', ',', ';', ':', '?', '!', "'", '.', '"', '-', '[', ']', '{', '}', '(', ')', '–', '—', '―', '~']
  Others = ['্', '়', 'ৗ', '‘', '’', '“', '”']

  BANGLA_VOCAB = sorted(list(set(Vowels + Vowel_signs + Consonants +  Operators + Punctuation_marks + Others)))
  return BANGLA_VOCAB

def foreign_character_normalization(word):
  BANGLA_VOCAB = bangla_vocabulary()
  normalized_word = ""

  for ch in word:
    if ch not in BANGLA_VOCAB:
      continue
    normalized_word += ch
  return normalized_word

def aligned_stateful_tokenizer(word):
  vocab = [ 'ঁ', 'ং', 'ঃ', 'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', '়', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ', 'ৗ', 'ড়', 'ঢ়', 'য়']
  n = len(word)
  i = 0
  j = n-1

  state = []
  tokens = []

  while i < n:
    subword = ""
    if word[i] in vocab:
      found = True
      while i < n and word[i] in vocab:
        subword += word[i]
        i += 1

    elif not(word[i] in vocab):
      found = False
      while i < n and not(word[i] in vocab):
        subword += word[i]
        i += 1

    state.append(found)
    tokens.append(subword)
  return state, tokens

def preprocess(word):
  preprocessed_word = foreign_character_normalization(word)
  return preprocessed_word


path = "/kaggle/input/text2ipa-mapping-trainset/previous_trainset_word_ipa_map_37807.csv"
df = pd.read_csv(path)

DICTIONARY = {}
vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ENGLISH_VOCAB = vocab + digits

problem = []
for index, row in df.iterrows():
  word = row['word']
  ipa = row['ipa']
  DICTIONARY[word] = ipa

# correcting incorrect annotaions
DICTIONARY["seen"] = ""
DICTIONARY["passage"] = ""
DICTIONARY["Writing"] = ""
DICTIONARY["Test"] = ""
DICTIONARY["B"] = ""
DICTIONARY["admissions"] = ""

print("Total train data", len(DICTIONARY))


import pandas as pd

url_test = "/kaggle/input/submission-csv/submission.csv"
df_test = pd.read_csv(url_test)
print("Shape:", df_test.shape)
df_test.head(2)


for index, row in df_test.iterrows():
  row_id = row['id']
  text = row['sentence']
  texts = text.split()

  for word in texts:
    if word in DICTIONARY.keys():
      continue

    normalized_word = foreign_character_normalization(word)
    state, tokens = aligned_stateful_tokenizer(normalized_word)

    if len(normalized_word) == 0:
      DICTIONARY[word] = ""
      continue

    for i in range(len(state)):
      if state[i]:
        tokenized_word = tokens[i]
        translated = decode_sequence(word_sentence(tokenized_word))
        trg = sentence_word(translated)
        trg = trg[7:]
        trg = trg[:-5]
        tokens[i] = trg

    value = "".join(tokens)
    DICTIONARY[word] = value

  print("----->", index)


print(len(DICTIONARY))


import pandas as pd

BANGLA_DIGIT = ['১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯', '০']

def load_dic(DICTIONARY):
  dic = {}
  for key in DICTIONARY:
    word = key
    ipa = DICTIONARY[key]
    if not(type(ipa) == type('cat')):
      ipa = word

    # Eleminiting bangla digits
    ipa_ = ""
    for ch in ipa:
      if ch not in BANGLA_DIGIT:
        ipa_ += ch
    dic[word] = ipa_

  print("Dictionary Loaded...")
  return dic

dic = load_dic(DICTIONARY)


def generate_submission(dic):
  url_test = "/kaggle/input/submission-csv/submission.csv"
  df_test = pd.read_csv(url_test)
  rows = []
  ipas = []

  for index, row in df_test.iterrows():
    row_id = row['id']
    text = row['sentence']
    texts = text.split()
    pred = []

    for word in texts:
      ipa = dic[word]
      pred.append(ipa)

    ipa_text = " ".join(pred)
    rows.append(row_id)
    ipas.append(ipa_text)

  return rows, ipas
rows, ipas = generate_submission(dic)
print(len(rows), len(ipas))


def submission_file(rows, ipas):
  data = {
    'row_id_column_name': rows,
    'ipa': ipas
  }

  df = pd.DataFrame(data, columns=data.keys())
  df.to_csv('/kaggle/working/submission2.csv', index=False)
submission_file(rows, ipas)


df1 = pd.read_csv('/kaggle/working/submission2.csv')
df1


df_test




