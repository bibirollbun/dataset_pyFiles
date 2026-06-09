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


import os
import pandas as pd
from IPython.display import clear_output


!mkdir ./aozorabunko
!wget https://github.com/aozorabunko/aozorabunko/archive/refs/heads/master.zip && unzip master.zip -d aozorabunko && rm master.zip
print('all unziped')


%cd aozorabunko


!cp ./aozorabunko-master/cards/**/files/*.html ./


import os
import pandas as pd
import chardet
from bs4 import BeautifulSoup
import re

def extract_info(html_content):
    """
    HTMLコンテンツから情報を抽出します。

    Args:
      html_content: HTMLコンテンツ

    Returns:
      辞書: title, author, text をキーとする辞書
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # titleの抽出
    title_tag = soup.find('h1', class_='title')
    title = title_tag.text.strip() if title_tag else ""

    # authorの抽出
    author_tag = soup.find('h2', class_='author')
    author = author_tag.text.strip() if author_tag else ""

    # textの抽出 (rubyタグの削除)
    main_text_div = soup.find('div', class_='main_text')
    if main_text_div:
        # rubyタグを削除
        for ruby in main_text_div.find_all('ruby'):
            ruby.decompose()
        text = main_text_div.text.strip()
        # 余分な改行や空白を削除
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[0-9０-９]', '0', text) # 数字を全て0に置換
        text = re.sub(r'[a-zA-Zａ-ｚＡ-Ｚ]', 'あ', text) # 英字を全てあに置換
        text = re.sub(r'[\u3000\n\t\r]', '', text) # 改行、タブ、復帰を削除
        text = re.sub(r'[@¥$%&]', '特殊', text) #


        # [ ]でくくられている部分を削除
        text = re.sub(r'【[^】]*】', '', text) # 【】
        text = re.sub(r'［[^］]*］', '', text) # ［］
        #《 》でくくられている部分を削除
        text = re.sub(r'《[^》]*》', '', text) # 《》
        text = re.sub(r'〈[^〉]*〉', '', text) # 〈〉

    else:
        text = ""

    return {"title": title, "author": author, "text": text}

def process_all_html_files():
    """
    カレントディレクトリのすべてのHTMLファイルから情報を抽出し、DataFrameにまとめます。

    Returns:
      DataFrame: 抽出された情報を含むDataFrame
    """
    all_data = []
    for filename in os.listdir():
        if filename.endswith('.html'):
            filepath = os.path.join(os.getcwd(), filename)  # カレントディレクトリを明示的に指定
            try:
                with open(filepath, 'rb') as f:
                    raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                if encoding is None:
                    encoding = 'shift_jis'  # デフォルトのエンコーディングを指定
                html_content = raw_data.decode(encoding)
                extracted_info = extract_info(html_content)
                all_data.append(extracted_info)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return pd.DataFrame(all_data)

if __name__ == '__main__':
    df = process_all_html_files()
    print(df.head())
    df.to_csv('aozora_data_5.csv', index=False)  # CSVファイルとして保存


df.head(30)


print(df['author'].value_counts()) # one of the authors you can take for fine-tuning 作家の一覧を表示、この中から一人選んでファインチューニングする


df_akutagawa = df[df['author'] == '芥川龍之介']  # 芥川龍之介を選んだ
df_akutagawa.head(10)


text_df = df_akutagawa[['text']]
text_df


%cd /kaggle/working/


!pip install -q -U pip
!pip install -q -U keras-nlp datasets
!pip install -q -U keras


import os
# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras

# Run at half precision.
#keras.config.set_floatx("bfloat16")




# Training Configurations
token_limit = 128
num_data_limit = 100
lora_name = "my_lora"
lora_rank = 4
lr_value = 1e-3
train_epoch = 15

model_id = "gemma3n-e4b-it"




import keras
import keras_nlp

import time

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma_lm.summary()

tick_start = 0

def tick():
    global tick_start
    tick_start = time.time()

def tock():
    print(f"TOTAL TIME ELAPSED: {time.time() - tick_start:.2f}s")

def text_gen(prompt):
    tick()
    input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    output = gemma_lm.generate(input, max_length=token_limit)
    print("\nGemma output:")
    print(output)
    tock()

# inference before fine-tuning
'''
text_gen("Write a Japanese short novel in Japanese")

text_gen("Translate the text below to Japanese.\n\"Hi, how can I get to the Tokyo museum?\"")
text_gen("Speak like a pirate. Teach me why the earth is flat.")
text_gen("Write a title")
text_gen("Write a poem")
'''
text_gen("Please write a short story like those written by Ryunosuke Akutagawa.")
text_gen("Write a Japanese short novel")
# text_gen("Write a Japanese short novel in Japanese")

# text_gen("Translate the text below to Japanese.\n\"Hi, how can I get to the Tokyo museum?\"")
text_gen("Speak like a pirate in Japanese. Teach me why the earth is flat.")
text_gen("Write a title written by Ryunosuke Akutagawa.")
text_gen("Write a poem written by Ryunosuke Akutagawa.")


text_gen("Speak like Ryunosuke Akutagawa in Japanese. Teach me why the earth is flat.")
text_gen("Write a Japanese title written by Ryunosuke Akutagawa.")
text_gen("Write a Japanese poem written by Ryunosuke Akutagawa.")


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)
import jax

def detoken(tokens):
  print(tokens)
  for x in tokens:
    word = tokenizer.detokenize(jax.numpy.array([x]))
    print(f"{x:6} -> {word}")

detoken(tokenizer("こんにちは。初めまして。今日は本当にいい天気ですね。"))


tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)

# example 芥川龍之介の小説

data = df_akutagawa['text'].tolist()
# data = []
train = []

for x in data:
  item = f"<start_of_turn>user\nWrite a short novel<end_of_turn>\n<start_of_turn>model\n{x}<end_of_turn>"
  length = len(tokenizer(item))
  # skip data if the token length is longer than our limit
  if length < token_limit:
    train.append(item)
    if(len(train)>=num_data_limit):
      break

print(len(train))
for i in int(len(train)):
    print(train[i])
    print("#########################################################")
    print("#########################################################")



# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=lora_rank)
gemma_lm.summary()

# Limit the input sequence length (to control memory usage).
gemma_lm.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_value,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    #model_name = f"/content/drive/MyDrive/{lora_name}_{lora_rank}_epoch{epoch+1}.lora.h5"
    #gemma_lm.backbone.save_lora_weights(model_name)

    # Evaluate
    text_gen("Please write a short story like those written by Ryunosuke Akutagawa in Japanese.")
    text_gen("Please write a short story like those written by Ryunosuke Akutagawa in English.")
    text_gen("Please write a short story in Arabic, similar to those written by Ryunosuke Akutagawa.")
history = gemma_lm.fit(train, epochs=train_epoch, batch_size=2, callbacks=[CustomCallback()])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.show()


gemma_lm.compile(sampler="top_k")
text_gen("Write a Japanese short novel in Japanese as if written by 芥川龍之介")
text_gen("Write a Japanese short novel")
# text_gen("Write a Japanese short novel in Japanese")

text_gen("Translate the text below to Japanese.\n\"hey, how can I get to the Tokyo museum?\"")
text_gen("Please write a short story like those written by Ryunosuke Akutagawa in Japanese.")
text_gen("Please write a short story like those written by Ryunosuke Akutagawa in English.")
text_gen("Please write a short story in Arabic, similar to those written by Ryunosuke Akutagawa.")

text_gen('芥川龍之介がGoogle本社を訪ねたときのあいさつを出力してください')


# モデル保存ディレクトリを作成
save_directory = "akutagawa_fine_tuned_model"
os.makedirs(save_directory, exist_ok=True)

# ファインチューニング済みモデルの重みを保存
gemma_lm.backbone.save_lora_weights(f"{save_directory}/{lora_name}_{lora_rank}_final.lora.h5")

# トークナイザの保存（推論時に必要）
# tokenizer.save_pretrained(save_directory)
gemma_lm.save_to_preset(save_directory)


# save_directory = "akutagawa_fine_tuned_model"  
# os.makedirs(save_directory, exist_ok=True)

# Save the model weights
gemma_lm.save_weights(os.path.join(save_directory, "model.weights.h5"))



# Save the tokenizer 
# tokenizer.save(os.path.join(save_directory, "tokenizer_config.json"))
#tokenizer.save_pretrained(save_directory)


# save the fine-tuned model locally

# Save the fine-tuned LoRA weights
# gemma_lm.backbone.save_lora_weights(f"/kaggle/working/akutagawa_fine_tuned_model/{lora_name}_{lora_rank}_final.lora.h5")

# Save the tokenizer (optional, but recommended)
# tokenizer.save_vocabulary("/kaggle/working/akutagawa_fine_tuned_model/")


print(f"Model saved to {save_directory}")

