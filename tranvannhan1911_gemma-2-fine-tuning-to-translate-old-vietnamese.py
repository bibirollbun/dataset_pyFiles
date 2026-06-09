!pip install rouge-score
!pip install nltk
!pip install --upgrade nltk
!pip install evaluate


import nltk
import subprocess

# Download and unzip wordnet
try:
    nltk.data.find('wordnet.zip')
except:
    nltk.download('wordnet', download_dir='/kaggle/working/')
    command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
    subprocess.run(command.split())
    nltk.data.path.append('/kaggle/working/')


import csv
import pandas as pd
import os


TRAIN_SIZE = 300
TEST_SIZE = 100


# def convert_txt_to_csv(txt_file_path, csv_file_path):
#     """
#     Chuyển đổi file .txt thành file .csv với 2 cột: nom và vietnamese.

#     Args:
#         txt_file_path (str): Đường dẫn tới file .txt đầu vào.
#         csv_file_path (str): Đường dẫn tới file .csv đầu ra.
#     """
#     try:
#         with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
#             lines = txt_file.readlines()

#         # Chuẩn bị dữ liệu cho file CSV
#         data = []
#         for line in lines:
#             if line.strip():
#                 # Tách dòng thành 2 phần bằng ký tự tab
#                 parts = line.split('\t')
#                 if len(parts) == 2:
#                     nom = parts[0].strip()
#                     vietnamese = parts[1].strip()
#                     data.append([nom, vietnamese])

#         # Ghi dữ liệu vào file .csv
#         with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow(['nom', 'vietnamese'])
#             writer.writerows(data)

#         print(f"Đã tạo file CSV tại: {csv_file_path}")

#     except Exception as e:
#         print(f"Lỗi xảy ra: {e}")


# def convert_all_txt_to_csv(input_folder, output_folder):
#     """
#     Duyệt qua tất cả các file .txt trong thư mục và chuyển đổi chúng sang .csv.

#     Args:
#         input_folder (str): Thư mục chứa các file .txt.
#         output_folder (str): Thư mục lưu các file .csv.
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
        
#     for file_name in os.listdir(input_folder):
#         if file_name.endswith('.txt'):
#             txt_file_path = os.path.join(input_folder, file_name)
#             csv_file_name = os.path.splitext(file_name)[0] + '.csv'
#             csv_file_path = os.path.join(output_folder, csv_file_name)
#             print(f"Đang chuyển đổi: {txt_file_path} -> {csv_file_path}")
#             convert_txt_to_csv(txt_file_path, csv_file_path)


# # Đường dẫn đến thư mục chứa file .txt và thư mục lưu file .csv
# input_folder = "/kaggle/input/nom-vietnamese-translation"
# csv_folder = "/kaggle/working"
# # convert_txt_to_csv(f"{input_folder}/DVSKTT-2 Ngoai ky toan thu.txt", f"{csv_folder}/data.csv")
# # Thực hiện chuyển đổi
# convert_all_txt_to_csv(input_folder, csv_folder)



# import os
# import pandas as pd

# def read_all_csv_to_dataframe(csv_folder):
#     """
#     Đọc tất cả các file CSV trong thư mục vào một DataFrame duy nhất.

#     Args:
#         csv_folder (str): Thư mục chứa các file CSV.

#     Returns:
#         pandas.DataFrame: DataFrame chứa dữ liệu từ tất cả các file CSV.
#     """
#     dataframes = []
    
#     for file_name in os.listdir(csv_folder):
#         if file_name.endswith('.csv'):
#             csv_file_path = os.path.join(csv_folder, file_name)
#             print(f"Đang đọc file: {csv_file_path}")
#             df = pd.read_csv(csv_file_path, encoding='utf-8')
#             dataframes.append(df)
    
#     # Gộp tất cả các DataFrame lại
#     combined_df = pd.concat(dataframes, ignore_index=True)
#     return combined_df

# df = read_all_csv_to_dataframe(csv_folder)

# print(len(df))
# print(df.head())


# save dataset
# df.to_csv("/kaggle/working/nom-vietnamese.csv", index=False)


df = pd.read_csv("/kaggle/input/nom-vietnamese-translate/nom-vietnamese.csv", encoding='utf-8')
df.head()


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
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
token_limit = 256
lora_name = "translator"
LORA_RANK = 16
LR_VALUE = 1e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1
TRAIN_EPOCH = 10
model_id = "gemma2_instruct_2b_en"



def get_prompt(prompt):
    return f"Dịch chữ nôm sau sang bản dịch tiếng việt (chỉ trả về văn bản bản dịch tiếng việt): {prompt}"


def get_full_prompt(prompt, response=None):
    prompt = get_prompt(prompt)
    if response is None:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"



tokenizer = keras_nlp.models.GemmaTokenizer.from_preset(model_id)
def is_valid_length(row):
    item = get_full_prompt(row['nom'], row['vietnamese'])
    return len(tokenizer(item)) < token_limit

# Áp dụng hàm kiểm tra từng dòng và lọc
df = df[df.apply(is_valid_length, axis=1)].reset_index(drop=True)

print(f"Số dòng hợp lệ: {len(df)}")


# Cố định tập dữ liệu giữa các lần chạy
# train_df = df.sample(TRAIN_SIZE, random_state=42)
# test_df = df.sample(TEST_SIZE, random_state=43)
train_df = df.iloc[:TRAIN_SIZE]
test_df = df.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
train = []

for _, x in train_df.iterrows():
    item = get_full_prompt(x['nom'], x['vietnamese'])
    train.append(item)
    
print(len(train))
print(train[0])


print(train_df.head())
print(test_df.head())


common_rows = pd.merge(train_df, test_df, how='inner')
print(f"\nSố dòng có cùng giá trị trong train_df và test_df: {len(common_rows)}")


import time

gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
gemma.summary()


def extract_response(output):
    start_marker = "<start_of_turn>model\n"
    end_marker = "<end_of_turn>"
    
    start_idx = output.find(start_marker) + len(start_marker)
    end_idx = output.find(end_marker, start_idx)
    
    if start_idx != -1 and end_idx != -1:
        return output[start_idx:end_idx].strip()
    return output


def translate_nom_to_literal(prompt):
    input = get_full_prompt(prompt)
    output = gemma.generate(input, max_length=token_limit)
    return extract_response(output)

print("target: ", df.iloc[0]["vietnamese"])
print("predicted: ", translate_nom_to_literal(df.iloc[0]["nom"]))

print("target: ", df.iloc[1]["vietnamese"])
print("predicted: ", translate_nom_to_literal(df.iloc[1]["nom"]))



from nltk.translate.bleu_score import sentence_bleu
# gemma = keras_nlp.models.GemmaCausalLM.from_preset(model_id)
# gemma.load_weights("/kaggle/working/translator_4_epoch5.lora.h5")
# gemma.summary()

test_df['predicted'] = test_df['nom'].apply(translate_nom_to_literal)
# test_df['target'] = test_df.apply(lambda row: get_full_prompt(row['nom'], row['vietnamese']), axis=1)
print(test_df)
print(test_df.iloc[0]['vietnamese'])
print(test_df.iloc[0]['predicted'])


def bleu_score(eval_df):
    bleu_scores = []
    
    for i, row in eval_df.iterrows():
        reference = row['vietnamese']
        prediction = row['predicted']
        score = sentence_bleu([reference.split()], prediction.split())
        bleu_scores.append(score)
    
    return sum(bleu_scores) / len(bleu_scores)
bleu_score_before_tunning = bleu_score(test_df)
print(f"BLEU Score trước finetuning : {bleu_score_before_tunning}")


from rouge_score import rouge_scorer

def rouge_score(eval_df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for i, row in eval_df.iterrows():
        reference = row['vietnamese']
        prediction = row['predicted']
        scores = scorer.score(reference, prediction)
        rouge_scores.append(scores)
    
    # Tính trung bình các ROUGE scores
    rouge1_avg = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    rouge2_avg = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    rougeL_avg = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    return rouge1_avg, rouge2_avg, rougeL_avg
rouge1_avg_before_tunning = rouge_score(test_df)
print(f"ROUGE-1 trung bình: {rouge1_avg_before_tunning[0]}")
print(f"ROUGE-2 trung bình: {rouge1_avg_before_tunning[1]}")
print(f"ROUGE-L trung bình: {rouge1_avg_before_tunning[2]}")


from nltk.translate.meteor_score import meteor_score

def meteor_score_eval(eval_df):
    meteor_scores = []

    for i, row in eval_df.iterrows():
        reference = row['vietnamese']
        prediction = row['predicted']
        # Tính METEOR score cho cặp tham chiếu và dự đoán
        score = meteor_score([reference.split()], prediction.split())
        meteor_scores.append(score)

    return sum(meteor_scores) / len(meteor_scores)

meteor_score_before_tuning = meteor_score_eval(test_df)
print(f"METEOR Score trước finetuning: {meteor_score_before_tuning}")



# Enable LoRA for the model and set the LoRA rank (4, 8 or 16).
gemma.backbone.enable_lora(rank=LORA_RANK)
gemma.summary()

# Limit the input sequence length (to control memory usage).
gemma.preprocessor.sequence_length = token_limit
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=LR_VALUE,
    weight_decay=WEIGHT_DECAY,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model_name = f"/kaggle/working/{lora_name}_{LORA_RANK}_epoch{epoch+1}.lora.h5"
    gemma.backbone.save_lora_weights(model_name)

    # # Evaluate
    print("\n")
    print("target: ", df.iloc[0]["vietnamese"])
    print("predicted: ", translate_nom_to_literal(df.iloc[0]["nom"]))
    
    print("target: ", df.iloc[1]["vietnamese"])
    print("predicted: ", translate_nom_to_literal(df.iloc[1]["nom"]))

start_train_time = time.time()
history = gemma.fit(train, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, callbacks=[CustomCallback()])
end_train_time = time.time()
print(f"Total training time: {end_train_time-start_train_time:.2f} seconds")



import matplotlib.pyplot as plt

print(history.history['loss'])

plt.plot(history.history['loss'])
plt.title('Training Loss Over Epochs')
plt.show()



print(history.history['sparse_categorical_accuracy'])

plt.plot(history.history['sparse_categorical_accuracy'])
plt.title('Training Sparse Categorical Accuracy Over Epochs')
plt.show()



test_df['predicted'] = test_df['nom'].apply(translate_nom_to_literal)
# test_df['target'] = test_df.apply(lambda row: get_full_prompt(row['nom'], row['vietnamese']), axis=1)
print(test_df)
print("target:", test_df.iloc[0]['vietnamese'])
print("predicted:", test_df.iloc[0]['predicted'])


bleu_score_after_tunning = bleu_score(test_df)
print(f"BLEU Score sau finetuning : {bleu_score_after_tunning}")


rouge1_avg_after_tunning = rouge_score(test_df)
print(f"ROUGE-1 trung bình: {rouge1_avg_after_tunning[0]}")
print(f"ROUGE-2 trung bình: {rouge1_avg_after_tunning[1]}")
print(f"ROUGE-L trung bình: {rouge1_avg_after_tunning[2]}")


meteor_score_after_tuning = meteor_score_eval(test_df)
print(f"METEOR Score sau finetuning: {meteor_score_after_tuning}")


print(history.history['loss'])
print(history.history['sparse_categorical_accuracy'])


import pandas as pd
results = {
    "Metric": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "LOSS", "SPARSE CATEGORICAL ACCURACY", "TRAIN TIME"],
    "Gemma 2": [
        2.4872380387216864e-80,
        0.38330782192232093,
        0.08283416205888852,
        0.2736587734690344,
        0.046394812190370296,
        0,
        0,
        "-"
    ],
    "1st tuning": [
        0.1505978850942885,
        0.7811066160052103,
        0.5674734811856526,
        0.7372987221949363,
        0.5191197242565135,
        0.554043173789978,
        0.7086907625198364,
        "715.13s"
    ],
    "2nd tuning": [
        0.19090855101650464,
        0.7677689194281221,
        0.5734528635752333,
        0.7342079464283501,
        0.5208456821555463,
        0.3561025559902191,
        0.808032214641571,
        "1356.47s"
    ],
    "3nd tuning": [
        0.2262997335682175,
        0.796754441985636,
        0.6246024891327907,
        0.7689454707603842,
        0.566332309975574,
        0.2720997631549835,
        0.8426046371459961,
        "1355.85s",
    ],
    "4-th tuning": [
        0.24792300788295077,
        0.7881659255563629,
        0.6316432455033653,
        0.7614855131263725,
        0.5678693903058963,
        0.19213862717151642,
        0.8921237587928772,
        "1355.85s",
    ],
    "5-th tuning": [
        0.3077114940827575,
        0.8522812162163638,
        0.7282150552635823,
        0.8382009743406661,
        0.6610018891975734,
        0.1955103874206543,
        0.880535900592804,
        "4304.25s",
    ],
}

# Chuyển kết quả thành DataFrame
df_results = pd.DataFrame(results)

# Hiển thị bảng kết quả
print(df_results)



from IPython.core.display import display, HTML

html_table = df_results.to_html(
    index=False,
    border=1,
    justify="center",
    classes="dataframe",
)

# Tạo style cho bảng
custom_style = """
<style>
    table.dataframe {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 16px;
        text-align: center;
    }
    table.dataframe th, table.dataframe td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    table.dataframe th {
        background-color: #f4f4f4;
    }
    table.dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    table.dataframe tr:hover {
        background-color: #f1f1f1;
    }
</style>
"""

# Hiển thị bảng với style
display(HTML(custom_style + html_table))

