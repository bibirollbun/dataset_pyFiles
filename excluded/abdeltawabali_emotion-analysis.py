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


# Importing Libraries
import pandas as pd


# Load data
train_df = pd.read_json("/kaggle/input/depi-r-2-emotion-analysis/train.json", lines=True)
val_df = pd.read_json("/kaggle/input/depi-r-2-emotion-analysis/validation.json", lines=True)
test_df = pd.read_json("/kaggle/input/depi-r-2-emotion-analysis/test.json", lines=True)

print("Train set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))


label_mapping = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}


print("Train label distribution:")
print(train_df['label'].value_counts())

print("Validation label distribution:")
print(val_df['label'].value_counts())



import matplotlib.pyplot as plt
from wordcloud import WordCloud

# حساب عدد الكلمات لكل تغريدة
val_df['word_count'] = val_df['text'].apply(lambda x: len(x.split()))
print("Average words per tweet (validation):", val_df['word_count'].mean())

# توليد سحابة كلمات للتغريدات في مجموعة التدريب
all_text_val = " ".join(val_df['text'].tolist())
wordcloud2 = WordCloud(width=800, height=400, background_color='white').generate(all_text_val)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud for Validation Tweets")
plt.show()



# حساب عدد الكلمات لكل تغريدة
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
print("Average words per tweet (train):", train_df['word_count'].mean())

# توليد سحابة كلمات للتغريدات في مجموعة التدريب
all_text = " ".join(train_df['text'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud for Training Tweets")
plt.show()


from transformers import AutoTokenizer

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_distil = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
print("BERT vocab size:", tokenizer_bert.vocab_size)
print("DistilBERT vocab size:", tokenizer_distil.vocab_size)
print("Roberta vocab size:", tokenizer_roberta.vocab_size)


from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
import pandas as pd


# عدد الفئات (في حالتنا 6 فئات)
num_labels = 6

# تحميل النموذج والتوكنيزر من bert-base-uncased
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# تحويل DataFrame إلى Dataset من Hugging Face
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# تعريف دالة التوكنزة للنموذج
def tokenize_function(example):
    return tokenizer_bert(example["text"], truncation=True, padding="max_length", max_length=64)

# تطبيق التوكنزة على المجموعتين
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# عرض الأعمدة الحالية لمجموعتي البيانات للتأكد
print("Train dataset columns:", train_dataset.column_names)
print("Validation dataset columns:", val_dataset.column_names)

# إعداد قائمة الأعمدة المراد إزالتها لمجموعة التدريب وإزالتها
columns_to_remove_train = [col for col in ["text", "word_count", "reviewTime"] if col in train_dataset.column_names]
train_dataset = train_dataset.remove_columns(columns_to_remove_train)

# إعداد قائمة الأعمدة المراد إزالتها لمجموعة التحقق وإزالتها
columns_to_remove_val = [col for col in ["text", "word_count", "reviewTime"] if col in val_dataset.column_names]
val_dataset = val_dataset.remove_columns(columns_to_remove_val)

# إعادة تسمية عمود "label" إلى "labels" إذا لم يكن كذلك
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")



# إعداد معلمات التدريب باستخدام Trainer API
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# بدء التدريب
trainer.train()

# تقييم النموذج
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)



import numpy as np
from sklearn.metrics import accuracy_score

# الحصول على التوقعات على مجموعة التحقق
val_predictions = trainer.predict(val_dataset)
# نحول التوقعات إلى فئات باستخدام argmax
val_preds = np.argmax(val_predictions.predictions, axis=1)

# حساب الدقة بمقارنة التوقعات مع التسميات الأصلية
accuracy = accuracy_score(val_dataset["labels"], val_preds)
print("Validation Accuracy:", accuracy)



import numpy as np
import pandas as pd
from datasets import Dataset

# تحويل DataFrame إلى Dataset من Hugging Face
test_dataset = Dataset.from_pandas(test_df)

# تطبيق دالة التوكنزة على مجموعة الاختبار
test_dataset = test_dataset.map(tokenize_function, batched=True)

# إزالة الأعمدة غير الضرورية من مجموعة الاختبار
columns_to_remove_test = [col for col in ["text", "word_count", "reviewTime"] if col in test_dataset.column_names]
test_dataset = test_dataset.remove_columns(columns_to_remove_test)

# الحصول على التوقعات لمجموعة الاختبار باستخدام trainer
test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=1)

# إنشاء DataFrame لملف الـ submission
submission_df = pd.DataFrame({
    "ID": test_df["id"],   # تأكد أن عمود "ID" موجود في test_df
    "label": test_preds     # التسميات المتوقعة من النموذج
})

# حفظ ملف الـ submission بصيغة CSV
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")







