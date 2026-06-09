pip install pandas numpy matplotlib seaborn


import pandas as pd

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")

print(train.shape)
train.head()


train.info()


train.describe(include="object")


train.isna().sum()


train.duplicated().any()


import seaborn as sns
import matplotlib.pyplot as plt

labeled = train[~train["Misconception"].isna()]
plt.figure(figsize=(10,6))
sns.countplot(y=labeled["Misconception"], order=labeled["Misconception"].value_counts().index)
plt.title("Distribution of Misconceptions")
plt.show()


incomplete = train[train["Misconception"] == "Incomplete"]

plt.figure(figsize=(8,5))
sns.countplot(x=incomplete["Category"], order=incomplete["Category"].value_counts().index)
plt.title("Distribution of Categories for Misconception = Incomplete")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()


incomplete = train[train["Misconception"] == "Incomplete"]

category_counts = incomplete["Category"].value_counts()
print(category_counts)


incomplete_false = incomplete[incomplete["Category"] == "False_Misconception"]
top_ids = incomplete_false["QuestionId"].value_counts().head(10).index
top_questions = incomplete_false[incomplete_false["QuestionId"].isin(top_ids)][["QuestionId", "QuestionText"]].drop_duplicates()

for i, row in top_questions.iterrows():
    print(f"QuestionId: {row['QuestionId']}\nQuestionText: {row['QuestionText']}\n")


incomplete_false = incomplete[incomplete["Category"] == "False_Misconception"]

for i, explanation in enumerate(incomplete_false["StudentExplanation"][:100], 1):
    print(f"{i}: {explanation}\n")


sns.countplot(y=train["Category"], order=train["Category"].value_counts().index)
plt.title("Distribution of Categories")
plt.show()


train['Correct'] = train['Category'].apply(lambda x: 'Correct' if 'True' in x else 'Incorrect')

proportions = train['Correct'].value_counts(normalize=True) * 100
print(proportions) 

sns.set_style("whitegrid")
plt.figure(figsize=(6,4))
sns.barplot(x=proportions.index, y=proportions.values, palette=['green','red'])
plt.ylabel("Proportion (%)")
plt.title("Proportion of Correct vs Incorrect Responses")
plt.show()



category_counts = train['Category'].value_counts(normalize=True) * 100

plt.figure(figsize=(8,5))
plt.bar(category_counts.index, category_counts.values, color=['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f'])
plt.ylabel("Proportion (%)")
plt.xlabel("Category")
plt.title("Proportion of every Category")
plt.xticks(rotation=45)
plt.ylim(0, max(category_counts.values)*1.1)  
plt.show()


train['Correct'] = train['Category'].apply(lambda x: 'Correct' if 'True' in x else 'Incorrect')
train['Has_Misconception'] = train['Category'].apply(lambda x: 'Misconception' if 'Misconception' in x else 'No Misconception')

plt.figure(figsize=(8,5))
sns.countplot(x='Correct', hue='Has_Misconception', data=train, palette=['#66c2a5','#fc8d62'])
plt.title("Combined Analysis: Correct vs Incorrect and Misconception")
plt.ylabel("Number of Responses")
plt.show()


sns.countplot(y=train["MC_Answer"], order=train["MC_Answer"].value_counts().index)
plt.title("Distribution of MC_Answer")
plt.show()


train['Explanation_len'] = train['StudentExplanation'].astype(str).apply(len)

plt.figure(figsize=(10,6))
sns.histplot(train['Explanation_len'], bins=50)
plt.title("Distribution of StudentExplanation Lengths")
plt.show()


train["Explanation_len"] = train["StudentExplanation"].apply(lambda x: len(str(x)))
longest_explanations = train.sort_values(by="Explanation_len", ascending=False)

longest_explanations[["StudentExplanation", "Explanation_len", "Category"]].head(10)



shortest_explanations = train.sort_values(by="Explanation_len", ascending=True)
shortest_explanations[["StudentExplanation", "Explanation_len", "Category"]].head(20)


plt.figure(figsize=(10,6))
sns.boxplot(x="Category", y="Explanation_len", data=train)
plt.title("Distribution of Explanation Length by Category")
plt.ylabel("Explanation Length (characters)")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.show()


from IPython.display import display, Math, Latex

tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'\\\(|\\\)|\\\[|\\\]', ' ', text)
    text = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', text)
    text = re.sub(r'\[Image:\s*(.*?)\]', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train["CleanedQuestion"] = train["QuestionText"].apply(clean_text)
train["CleanedMcAnswer"] = train["MC_Answer"].apply(clean_text)



i = 0

print("ğŸ”¹ Original")
print("Q:", train.loc[i, "QuestionText"])
print("A:", train.loc[i, "MC_Answer"])
print("\nğŸ”¹ Cleaned")
print("Q:", train.loc[i, "CleanedQuestion"])
print("A:", train.loc[i, "CleanedMcAnswer"])



train['Misconception'] = train['Misconception'].fillna('None')


train['Misconception'].isna().sum()


train = train.drop(columns=['row_id'])


train['is_short_explanation'] = (train['Explanation_len'] <= 6).astype(int)


train['Label'] = train['Category'] + ':' + train['Misconception']


train[['truth', 'type']] = train['Category'].str.split('_', expand=True)
train = pd.get_dummies(train, columns=['truth', 'type'], dtype=int)


import transformers
print(transformers.__version__)


!pip uninstall -y transformers
!pip install transformers==4.26.1


import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def embed_texts(texts, batch_size=32):
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean of token embeddings instead of pooler_output
            emb = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        embeddings_list.append(emb.cpu())
    return torch.cat(embeddings_list, dim=0)


question_emb = embed_texts(train["CleanedQuestion"].astype(str).tolist())
mcanswer_emb = embed_texts(train["CleanedMcAnswer"].astype(str).tolist())
student_emb = embed_texts(train["StudentExplanation"].astype(str).tolist())


concatenated_emb = torch.cat([question_emb, mcanswer_emb, student_emb], dim=1)
print(concatenated_emb[:5])

