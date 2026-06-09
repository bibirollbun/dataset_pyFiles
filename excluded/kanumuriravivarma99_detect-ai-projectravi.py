import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns


data = '/kaggle/input/llm-detect-ai-generated-text/'

for dirname, _, filenames in os.walk(data):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train_prompts = pd.read_csv(data + "train_prompts.csv")
print(df_train_prompts.info())
df_train_prompts.head()


df_train_essays = pd.read_csv(data + "train_essays.csv")
print(df_train_essays.info())
df_train_essays.head()


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays,
                   x="prompt_id")

abs_values = df_train_essays['prompt_id'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of prompt ID")


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays,
                   x="generated")

abs_values = df_train_essays['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Text (0 = Human, 1 = AI)")


import kagglehub
newtest =  "/kaggle/input/newtestdata/test_essays_combined.csv"
df_test_essays = pd.read_csv(newtest)
print(df_test_essays.info())
df_test_essays.head()


import kagglehub

# Download latest version
df_train_essays_extended = kagglehub.dataset_download("radek1/llm-generated-essays")

print("Path to extended essays:", df_train_essays_extended)


df_train_essays_extended = '/kaggle/input/llm-generated-essays'

for dirname, _, filenames in os.walk(df_train_essays_extended):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train_essays_extended = pd.read_csv('/kaggle/input/llm-generated-essays/ai_generated_train_essays.csv')
df_train_essays_extendedgpt = pd.read_csv('/kaggle/input/llm-generated-essays/ai_generated_train_essays_gpt-4.csv')
df_train_essays_extended.info()
df_train_essays_extendedgpt.info()


df_train_essays_extended




df_train_essays_extendedgpt


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays_extended,
                   x="generated")

abs_values = df_train_essays_extended['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Generated Text")


f, ax = plt.subplots(figsize=(12, 4))

sns.despine()
ax = sns.countplot(data=df_train_essays_extendedgpt,
                   x="generated")

abs_values = df_train_essays_extendedgpt['generated'].value_counts().values

ax.bar_label(container=ax.containers[0], labels=abs_values)

ax.set_title("Distribution of Generated Text")


df_train_essays_combined = pd.concat([df_train_essays_extended[["text", "generated"]], df_train_essays[["text", "generated"]],df_train_essays_extendedgpt[["text","generated"]]])

df_train_essays_combined.info()


df_train_essays_combined


sns.countplot(data=df_train_essays_combined, x='generated')
plt.title("Distribution in Combined Dataset")
plt.show()



# Visualizing the most common words for AI vs human with wordcloud

from wordcloud import WordCloud

ai_text = " ".join(df_train_essays_combined.loc[df_train_essays_combined['generated'] == 1, 'text'])
human_text = " ".join(df_train_essays_combined.loc[df_train_essays_combined['generated'] == 0, 'text'])

ai_wordcloud = WordCloud(width=800, height=400).generate(ai_text)
human_wordcloud = WordCloud(width=800, height=400).generate(human_text)

# Display them
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].imshow(ai_wordcloud, interpolation='bilinear')
ax[0].set_title('AI-generated Text')
ax[0].axis('off')
ax[1].imshow(human_wordcloud, interpolation='bilinear')
ax[1].set_title('Human-written Text')
ax[1].axis('off')
plt.show()


import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

df_train_essays_combined['cleaned_text'] = df_train_essays_combined['text'].apply(preprocess)


#test we used to see which feature technique between bag of words and tf-idf would be best to use
# because tf-idf had a higher accuracy, that's the one we used
''' 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    df_train_essays_combined['text'], df_train_essays_combined['generated'], test_size=0.3, random_state=42)

bagOfWords = CountVectorizer()
X_train_bow = bagOfWords.fit_transform(X_train)
X_test_bow = bagOfWords.transform(X_test)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf_bow = LogisticRegression(random_state=42, max_iter=1000)
clf_tfidf = LogisticRegression(random_state=42,max_iter=1000)

clf_bow.fit(X_train_bow, y_train)
clf_tfidf.fit(X_train_tfidf, y_train)


y_pred_bow = clf_bow.predict(X_test_bow)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)

accuracy_bow = accuracy_score(y_test, y_pred_bow)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

print("Bag of Words Accuracy: {:.2f}%".format(accuracy_bow * 100))
print("TF-IDF Accuracy: {:.2f}%".format(accuracy_tfidf * 100))
'''


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_train_essays_combined['cleaned_text'])
y = df_train_essays_combined['generated']  # 0 for Human, 1 for AI


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


#SVM 
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("SVM Classification Report:\n")
print(classification_report(y_test, y_pred_svm))


#RandomForest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred_rf))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test)
plt.title("SVM Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.show()


pip install -q transformers sentence-transformers


from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Use cleaned_text for embedding
X_bert = bert_model.encode(df_train_essays_combined['cleaned_text'].tolist(), show_progress_bar=True)



X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    X_bert, y, test_size=0.4, random_state=42)


rf_bert = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bert.fit(X_train_bert, y_train_bert)
y_pred_bert = rf_bert.predict(X_test_bert)

print("Random Forest on BERT Embeddings:\n")
print(classification_report(y_test_bert, y_pred_bert))


def predict_texts(texts):
    embeddings = bert_model.encode(texts)
    preds = rf_bert.predict(embeddings)
    return ["AI-generated" if p == 1 else "Human-written" for p in preds]
    

def predict_probability(texts):
    embeddings = bert_model.encode(texts)
    probs = rf_bert.predict_proba(embeddings)[:, 1] 
    return probs
    
sample_texts = [
    "Today, we explore the wonders of machine learning.",
    "Explanatory Essay: The Advantages of Limiting Car Usage.  Limiting car usage is becoming increasingly popular worldwide due to its numerous benefits for individuals and the environment. This essay explores the key advantages of reducing reliance on cars. A major benefit is the reduction in greenhouse gas emissions. Passenger cars contribute significantly to emissions in regions like Europe and the United States. Cutting back on car usage helps combat climate change and supports environmental goals such as those promoted by former President Obama. Improved air quality is another critical advantage. Cities like Paris and Bogota have implemented driving bans during high smog levels, resulting in cleaner air and less congestion. Encouraging public transit, cycling, and walking can greatly enhance urban air quality, especially in densely populated areas. Limiting car use also reduces traffic congestion. In cities like Vauban, Germany, where car access is restricted, traffic flows more smoothly and roads are safer. Less congestion means fewer accidents and less time spent in traffic. Urban planning also benefits from reduced car dependence. Smart city designs prioritize access to public transportation and walkable neighborhoods, reducing the need for extensive parking spaces and making communities more livable.  Finally, there’s a cultural shift toward sustainable transport. Many young people today prefer alternatives to car ownership, favoring public transit, biking, and car-sharing services. This shift encourages innovation and supports long-term sustainability. In conclusion, limiting car usage offers clear benefits: it lowers emissions, improves air quality, reduces congestion, supports better urban planning, and encourages sustainable behavior. As more cities adopt this approach, we move toward a cleaner, healthier, and more efficient future."
]

sample_texts_processed = [preprocess(text) for text in sample_texts] #using preprocessing function to clean data 
print(predict_texts(sample_texts_processed))
print(predict_probability(sample_texts_processed))

sample_texts2 = [
    # Human‑written
    "Over the past decade, renewable energy sources like wind and solar have become more cost‑competitive, leading to widespread adoption and significant reductions in carbon emissions.",
    "My grandmother’s handwritten recipes always included a pinch of love—she believed that no matter how precise you were with measurements, a warm heart was the secret ingredient.",
    "During my summer internship at the cybersecurity firm, I learned to analyze network logs for anomalies, write custom detection scripts, and contribute to our team’s threat‑intelligence dashboard.",

    # AI‑generated
    "The convergence of quantum computing and blockchain architectures heralds a new paradigm in secure decentralized protocols, offering unparalleled resistance to classical cryptographic attacks.",
    "In the realm of modern gastronomy, the juxtaposition of umami‑rich broths with molecular gastronomy techniques orchestrates an immersive culinary symphony that transcends traditional palatal expectations.",
    "Architectural sustainability is not merely a trend but an evolutionary necessity, where biomimicry and adaptive façade engineering collaborate to optimize thermal efficiency and occupant well‑being."
]
''' 
sample_texts2_processed = [preprocess(text) for text in sample_texts2]
print(predict_texts(sample_texts2_processed))
print(predict_probability(sample_texts2_processed))
'''

sample_texts3 = [
    "Phones and Driving don't mix. Driving with a phone causes more wrecks every year. Law enforcement have noticed the unsafety of having a phone while driving and made it where it is illegal.Phone is a safety issue not just to the people in the car but to the one driving as well. Phones make it where the drive get sidetrack and can cause a wreck, and affect everyone life in the vehicle not just yourself. One thing that is big while driving that is a safety issues, is texting while driving. Texting on the phone is the number one reason people use their phone while driving on the road.Driving while on the phone can be the number one thing that causes death in a car by accident. When you're driving remember you got life of other people in your hand.",
    "When I was just learning how to play soccer I couldn't kick the ball correctly. Every single time I tried to my foot would slide under the ball causing the ball to go straight up in the air. Due to that disheartening fact I tended to get made fun of.....ALOT. I mean who can't kick a soccer ball correctly. So the best way I saw fit to help fix the problem at hand, was to ask the coach, my dad, and my best friend for advice. That would hopefully help me kick the ball better. So one thing I think is very important to help anybody learn is to ask more than one person for advice that in the future can help guide you into making better choices",
    "Seeking multiple opinions can help someone make a better choice by making them think about the outcome of there choice because there choice might give them a bad future, they can make sure that it is the best choice for them, and can help them realize what they want to do in life. If they make good choices throughout there whole life it can make them end up with great friends, house, and family.My first reason is so that they do not make a choice that will give them a bad future. Sometimes, people do not think before they make a choice. Therefore, if they listen to other peoples opinions it can help them realize that they will end up in a bad place or situation. For example, if they hang out around bad people it can get them into doing bad stuff and possibility make them end up in jail, but if you listen to someone that says not to hang out with them before you started hanging out with those people then they would not end up in jail. If they do not listen to there friend if they say not to hangout with them because they are bad, they will lose that friend that cared about them.",
    "I believe that successful people try new things and take risks rather than only doing what they already know how to do well. I have seen many successful people throughout my life and I have learned that they are not afraid to try new things and take risks. They know that if they are not successful at first, they can always try again and again until they are successful.One of the most successful people I have ever met was Steve Jobs. Steve Jobs was the co-founder of Apple and he was responsible for developing the Macintosh computer and the iPod. He was also responsible for creating the iPhone and the iPad. Steve Jobs was a risk taker and he knew that if he did not try new things and take risks, he would not be successful. He knew that if he was not successful at first, he could always try again and again until he was successful.",
    "I generally agree with this statement. Many advertisements make products seem much better than they really are. One reason is that advertisers often focus on the features of the product rather than the drawbacks. For example, an advertisement for a new car might focus on the features that make the car unique, such as the new engine or the sleek design. However, the advertisement might not mention the fact that the car is also very expensive and may not be suitable for everyone.",
    "I believe that in twenty years, there will be fewer cars in use than there are today. There are many reasons for this, the most significant of which is the increasing popularity of alternative transportation methods, such as bicycles and public transportation.Another reason is the increasing popularity of electric vehicles. These vehicles are not only more environmentally friendly, but they also require much less maintenance than traditional cars. In addition, they are much cheaper to operate than traditional cars."
]
''' 
sample_texts3_processed = [preprocess(text) for text in sample_texts3]
print(predict_texts(sample_texts3_processed))
print(predict_probability(sample_texts3_processed))
''' 


new_df = pd.DataFrame(sample_texts2, columns=["text"])
df_test_essays = pd.concat([df_test_essays, new_df], ignore_index=True)


ids = df_test_essays["id"]
texts = df_test_essays["text"].tolist()

# Encode texts with BERT
X_submission = bert_model.encode(texts)

# Predict probability that each essay was generated
probability = rf_bert.predict_proba(X_submission)[:, 1]

# Create and save submission DataFrame
submission_df = pd.DataFrame({
    "id": ids,
    "generated": probability
})

print(submission_df.head())
submission_df.to_csv("submission.csv", index=False)

