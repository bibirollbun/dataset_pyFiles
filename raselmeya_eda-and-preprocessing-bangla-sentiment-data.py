import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from plotly import graph_objs as go


bn_sentiment_data= pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")


# Check data
bn_sentiment_data.head()


# Check the shape of the DataFrame
print("Shape of the DataFrame:", bn_sentiment_data.shape)




# Examine the data types of each column
print("\nData Types of each column:\n", bn_sentiment_data.dtypes)




# Generate descriptive statistics for numerical features
print("\nDescriptive statistics for numerical features:\n", bn_sentiment_data.describe())




# Analyze categorical features
for col in ['sentiment']:  # Only 'sentiment' is categorical
    print(f"\nValue counts for '{col}':\n", bn_sentiment_data[col].value_counts())
    print(f"\nUnique values for '{col}':\n", bn_sentiment_data[col].unique())


# Count missing values in each column
missing_values = bn_sentiment_data.isna().sum()

# Display the results
print("Missing values per column:\n", missing_values)


# Remove rows with any NaN values
bn_sentiment_data = bn_sentiment_data.dropna()


# Drop irreverent Columns
bn_sentiment_data= bn_sentiment_data.drop(columns=['id'])

# check 
bn_sentiment_data.head()


print("Sentiments:")
print(bn_sentiment_data.sentiment.unique())
sentiment_list=bn_sentiment_data.sentiment.unique()



# Plot the distribution of sentiment classes
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=bn_sentiment_data, palette='viridis')
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()



# Plot a pie chart for sentiment class proportions
sentiment_counts = bn_sentiment_data['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2', len(sentiment_counts)), startangle=90)
plt.title('Sentiment Class Proportions')
plt.ylabel('')  # Hide the y-axis label
plt.show()



# Add a new column 'text_length' for the length of each text
bn_sentiment_data['text_length'] = bn_sentiment_data['text'].apply(len)

# Boxplot to show text length distribution by sentiment
plt.figure(figsize=(8, 6))
sns.boxplot(x='sentiment', y='text_length', data=bn_sentiment_data, palette='Set3')
plt.title('Text Length Distribution by Sentiment Class')
plt.xlabel('Sentiment')
plt.ylabel('Text Length')
plt.show()



# Add a new column 'word_count' for the word count of each text
bn_sentiment_data['word_count'] = bn_sentiment_data['text'].apply(lambda x: len(x.split()))

# Boxplot to show word count distribution by sentiment
plt.figure(figsize=(8, 6))
sns.boxplot(x='sentiment', y='word_count', data=bn_sentiment_data, palette='Set2')
plt.title('Word Count Distribution by Sentiment Class')
plt.xlabel('Sentiment')
plt.ylabel('Word Count')
plt.show()



def text_to_word_list(text):
    text = text.split()
    return text

def replace_strings(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\u00C0-\u017F"          #latin
                           u"\u2000-\u206F"          #generalPunctuations
                               
                           "]+", flags=re.UNICODE)
    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)
    #latin_pattern=re.compile('[A-Za-z\u00C0-\u00D6\u00D8-\u00f6\u00f8-\u00ff\s]*',)
    
    text=emoji_pattern.sub(r'', text)
    text=english_pattern.sub(r'', text)

    return text

def remove_punctuations(my_str):
    # define punctuation
    punctuations = '''````Â£|Â¢|Ã‘+-*/=EROeroà§³à§¦à§§à§¨à§©à§ªà§«à§¬à§­à§®à§¯012â€“34567â€¢89à¥¤!()-[]{};:'"â€œ\â€™,<>./?@#$%^&*_~â€˜â€”à¥¥â€�â€°ğŸ¤£âš½ï¸�âœŒï¿½ï¿°à§·ï¿°'''
    
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char

    # display the unpunctuated string
    return no_punct



def joining(text):
    out=' '.join(text)
    return out

def preprocessing(text):
    out=remove_punctuations(replace_strings(text))
    return out



bn_sentiment_data['cleanText'] = bn_sentiment_data.text.apply(lambda x: preprocessing(str(x)))


bn_stop_word =pd.read_excel('/kaggle/input/bangla-stopwords/stopwords_bangla.xlsx')
stopwords = bn_stop_word['words'].tolist()


display(bn_stop_word)


def stopwordRemoval(text):    
    x=str(text)
    l=x.split()

    stm=[elem for elem in l if elem not in stopwords]
    
    out=' '.join(stm)
    
    return str(out)



bn_sentiment_data['cleanText'] = bn_sentiment_data.cleanText.apply(lambda x: stopwordRemoval(str(x)))


#make sure to turn on internet on your kernel
#importing stemmer
!pip install bangla-stemmer
from bangla_stemmer.stemmer import stemmer
## stemmer function
def stemText (x):
  stmr = stemmer.BanglaStemmer()
  words=x.split(' ')
  stm = stmr.stem(words)
  words=(' ').join(stm)
  return words


bn_sentiment_data['cleanText'] = bn_sentiment_data.cleanText.apply(lambda x: stemText(str(x)))


display(bn_sentiment_data)



count_text = bn_sentiment_data.groupby('sentiment').count()['cleanText'].reset_index().sort_values(by='cleanText',ascending=False)
count_text.style.background_gradient(cmap='Purples')


fig = go.Figure(go.Funnelarea(
    text =count_text.sentiment,
    values = count_text.cleanText,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution on Train Set"}
    ))
fig.show()


# Assuming sentiment_list is defined and contains the sentiment classes
for i in sentiment_list:
    temp = bn_sentiment_data.loc[bn_sentiment_data['sentiment'] == str(i)].copy() 
    temp['temp_list'] = temp['cleanText'].apply(lambda x: str(x).split())
    
    # Get the most common words
    top = Counter([item for sublist in temp['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words', 'count']
    
    # Display the result with styling
    temp.style.background_gradient(cmap='Blues')
    temp = temp.style.set_caption('Top 20 Words In ' + str(i) + " Sentiment")
    display(temp)



bn_sentiment_data['temp_list'] = bn_sentiment_data['cleanText'].apply(lambda x:str(x).split())
top = Counter([item for sublist in bn_sentiment_data['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')



# Create figure with title
fig = go.Figure(layout=dict(title=dict(text="Text Length Histogram of Trainset")))

# Add histogram trace
fig.add_trace(go.Histogram(x=bn_sentiment_data['text_length']))

# Update layout to add x and y labels
fig.update_layout(
    xaxis_title="Text Length",  # X-axis label
    yaxis_title="Frequency",    # Y-axis label
)

# Show the plot
fig.show()


fig = go.Figure(layout=dict(title=dict(text="Text Length Histogram of Trainset")))
fig.add_trace(go.Histogram(x=bn_sentiment_data['word_count']))
# Update layout to add x and y labels
fig.update_layout(
    xaxis_title="Word Count",  # X-axis label
    yaxis_title="Frequency",    # Y-axis label
)

# Show the plot
fig.show()


import nltk
from nltk.util import ngrams


# Function to create top 20 n-grams
def get_ngrams(data, n):
    all_words = []
    for i in range(len(data)):
        temp = data["cleanText"][i].split()
        for word in temp:
            all_words.append(word)

    # Create n-grams
    tokenized = all_words
    esBigrams = ngrams(tokenized, n)

    esBigram_wordlist = nltk.FreqDist(esBigrams)
    top20 = esBigram_wordlist.most_common(20)
    top20 = dict(top20)
    df_ngrams = pd.DataFrame(sorted(top20.items(), key=lambda x: x[1], reverse=True))
    df_ngrams.columns = ['Ngram', 'count']
    return df_ngrams


# Function to visualize the top 20 n-grams
def show(train):
    display(train.head(20))




for i in sentiment_list:
    # Using .copy() to avoid SettingWithCopyWarning
    temp = bn_sentiment_data.loc[bn_sentiment_data['sentiment'] == str(i)].copy()
    
    temp['temp_list'] = temp['cleanText'].apply(lambda x: str(x).split())
    temp.reset_index(drop=True, inplace=True)

    # Get unigrams
    train_unigrams = get_ngrams(temp, 1)
    print("\t\t\t====== Unigrams of " + str(i) + "======")   
    show(train_unigrams)



for i in sentiment_list:
    # Using .copy() to avoid SettingWithCopyWarning
    temp = bn_sentiment_data.loc[bn_sentiment_data['sentiment'] == str(i)].copy()
    
    temp['temp_list'] = temp['cleanText'].apply(lambda x: str(x).split())
    temp.reset_index(drop=True, inplace=True)

    # Get unigrams
    train_unigrams = get_ngrams(temp, 2)
    print("\t\t\t====== Unigrams of " + str(i) + "======")   
    show(train_unigrams)



# Importing wordcloud for plotting word clouds and textwrap for wrapping longer text
from wordcloud import WordCloud
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Function for generating word clouds
def generate_wordcloud(data,title):
  data = [tuple(x) for x in data.values]
  wc = WordCloud(font_path="/kaggle/input/siyam-rupali-text/Siyamrupali.ttf",width=1080, height=720, max_words=150,colormap="Dark2").generate_from_frequencies(dict(data))
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap("Word Cloud of "+title,60)),fontsize=13)
  plt.show()



for i in sentiment_list:
    temp=bn_sentiment_data.loc[bn_sentiment_data['sentiment'] == str(i)]
    #display(temp)
    temp['temp_list'] = temp['text'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in temp['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(100))
    temp.columns = ['Common_words','count']
    generate_wordcloud(temp,str(i))
    




