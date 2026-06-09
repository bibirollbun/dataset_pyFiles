
# remove warnings
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
# for printing HTML
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import seaborn as sns
!pip install distance


import zipfile
import pandas as pd

# Path to the zipped dataset
zip_file_path = '/kaggle/input/quora-question-pairs/train.csv.zip'

# Unzipping the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('/mnt/data')  # Extracting the contents

# Load the unzipped CSV file into a DataFrame
csv_file_path = '/mnt/data/train.csv'
df = pd.read_csv(csv_file_path).sample(20000, random_state=1)



# Display the first few rows of the dataframe
df.head(10)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from sklearn.preprocessing import LabelEncoder
import missingno as msno

def styled_heading(text, background_color='#14adc6', text_color='white'):
    """Generate an HTML string for a styled heading."""
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        font-family: 'Montserrat', sans-serif;
        color: {text_color};
        padding: 15px;
        font-size: 30px;
        font-weight: bold;
        line-height: 1;
        border-radius: 20px 20px 0 0;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px dashed {background_color};
    ">
        {text}
    </div>
    """


def style_table(dff):
    """Style a DataFrame as HTML."""
    return dff.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "purple")]}
    ]).set_properties(**{"text-align": "center"}).hide(axis="index").to_html()



def dataset_analysis(df, head_color='blue', text_color='white'):
    """Analyze the dataset by displaying the top rows, summary, info, and visualizations."""
    try:
        # Display the top 5 rows of the dataset
        display(HTML(styled_heading('Top 5 rows of the dataset', head_color, text_color)))
        display(HTML(style_table(df.head(5))))

        # Display the statistical summary of the dataset
        display(HTML(styled_heading('Statistical Summary of dataset', head_color, text_color)))
        display(HTML(style_table(df.describe())))

        # Display dataset info
        display(HTML(styled_heading('Dataset Info', head_color, text_color)))
        df.info()

        # Display number of rows and columns
        display(HTML(styled_heading('No of rows and columns', head_color, text_color)))
        categorical_count = df.select_dtypes(include='object').shape[1]
        numerical_count = df.select_dtypes(exclude='object').shape[1]
        print(f'There are {categorical_count} categorical columns and {numerical_count} numerical columns.')

        # Calculate missing percentages
        missing_percentages = df.isnull().sum() / len(df) * 100
        missing_percentages = missing_percentages.apply(lambda x: f'{x:.2f}%')

        # Plot and display null values heatmap
        display(HTML(styled_heading('Plotting Null values', head_color, text_color)))
        plt.figure(figsize=(18, 9))
        sns.heatmap(df.isnull(), cmap='RdBu', yticklabels=False)
        plt.title("Heatmap of Missing Values")
        plt.show()

        # Plotting Non-Null values
        display(HTML(styled_heading('Plotting the Non-Null values', head_color, text_color)))
        msno.bar(df)
        plt.title("Non-Null Values Bar Chart")
        plt.show()

        # Plot and display the correlation matrix
        display(HTML(styled_heading('Plotting Correlation matrix', head_color, text_color)))

        # Ensure uniform types within columns before encoding
        df_cor = df.copy()
        for column in df_cor.columns:
            if df_cor[column].dtype == 'object':
                df_cor[column] = df_cor[column].astype(str)

        # Label encode categorical columns
        for column in df_cor.columns:
            if df_cor[column].dtype == 'object':
                df_cor[column] = LabelEncoder().fit_transform(df_cor[column])

        plt.figure(figsize=(18, 9), facecolor='none')
        cmap = sns.diverging_palette(230, 30, as_cmap=True)

        sns.heatmap(
            df_cor.corr(),
            annot=True,
            cmap=cmap,
            square=False,
            linewidths=.9,
            fmt='.2f',
            annot_kws={"size": 12},
        )
        plt.title("Correlation Matrix Heatmap")
        plt.show()

        # Optional: Drop rows with missing values in the 'status' column if applicable
        # df.dropna(subset=['status'], inplace=True)

        df.columns = df.columns.str.strip()
    
    except Exception as e:
        print(f"An error occurred during dataset analysis: {e}")

# Example usage:
# df = pd.read_csv('your_dataset.csv')
dataset_analysis(df, 'blue', 'white')



df_dummy = df.copy()
df_dummy['is_duplicate'] = df_dummy['is_duplicate'].map({1: 'Duplicated', 0: 'Not_duplicated'})
duplics = df_dummy['is_duplicate'].value_counts()
fig = px.pie(
    df,
    values=duplics.values,
    names=duplics.index,
    title='No Of Duplicate Questions',
    color_discrete_sequence=px.colors.sequential.Purp
)

# Update trace properties
fig.update_traces(
    textposition='outside',  # Place labels outside the pie slices
    textinfo='label+value',  # Display both continent name and percentage value on labels
    pull=[0.1, 0]  # Separate certain slices outwards (first continent by 0.1)
)

# Update layout properties
fig.update_layout(
    height=600,  # Set chart height in pixels
    width=1100,   # Set chart width in pixels
    plot_bgcolor='#111',  # Set the background color of the plot area
    paper_bgcolor='#111',  # Set the background color of the entire plot
    font_color='white',  # Set the font color
)

# Display the chart
fig.show()


import pandas as pd

# Assuming df is your DataFrame containing 'qid1' and 'qid2'
# Sample DataFrame for demonstration (uncomment and modify accordingly)
# df = pd.DataFrame({
#     'qid1': [...],
#     'qid2': [...]
# })

# Combine 'qid1' and 'qid2' into a single Series for analysis
qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())

# Count occurrences of each unique question ID
a = qid.value_counts()

# Initialize counters for duplicated and unique question IDs
duplicated = 0
unique = 0

# Iterate through the counts of each question ID
for question_id, count in a.items():
    if count > 1:
        duplicated += 1  # Increment duplicated count if more than one occurrence
    else:
        unique += 1  # Increment unique count if exactly one occurrence

# Calculate the total number of questions (both duplicated and unique)
total_questions = duplicated + unique

# Output the results with detailed messages
print(f'Total questions in the dataset are --> {total_questions}')
print(f'Number of duplicated questions in the dataset are --> {duplicated}')
print(f'Number of unique questions in the dataset are --> {unique}')

# Additional output to clarify the types of questions found
print("\nDetailed Breakdown:")
print(f"{'Question Type':<20} | {'Count':<10}")
print("-" * 30)
print(f"{'Duplicated Questions':<20} | {duplicated:<10}")
print(f"{'Unique Questions':<20} | {unique:<10}")
print(f"{'Total Questions':<20} | {total_questions:<10}")



df.dropna(subset=['question1', 'question2'], inplace=True)


b = 0
for i in df['question1']:
    print(i)
    b += 1
    if b == 3:
        break 





import re
from bs4 import BeautifulSoup

def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q
    
df['question1'] = df['question1'].apply(preprocess)
df['question2'] = df['question2'].apply(preprocess)





len1 = []
for question in df['question1']:
    length = len(question)
    len1.append(length)
df['question1_length'] = len1



len2 = []
for i in df['question2']:
    length = len(i)
    len2.append(length)
df['question2_length'] = len2



# Calculate no of words on question1 and making a new column based on this
question1words = []
for i in df['question1']:
    words = 0
    for j in i.split():
        words += 1
    question1words.append(words)
df['Question1 Words'] = question1words



# Calculate no of words on question2 and making a new column based on this
question2words = []
for i in df['question2']:
    words = 0
    for j in i.split():
        words += 1
    question2words.append(words)
        
df['Question2 Words'] = question2words



# Total Words in question 1 and question 2
df['Total words in question1&2'] = df['Question2 Words'] + df['Question1 Words']



# Sum of unique words in Question1 and Question2
unique_list = []
for question1, question2 in zip(df['question1'], df['question2']):
    words_a = set(question1.lower().split())
    words_b = set(question2.lower().split())
    unique_set = words_a.union(words_b)
#     unique_a = len(set(words_a))
#     unique_b = len(set(words_b)) 
#     uniuqe = words_a + words_b
    unique_list.append(len(unique_set))
df['unique words in Q1&Q2'] = unique_list



# Advanced Features
from nltk.corpus import stopwords

def fetch_token_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features


token_features = df.apply(fetch_token_features, axis=1)

df["cwc_min"]       = list(map(lambda x: x[0], token_features))
df["cwc_max"]       = list(map(lambda x: x[1], token_features))
df["csc_min"]       = list(map(lambda x: x[2], token_features))
df["csc_max"]       = list(map(lambda x: x[3], token_features))
df["ctc_min"]       = list(map(lambda x: x[4], token_features))
df["ctc_max"]       = list(map(lambda x: x[5], token_features))
df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
df["first_word_eq"] = list(map(lambda x: x[7], token_features))



import distance

def fetch_length_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0] * 3
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    
    strs = list(distance.lcsubstrings(q1, q2))
    if strs:  # Check if strs is not empty
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0.0  # Set to 0 if there are no common substrings
    
    return length_features


length_features = df.apply(fetch_length_features, axis=1)

df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
df['mean_len'] = list(map(lambda x: x[1], length_features))
df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))



final_df = df.drop(columns=['qid1', 'qid2', 'question1', 'question2', 'id'])


ques_df = df[['question1','question2']]


# from sklearn.feature_extraction.text import CountVectorizer
# questions = list(ques_df['question1']) + list(ques_df['question2'])
# cv = CountVectorizer(max_features=3000)
# q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)



# temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
# temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
# temp_df = pd.concat([temp_df1, temp_df2], axis=1)



# new_df = pd.concat([temp_df, temp_df], axis=1)
# new_df.head()




# new_df['is_duplicate'] = df['is_duplicate']


# X = new_df.drop(columns=['is_duplicate'], axis=1)
# y = new_df['is_duplicate']
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)




# from xgboost import XGBClassifier
# xgb = XGBClassifier()
# xgb.fit(X_train,y_train)
# y_pred1 = xgb.predict(X_test)
# accuracy_score(y_test,y_pred1)


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Assuming ques_df is already defined with 'question1' and 'question2'
questions = list(ques_df['question1']) + list(ques_df['question2'])
cv = CountVectorizer(max_features=3000)

# Fit and transform questions
q_arr = cv.fit_transform(questions).toarray()

# Split the array into two parts for question1 and question2
q1_arr, q2_arr = np.vsplit(q_arr, 2)

# Create DataFrames for both question arrays
temp_df1 = pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index=ques_df.index)

# Concatenate without duplicating features
new_df = pd.concat([temp_df1.add_prefix('q1_'), temp_df2.add_prefix('q2_')], axis=1)

# Add the target variable
new_df['is_duplicate'] = df['is_duplicate']

X = new_df.drop(columns=['is_duplicate'], axis=1)
y = new_df['is_duplicate']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)

import optuna
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Define the objective function for hyperparameter optimization
def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
    }

    xgb = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Run the Optuna study to find the best hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)





# Print the best hyperparameters found
print("Best hyperparameters: ", study.best_params)

# Train the XGBClassifier with the best found hyperparameters
best_params = study.best_params
xgb_best = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
xgb_best.fit(X_train, y_train)

# Make predictions and evaluate the accuracy
y_pred_best = xgb_best.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

print("Accuracy with best hyperparameters:", accuracy_best)


