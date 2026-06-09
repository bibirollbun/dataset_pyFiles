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


from sklearn.pipeline import Pipeline
import spacy


def load_models():    
    with open('/kaggle/input/predicter/tensorflow2/default/1/modelv2.pkl', 'rb') as model_file:
        model = joblib.load(model_file)

    # Carregar o vectorizer
    with open('/kaggle/input/predicter/tensorflow2/default/1/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = joblib.load(vectorizer_file)
    return model,vectorizer

def predict(text):
    model, vectorizer = load_models()

    X = vectorizer.transform([text])

    if hasattr(X, "toarray"):
        X = X.toarray()

    prediction = model.predict(X)
    return prediction


def remove_excessive_spaces(text: str) -> str:
    """
    This function removes excessive spaces from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with excessive spaces removed.
    """
    return re.sub(r'\s+', ' ', text).strip() 

def remove_repeated_non_word_characters(text: str) -> str:
    """
    This function removes repeated non-word characters from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with repeated non-word characters removed.
    """
    return re.sub(r'(\W)\1+', r'\1', text).strip()

def remove_first_line_from_text(text: str) -> str:
    """
    This function removes the first line from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with the first line removed.
    """
    return re.sub(r'^.*\n', '', text).strip()

def remove_last_line_from_text(text: str) -> str:
    """
    This function removes the last line from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with the last line removed.
    """
    return re.sub(r'\n.*$', '', text).strip()

def fix_isolated_commas_in_text(text: str) -> str:
    """
    This function fixes isolated commas in the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with isolated commas fixed.
    """
    text = re.sub(r' ([.,:;!?])', r'\1', text)
    return text.strip()

def keep_words_longer_than(text: str, min_length: int = 2) -> str:
    """
    This function keeps only the words in the text that are longer than a given length.

    Args:
        text (str): The input text.
        min_length (int, optional): The minimum length of the words to keep. Defaults to 2.

    Returns:
        str: The text with only the words longer than the given length.
    """
    return ' '.join([word for word in text.split() if len(word) > min_length])

def keep_only_alphabet_characters(text: str) -> str:
    """
    This function keeps only the alphabet characters in the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with only the alphabet characters.
    """
    return re.sub(r'[^a-zA-Z]', ' ', text).strip()

def remove_accents_from_text(text: str) -> str:
    """
    This function removes accents from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with accents removed.
    """
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def lemmatize_text_with_spacy(text: str) -> str:
    """
    This function lemmatizes the text using the Spacy library.

    Args:
        text (str): The input text.

    Returns:
        str: The lemmatized text.
    """
    nlp_spacy = spacy.load('en_core_web_sm')
    doc = nlp_spacy(text)
    return ' '.join([token.lemma_ for token in doc])


pipeline_clean_text = Pipeline([
    ('remove_first_line_from_text', FunctionTransformer(remove_first_line_from_text)),
    ('remove_last_line_from_text', FunctionTransformer(remove_last_line_from_text)),
    ('remove_excessive_spaces', FunctionTransformer(remove_excessive_spaces)),
    ('remove_repeated_non_word_characters', FunctionTransformer(remove_repeated_non_word_characters)),
    ('fix_isolated_commas_in_text', FunctionTransformer(fix_isolated_commas_in_text)),
])


# Cell to read your dataframe
# df = pd.read_csv('path')


# df['text'] = df['text'].apply(pipeline_clean_text.transform)


predictions = []
for text in df['text']:
    predictions.append(predict(text))


comparation_data = df.copy(deep=True)
comparision_data["predictions"] = [np.round(pred) for pred in predictions]


# comparation_data.to_csv('path')

