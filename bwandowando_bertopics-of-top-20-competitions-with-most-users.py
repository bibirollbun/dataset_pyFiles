%pip install -U plotly==6.1.2 datamapplot==0.5.1 bertopic umap-learn swifter langdetect lingua-language-detector kaleido -q 


from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")


!python -m spacy download en_core_web_lg


import gc
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import re
import swifter
import torch
import warnings


from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime
from hdbscan import HDBSCAN
from html import unescape
from IPython.display import clear_output
from langdetect import detect, DetectorFactory
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path
from plotly.graph_objs._figure import Figure
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from slugify import slugify
from spacy.lang.en import stop_words
from transformers import AutoTokenizer
from umap import UMAP


@dataclass
class CompetitionGraphs:
    topic_chart: Figure
    topics_over_time_chart: Figure
    topics_data_map_fig: Figure

@dataclass
class CompetitionBERTTopicInfo:
    competition_id:int
    forum_id: int
    competition_name: str
    topic_model: BERTopic    
    topics: list
    probs: np.ndarray
    competition_graphs: CompetitionGraphs
    
@dataclass
class EmbeddingModelSettings:
    batch_size: int = 24
    context_size: int = 2048
    padding:str = "longest"
    
@dataclass
class BERTTopicSettings:
    topic_count: str = "auto"
    top_n_words: int = 10
    topic_time_bins:int = 200


DetectorFactory.seed = 0
pd.set_option('display.html.use_mathjax', False)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pio.renderers.default = 'iframe' #https://www.kaggle.com/discussions/product-announcements/549950
warnings.filterwarnings("ignore")


CURRENT_YEAR_ONLY = False #this will only get this year's and last's data
DATASET_ROOT_DIRECTORY = "/kaggle/input"
MINIMUM_REWARD = 10000
MONTHS_INCLUDE_TIL_COMPETITION_END = 1
PARENT_OUTPUT_DIRECTORY = "/kaggle/working"
OUTPUT_DIRECTORY = "/{PARENT_OUTPUT_DIRECTORY}/{model_shard}/{time_shard}/{model_shard}"
RANK_CUTOFF = 20 #in my local this is 20
TITLE_ONLY = False
WHITESPACE_PATTERN = re.compile(r"\s+")

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
CONTEXT_SIZE=768
MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'


CURRENCY_SYMBOL_MAPPINGS = {"USD":"$" ,"EUR":"€" ,"GBP":"£"}

COMPETITION_COLUMNS = [
 'CompetitionId',
 'Title',
 'HostSegmentTitle',
 'ForumId',
 'RewardType',
 'RewardQuantity',
 'EnabledDate',
 'DeadlineDate',                       
 'TotalCompetitors',
]

REPLACEMENTS = {
    "cv": "cross-validation",
    "cross validation": "cross-validation",
    "data set": "dataset",
    "kernels": "notebooks",
    "kernel": "notebook",
    "lb": "leaderboard",
    "leader board": "leaderboard",
    "leader-board": "leaderboard",
    "lgb":"lightgbm",    
    "missing value": "missing values",
    "na values": "missing values",
    "null values": "missing values",
    "null value": "missing values",
    "nan values": "missing values",
    "nan value": "missing values",
    "post processing": "post-processing",    
    "pre trained": "pre-trained",
    "team mate": "teammate",
    "values missing": "missing values",        
    "xgb":"xgboost"
}

TOPIC_COLUMNS = ['ParentForumId',
 'ParentForumTitle',
 'ForumId',
 'ForumTitle',
 'CompetitionId',                 
 'ForumTopicId',
 'ForumMessageId',
 'Score',
 'CreationDate',
 'CreationDateYear',
 'ForumTopicTitle',
 'DisplayName',
 'Rank',
 'CleanedTopicAndMessage',
 'EnabledDate',                 
 'DeadlineDate'
]


time_shard = datetime.today().strftime("%Y%m%d")
model_shard = slugify(MODEL_NAME)


embedding_model_settings = EmbeddingModelSettings(context_size=CONTEXT_SIZE)
detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
model.max_seq_length = embedding_model_settings.context_size


def remove_carriage_returns(x):
    return WHITESPACE_PATTERN.sub(" ", x).strip()    
    
def unify_multiple_whitespaces(x):
    """replace multiple whitespaces with just one"""
    return re.sub(' {2,}', ' ', str(x))    

def clean_html(x):
    """Unescape string then remove html parts"""
    soup = BeautifulSoup(unescape(str(x)), 'lxml')
    return soup.text

def remove_urls(x):
    """remove urls from string"""
    cleaned_string = re.sub(r'http\S+', '<url>', str(x), flags=re.MULTILINE)
    return cleaned_string

def remove_repeating_non_alnum_chars(text):
    pattern = r'([^a-zA-Z0-9])\1{2,}'
    return re.sub(pattern, r'\1', text)

def replace_repeating_chars(text):
    pattern = r'(.)\1{4,}'
    return re.sub(pattern, r'\1\1\1\1', text)

def remove_emojis(data):
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U00002300-\U000023FF"  # Miscellaneous Technical
        u"\U0001F1E6-\U0001F1FF"  # Regional Indicator Symbols
        u"\U0000FE0F"             # Variation Selector-16
        u"\U0000200D"             # Zero Width Joiner
        "]+", flags=re.UNICODE)
    return re.sub(emoj, '', data)


def replace_words(text, replacements):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replacements.keys()) + r')\b', re.IGNORECASE)

    def replace_match(match):
        return replacements[match.group(0).lower()]
    
    return pattern.sub(replace_match, text)

def replace_ordinals(text):
    pattern = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)
    
    def replace_match(match):
        return "nth"
    
    # Perform the replacement
    return pattern.sub(replace_match, text)


def lingua_detect(x:str)->str:
    result = detector.detect_language_of(x)

    if(result!=None):
        return detector.detect_language_of(x).iso_code_639_1.name
    else:
        return "UNKNOWN"    
    
def langdetect_detect(x:str)->str:
    try:
        result = detect(x)

        if(result!=None):
            result = result.upper()

            if((result == "ZH-CN")|(result == "ZH-TW")):
                return "ZH"
            else:
                return result

        else:
            return "UNKNOWN"
    except:
        return "UNKNOWN"  


def get_token_count(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def cleanup():
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    print("Clean-up done")


def clean_text_field(df, field_name):
    df[f"Cleaned{field_name}"] = df[field_name].swifter.apply(clean_html)
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(remove_carriage_returns)
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(remove_urls)
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(remove_repeating_non_alnum_chars)
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(replace_repeating_chars)        
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(remove_emojis)
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(lambda x: replace_words(x, REPLACEMENTS))
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(replace_ordinals)
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].swifter.apply(unify_multiple_whitespaces)    
    df[f"Cleaned{field_name}"] = df[f"Cleaned{field_name}"].str.strip()
    return df


def shade_figure(fig, start_x, end_y, title, color="red", opacity=0.075):
    fig.add_vrect(x0=start_x
                  , x1=end_y
                  , annotation_text=title
                  , annotation_position="top left"
                  , annotation_textangle = 90                  
                  , annotation=dict(font_size=14, font_family="Courier New")
                  , fillcolor=color, opacity=opacity
                  , line_width=1)

    return fig


df_competitions = pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/meta-kaggle/Competitions.csv")
df_forum_topics = pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/meta-kaggle/ForumTopics.csv")
df_forum_messages = pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/meta-kaggle/ForumMessages.csv")
df_forums = pd.read_csv(f"{DATASET_ROOT_DIRECTORY}/meta-kaggle/Forums.csv")


df_competitions = df_competitions.rename(columns={"Id":"CompetitionId"})
df_forums = df_forums.rename(columns={"Id":"ForumId"})
df_forum_topics = df_forum_topics.rename(columns={"Id":"ForumTopicId","Title":"ForumTopicTitle" })
df_forum_messages = df_forum_messages.rename(columns={"Id":"ForumMessageId"})


df_competitions = df_competitions[COMPETITION_COLUMNS]
df_competitions = df_competitions[df_competitions["HostSegmentTitle"].isin(["Featured","Research"])]
df_competitions = df_competitions[df_competitions["RewardType"].isin(["USD","EUR","GBP"])]
df_competitions["RewardType"] = df_competitions["RewardType"].map(CURRENCY_SYMBOL_MAPPINGS)
df_competitions["ForumId"] = df_competitions["ForumId"].fillna(-1).astype("int")


current_datetime = pd.to_datetime(datetime.now())
df_competitions["EnabledDate"] = pd.to_datetime(df_competitions["EnabledDate"])
df_competitions["DeadlineDate"] = df_competitions["DeadlineDate"].fillna(current_datetime)
df_competitions["CutOffDate"] = pd.to_datetime(df_competitions["DeadlineDate"]) + pd.DateOffset(months=MONTHS_INCLUDE_TIL_COMPETITION_END)
df_competitions["EnabledDateYear"] = df_competitions["EnabledDate"].dt.year


df_competitions["Rank"] = df_competitions["TotalCompetitors"].rank(method="first", ascending=False).astype("int")
df_competitions = df_competitions[df_competitions["RewardQuantity"] >= MINIMUM_REWARD].reset_index(drop=True)
df_competitions = df_competitions[df_competitions["Rank"]  <= RANK_CUTOFF].reset_index(drop=True)
df_competitions = df_competitions[df_competitions["ForumId"]>0]
df_competitions["DisplayName"]  = "🏆 " + df_competitions["Title"] \
                                + "| 💵 " + df_competitions["RewardType"] + df_competitions["RewardQuantity"].astype("int").astype("str")  \
                                + "| 👥 " + df_competitions["TotalCompetitors"].astype("int").astype("str")  \
                                + "| 📅 " + df_competitions["EnabledDate"].dt.strftime("%b %Y")


df_parent_forum = df_forums.copy(deep=True)
df_parent_forum = df_parent_forum.rename(columns={"Title":"ParentForumTitle"})
df_parent_forum = df_parent_forum.drop(columns=["ParentForumId"])
df_parent_forum = df_parent_forum.rename(columns={"ForumId":"ParentForumId"})
df_parent_and_child_forum = pd.merge(df_parent_forum, df_forums, left_on=["ParentForumId"], right_on=["ParentForumId"], suffixes=("_parent", "_child"))
df_parent_and_child_forum = df_parent_and_child_forum.rename(columns={"Title":"ForumTitle"})
df_parent_and_child_forum = df_parent_and_child_forum.sort_values(["ParentForumId","ForumId"]).reset_index(drop=True)


df_forum_topics["CreationDateYear"] = pd.to_datetime(df_forum_topics["CreationDate"]).dt.year

if(CURRENT_YEAR_ONLY):
    previous_year = df_forum_topics["CreationDateYear"].max()
    df_forum_topics = df_forum_topics[df_forum_topics["CreationDateYear"] >= previous_year].reset_index(drop=True)    


df_forum_topics = df_forum_topics[df_forum_topics["ForumTopicTitle"].notnull()]
df_forum_topics = pd.merge(df_forum_topics, df_competitions, left_on = ["ForumId"], right_on = ["ForumId"])


df_forum_topic_messages = pd.merge(df_forum_topics, df_forum_messages , left_on=["ForumTopicId","FirstForumMessageId"], right_on=["ForumTopicId","ForumMessageId"], suffixes=("_topic","_message"))


df_forum_topic_messages = df_forum_topic_messages[df_forum_topic_messages["CreationDate"] <= df_forum_topic_messages["CutOffDate"]].reset_index(drop=True)


df_forum_topic_messages = pd.merge(df_parent_and_child_forum, df_forum_topic_messages, left_on = ["ForumId"], right_on = ["ForumId"])
df_forum_topic_messages = df_forum_topic_messages[df_forum_topic_messages["ParentForumTitle"].isin(["Active Competitions","Past Competitions"])].reset_index(drop=True)


df_forum_topics = None
df_forum_messages = None

del df_forum_topics
del df_forum_messages

cleanup()


if(TITLE_ONLY):
    df_forum_topic_messages["TopicAndMessage"] = df_forum_topic_messages["ForumTopicTitle"].fillna("").astype("str")
else:
    df_forum_topic_messages["TopicAndMessage"] = df_forum_topic_messages["ForumTopicTitle"].fillna("").astype("str") + ". " + df_forum_topic_messages["Message"].fillna("").astype("str")    


print(df_forum_topic_messages.shape[0])
df_forum_topic_messages["lingua_language"] = df_forum_topic_messages["TopicAndMessage"].swifter.apply(lingua_detect)
df_forum_topic_messages = df_forum_topic_messages[df_forum_topic_messages["lingua_language"] == "EN"]

df_forum_topic_messages["langdetect_language"] = df_forum_topic_messages["TopicAndMessage"].swifter.apply(langdetect_detect)
df_forum_topic_messages = df_forum_topic_messages[df_forum_topic_messages["langdetect_language"] == "EN"].reset_index(drop=True)
print(df_forum_topic_messages.shape[0])


df_forum_topic_messages = clean_text_field(df_forum_topic_messages,"TopicAndMessage")
df_forum_topic_messages = df_forum_topic_messages.drop(columns=["KernelId"], errors="ignore")
df_forum_topic_messages = df_forum_topic_messages.reset_index(drop=True)


df_forum_topic_messages = df_forum_topic_messages[TOPIC_COLUMNS]


df_parent_and_child_forum = None
df_combined = None

del df_parent_and_child_forum
del df_combined

cleanup()


df_forum_topic_messages["TokenCount"] = df_forum_topic_messages["CleanedTopicAndMessage"].swifter.apply(get_token_count)
print((df_forum_topic_messages[df_forum_topic_messages["TokenCount"] <= embedding_model_settings.context_size].shape[0] / df_forum_topic_messages.shape[0]) * 100)


df_forum_topic_messages.shape


df_forum_topic_messages['CleanedTopicAndMessage'] = df_forum_topic_messages['CleanedTopicAndMessage'].str.replace("<url>","")


df_forum_topic_messages.sample(1).reset_index(drop=True)


text_embeddings = model.encode(df_forum_topic_messages['CleanedTopicAndMessage']
                               , convert_to_tensor=True
                               , show_progress_bar=True
                               , batch_size=embedding_model_settings.batch_size
                               , normalize_embeddings=True
                               , max_length=embedding_model_settings.context_size
                               , truncation = True
                               , padding=embedding_model_settings.padding )


df_topics_embeddings_exploded = pd.DataFrame(list(text_embeddings.to('cpu').detach().numpy()), index= df_forum_topic_messages.index)
df_topics_embeddings_exploded.columns = [f"Vector_{x+1:04d}" for x in df_topics_embeddings_exploded.columns.tolist()]
df_topics_embeddings_exploded = pd.merge(df_forum_topic_messages[["ForumTopicId"]], df_topics_embeddings_exploded, left_index=True, right_index=True)


text_embeddings = None
del text_embeddings
cleanup()


session_output_directory = OUTPUT_DIRECTORY.format(PARENT_OUTPUT_DIRECTORY=PARENT_OUTPUT_DIRECTORY, model_shard=model_shard,time_shard=time_shard)

topic_model_output_directory = f"{session_output_directory}/topic_model"
tabular_data_output_directory = f"{session_output_directory}/tabular_data"

topic_directory = f"{PARENT_OUTPUT_DIRECTORY}/topic"
topic_over_time_directory = f"{PARENT_OUTPUT_DIRECTORY}/topic_over_time"
data_map_directory = f"{PARENT_OUTPUT_DIRECTORY}/data_map"

Path(topic_model_output_directory).mkdir(parents=True, exist_ok=True)
Path(tabular_data_output_directory).mkdir(parents=True, exist_ok=True)
Path(topic_directory).mkdir(parents=True, exist_ok=True)
Path(topic_over_time_directory).mkdir(parents=True, exist_ok=True)
Path(data_map_directory).mkdir(parents=True, exist_ok=True)


if(not CURRENT_YEAR_ONLY):
    df_forum_topic_messages = df_forum_topic_messages.sort_values(["ForumTopicId"]).reset_index(drop=True)
    df_topics_embeddings_exploded = df_topics_embeddings_exploded.sort_values(["ForumTopicId"]).reset_index(drop=True)
    
    df_forum_topic_messages.to_parquet(f"{tabular_data_output_directory}/df_forum_topic_message.parquet", index=False)
    df_topics_embeddings_exploded.to_parquet(f"{tabular_data_output_directory}/df_topics_embeddings_exploded.parquet", index=False)


if (not ('df_forum_topic_messages' in locals())|('df_forum_topic_messages' in globals())):
    print("Loading parquet from disk")
    df_forum_topic_messages = pd.read_parquet(f"{tabular_data_output_directory}/df_forum_topic_message.parquet")
    df_topics_embeddings_exploded= pd.read_parquet(f"{tabular_data_output_directory}/df_topics_embeddings_exploded.parquet")
else:
    print("Skipping loading from disk")


BERT_topic_settings = BERTTopicSettings()    

umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(ngram_range=(2, 2), stop_words=list(stop_words.STOP_WORDS), min_df=2)


keybert_model = KeyBERTInspired()
pos_model = PartOfSpeech("en_core_web_lg")
mmr_model = MaximalMarginalRelevance(diversity=0.3)

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
    "MMR": mmr_model,
    "POS": pos_model
}


cleanup()


df_forum_topic_messages = df_forum_topic_messages.sort_values(["Rank","ForumId","ForumTopicId","ForumMessageId"]).reset_index(drop=True)


competition_forum_id_collection = df_forum_topic_messages["ForumId"].unique().tolist()


def generate_bertopic_object():
    return BERTopic(
      # Pipeline models
      embedding_model=model,
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer_model,
      representation_model=representation_model,

      # Hyperparameters
      top_n_words=BERT_topic_settings.top_n_words,
      verbose=True,
      nr_topics=BERT_topic_settings.topic_count,
      calculate_probabilities = False,
      min_topic_size = 10
    )    


competition_BERTTopic_collection = []
for index, forum_id in enumerate(competition_forum_id_collection):
    print(f"{index+1} of {len(competition_forum_id_collection)}")
    topic_model = generate_bertopic_object()
    
    inner_messages = df_forum_topic_messages[df_forum_topic_messages["ForumId"] == forum_id]\
                                    .sort_values(["ForumTopicId", "ForumMessageId"]) \
                                    .reset_index(drop=True)
    
    inner_messages["EnabledDate"] = pd.to_datetime(inner_messages["EnabledDate"])
    inner_messages["DeadlineDate"] = pd.to_datetime(inner_messages["DeadlineDate"])
    inner_messages["CreationDate"] = pd.to_datetime(inner_messages["CreationDate"])
    
    
    inner_embeddings = df_topics_embeddings_exploded[df_topics_embeddings_exploded["ForumTopicId"].isin(inner_messages["ForumTopicId"])]\
                                    .sort_values(["ForumTopicId"]) \
                                    .reset_index(drop=True)
    
    competition_name = inner_messages[inner_messages["ForumId"]==forum_id].iloc[0]["DisplayName"]    
    competition_id = inner_messages[inner_messages["ForumId"]==forum_id].iloc[0]["CompetitionId"]
    
    enabled_date = inner_messages[inner_messages["ForumId"]==forum_id].iloc[0]["EnabledDate"]
    deadline_date = inner_messages[inner_messages["ForumId"]==forum_id].iloc[0]["DeadlineDate"]
    cut_off_date = inner_messages[inner_messages["ForumId"]==forum_id]["CreationDate"].max()
    
    topics, probs = topic_model.fit_transform(inner_messages["CleanedTopicAndMessage"] \
                                              , inner_embeddings[inner_embeddings.columns.tolist()[1:]].to_numpy())
    
    ## --------------------------------------
    topics_barchart_fig = topic_model.visualize_barchart()
    topics_barchart_fig.update_layout(title=competition_name)
    
    topics_data_map_fig = topic_model.visualize_document_datamap(inner_messages["CleanedTopicAndMessage"] \
                                              , embeddings=inner_embeddings[inner_embeddings.columns.tolist()[1:]].to_numpy() \
                                              , title=remove_emojis(competition_name))
    
    topics_over_time = topic_model.topics_over_time(inner_messages["CleanedTopicAndMessage"] \
                                                    , inner_messages["CreationDate"] \
                                                    , nr_bins=BERT_topic_settings.topic_time_bins if inner_messages["CreationDate"].nunique() >= BERT_topic_settings.topic_time_bins else inner_messages["CreationDate"].nunique())
    
    topics_over_time_fig = topic_model.visualize_topics_over_time(topics_over_time)
    topics_over_time_fig.update_layout(title=competition_name)
    
    topics_over_time_fig = shade_figure(topics_over_time_fig, enabled_date, deadline_date, "Competition Active", color="green") 
    topics_over_time_fig = shade_figure(topics_over_time_fig, deadline_date, cut_off_date, "Competition End", color="red") 
    
    
    ## --------------------------------------
    competition_graphs = CompetitionGraphs(
        topic_chart= topics_barchart_fig
        ,topics_over_time_chart = topics_over_time_fig
        ,topics_data_map_fig=topics_data_map_fig
    )
    
    ## --------------------------------------
    competition_bert_topic_info = CompetitionBERTTopicInfo(
        competition_id = competition_id
        , forum_id = forum_id
        , competition_name = competition_name
        , topics = topics
        , probs = probs
        , topic_model = topic_model
        , competition_graphs = competition_graphs
    )
    
    ## --------------------------------------
    competition_BERTTopic_collection.append(competition_bert_topic_info)    
    
    ## --------------------------------------
    print("*" * 100) 

    topics_barchart_fig = None
    topics_over_time_fig = None
    topics_data_map_fig = None
    
clear_output(wait=True)


df_topic_model_collection = []

for index, bertopic_instance in enumerate(competition_BERTTopic_collection):
    print("*" * 100) 
    print(f"Writing BERTOPIC #{index+1} plotly graphs into HTML")
    print(bertopic_instance.competition_name)
    fig = bertopic_instance.competition_graphs.topic_chart
    fig.update_layout(width=1500, height=750, autosize=True)
    fig.write_html(f"{topic_directory}/{index+1:02d}_{slugify(bertopic_instance.competition_name)}.html")
    
    fig = bertopic_instance.competition_graphs.topics_over_time_chart
    fig.update_layout(width=1500, height=750, autosize=True)
    fig.write_html(f"{topic_over_time_directory}/{index+1:02d}_{slugify(bertopic_instance.competition_name)}.html")
    
    fig = bertopic_instance.competition_graphs.topics_data_map_fig
    fig.savefig(f"{data_map_directory}/{index+1:02d}_{slugify(bertopic_instance.competition_name)}.png", bbox_inches="tight")
    
    
    df_topic_model = bertopic_instance.topic_model.get_topic_info()
    df_topic_model["CompetitionId"] = bertopic_instance.competition_id
    df_topic_model["ForumId"] = bertopic_instance.forum_id
    df_topic_model["CompetitionName"] = bertopic_instance.competition_name
    df_topic_model_collection.append(df_topic_model.copy(deep=True))
    
# combine the individual dataframes per competition    
df_combined = pd.concat(df_topic_model_collection)
columns = df_combined.columns.tolist()[-3:]
columns.extend(df_combined.columns.tolist()[:-3])
df_combined = df_combined[columns].sort_values(["CompetitionId","Topic"]).reset_index(drop=True)
df_combined.to_parquet(f"{tabular_data_output_directory}/df_competition_topics.parquet", index=False)


df_combined.sample(5).reset_index(drop=True)


if('iframe' != pio.renderers.default):
    for index, bertopic_instance in enumerate(competition_BERTTopic_collection):
        print("*" * 100) 
        print(f"Writing BERTOPIC #{index+1} plotly graphs into HTML")
        print(bertopic_instance.competition_name)
        fig = bertopic_instance.competition_graphs.topic_chart
        fig.update_layout(width=1500, height=750, autosize=True)
        fig.show()
else:
    print("Skipping printing images")


if('iframe' != pio.renderers.default):
    for index, bertopic_instance in enumerate(competition_BERTTopic_collection):
        print("*" * 100) 
        print(f"Writing BERTOPIC #{index+1} plotly graphs into HTML")
        print(bertopic_instance.competition_name)
        fig = bertopic_instance.competition_graphs.topics_over_time_chart
        fig.update_layout(width=1500, height=750, autosize=True)
        fig.show()
else:
    print("Skipping printing images")        

