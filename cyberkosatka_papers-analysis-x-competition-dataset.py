!pip install -q requests beautifulsoup4 pandas


import kagglehub
import more_itertools
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns


COLORS = {
    "good": "#66DE93",
    "bad": "#FF616D",
    "neutral": "#FFEAC9",
    "bg":      "#F5F5F5",
    "law": "#808080",     
    "ad": "#9147FF"  
}

rule_palette = {1: COLORS["bad"], 
                0: COLORS["good"],
                'No legal advice: Do not offer or request legal advice.': COLORS["law"], 
                'No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.': COLORS["ad"]}


train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')


print(f'The train dataset contains {len(train_data)} rows')


train_data.groupby(['rule']).size().reset_index(name='counts')


train_data.groupby(['rule', 'rule_violation']).size().reset_index(name='counts')


train_data['subreddit'].nunique()


cluster_subreddits = {
    'C0': ['conspiracy', 'Android', 'atheism', 'Incels', 'PurplePillDebate', 'IAmA', 'canada', 'tifu', 'india', 'SubredditDrama',
           'dataisbeautiful', 'pics', 'LifeProTips', 'hiphopheads', 'fantasyfootball', 'explainlikeimfive', 'worldnews', 'SandersForPresident'],
    'C1': ['CanadaPolitics', 'spacex', 'changemyview', 'NeutralPolitics', 'personalfinance', 'AskHistorians', 'history', 'whatisthisthing',
           'science', 'Games', 'philosophy', 'space', 'Futurology', 'syriancivilwar', 'legaladvice', 'PoliticalDiscussion', 'AskTrumpSupporters',
           'TheSilphRoad', 'Christianity', 'DIY', 'OutOfTheLoop', 'UpliftingNews'],
    'C2': ['DestinyTheGame', 'hearthstone', 'Overwatch', 'jailbreak', '2007scape', 'wow'],
    'C3': ['CFB', 'me_irl', 'books', 'movies', 'nba', 'nfl', 'asoiaf', 'pokemon', 'MMA', 'relationships', 'AskWomen', 'food', 'pcmasterrace',
           'Showerthoughts', 'GlobalOffensiveTrade', 'pokemongo', 'leagueoflegends', 'depression', 'gonewild', 'hillaryclinton', 'SuicideWatch',
           'The_Donald', 'gaming', 'GlobalOffensive', 'anime', 'politics', 'photoshopbattles', 'television', 'ShitRedditSays', 'GetMotivated',
           'aww', 'EnoughTrumpSpam', 'sex', 'gameofthrones', 'TwoXChromosomes', 'funny', 'nottheonion', 'europe', 'LateStageCapitalism', 'news',
           'technology', 'soccerstreams', 'socialism'],
    'C4': ['churning', 'NSFW_GIF', 'pokemontrades', 'nosleep'],
    'C5': ['videos', 'OldSchoolCool', 'gifs'],
    'C6': ['AskReddit'],
    'C7': ['BlackPeopleTwitter'],
    'C8': ['askscience'],
    'C9': ['creepyPMs']
}


print(train_data['subreddit'].unique())


def assign_cluster(subreddit):
    for cluster, subreddits in cluster_subreddits.items():
        if subreddit in subreddits:
            return cluster
    return None 

train_data['cluster'] = train_data['subreddit'].apply(assign_cluster)



train_data.head(5)


cluster_order = list(cluster_subreddits.keys())

plt.figure(figsize=(12, 8))
sns.countplot(data=train_data, x='cluster', hue='rule', palette=rule_palette, order=cluster_order)
plt.title("Subreddits splitted by different rule violations")
plt.xlabel('Subreddit')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))

sns.countplot(data=train_data, x='cluster', hue='rule_violation', palette=rule_palette, order=cluster_order)

plt.title("Top 20 Most Popular Subreddits with Rule Violation Proportions")
plt.xlabel('Subreddit')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Rule Violation', loc='upper right', labels=['No Violation (0)', 'Violation (1)'])
plt.tight_layout()
plt.show()



top_subreddits = train_data['subreddit'].value_counts().head(20).index


plt.figure(figsize=(12, 8))
sns.countplot(data=train_data, x='subreddit', hue='rule', order=top_subreddits, palette=rule_palette)
plt.title("Subreddits splitted by different rule violations")
plt.xlabel('Subreddit')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

