import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


score_path = "/kaggle/input/you-cant-please-them-all-api-essays/score.csv"
score_df = pd.read_csv(score_path)


score_df.head()


avg_q_by_mood = score_df.groupby('mood')['avg_q'].mean()
avg_q_by_mood.plot(kind='bar', title='avg_q by Emotion')
plt.xlabel('Emotion')
plt.ylabel('avg_q')
plt.xticks(rotation=45)
plt.show()


avg_q_by_mood = score_df.groupby('style')['avg_q'].mean()
avg_q_by_mood.plot(kind='bar', title='avg_q by Style')
plt.xlabel('Style')
plt.ylabel('avg_q')
plt.xticks(rotation=45)
plt.show()


avg_q_by_mood = score_df.groupby('profession')['avg_q'].mean()
avg_q_by_mood.plot(kind='bar', title='avg_q by Profession')
plt.xlabel('Profession')
plt.ylabel('avg_q')
plt.xticks(rotation=45)
plt.show()


avg_q_by_mood = score_df.groupby('mood')['avg_variance'].mean()
avg_q_by_mood.plot(kind='bar', title='avg_variance by Emotion', color='red')
plt.xlabel('Emotion')
plt.ylabel('avg_variance')
plt.xticks(rotation=45)
plt.show()


avg_q_by_mood = score_df.groupby('style')['avg_variance'].mean()
avg_q_by_mood.plot(kind='bar', title='avg_variance by Style',color='red')
plt.xlabel('Style')
plt.ylabel('avg_variance')
plt.xticks(rotation=45)
plt.show()


avg_q_by_mood = score_df.groupby('profession')['avg_variance'].mean()
avg_q_by_mood.plot(kind='bar', title='avg_variance by Profession',color='red')
plt.xlabel('Profession')
plt.ylabel('avg_variance')
plt.xticks(rotation=45)
plt.show()


grouped_df = score_df.groupby(['style', 'profession', 'mood'])[['llama-3.2-3b_score', 'qwen-2.5-3b_score', 'gemma-2-2b_score']].mean().reset_index()

melted_df = pd.melt(grouped_df, id_vars=['style', 'profession', 'mood'], 
                    value_vars=['llama-3.2-3b_score', 'qwen-2.5-3b_score', 'gemma-2-2b_score'],
                    var_name='model', value_name='score')

plt.figure(figsize=(12, 6))
sns.barplot(data=melted_df, x='mood', y='score', hue='model', errorbar=None)
plt.title('Emotion Preference')
plt.ylabel('Score')
plt.xlabel('Emotion')
plt.legend(title='model')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(data=melted_df, x='style', y='score', hue='model', errorbar=None)
plt.title('Style Preference')
plt.ylabel('Score')
plt.xlabel('Style')
plt.legend(title='model')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(data=melted_df, x='profession', y='score', hue='model', errorbar=None)
plt.title('Profession Preference')
plt.ylabel('Score')
plt.xlabel('Profession')
plt.legend(title='model')
plt.xticks(rotation=45)
plt.show()

