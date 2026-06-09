import warnings
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


train.head()


test.head()


"Train Shapes", train.shape, "\n Test Shapes", test.shape


targets = ["Category", "Misconception"]


list(train['Category'].unique())


train['is_correct'] = train['Category'].str.startswith('True')


plt.figure(figsize=(8, 4))
sns.heatmap(train.isnull(), cbar=False, cmap="YlGnBu")
plt.title("75% of Misconceptions are empty")
plt.show()
train['Misconception'] = train['Misconception'].fillna('NA')


plt.figure(figsize=(10, 5))
sns.countplot(data=train, x='Category', order=train['Category'].value_counts().index)
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=train, x='is_correct', order=train['is_correct'].value_counts().index)
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train[train['Misconception'] != 'NA'], x='Misconception', order=train['Misconception'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title("Our first 3 predictions?")
plt.show()


train['question_len'] = train['QuestionText'].apply(lambda x: len(x.split(" ")))
train['explanation_len'] = train['StudentExplanation'].apply(lambda x: len(x.split(" ")))
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(train['question_len'], bins=30, ax=axs[0], kde=True)
axs[0].set_title("15 Questions - Max 45 words")
sns.histplot(train['explanation_len'], bins=30, ax=axs[1], kde=True, color="orange")
axs[1].set_title("Explanations - < Max 350 words")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train, x='Category', y='explanation_len')
plt.title("Explanations length don't impact the Category")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train, x='is_correct', y='explanation_len')
plt.title("Explanations length don't impact the Correct/False")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train, x='Misconception', y='explanation_len')
plt.xticks(rotation=90)
plt.title("Explanations length don't impact the Category")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train[(train['Misconception']!='NA')&(train['explanation_len']<=75)], x='Misconception', y='explanation_len')
plt.xticks(rotation=90)
plt.title("Explanations length can be used for Misconception")
plt.show()


train['Category']


categories = list(train['Category'].unique())


for category in categories:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train[(train['explanation_len']<=75)&(train['Category']==category)], x='Misconception', y='explanation_len')
    plt.xticks(rotation=90)
    plt.title(f"{category.upper()}")
    plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(data=train[(train['explanation_len']>=75)], x='Misconception', y='explanation_len')
plt.xticks(rotation=5)
plt.title("Outliers? is questions need short explanations enough?")
plt.show()


combinations = {}
for category, possible_submissions in train[['Category','Misconception']].groupby(['Category']).agg(set).to_dict()['Misconception'].items():
    print(category, possible_submissions, len(possible_submissions))
    combinations[category] = possible_submissions


options = list(dict(train['Misconception'].value_counts()).keys())
str(options)


def calculate_map_at_3(true_labels, predictions):
    merged = true_labels.merge(predictions, on='row_id', how='inner')
    merged['true_label'] = merged.apply(lambda row: f"{row['Category']}:{row['Misconception']}", axis=1)
    ap_scores = []
    
    for _, row in merged.iterrows():
        true_label = row['true_label']
        pred_labels = row['Category:Misconception'].split()[:3]
        correct = 0
        ap = 0.0
        seen_correct = False
        
        for k, pred in enumerate(pred_labels, 1):
            if pred == true_label and not seen_correct:
                correct += 1
                precision = correct / k
                ap += precision
                seen_correct = True
        ap_scores.append(ap if correct > 0 else 0.0)
    
    return np.mean(ap_scores)

train['Category:Misconception'] = 'True_Correct:NA False_Neither:NA False_Misconception:Incomplete'
map_score = calculate_map_at_3(
    true_labels=train[['row_id', 'Category', 'Misconception']],
    predictions=train[['row_id', 'Category:Misconception']]
)


print(f"MAP@3 Score: {map_score:.4f} -- Close to LB sample submisison")


best_guess_options = ["True_Correct:NA","False_Correct:NA","True_Neither:NA","False_Neither:NA"] + [f'True_Misconception:{x}' for x in options[0:4]] + [f'False_Misconception:{x}' for x in options[0:4]]
len(best_guess_options)


from tqdm import tqdm
submissions = list(itertools.combinations(best_guess_options, 3))
scores = []
for submisison in tqdm(submissions, total=len(submissions)):
    value = ' '.join(submisison)
    train['Category:Misconception'] = value
    map_score = calculate_map_at_3(
        true_labels=train[['row_id', 'Category', 'Misconception']],
        predictions=train[['row_id', 'Category:Misconception']]
    )
    scores.append({"pred": value, "map_score": map_score})
    print(map_score, value)


best_score = pd.DataFrame(scores)
best_score = best_score.sort_values(['map_score'], ascending=False).reset_index(drop=True)
best_score


best_score['cv_map_score'] = best_score['map_score'].apply(lambda x: int(x*1000)/1000)
best_score['difference'] = best_score['cv_map_score'].diff().abs().fillna(1)
selected_best_score = best_score[best_score['difference'] >= 0.001].reset_index(drop=True)[['pred','map_score','cv_map_score']][0:50]

selected_best_score.sort_values(['map_score']).reset_index(drop=True)


selected_best_score.to_csv("submissions_cv_scores.csv", index=False)


value = best_score.loc[1, 'pred']
value


sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = value
sub


sub.to_csv("submission.csv", index=False)




