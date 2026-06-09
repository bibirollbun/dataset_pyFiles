import os
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt



BASE_PATH = "../input/petfinder-pawpularity-score/"
train_df = pd.read_csv(f'{BASE_PATH}train.csv')
print(len(train_df))
train_df.head()


def meta_feature_samples(feature):
    colors = ["#ED2938", "#B25F4A", "#77945C", "#3BCA6D", "#00FF7F"]
    figs = plt.figure(constrained_layout=True, figsize=(15, 12))
    subfigs = figs.subfigures(5, 2, hspace=0.07)
    for idx, fig in enumerate(subfigs,1):
        axes = fig[0].subplots(1, 3)
        fig[0].supylabel(f'<{idx*20}', fontweight='bold')
        fig[0].set_facecolor(colors[idx-1])
        train_sub = train_df[(train_df['Pawpularity']<=idx*20) & ((idx-1)*20<=train_df['Pawpularity']) & 
                          (train_df[feature]==0)
                         ].sample(3, random_state=0)
        for image_id, ax in zip(train_sub['Id'], axes):
            ax.imshow(Image.open(os.path.join(BASE_PATH, 'train', image_id + '.jpg')))
            ax.set_xticks([])
            ax.set_yticks([])

        axes = fig[1].subplots(1, 3)
        fig[1].set_facecolor(colors[idx-1])

        train_sub = train_df[(train_df['Pawpularity']<=idx*20) & ((idx-1)*20<=train_df['Pawpularity']) & (train_df[feature]==1)].sample(3, random_state=0)
        for image_id, ax in zip(train_sub['Id'], axes):
            ax.imshow(Image.open(os.path.join(BASE_PATH, 'train', image_id + '.jpg')))
            ax.set_xticks([])
            ax.set_yticks([])
    figs.suptitle(f'{feature} 0 & 1 samples', fontweight='bold', fontsize=20)        
    figs.supylabel('Pawpularity', fontsize=18)


    plt.show()


#Let's start with just one variable to demonstrate
variable = 'Eyes'
print(train_df[variable].value_counts())
fig, ax = plt.subplots(1,1)
sns.histplot(train_df, x="Pawpularity", hue=variable, kde=True)
plt.suptitle(variable, fontsize=20, fontweight='bold')
fig.show()
meta_feature_samples(variable)


#Now lets do the same for all the variables with a simple for loop:

#get a the column names into a list
feature_variables = train_df.columns.values.tolist()

#for each of the feature variables, doesn't include Id and Pawpularity by using [1:-1]
#show a boxplot and distribution plot against pawpularity
for variable in feature_variables[1:-1]:
    print(train_df[variable].value_counts())
    fig, ax = plt.subplots(1,1)
    sns.histplot(train_df, x="Pawpularity", hue=variable, kde=True)
    plt.suptitle(variable, fontsize=20, fontweight='bold')
    fig.show()
    meta_feature_samples(variable)


TARGET = "Pawpularity"
FEATURES = [col for col in train_df.columns if col not in ['Id', TARGET]]
print(f'TARGET:{TARGET}')
print(f'FEATURES:{FEATURES}')


corr_matrix = train_df[FEATURES + [TARGET]].corr()
corr_matrix


target_corr = corr_matrix[TARGET][:-1]
target_corr

