import numpy as np
import pandas as pd
import os 

def extract_real_and_fake_articles(req="train"):
    df = pd.read_csv(f"/kaggle/input/fake-or-real-the-impostor-hunt/data/{req}.csv")
    Fake_df = pd.DataFrame()
    Real_df = pd.DataFrame()
    
    folders = {}
    for root, dirs, files in os.walk(f'/kaggle/input/fake-or-real-the-impostor-hunt/data/{req}'):
        if files:  
            folder_name = os.path.basename(root)
            folders[folder_name] = files

    sorted_folders = dict(sorted(folders.items()))
    Reals = []
    for article_names, text_file in sorted_folders.items():
        folder_id = int(article_names.split('_')[-1])
        Reals.append(sorted_folders[article_names][df[df['id'] == folder_id]['real_text_id'].values[0] - 1])

    Fakes = []
    count = 0
    for articles in sorted_folders.keys():
        for i in range(len(sorted_folders[articles])):
            if not sorted_folders[articles][i] == Reals[count]:
                Fakes.append(sorted_folders[articles][i])
        count += 1

    base_path = f'/kaggle/input/fake-or-real-the-impostor-hunt/data/{req}'
    content = []
    for folder_id, articles in zip(range(0, 96), sorted_folders.keys()): 
        real_file = Reals[folder_id]
        full_path = os.path.join(base_path, articles, real_file)
        with open(full_path, 'r', encoding='utf-8') as f:
            content.append(f.read())
    Real_df['Real_Text'] = content

    content = []
    for folder_id, articles in zip(range(0, 96), sorted_folders.keys()): 
        fake_file = Fakes[folder_id]
        full_path = os.path.join(base_path, articles, fake_file)
        with open(full_path, 'r', encoding='utf-8') as f:
            content.append(f.read())
    Fake_df['Fake_Text'] = content

    return Real_df, Fake_df



Real_df_2 = pd.DataFrame()
Fake_df_2 = pd.DataFrame()
req="train"
Real_df_2,Fake_df_2 = extract_real_and_fake_articles(req)

