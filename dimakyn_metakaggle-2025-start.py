import os
import copy
import uuid

import pandas as pd

import plotly.graph_objects as go
from IPython.display import IFrame, display, HTML


import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)


for dirname, _, filenames in os.walk(MK_PATH):
    for filename in filenames:
        print(filename)


def get_file_size(filename):
    fpath = os.path.join(MK_PATH, filename)
    size = os.path.getsize(fpath) / (1024**2)
    return size


def get_general_info_files(filename):
    file_size = get_file_size(filename)
    print(f"[INFO] File size => {file_size:.2f} M")
    if file_size < 300:
        df = pd.read_csv(f"{MK_PATH}/{filename}")
        print(f"[INFO] Shape => {df.shape}")
    else:
        df = pd.read_csv(f"{MK_PATH}/{filename}", nrows=1000)
        print(f"[WARN] open only 1000 rows.")
    print(f"[INFO] Columns => {df.columns}")
    print("----------------------------------------------------------------------------")

    display(HTML(df.head(10).to_html(index=False)))

    print("\n=============================================================================\n")


USERS_FILES = ["Users.csv", "UserAchievements.csv", "UserOrganizations.csv", "UserFollowers.csv", "Teams.csv", "TeamMemberships.csv", "Organizations.csv"]

for filename in USERS_FILES:
    
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


COMPETITION_FILES = ["Competitions.csv", "CompetitionTags.csv", "Submissions.csv"]

for filename in COMPETITION_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


KERNELS_FILES = ["Kernels.csv", "KernelVersions.csv", "KernelVotes.csv", "KernelLanguages.csv", "KernelAcceleratorTypes.csv", "KernelTags.csv"]

for filename in KERNELS_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


DATASETS_FILES = ["Datasets.csv", "DatasetVersions.csv", "DatasetVotes.csv", "DatasetTags.csv", "DatasetTasks.csv", "DatasetTaskSubmissions.csv"]

for filename in DATASETS_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


MODELS_FILES = ["Models.csv", "ModelVersions.csv", "ModelVariations.csv", "ModelVariationVersions.csv", "ModelVotes.csv", "ModelTags.csv"]

for filename in MODELS_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


FORUM_FILES = ["Forums.csv", "ForumTopics.csv", "ForumMessages.csv", "ForumMessageVotes.csv", "ForumMessageReactions.csv"]

for filename in FORUM_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


KERNEL_FILES = ["KernelVersionCompetitionSources.csv", "KernelVersionKernelSources.csv", "KernelVersionDatasetSources.csv", "KernelVersionModelSources.csv"]

for filename in KERNEL_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


TAGS_FILES = ["Tags.csv"]

for filename in TAGS_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


EPISODES_FILES = ["Episodes.csv", "EpisodeAgents.csv"]

for filename in EPISODES_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)


DATASOURCE_FILES = ["Datasources.csv"]

for filename in DATASOURCE_FILES:
    print(f"[INFO] File name => {filename}")
    get_general_info_files(filename)




