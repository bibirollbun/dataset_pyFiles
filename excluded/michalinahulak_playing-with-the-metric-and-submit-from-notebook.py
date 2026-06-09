from typing import List, Set
import pandas as pd
import csv
import random

def f1_score_per_image(true_labels: Set[int], pred_labels: Set[int]) -> float:
    """
    Compute the F1 score for a single image.
    true_labels: set of ground truth species (e.g., {1, 4, 9})
    pred_labels: set of predicted species (e.g., {1, 4, 5})
    """
    tp = len(true_labels & pred_labels)  # True Positives
    fp = len(pred_labels - true_labels)  # False Positives
    fn = len(true_labels - pred_labels)  # False Negatives

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def macro_f1_per_transect(
    transect_true: List[Set[int]],
    transect_pred: List[Set[int]]
) -> float:
    """
    Compute the average F1 score for a single transect.
    transect_true: list of sets with ground truth labels for each image in the transect
    transect_pred: list of sets with predicted labels for each image in the transect
    """
    assert len(transect_true) == len(transect_pred)
    scores = [
        f1_score_per_image(t, p)
        for t, p in zip(transect_true, transect_pred)
    ]
    return sum(scores) / len(scores)


def final_macro_f1_score(
    all_true: List[List[Set[int]]],
    all_pred: List[List[Set[int]]]
) -> float:
    """
    Compute the final score: the average F1 score across all transects.
    all_true: list of transects, each being a list of sets of ground truth labels
    all_pred: list of transects, each being a list of sets of predicted labels
    """
    assert len(all_true) == len(all_pred)
    transect_scores = [
        macro_f1_per_transect(true, pred)
        for true, pred in zip(all_true, all_pred)
    ]
    return sum(transect_scores) / len(transect_scores)



# example
true = [
    [ {1, 2}, {3} ],            # transekt 1
    [ {4, 5}, {6, 7, 8} ]       # transekt 2
]

pred = [
    [ {1, 2}, {2, 3} ],         # transekt 1
    [ {4}, {6, 9} ]             # transekt 2
]

score = final_macro_f1_score(true, pred)
print(f"Final macro-averaged F1 score: {score:.4f}")


test = pd.read_csv('/kaggle/input/plantclef-2025/PlantCLEF2025_test.csv', sep = ';')


test


species = pd.read_csv('/kaggle/input/plantclef-2025/species_ids.csv')
species


all_species = list(species['species_id'].unique())


def random_species_subset(species_pool, min_n=1, max_n=5):
    n = random.randint(min_n, max_n)
    return random.sample(species_pool, k=n)


df_run = pd.DataFrame()

df_run['quadrat_id']=test["quadrat_id"]

df_run['species_ids'] = df_run['quadrat_id'].apply(lambda _: random_species_subset(all_species))
df_run


df_run.to_csv("submission.csv", sep=',', index=False, quoting=csv.QUOTE_ALL)

