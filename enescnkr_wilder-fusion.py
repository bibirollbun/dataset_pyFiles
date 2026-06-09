!pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop
!pip install git+https://github.com/WildlifeDatasets/wildlife-tools


import os
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T
from wildlife_datasets.datasets import AnimalCLEF2025
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration

sample_submission = pd.read_csv('/kaggle/input/animal-clef-2025/sample_submission.csv')


# Loading the dataset
root = '/kaggle/input/animal-clef-2025'
dataset = AnimalCLEF2025(root, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')
n_query = len(dataset_query)


# Select identities with a lot of samples
identity_counts = dataset_database.metadata['identity'].value_counts()
identity_counts = np.ceil(identity_counts[identity_counts >= 5] / 25).astype(int)
identity_counts[identity_counts > 5] = 5

# Get sample rows for selected identities
calibration_indexes = []
for identity, sample_count in zip(identity_counts.index, identity_counts):
    calibration_indexes.extend(dataset_database.metadata[dataset_database.metadata['identity'] == identity].sample(sample_count, random_state=42).index)

# Better calibration dataset wih 749 rows
calibration_dataset = dataset_database.metadata.loc[calibration_indexes, :]
dataset_calibration = AnimalCLEF2025(root, df=calibration_dataset, load_label=True)
n_calibration = calibration_dataset.shape[0]
print(n_calibration)


# Loading the models
name = 'hf-hub:BVRA/MegaDescriptor-L-384'
model = timm.create_model(name, num_classes=0, pretrained=True)
device = 'cpu'

matcher_aliked = SimilarityPipeline(
    matcher = MatchLightGlue(features='aliked', device=device, batch_size=16),
    extractor = AlikedExtractor(),
    transform = T.Compose([
        T.Resize([512, 512]),
        T.ToTensor()
    ])
    calibration = IsotonicCalibration()
)

matcher_mega = SimilarityPipeline(
    matcher = CosineSimilarity(),
    extractor = DeepFeatures(model=model, device=device, batch_size=16),
    transform = T.Compose([
        T.Resize([384, 384]),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),
    calibration = IsotonicCalibration()
)


# Calibrating the WildFusion
wildfusion = WildFusion(calibrated_pipelines = [matcher_aliked, matcher_mega], priority_pipeline = matcher_mega)
wildfusion.fit_calibration(dataset_calibration, dataset_calibration)


# Compute WildFusion similarity
num_of_pairs = 42
similarity = wildfusion(dataset_query, dataset_database, B=num_of_pairs)

pred_idx = similarity.argsort(axis=1)[:,-1]
pred_scores = similarity[range(n_query), pred_idx]


for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    predictions = dataset_database.labels_string[pred_idx]
    predictions[pred_scores < threshold] = 'new_individual'
    sample_submission['identity'] = predictions
    sample_submission.to_csv(f'sample_submission_{int(threshold*100)}.csv', index=False)

