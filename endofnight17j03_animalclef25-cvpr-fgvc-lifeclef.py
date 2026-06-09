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

def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)


root = '/kaggle/input/animal-clef-2025'
transform_display = T.Compose([
    T.Resize([384, 384]),
    ])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


dataset = AnimalCLEF2025(root, transform=transform_display, load_label=True)


dataset.plot_grid()
dataset.metadata


dataset.metadata[['dataset', 'split']].value_counts()


idx = dataset.metadata['identity'].str.startswith('SeaTurtleID2022')
idx[idx.isnull()] = False
dataset.plot_grid(idx=idx);


# Loading the dataset
dataset = AnimalCLEF2025(root, transform=transform, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')
n_query = len(dataset_query)


#Loading the model
name = 'hf-hub:BVRA/MegaDescriptor-L-384'
device = 'cuda'
model = timm.create_model(name, num_classes=0, pretrained=True)
extractor = DeepFeatures(model, device=device, batch_size=32, num_workers=0)
features_database = extractor(dataset_database)
features_query = extractor(dataset_query)


similarity = CosineSimilarity()(features_query, features_database)


pred_idx = similarity.argsort(axis=1)[:,-1]
pred_scores = similarity[range(n_query), pred_idx]


new_individual = 'new_individual'
threshold = 0.6
labels = dataset_database.labels_string
predictions = labels[pred_idx]
predictions[pred_scores < threshold] = new_individual
create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv')


root = '/kaggle/input/animal-clef-2025'
transform_display = T.Compose([
    T.Resize([384, 384]),
])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transforms_aliked = T.Compose([
    T.Resize([512, 512]),
    T.ToTensor()
])


dataset = AnimalCLEF2025(root, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')
dataset_calibration = AnimalCLEF2025(root, df=dataset_database.metadata[:100], load_label=True) 

n_query = len(dataset_query)


name = 'hf-hub:BVRA/MegaDescriptor-L-384'
model = timm.create_model(name, num_classes=0, pretrained=True)
device = 'cuda'

matcher_aliked = SimilarityPipeline(
    matcher = MatchLightGlue(features='aliked', device=device, batch_size=16),
    extractor = AlikedExtractor(),
    transform = transforms_aliked,
    calibration = IsotonicCalibration()
)

matcher_mega = SimilarityPipeline(
    matcher = CosineSimilarity(),
    extractor = DeepFeatures(model=model, device=device, batch_size=16),
    transform = transform,
    calibration = IsotonicCalibration()
)


wildfusion = WildFusion(calibrated_pipelines = [matcher_aliked, matcher_mega], priority_pipeline = matcher_mega)
wildfusion.fit_calibration(dataset_calibration, dataset_calibration)

# Compute WildFusion similarity
similarity = wildfusion(dataset_query, dataset_database, B=25)

pred_idx = similarity.argsort(axis=1)[:,-1]
pred_scores = similarity[range(n_query), pred_idx]

labels = dataset_database.labels_string


for threshold in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]:
    predictions = labels[pred_idx]
    predictions[pred_scores < threshold] = 'new_individual'
    create_sample_submission(dataset_query, predictions, file_name=f'sample_submission_new{int(threshold*100)}.csv')

