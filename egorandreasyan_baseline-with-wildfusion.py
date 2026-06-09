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
    T.Resize([224, 224]),
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


# Loading the dataset
dataset = AnimalCLEF2025(root, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')
dataset_calibration = AnimalCLEF2025(root, df=dataset_database.metadata[:256], load_label=True) 

n_query = len(dataset_query)


# Loading the models
name = 'hf-hub:BVRA/MegaDescriptor-T-224'
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


# Calibrating the WildFusion
wildfusion = WildFusion(calibrated_pipelines = [matcher_aliked, matcher_mega], priority_pipeline = matcher_mega)
wildfusion.fit_calibration(dataset_calibration, dataset_calibration)


# Compute WildFusion similarity
similarity = wildfusion(dataset_query, dataset_database, B=32)


pred_idx = similarity.argsort(axis=1)[:,-1]
pred_scores = similarity[range(n_query), pred_idx]


new_individual = 'new_individual'
threshold = 0.45
labels = dataset_database.labels_string
predictions = labels[pred_idx]
predictions[pred_scores < threshold] = new_individual
create_sample_submission(dataset_query, predictions, file_name='sample_submission_S.csv')




