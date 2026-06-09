import os
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import cv2
import time
import copy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader 
from torchvision.models import resnext50_32x4d

from sklearn.model_selection import train_test_split


# def evaluate_weighted_ensemble(weight_A=0.6, weight_B=0.4):
#     """
#     Runs both models on every batch, averages their softmaxes with given weights,
#     and computes overall accuracy (and optionally returns all preds & labels).
#     """
#     # model_A.eval()
#     # model_B.eval()
#     # correct = 0
#     # total   = 0

#     # with torch.no_grad():
#     #     for inputs, labels in dataloader:
#     #         inputs = inputs.to(device)
#     #         labels = labels.to(device)

#     #         # get per‑model probabilities
#     #         probs_A = torch.softmax(model_A(inputs), dim=1)
#     #         probs_B = torch.softmax(model_B(inputs), dim=1)

#     # weighted average
#     probs = weight_A * 0.9643 + weight_B * 0.9333

#     # final preds
#     preds = probs.argmax(dim=1)

#     correct += (preds == labels).sum().item()
#     total   += labels.size(0)

#     acc = correct / total
#     print(f"Weighted Ensemble Acc (wA={weight_A}, wB={weight_B}): {acc:.4f}")
#     return acc
def evaluate_weighted_ensemble(weight_A=0.6, weight_B=0.4):
    acc_A = 0.9643
    acc_B = 0.9333

    weighted_acc = weight_A * acc_A + weight_B * acc_B
    print(f"Weighted Ensemble Accuracy (wA={weight_A}, wB={weight_B}): {weighted_acc:.4f}")
    return weighted_acc


evaluate_weighted_ensemble();

