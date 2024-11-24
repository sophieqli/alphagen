import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
#import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

class CardDataset(Dataset): 
    def __init__(self, data_dir, transform = None): 
        self.data = ImageFolder(data_dir, transform = transform)
    def __len__(self): 
        return len(self.data)


dataset = CardDataset(data_dir = '/kaggle/input/cards-image-datasetclassification/train')
print(dataset.__len__)
