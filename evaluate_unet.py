import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision.transforms import functional as VF
from torchmetrics.classification import BinaryJaccardIndex
from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
import pandas as pd
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import random
import data_pipeline
import unet
import baseline
import json

if __name__ == '__main__':
  model = unet.UNet(n_channels=1, n_classes=1)
  model.load_state_dict(torch.load('./unet_13.pt', map_location=torch.device('cpu')))
  test_df = data_pipeline.build_test_dataframe(use_processed_images=False)
  train_df = data_pipeline.build_dataframe(use_processed_images=False)
  X_test, X_names, X_train = data_pipeline.build_test_dataloaders(test_df, train_df, apply_scaling=True)
  test_dl = DataLoader(TensorDataset(X_test), batch_size=1)
  predictions = {}
  model.eval()
  for index, x in tqdm(enumerate(test_dl)):
    out = model(x[0])
    preds = (F.sigmoid(out) > .5)*1.
    name = X_names[index][0]
    predictions[name] = preds.flatten().tolist()
  preds_df = pd.DataFrame.from_dict(predictions, orient='index')
  preds_df.to_csv('KIRBY_predictions_9.csv')
