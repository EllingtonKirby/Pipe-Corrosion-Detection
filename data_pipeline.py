import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.transforms import v2
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import RobustScaler
import json

class WellsDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = False
        self.scaler = RobustScaler()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def build_dataframe():
    data_dir = './train/images/'
    data_dict = []
    pattern = r'well_(\d+)_patch_(\d+)\.npy'
    for i, filename in enumerate(os.listdir(data_dir)):
        example = np.load(data_dir + filename)
        match = re.match(pattern, filename)
        if match:
            well_number = int(match.group(1))  # Extract well number
            patch_number = int(match.group(2)) # Extract patch number
        else:
            print("Filename format does not match the expected pattern.")  

        data_dict.append((filename, well_number, patch_number, example.flatten()))

    df = pd.DataFrame(data=data_dict, columns=['filename', 'well_number', 'patch_number', 'data'])

    df_y = pd.read_csv('train/y_train.csv')
    df_y['Unnamed: 0'] = df_y['Unnamed: 0'] + '.npy'

    # Create single dataframe with data and labels as lists
    merged = pd.merge(df, df_y, how='left', left_on='filename', right_on='Unnamed: 0')
    data_columns = [str(i) for i in range(1296)]
    labeled_data = merged[data_columns].to_numpy()
    # Convert missing labels to all zeros
    labeled_data = np.nan_to_num(labeled_data)
    merged['labels'] = labeled_data.tolist()
    merged = merged.rename(columns={'Unnamed: 0':'label_name'})
    merged = merged.fillna(0.0)

    data = torch.from_numpy(np.vstack(merged['data'].to_numpy(dtype=np.ndarray)))
    # Remove corruputed samples
    outliers = ((data.min(dim=1, keepdim=True).values == -999.2500) == True).flatten()

    merged = merged.drop(merged.loc[outliers.tolist()].index)
    return merged

def build_dataloaders(dataframe):
    data = torch.from_numpy(np.vstack(dataframe['data'].to_numpy()))
    data = torch.nan_to_num(data)
    labels = torch.from_numpy(np.vstack(dataframe['labels'].to_numpy()))

    p = np.random.permutation(len(data))
    with open('train_set_permutation.json', 'w') as f:
        # Write permutation to file so that we can re-apply the same transform later
        json.dump(p.tolist(), f)

    data, labels = data[p], labels[p]

    offset = int(len(data) * .8)
    X_train, X_valid = data[:offset], data[offset:]
    Y_train, Y_valid = labels[:offset].float().reshape(-1, 1, 36, 36), labels[offset:].float().reshape(-1, 1, 36, 36)

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = torch.tensor(scaler.transform(X_train)).float().reshape(-1, 1, 36, 36)
    X_valid = torch.tensor(scaler.transform(X_valid)).float().reshape(-1, 1, 36, 36)

    # Applying augmentations selectively does not improve performance
    # num_non_zero_y = torch.count_nonzero(labels[:offset], axis=1)
    examples_to_augment = X_train
    labels_to_augment = Y_train

    rolled_x, rolled_y = [], []
    for i in range(1, 36):
        rolled_x.append(torch.roll(examples_to_augment, i, dims=3))
        rolled_y.append(torch.roll(labels_to_augment, i, dims=3))

    X_train, Y_train = torch.vstack((X_train, *rolled_x)), torch.vstack((Y_train, *rolled_y))

    flipper = v2.RandomVerticalFlip(1)
    X_train, Y_train = torch.vstack((X_train, flipper(examples_to_augment))), torch.vstack((Y_train, flipper(labels_to_augment)))

    train_dataset = WellsDataset(X_train, Y_train, None)
    valid_dataset = WellsDataset(X_valid, Y_valid, None)

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128)

    return train_dataloader, valid_dataloader

def build_test_dataframe():
  data_dir = './test/images/'
  data_dict = []
  pattern = r'well_(\d+)_patch_(\d+)\.npy'
  for i, filename in enumerate(os.listdir(data_dir)):
      example = np.load(data_dir + filename)
      match = re.match(pattern, filename)
      name = filename[:-4]
      if match:
          well_number = int(match.group(1))  # Extract well number
          patch_number = int(match.group(2)) # Extract patch number
      else:
          print("Filename format does not match the expected pattern.")  

      data_dict.append((name, well_number, patch_number, example.flatten()))

  df = pd.DataFrame(data=data_dict, columns=['filename', 'well_number', 'patch_number', 'data'])
  return df.sort_values(by=['well_number','patch_number'])

def build_dataloaders_for_classiication(train_dataframe):
    data = torch.from_numpy(np.vstack(train_dataframe['data'].to_numpy()))
    data = torch.nan_to_num(data)
    labels = torch.from_numpy(np.vstack(train_dataframe['well_number'].to_numpy())).squeeze() - 1

    p = np.random.permutation(len(data))
    with open('train_set_permutation.json', 'w') as f:
        # Write permutation to file so that we can re-apply the same transform later
        json.dump(p.tolist(), f)

    data, labels = data[p], labels[p]
    offset = int(len(data) * .5)
    X_train, X_valid = data[:offset], data[offset:]
    Y_train, Y_valid = labels[:offset], labels[offset:]

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = torch.tensor(scaler.transform(X_train)).float().reshape(-1, 1, 36, 36)
    X_valid = torch.tensor(scaler.transform(X_valid)).float().reshape(-1, 1, 36, 36)

    examples_to_augment = X_train
    labels_to_augment = Y_train

    rolled_x, rolled_y = [], []
    for i in range(1, 36):
        rolled_x.append(torch.roll(examples_to_augment, i, dims=3))
        rolled_y.append(labels_to_augment)

    X_train, Y_train = torch.vstack((X_train, *rolled_x)), torch.hstack((Y_train, *rolled_y))

    flipper = v2.RandomVerticalFlip(1)
    X_train, Y_train = torch.vstack((X_train, flipper(examples_to_augment))), torch.hstack((Y_train, labels_to_augment))

    train_dataset = WellsDataset(X_train, Y_train, None)
    valid_dataset = WellsDataset(X_valid, Y_valid, None)

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128)

    return train_dataloader, valid_dataloader


def individually_scale_all_images():
    data_dir = './train/images/'
    output_dir = './train/processed_images/'
    print(f"Scaling {data_dir}")
    for _, filename in enumerate(os.listdir(data_dir)):
        print(f"Processing: {filename}")
        example = np.load(data_dir + filename)
        scaled = RobustScaler().fit_transform(example)
        np.save(output_dir + filename, scaled)
    
    data_dir = './test/images/'
    output_dir = './test/processed_images/'
    print(f"Scaling {data_dir}")
    for _, filename in enumerate(os.listdir(data_dir)):
        print(f"Processing: {filename}")
        example = np.load(data_dir + filename)
        scaled = RobustScaler().fit_transform(example)
        np.save(output_dir + filename, scaled)

if __name__ == '__main__':
    print("Data pipeline invoked directly, applying robust scaler to all images individually")
    individually_scale_all_images()
    print("Done")
