import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import RobustScaler
import json
import collections
torchvision.disable_beta_transforms_warning()
    
def build_dataframe(use_processed_images=True, limit_well_number=None):
    if use_processed_images:
        data_dir = './train/processed_images/'
    else:
        data_dir = './train/images/'
    data_dict = []
    pattern = r'well_(\d+)_patch_(\d+)\.npy'
    for i, filename in enumerate(os.listdir(data_dir)):
        example = np.nan_to_num(np.load(data_dir + filename))
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
    outliers = ((data.min(dim=1, keepdim=True).values < -10) == True).flatten()
    merged = merged.drop(merged.loc[outliers.tolist()].index)

    if limit_well_number != None:
        merged = merged[merged['well_number'] == limit_well_number]

    return merged

def build_test_dataframe(use_processed_images=True, limit_well_number=None):
    if use_processed_images:
        data_dir = './test/processed_images/'
    else:
        data_dir = './test/images/'

    data_dict = []
    pattern = r'well_(\d+)_patch_(\d+)\.npy'
    for i, filename in enumerate(os.listdir(data_dir)):
        example = np.nan_to_num(np.load(data_dir + filename))
        match = re.match(pattern, filename)
        name = filename[:-4]
        if match:
            well_number = int(match.group(1))  # Extract well number
            patch_number = int(match.group(2)) # Extract patch number
        else:
            print("Filename format does not match the expected pattern.")  

        data_dict.append((name, well_number, patch_number, example.flatten()))

    df = pd.DataFrame(data=data_dict, columns=['filename', 'well_number', 'patch_number', 'data'])
    data = torch.from_numpy(np.vstack(df['data'].to_numpy(dtype=np.ndarray)))

    if limit_well_number != None:
        df = df[df['well_number'] == limit_well_number]

    return df

def build_ensemble_dataloaders():
    train_df = build_dataframe(use_processed_images=False, limit_well_number=None)
    data = torch.from_numpy(np.vstack(train_df['data'].to_numpy()))
    data = torch.nan_to_num(data)
    labels = torch.from_numpy(np.vstack(train_df['labels'].to_numpy()))
    wells = torch.from_numpy(np.vstack(train_df['well_number'].to_numpy()))
    
    p = np.random.permutation(len(data))
    data, labels = data[p], labels[p]
    offset = int(len(data) * .8)
    X_train, X_valid = data[:offset].float().reshape(-1, 1, 36, 36), data[offset:].float().reshape(-1, 1, 36, 36)
    Y_train, Y_valid = labels[:offset].float().reshape(-1, 1, 36, 36), labels[offset:].float().reshape(-1, 1, 36, 36)
    Wells_train, Wells_valid = wells[:offset].float().reshape(-1, 1), wells[offset:].float().reshape(-1, 1)
    
    well_groups = [3, 7, 13, 'rest']
    well_group_dataloaders = {}
    
    for well in well_groups:
        if well != 'rest':
            well_mask = Wells_train == well
            # x_well = 
    
    
class WellsDataset(Dataset):
    def __init__(self, data, labels, transform, wells=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        if (self.transform):
            self.flipper = v2.RandomVerticalFlip(1)
        self.wells = wells

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image, label = self.transform(image, label, self.flipper)
        if self.wells != None:
            return image, label, self.wells[idx]
        return image, label
    
def image_label_transforms(image, label, flipper):
    axis = 2
    roll_distance = np.random.randint(0, 36)
    image = torch.roll(image, roll_distance, dims=axis)
    label = torch.roll(label, roll_distance, dims=axis)

    flip = np.random.randint(2) % 2 == 0
    if (flip):
        image = flipper(image)
        label = flipper(label)

    # image, _ = cutout(image, torch.zeros_like(image), size=9)

    return image, label

def cutout(image: torch.Tensor, label: torch.Tensor, size):
    image_c = image.clone()
    label_c = label.clone()
    h, w = image.shape[-2:]

    y = np.random.randint(0, h - size - 1)
    x = np.random.randint(0, w - size - 1)

    image_c[:, y:y + size, x:x + size] = 0
    label_c[:, y:y + size, x:x + size] = 0

    return image_c, label_c

def just_image_transforms(image, label, flipper):
    axis = 2
    roll_distance = np.random.randint(0, 36)
    image = torch.roll(image, roll_distance, dims=axis)

    flip = np.random.randint(2) % 2 == 0
    if (flip):
        image = flipper(image)

    return image, label

def build_dataloaders(dataframe, apply_scaling=False, apply_bulk_data_augmentations=False, split_train=False):
    data = torch.from_numpy(np.vstack(dataframe['data'].to_numpy()))
    data = torch.nan_to_num(data)
    labels = torch.from_numpy(np.vstack(dataframe['labels'].to_numpy()))

    if split_train:
        p = np.random.permutation(len(data))
        data, labels = data[p], labels[p]
        with open('train_set_permutation.json', 'w') as f:
            # Write permutation to file so that we can re-apply the same transform later
            json.dump(p.tolist(), f)

        offset = int(len(data) * .8)
        X_train, X_valid = data[:offset].float().reshape(-1, 1, 36, 36), data[offset:].float().reshape(-1, 1, 36, 36)
        Y_train, Y_valid = labels[:offset].float().reshape(-1, 1, 36, 36), labels[offset:].float().reshape(-1, 1, 36, 36)
    else:
        X_train, X_valid = data.float().reshape(-1, 1, 36, 36), torch.zeros(0)
        Y_train, Y_valid = labels.float().reshape(-1, 1, 36, 36), torch.zeros(0)

    if (apply_scaling):
        X_train, X_valid = X_train.reshape(-1, 36*36), X_valid.reshape(-1, 36*36)
        scaler = RobustScaler().fit(X_train)
        X_train = torch.tensor(scaler.transform(X_train)).float().reshape(-1, 1, 36, 36)
        if (split_train):
            X_valid = torch.tensor(scaler.transform(X_valid)).float().reshape(-1, 1, 36, 36)

    if apply_bulk_data_augmentations:
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
    else:
        train_dataset = WellsDataset(X_train, Y_train, image_label_transforms)
        valid_dataset = WellsDataset(X_valid, Y_valid, None)

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128)

    return train_dataloader, valid_dataloader

def build_dataloaders_weighted(tau):
    dataframe = build_dataframe(use_processed_images=False)
    X = torch.from_numpy(np.vstack(dataframe['data'].to_numpy()))
    X = torch.nan_to_num(X)
    Y = torch.from_numpy(np.vstack(dataframe['labels'].to_numpy()))

    well_numbers = dataframe['well_number']
    wells = torch.from_numpy(np.vstack(well_numbers.to_numpy())) - 1
    well_mean_weight = np.mean(well_numbers.value_counts().values)
    
    sample_weight = {well: ratio/well_mean_weight for well, ratio in well_numbers.value_counts().items()}
    sample_weight = {well: min(1/ratio, tau) for well, ratio in sample_weight.items()}
    print("Sample weights: ", sample_weight)
    sample_weight = collections.OrderedDict(sorted(sample_weight.items()))
    sample_weight = torch.tensor(list(sample_weight.values()))
    wells = sample_weight[wells].flatten()

    X_train = X.float()
    Y_train= Y.float()
    Wells_train = wells.float().reshape(-1, 1, 1, 1)

    # Scale
    scaler = RobustScaler().fit(X_train)
    X_train = torch.from_numpy(scaler.transform(X_train)).float().reshape(-1, 1, 36, 36)
    Y_train = Y_train.reshape(-1, 1, 36, 36)

    train_dataset = WellsDataset(X_train, Y_train, transform=image_label_transforms,  wells=Wells_train)
    valid_dataset = WellsDataset(torch.zeros(0), torch.zeros(0), transform=None, wells=torch.zeros(0))

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128)

    return train_dataloader, valid_dataloader

def build_test_dataloaders(test_dataframe, train_dataframe, apply_scaling=False):
    test_data = torch.from_numpy(np.vstack(test_dataframe['data'].to_numpy()))
    test_data = torch.nan_to_num(test_data)
    X_names = np.vstack(test_dataframe['filename'].to_numpy())

    train_data = torch.from_numpy(np.vstack(train_dataframe['data'].to_numpy()))
    train_data = torch.nan_to_num(train_data)
    Y_train = torch.from_numpy(np.vstack(train_dataframe['labels'].to_numpy()))

    if apply_scaling:
        train_data, test_data = train_data.reshape(-1, 36*36), test_data.reshape(-1, 36*36)
        scaler = RobustScaler().fit(train_data)
        train_data = torch.tensor(scaler.transform(train_data)).float().reshape(-1, 1, 36, 36)
        test_data = torch.tensor(scaler.transform(test_data)).float().reshape(-1, 1, 36, 36)

    X_test = test_data.float().reshape(-1, 1, 36, 36)
    X_train = train_data.float().reshape(-1, 1, 36, 36)
    return X_test, X_names, X_train, Y_train


def build_dataloaders_for_classiication(train_dataframe, apply_scaling=False, apply_bulk_data_augmentations=False):
    data = torch.from_numpy(np.vstack(train_dataframe['data'].to_numpy()))
    data = torch.nan_to_num(data)
    labels = torch.from_numpy(np.vstack(train_dataframe['well_number'].to_numpy())).squeeze() - 1

    p = np.random.permutation(len(data))

    data, labels = data[p], labels[p]
    offset = int(len(data) * .8)
    X_train, X_valid = data[:offset].float().reshape(-1, 1, 36, 36), data[offset:].float().reshape(-1, 1, 36, 36)
    Y_train, Y_valid = labels[:offset], labels[offset:]

    if (apply_scaling):
        with open('classification_set_permutation.json', 'w') as f:
            # Write permutation to file so that we can re-apply the same transform later
            json.dump(p.tolist(), f)
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = torch.tensor(scaler.transform(X_train)).float().reshape(-1, 1, 36, 36)
        X_valid = torch.tensor(scaler.transform(X_valid)).float().reshape(-1, 1, 36, 36)

    if apply_bulk_data_augmentations:
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
    else:
        train_dataset = WellsDataset(X_train, Y_train, just_image_transforms)
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
