import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import unet
from train_unet import DEVICE
import data_pipeline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

def build_tensors(test_dataframe, train_dataframe, apply_scaling=False):
    test_data = torch.from_numpy(np.vstack(test_dataframe['data'].to_numpy()))
    test_data = torch.nan_to_num(test_data)
    X_names = np.vstack(test_dataframe['filename'].to_numpy())

    train_data = torch.from_numpy(np.vstack(train_dataframe['data'].to_numpy()))
    train_data = torch.nan_to_num(train_data)
    Y_train = torch.from_numpy(np.vstack(train_dataframe['labels'].to_numpy()))

    train_data, test_data = train_data.reshape(-1, 36*36), test_data.reshape(-1, 36*36)
    scaler = RobustScaler().fit(train_data)
    train_data = torch.tensor(scaler.transform(train_data)).float().reshape(-1, 1, 36, 36)
    test_data = torch.tensor(scaler.transform(test_data)).float().reshape(-1, 1, 36, 36)

    X_test = test_data.float().reshape(-1, 1, 36, 36)
    X_train = train_data.float().reshape(-1, 1, 36, 36)
    return X_test, X_names, X_train, Y_train


test_df = data_pipeline.build_test_dataframe(use_processed_images=False, limit_well_number=None)

test_data = torch.from_numpy(np.vstack(test_df['data'].to_numpy()))
test_data = torch.nan_to_num(test_data)
# We can try this if it works for the default case
# count_less_than_neg_10 = torch.sum(test_data < -10, dim=1)
# total_elements_per_row = test_data.size(1)
# ratios = count_less_than_neg_10.float() / total_elements_per_row
# outliers = (ratios > .05).flatten()
# test_data[test_data < -10] = 0

outliers = ((test_data.min(dim=1, keepdim=True).values < -10) == True).flatten()
test_df = test_df.drop(test_df.loc[outliers.tolist()].index)

train_df = data_pipeline.build_dataframe(use_processed_images=False, limit_well_number=None)

X_test, X_names, X_train, Y_train = build_tensors(test_df, train_df, apply_scaling=True)

target_dataset = data_pipeline.WellsDataset(X_test, torch.zeros(len(X_test), 1), transform=None, wells=None)
target_dl = DataLoader(target_dataset, batch_size=128)

source_dataset = data_pipeline.WellsDataset(X_train, torch.ones(len(X_train), 1), transform=None, wells=None)
source_dl = DataLoader(source_dataset, batch_size=128)

adversarial_loss = torch.nn.BCELoss(reduction='mean')

# Define your UNet, discriminator, and optimizer
generator = unet.UNet(n_channels=1, n_classes=1, n_steps=4, with_pl=True)
generator.load_state_dict(torch.load('./checkpoints/unet/unet_16.pt', map_location=DEVICE))
generator = generator.to(DEVICE)

discriminator = unet.Unet_Discriminator(n_channels=1, n_classes=1).to(DEVICE)

optimizer_generator = optim.Adam(generator.parameters())
optimizer_discriminator = optim.Adam(discriminator.parameters())

num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    source_dl = DataLoader(source_dataset, batch_size=128, shuffle=True)
    source_iterator = iter(source_dl)

    discriminator_losses = []
    generator_losses = []

    discriminator_source_accs = []
    discriminator_target_accs = []
    generator_accs = []

    for (target_data, target_labels) in tqdm(target_dl):
        # Train the discriminator
        source_data, source_labels = next(source_iterator)

        discriminator.zero_grad()
        optimizer_discriminator.zero_grad()
    
        source_images = source_data.to(DEVICE)
        source_labels = source_labels.to(DEVICE)

        target_images = target_data.to(DEVICE)
        target_labels = target_labels.to(DEVICE)

        source_outputs, _ = generator(source_images)
        source_outputs = source_outputs.detach()
        target_outputs, _ = generator(target_images)
        target_outputs = target_outputs.detach()

        discrim_on_source = discriminator(source_outputs)
        discrim_on_target = discriminator(target_outputs)

        discriminator_loss = adversarial_loss(discrim_on_source, source_labels) + adversarial_loss(discrim_on_target, target_labels)
        discriminator_losses.append(discriminator_loss.item())
        discriminator_loss.backward()

        discrim_source_accuracy = torch.sum((torch.round(discrim_on_source) == source_labels)) / len(source_labels)
        discrim_target_accuracy = torch.sum((torch.round(discrim_on_target) == target_labels)) / len(target_labels)

        discriminator_source_accs.append(discrim_source_accuracy.cpu().detach())
        discriminator_target_accs.append(discrim_target_accuracy.cpu().detach())

        optimizer_discriminator.step()

        # Train the generator
        generator.zero_grad()
        optimizer_generator.zero_grad()

        outputs, _ = generator(target_images)

        discrim_from_gen = discriminator(outputs)

        generator_loss = adversarial_loss(discrim_from_gen, target_labels)
        generator_losses.append(generator_loss.item())
        generator_loss.backward()

        generator_accuracy = torch.sum((torch.round(discrim_from_gen) == target_labels)) / len(target_labels)
        generator_accs.append(generator_accuracy.cpu().detach())

        optimizer_generator.step()

    print("-"*100)
    print(f"epoch: {epoch}")
    print(f"Losses:")
    print(f"Discriminator loss: {np.mean(discriminator_losses)}")
    print(f"Generator loss: {np.mean(generator_losses)}")
    print(f"Accs:")
    print(f"Discrim on Source: {np.mean(discriminator_source_accs)}")
    print(f"Discrim on Target: {np.mean(discriminator_target_accs)}")
    print(f"Generator from Target: {np.mean(generator_accs)}")


torch.save(generator.state_dict(), './unet_16_generator.pt')
torch.save(discriminator.state_dict(), './discriminator_1.pt')
