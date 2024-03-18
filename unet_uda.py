import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import unet
from train_unet import DEVICE, DiceBCELoss, PseduoLabelBCELoss
from torchmetrics.classification import BinaryJaccardIndex
import data_pipeline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

def diff_augment(image, label, flipper):
    axis = 2
    roll_distance = np.random.randint(0, 36)
    image = torch.roll(image, roll_distance, dims=axis)
    label = torch.roll(label, roll_distance, dims=axis)

    flip = np.random.randint(2) % 2 == 0
    if (flip):
        image = flipper(image)
        label = flipper(label)

    return image, label

def build_tensors(test_dataframe, train_dataframe):
    test_data = torch.from_numpy(np.vstack(test_dataframe['data'].to_numpy()))
    test_data = torch.nan_to_num(test_data)
    
    outliers = ((test_data.min(dim=1, keepdim=True).values < -10) == True).flatten()
    test_df = test_dataframe.drop(test_dataframe.loc[outliers.tolist()].index)
    
    test_data = torch.from_numpy(np.vstack(test_df['data'].to_numpy()))
    test_data = torch.nan_to_num(test_data)

    X_names = np.vstack(test_df['filename'].to_numpy())

    train_data = torch.from_numpy(np.vstack(train_dataframe['data'].to_numpy()))
    train_data = torch.nan_to_num(train_data)
    Y_train = torch.from_numpy(np.vstack(train_dataframe['labels'].to_numpy()))

    train_data, test_data = train_data.reshape(-1, 36*36), test_data.reshape(-1, 36*36)
    scaler = RobustScaler().fit(train_data)
    train_data = torch.tensor(scaler.transform(train_data)).float().reshape(-1, 1, 36, 36)
    test_data = torch.tensor(scaler.transform(test_data)).float().reshape(-1, 1, 36, 36)

    X_test = test_data.float().reshape(-1, 1, 36, 36)
    X_train = train_data.float().reshape(-1, 1, 36, 36)
    
    p = np.random.permutation(X_train)
    X_train = X_train[p]
    Y_train = Y_train[p]

    return X_test, X_names, X_train, Y_train


test_df = data_pipeline.build_test_dataframe(use_processed_images=False, limit_well_number=None)
train_df = data_pipeline.build_dataframe(use_processed_images=False, limit_well_number=None)

X_test, X_names, X_train, Y_train = build_tensors(test_df, train_df)

target_dataset = data_pipeline.WellsDataset(X_test, torch.zeros(len(X_test), 1, 36, 36), transform=diff_augment, wells=None)
target_dl = DataLoader(target_dataset, batch_size=128)

source_dataset = data_pipeline.WellsDataset(X_train, Y_train.float().reshape(-1, 1, 36, 36), transform=diff_augment, wells=None)
source_dl = DataLoader(source_dataset, batch_size=128)

source_dataset_val = data_pipeline.WellsDataset(X_train[:len(X_test)], Y_train[:len(X_test)].float().reshape(-1, 1, 36, 36), transform=None, wells=None)
source_dl_val = DataLoader(source_dataset_val, batch_size=128)

# Define your UNet, discriminator, and optimizer
generator = unet.UNet(n_channels=1, n_classes=1, n_steps=4, with_pl=True)
generator.load_state_dict(torch.load('./checkpoints/unet/unet_16.pt', map_location=DEVICE))
generator = generator.to(DEVICE)

discriminator = unet.Unet_Discriminator(n_classes=1).to(DEVICE)

adversarial_weight_source = 1.0 
adversarial_weight_target = 1.0 
lmbda = 1.0

optimizer_generator = optim.Adam(generator.parameters(), lr=.0001)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=.001)

adversarial_loss = torch.nn.BCELoss(reduction='mean')
metric = BinaryJaccardIndex().to(DEVICE)
dice_criterion = DiceBCELoss().to(DEVICE)
pseudo_labeling_criterion = PseduoLabelBCELoss()

num_epochs = 30

def train_emin_uda():
    # Training loop
    for epoch in range(num_epochs):
        source_dl = DataLoader(source_dataset, batch_size=128, shuffle=True)
        source_iterator = iter(source_dl)

        generator_losses = []

        for (target_data, _) in tqdm(target_dl):
            # Train the discriminator
            source_data, source_labels = next(source_iterator)
            
            optimizer_generator.zero_grad()

            if len(target_data) != len(source_data):
                # end of batch
                source_data = source_data[0:len(target_data)]
                source_labels = source_labels[0:len(target_data)]
        
            source_images = source_data.to(DEVICE)
            source_labels = source_labels.to(DEVICE)

            target_images = target_data.to(DEVICE)

            source_preds, source_class_pred, _ = generator(source_images)
            source_dice_loss = dice_criterion(source_preds, source_labels, weights=None)
            source_class_loss = pseudo_labeling_criterion(source_class_pred, source_labels, weights=None)
            
            target_preds, target_class_preds, _ = generator(target_images)
            target_output = F.sigmoid(target_preds)
            target_loss = (-target_output * torch.log(target_output + 1e-5)).sum(1).mean()
            target_class_output = F.sigmoid(target_class_preds)
            target_class_loss = (-target_class_output * torch.log(target_class_output + 1e-5)).sum(1).mean()

            loss = source_dice_loss + source_class_loss + (target_loss + target_class_loss) * lmbda
            loss.backward()

            generator_losses.append(loss.item())
            optimizer_generator.step()

        # test Validation of model on rest of training set
        with torch.no_grad():
            generator.eval()
            train_iou = []
            for (input, labels) in tqdm(source_iterator):
                input = input.to(DEVICE)
                labels = labels.to(DEVICE)
                preds,pseudo_label,_ = generator(input)

                dice_loss = dice_criterion(preds, labels, weights=None)
                class_loss = pseudo_labeling_criterion(pseudo_label, labels, weights=None)
                loss = dice_loss + class_loss
                train_iou.append(metric(preds, labels).item())


        print("-"*100)
        print(f"epoch: {epoch}")
        print(f"Losses:")
        print(f"Generator loss:      {np.mean(generator_losses)}")
        print(f"Accs:")
        print(f"Generator train IoU: {np.mean(train_iou)}")

    torch.save(generator.state_dict(), './unet_16_uda_emin_1.pt')

def train_adversarial_uda():
    # Training loop
    for epoch in range(num_epochs):
        source_iterator = iter(source_dl)

        discriminator_losses = []
        generator_losses = []

        discriminator_source_accs = []
        discriminator_target_accs = []
        generator_accs = []

        for (target_data, _) in tqdm(target_dl):
            # Train the discriminator
            source_data, _ = next(source_iterator)
            source_labels = torch.ones(len(target_data), 1)
            target_labels = torch.zeros(len(target_data), 1)

            optimizer_discriminator.zero_grad()
            generator.eval()

            if len(target_data) != len(source_data):
                # end of batch
                source_data = source_data[0:len(target_data)]
                source_labels = source_labels[0:len(target_data)]
        
            source_images = source_data.to(DEVICE)
            source_labels = source_labels.to(DEVICE)

            target_images = target_data.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            _, _, source_features = generator(source_images)
            source_features = source_features.detach()
            _, _, target_features = generator(target_images)
            target_features = target_features.detach()

            discrim_on_source = discriminator(source_features)
            discrim_on_target = discriminator(target_features)

            discriminator_loss_source = adversarial_loss(discrim_on_source, source_labels) * adversarial_weight_source
            discriminator_loss_target = adversarial_loss(discrim_on_target, target_labels) * adversarial_weight_target
            discriminator_loss = discriminator_loss_source + discriminator_loss_target
            discriminator_losses.append(discriminator_loss.item())
            discriminator_loss.backward()

            discrim_source_accuracy = torch.sum((torch.round(discrim_on_source) == source_labels)) / len(source_labels)
            discrim_target_accuracy = torch.sum((torch.round(discrim_on_target) == target_labels)) / len(target_labels)

            discriminator_source_accs.append(discrim_source_accuracy.cpu().detach())
            discriminator_target_accs.append(discrim_target_accuracy.cpu().detach())

            optimizer_discriminator.step()

            # Train the generator
            generator.train()
            optimizer_generator.zero_grad()

            # _, _, source_features = generator(source_images)
            _, _, target_features = generator(target_images)

            # discrim_from_gen_source = discriminator(source_features)
            discrim_from_gen_target = discriminator(target_features)

            # generator_loss_source = adversarial_loss(discrim_from_gen_source, source_labels) * adversarial_weight_source
            generator_loss_target = adversarial_loss(discrim_from_gen_target, source_labels) * adversarial_weight_target
            # generator_loss = generator_loss_source + generator_loss_target
            generator_loss = generator_loss_target
            generator_losses.append(generator_loss.item())
            generator_loss.backward()

            # generator_accuracy = torch.sum((torch.round(discrim_from_gen_source) == source_labels)) + torch.sum((torch.round(discrim_from_gen_target) == source_labels))
            generator_accuracy = torch.sum((torch.round(discrim_from_gen_target) == source_labels))
            generator_accuracy = generator_accuracy.item()
            # generator_accuracy /= (len(target_labels) + len(source_labels))
            generator_accuracy /= (len(target_labels))
            generator_accs.append(generator_accuracy)

            optimizer_generator.step()

        # test Validation of model on rest of training set
        train_iou = []
        for (input, labels) in tqdm(source_dl_val):
            input = input.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer_generator.zero_grad()
            preds,pseudo_label,_ = generator(input)

            dice_loss = dice_criterion(preds, labels, weights=None)
            class_loss = pseudo_labeling_criterion(pseudo_label, labels, weights=None)
            loss = dice_loss + class_loss
            loss.backward()
            train_iou.append(metric(preds, labels).item())
            optimizer_generator.step()


        print("-"*100)
        print(f"epoch: {epoch}")
        print(f"Losses:")
        print(f"Discriminator loss: {np.mean(discriminator_losses)}")
        print(f"Generator loss: {np.mean(generator_losses)}")
        print(f"Accs:")
        print(f"Discrim on Source:          {np.mean(discriminator_source_accs)}")
        print(f"Discrim on Target:          {np.mean(discriminator_target_accs)}")
        print(f"Generator tricking discrim: {np.mean(generator_accs)}")
        print(f"Generator train IoU:        {np.mean(train_iou)}")


    torch.save(generator.state_dict(), './unet_16_uda_3_augments_only_discrim_target.pt')
    torch.save(discriminator.state_dict(), './discriminator_4.pt')

train_adversarial_uda()
