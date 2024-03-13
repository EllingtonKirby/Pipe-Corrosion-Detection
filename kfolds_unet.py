import torch
from torch.utils.data import DataLoader
import numpy as np
import train_unet
import unet
import data_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import collections

def main():
  dataframe = data_pipeline.build_dataframe(use_processed_images=False)
  X = torch.from_numpy(np.vstack(dataframe['data'].to_numpy()))
  X = torch.nan_to_num(X)
  Y = torch.from_numpy(np.vstack(dataframe['labels'].to_numpy()))
  
  well_numbers = dataframe['well_number']
  wells = torch.from_numpy(np.vstack(well_numbers.to_numpy())) - 1
  well_mean_weight = np.mean(dataframe['well_number'].value_counts().values)
  sample_weight = {well: ratio/well_mean_weight for well, ratio in dataframe['well_number'].value_counts().items()}
  sample_weight = {well: 1/ratio for well, ratio in sample_weight.items()}
  sample_weight = collections.OrderedDict(sorted(sample_weight.items()))
  sample_weight = torch.tensor(list(sample_weight.values()))
  wells = sample_weight[wells].flatten()

  splits = KFold(n_splits=5, shuffle=True)

  models = [4] # Number of UNet steps

  per_model_losses = {}
  per_model_metrics = {}
  for steps in models:    
    model_losses = []
    model_metrics = []
    for fold, (train_indices, valid_indices) in enumerate(splits.split(X=X, y=Y)):
      model = unet.UNet(n_channels=1, n_classes=1, n_steps=steps).to(torch.device('cuda:0'))
      print("-"*100)
      print(f"Unet with Steps: {model.n_steps}, Fold: {fold}")
      X_train, X_valid = X[train_indices].float().reshape(-1, 36*36), X[valid_indices].float().reshape(-1, 36*36)
      Y_train, Y_valid = Y[train_indices].float(), Y[valid_indices].float()
      Wells_train, Wells_valid = wells[train_indices].float().reshape(-1, 1, 1, 1), wells[valid_indices].float().reshape(-1, 1, 1, 1)

      # Scale
      scaler = RobustScaler().fit(X_train)
      X_train, X_valid = torch.from_numpy(scaler.transform(X_train)), torch.from_numpy(scaler.transform(X_valid))
      X_train, X_valid = X_train.float().reshape(-1, 1, 36, 36), X_valid.float().reshape(-1, 1, 36, 36)
      Y_train, Y_valid = Y_train.reshape(-1, 1, 36, 36), Y_valid.reshape(-1, 1, 36, 36)

      train_dataset = data_pipeline.WellsDataset(X_train, Y_train, transform=data_pipeline.image_label_transforms,  wells=Wells_train)
      valid_dataset = data_pipeline.WellsDataset(X_valid, Y_valid, transform=None, wells=Wells_valid)

      train_dataloader = DataLoader(train_dataset, batch_size=128)
      valid_dataloader = DataLoader(valid_dataset, batch_size=128)
      
      # Train
      _, _, _, valid_losses, valid_metrics = train_unet.train_local_weighted(model, train_dataloader, valid_dataloader, lr=.001, num_epochs=50)
      model_losses.append(valid_losses[-1])
      model_metrics.append(valid_metrics[-1])

    per_model_losses[steps] = np.mean(model_losses)
    per_model_metrics[steps] = np.mean(model_metrics)
    print(f"Finish folds for Model with Steps: {steps}")
    print(f"Average Loss: {per_model_losses[steps]}")
    print(f"Average IoU   {per_model_metrics[steps]}")
  
  print("-"*100)
  print("Finished full KFolds testing")
  for key in per_model_losses.keys():
    print(f"UNet(steps={key}): loss={per_model_losses[key]}, IoU={per_model_metrics[key]}")

if __name__=='__main__':
  main()


    