import torch
from torch.utils.data import DataLoader
import numpy as np
import data_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import collections
import train_resnet
from backboned_unet import Unet

def main():
  dataframe = data_pipeline.build_dataframe(use_processed_images=False)
  X = torch.from_numpy(np.vstack(dataframe['data'].to_numpy()))
  X = torch.nan_to_num(X)
  Y = torch.from_numpy(np.vstack(dataframe['labels'].to_numpy()))
  
  splits = KFold(n_splits=5, shuffle=True)

  models = [2] # Number of Recurrence steps

  per_model_losses = {}
  per_model_metrics = {}
  for t_steps in models:    
    model_losses = []
    model_metrics = []
    for fold, (train_indices, valid_indices) in enumerate(splits.split(X=X, y=Y)):
      model = Unet(backbone_name='resnet18', classes=1, decoder_filters=(512, 256, 128, 64, 32), pretrained=False)
      
      print("-"*100)
      print(f"ResNet50 Net Fold: {fold}")
      X_train, X_valid = X[train_indices].float().reshape(-1, 36*36), X[valid_indices].float().reshape(-1, 36*36)
      Y_train, Y_valid = Y[train_indices].float(), Y[valid_indices].float()

      # Scale
      scaler = RobustScaler().fit(X_train)
      X_train, X_valid = torch.from_numpy(scaler.transform(X_train)), torch.from_numpy(scaler.transform(X_valid))
      X_train, X_valid = X_train.float().reshape(-1, 1, 36, 36), X_valid.float().reshape(-1, 1, 36, 36)
      Y_train, Y_valid = Y_train.reshape(-1, 1, 36, 36), Y_valid.reshape(-1, 1, 36, 36)

      train_dataset = data_pipeline.WellsDataset(X_train, Y_train, 
                                                 transform=lambda x,y,z: data_pipeline.image_label_transforms(x,y,z,apply_cutout=False))
      valid_dataset = data_pipeline.WellsDataset(X_valid, Y_valid, transform=None)

      train_dataloader = DataLoader(train_dataset, batch_size=64)
      valid_dataloader = DataLoader(valid_dataset, batch_size=64)
      
      # Train
      _, _, _, valid_losses, valid_metrics = train_resnet.train_local(model, train_dataloader, valid_dataloader, num_epochs=50)
      model_losses.append(valid_losses[-1])
      model_metrics.append(valid_metrics[-1])

    per_model_losses[t_steps] = np.mean(model_losses)
    per_model_metrics[t_steps] = np.mean(model_metrics)
    print(f"Finish folds for Model with t_steps: {t_steps}")
    print(f"Average Loss: {per_model_losses[t_steps]}")
    print(f"Average IoU   {per_model_metrics[t_steps]}")
  
  print("-"*100)
  print("Finished full KFolds testing")
  for key in per_model_losses.keys():
    print(f"R2U(): loss={per_model_losses[key]}, IoU={per_model_metrics[key]}")

if __name__=='__main__':
  main()


    