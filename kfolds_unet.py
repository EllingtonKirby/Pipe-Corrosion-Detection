import torch
import numpy as np
import train_unet
import unet
import data_pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

def main():
  dataframe = data_pipeline.build_dataframe(use_processed_images=False)
  X = torch.from_numpy(np.vstack(dataframe['data'].to_numpy()))
  X = torch.nan_to_num(X)
  Y = torch.from_numpy(np.vstack(dataframe['labels'].to_numpy()))
  splits = KFold(n_splits=5, shuffle=True).split(X=X, y=Y)

  models = [
    unet.UNet(1, 1, n_steps=1),
    unet.UNet(1, 1, n_steps=2),
    unet.UNet(1, 1, n_steps=3),
    unet.UNet(1, 1, n_steps=4),
  ]

  per_model_losses = {}
  per_model_metrics = {}
  for model in models:
    model_losses = []
    model_metrics = []
    for fold, (train_indices, valid_indices) in enumerate(splits):
      print("-"*100)
      print(f"Unet with Steps: {model.n_steps}, Fold: {fold}")
      X_train, X_valid = X[train_indices].float().reshape(1, -1), X[valid_indices].float().reshape(1, -1)
      Y_train, Y_valid = Y[train_indices].float(), Y[valid_indices].float()

      # Scale
      scaler = RobustScaler().fit(X_train)
      X_train, X_valid = scaler.transform(X_train), scaler.transform(X_valid)
      X_train, X_valid = X_train.reshape(-1, 1, 36, 36), X_valid.reshape(-1, 1, 36, 36)
      Y_train, Y_valid = Y_train.reshape(-1, 1, 36, 36), Y_valid.reshape(-1, 1, 36, 36)

      train_dl = data_pipeline.WellsDataset(X_train, Y_train, transform=data_pipeline.image_label_transforms)
      valid_dl = data_pipeline.WellsDataset(X_valid, Y_valid, transform=None)
      
      # Train
      _, _, _, valid_losses, valid_metrics = train_unet.train_local(model, train_dl, valid_dl, lr=.001, num_epochs=100)
      model_losses.append(valid_losses[-1])
      model_metrics.append(valid_metrics[-1])
      
    per_model_losses[model.n_steps] = torch.mean(model_losses)
    per_model_metrics[model.n_steps] = torch.mean(model_metrics)
    print(f"Finish folds for Model with Steps: {model.n_steps}")
    print(f"Average Loss: {per_model_losses[model.n_steps]}")
    print(f"Average IoU   {per_model_metrics[model.n_steps]}")
  
  print("-"*100)
  print("Finished full KFolds testing")
  for key in per_model_losses.keys():
    print(f"UNet(steps={key}): loss={per_model_losses[key]}, IoU={per_model_metrics[key]}")

if __name__=='__main__':
  main()


    