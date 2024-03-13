import torch
from torch.utils.data import DataLoader
import numpy as np
import train_r2u_att
import r2u_att
import data_pipeline
import attunet
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

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
      model = r2u_att.R2AttU_Net(img_ch=1, output_ch=1).to(torch.device('cuda:0'))
      print("-"*100)
      print(f"R2UAtt Net Fold: {fold}")
      X_train, X_valid = X[train_indices].float().reshape(-1, 36*36), X[valid_indices].float().reshape(-1, 36*36)
      Y_train, Y_valid = Y[train_indices].float(), Y[valid_indices].float()

      # Scale
      scaler = RobustScaler().fit(X_train)
      X_train, X_valid = torch.from_numpy(scaler.transform(X_train)), torch.from_numpy(scaler.transform(X_valid))
      X_train, X_valid = X_train.float().reshape(-1, 1, 36, 36), X_valid.float().reshape(-1, 1, 36, 36)
      Y_train, Y_valid = Y_train.reshape(-1, 1, 36, 36), Y_valid.reshape(-1, 1, 36, 36)

      train_dataset = data_pipeline.WellsDataset(X_train, Y_train, transform=data_pipeline.image_label_transforms)
      valid_dataset = data_pipeline.WellsDataset(X_valid, Y_valid, transform=None)

      train_dataloader = DataLoader(train_dataset, batch_size=128)
      valid_dataloader = DataLoader(valid_dataset, batch_size=128)
      
      # Train
      _, _, _, valid_losses, valid_metrics = train_r2u_att.train_local(model, train_dataloader, valid_dataloader, lr=.001, num_epochs=50)
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
    print(f"R2UAtt(): loss={per_model_losses[key]}, IoU={per_model_metrics[key]}")

if __name__=='__main__':
  main()


    