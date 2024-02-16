import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import data_pipeline
import unet

global DEVICE
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda:0')
  print("CUDA is available and is used")
elif not torch.backends.mps.is_available():
  if not torch.backends.mps.is_built():
    print("MPS not available because the current PyTorch install was not "
          "built with MPS enabled.")
  else:
      print("MPS not available because the current MacOS version is not 12.3+ "
          "and/or you do not have an MPS-enabled device on this machine.")
  DEVICE = torch.device('cpu')
  print("CUDA and MPS are not available, switching to CPU.")
else:
  DEVICE = torch.device("mps")
  print("CUDA not available, switching to MPS")

if __name__ == '__main__':
  model = unet.UNet(n_channels=1, n_classes=1)
  model.load_state_dict(torch.load('./unet_13.pt'))
  test_df = data_pipeline.build_test_dataframe(use_processed_images=False)
  train_df = data_pipeline.build_dataframe(use_processed_images=False)
  X_test, X_names, X_train = data_pipeline.build_test_dataloaders(test_df, train_df, apply_scaling=True)
  test_dl = DataLoader(TensorDataset(X_test), batch_size=1)
  predictions = {}
  model.eval()
  for index, x in tqdm(enumerate(test_dl)):
    x = x[0].to(DEVICE)
    out = model(x)
    preds = (F.sigmoid(out) > .5)*1.
    name = X_names[index][0]
    predictions[name] = preds.detach().flatten().tolist()
  preds_df = pd.DataFrame.from_dict(predictions, orient='index')
  preds_df.to_csv('KIRBY_predictions_9.csv')
