import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pandas as pd
from tqdm import tqdm
import data_pipeline
import unet
import numpy as np
from skimage import morphology

if __name__ == '__main__':
  model = unet.UNet(n_channels=1, n_classes=1, n_steps=4)
  model.load_state_dict(torch.load('./checkpoints/unet/unet_15.pt'))
  model = model.cuda()
  test_df = data_pipeline.build_test_dataframe(use_processed_images=False, limit_well_number=None)
  train_df = data_pipeline.build_dataframe(use_processed_images=False, limit_well_number=None)

  test_data = torch.from_numpy(np.vstack(test_df['data'].to_numpy()))
  test_data = torch.nan_to_num(test_data)
  outliers = ((test_data.min(dim=1, keepdim=True).values < -10) == True).flatten()

  X_test, X_names, X_train, Y_train = data_pipeline.build_test_dataloaders(test_df, train_df, apply_scaling=True)
  test_dl = DataLoader(TensorDataset(X_test), batch_size=1)
  with torch.no_grad():
    predictions = {}
    model.eval()
    for index, x in tqdm(enumerate(test_dl)):
      input = x[0].cuda()
      out = model(input)
      preds = (F.sigmoid(out) > .5)*1.
      preds = preds.cpu().detach()
      preds = morphology.binary_dilation(preds)
      if outliers[index]:
        preds = torch.zeros_like(out).cpu().detach()
      name = X_names[index][0]
      predictions[name] = preds.flatten().tolist()
  preds_df = pd.DataFrame.from_dict(predictions, orient='index')
  preds_df.to_csv('./outputs/KIRBY_predictions_14.csv')
