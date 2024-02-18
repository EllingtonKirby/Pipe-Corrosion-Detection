import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pandas as pd
from tqdm import tqdm
import data_pipeline
import unet

if __name__ == '__main__':
  model = unet.UNet(n_channels=1, n_classes=1, n_steps=4)
  model.load_state_dict(torch.load('./checkpoints/unet/unet_15.pt', map_location=torch.device('cpu')))
  test_df = data_pipeline.build_test_dataframe(use_processed_images=False)
  train_df = data_pipeline.build_dataframe(use_processed_images=False)
  X_test, X_names, X_train = data_pipeline.build_test_dataloaders(test_df, train_df, apply_scaling=True)
  test_dl = DataLoader(TensorDataset(X_test), batch_size=1)
  with torch.no_grad():
    predictions = {}
    model.eval()
    for index, x in tqdm(enumerate(test_dl)):
      out = model(x[0])
      preds = (F.sigmoid(out) > .5)*1.
      name = X_names[index][0]
      predictions[name] = preds.flatten().tolist()
  preds_df = pd.DataFrame.from_dict(predictions, orient='index')
  preds_df.to_csv('KIRBY_predictions_12.csv')
