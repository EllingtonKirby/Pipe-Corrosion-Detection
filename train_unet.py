import torch
import torch.nn as nn
import torch.nn.functional as F
import unet
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
from data_pipeline import build_dataframe, build_dataloaders

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


class DiceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(DiceLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    return 1 - dice


def train(train_dataloader, validation_dataloader, num_epochs, lr, from_ckpt=None):
  model = unet.UNet(n_channels=1, n_classes=1).to(DEVICE)
  if from_ckpt:
    model.load_state_dict(torch.load(from_ckpt))
  optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
  metric = BinaryJaccardIndex().to(DEVICE)
  criterion = DiceLoss().to(DEVICE)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
  for e in range(num_epochs):
    train_loss = 0
    train_iou = 0
    for input, labels in tqdm(iter(train_dataloader)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      preds = model(input)
      loss = criterion(preds, labels)
      iou = metric(preds, labels)
      train_loss += loss
      train_iou += iou
      loss.backward()
      optimizer.step()

    valid_loss = 0
    valid_iou = 0
    with torch.no_grad():
      for input, labels in tqdm(iter(validation_dataloader)):
        input = input.to(DEVICE)
        labels = labels.to(DEVICE)
        preds = model(input)
        loss = criterion(preds, labels)
        iou = metric(preds, labels)
        valid_loss += loss
        valid_iou += iou
    
    print(f'Epoch: {e}')
    print(f'Train loss:      {train_loss / len(train_dataloader)}')
    print(f'Validation loss: {valid_loss/ len(validation_dataloader)}')
    print(f'Train intersection over union:      {train_iou/ len(train_dataloader)}')
    print(f'Validation intersection over union: {valid_iou/ len(validation_dataloader)}')
    scheduler.step(valid_iou / len(validation_dataloader))
  torch.save(model.state_dict(), 'unet_7.pt')

if __name__ == '__main__':
  train_dl, valid_dl = build_dataloaders(build_dataframe())
  train(train_dl, valid_dl, 50, 0.001, './unet_7.pt')
