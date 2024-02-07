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
    # inputs = F.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    print("Dice Loss")
    print("Inputs: ", inputs)
    print("Target: ", inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    print("Inputs: ", inputs)
    print("Target: ", inputs)
    intersection = (inputs * targets).sum()                            
    print("Intersection: ", intersection)
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    print("Dice: ", dice)
    return 1 - dice


def train(train_dataloader, validation_dataloader, num_epochs, lr):
  model = unet.UNet(n_channels=1, n_classes=2).to(DEVICE)
  optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
  metric = BinaryJaccardIndex().to(DEVICE)
  criterion = DiceLoss().to(DEVICE)
  predictor = nn.Softmax(dim=1)
  for e in range(num_epochs):
    model.train()
    train_loss = 0
    train_iou = 0
    for input, labels in tqdm(iter(train_dataloader)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      output = model(input)
      print("Output: ", output)
      preds = predictor(output)
      print("Preds1: ", preds)
      preds = torch.argmax(preds, dim=1)
      preds("Preds2: ", preds)
      loss = criterion(preds, labels)
      iou = metric(preds.view(-1, 1, 36, 36), labels)
      train_loss += loss
      train_iou += iou
      loss.backward()
      optimizer.step()

    model.eval()
    valid_loss = 0
    valid_iou = 0
    with torch.no_grad():
      for input, labels in tqdm(iter(validation_dataloader)):
        input = input.to(DEVICE)
        labels = labels.to(DEVICE)
        out = model(input)
        preds = torch.argmax(predictor(out), dim=1)
        loss = criterion(preds.view(-1, 1, 36, 36), labels)
        iou = metric(preds, labels)
        valid_loss += loss
        valid_iou += iou
    
    print(f'Epoch: {e}')
    print(f'Train loss:      {train_loss / len(train_dataloader)}')
    print(f'Validation loss: {valid_loss/ len(validation_dataloader)}')
    print(f'Train intersection over union:      {train_iou/ len(train_dataloader)}')
    print(f'Validation intersection over union: {valid_iou/ len(validation_dataloader)}')
  torch.save(model.state_dict(), 'unet_1.pt')

if __name__ == '__main__':
  train_dl, valid_dl = build_dataloaders(build_dataframe())
  train(train_dl, valid_dl, 30, 0.001)
