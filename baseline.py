import torch
import torch.nn as nn
from torchvision.transforms import functional as VF
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


class Baseline(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    num_channels = 64
    self.feature_extractor = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=num_channels*2, out_channels=1, kernel_size=3, padding=1),
    )
    self.classifier = nn.Sequential(
      nn.Sigmoid()
    )
  
  def forward(self, input):
    x = self.feature_extractor(input)
    return self.classifier(x)
  
class DiceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(DiceLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    print('Dice Loss')
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    print("Inputs ", inputs.sum())
    print("Targets ", targets.sum())
    intersection = (inputs * targets).sum()                            
    print("Intersection ", intersection)
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    print("Dice ", dice)
    return 1 - dice


def train(train_dataloaer, validation_dataloader, num_epochs, lr):
  model = Baseline().to(DEVICE)
  optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
  metric = BinaryJaccardIndex().to(DEVICE)
  criterion = DiceLoss().to(DEVICE)
  for e in range(num_epochs):
    model.train()
    train_loss = 0
    train_iou = 0
    for input, labels in tqdm(iter(train_dataloaer)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      output = model(input)
      loss = criterion(output, labels)
      train_loss += loss.detach().item()
      iou = metric(output, labels)
      train_iou += iou.detach().item()
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
        loss = criterion(out, labels)
        iou = metric(out, labels)
        valid_loss += loss.detach().item()
        valid_iou += iou.detach().item()
    
    print(f'Epoch: {e}')
    print(f'Train loss:      {train_loss}')
    print(f'Validation loss: {valid_loss}')
    print(f'Train intersection over union:      {train_iou}')
    print(f'Validation intersection over union: {valid_iou}')
  torch.save(model.state_dict(), 'baseline_model')

if __name__ == '__main__':
  train_dl, valid_dl = build_dataloaders(build_dataframe())
  train(train_dl, valid_dl, 30, 0.001)
