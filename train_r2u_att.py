import torch
import torch.nn as nn
import torch.nn.functional as F
import r2u_att
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
from data_pipeline import build_dataframe, build_dataloaders
from train_unet import DEVICE

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

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, weights, smooth=1):
        if weights == None:
           weights = torch.ones(targets.shape[0], 1, 1, 1).to(DEVICE)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        inputs_flat = inputs
        targets_flat = targets

        intersection = (inputs_flat * targets_flat)
        dice_loss = 1 - (2 * intersection + smooth) / (inputs_flat.sum(dim=0) + targets_flat.sum(dim=0) + smooth)
        bce_loss = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='none')
        combined_loss = bce_loss + dice_loss
        
        return (combined_loss*weights).sum() / len(targets)
    
class PseduoLabelBCELoss(nn.Module):
    def __init__(self):
          super(PseduoLabelBCELoss, self).__init__()

    def forward(self, pseudo_label, targets, weights):
      if weights == None:
            weights = torch.ones(targets.shape[0], 1, 1, 1).to(DEVICE)
          
      tensors = targets.flatten(start_dim=1)
      contains_ones = (tensors == 1).any(dim=1)
      labels = contains_ones.int()
      bce = F.binary_cross_entropy_with_logits(pseudo_label, labels.view(-1,1).float(), reduction='none')
      return (bce * weights).sum() / len(targets)
    
class VerboseReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(VerboseReduceLROnPlateau, self).__init__(*args, **kwargs)

    def step(self, metrics, epoch=None):
        old_lr = self.optimizer.param_groups[0]['lr']
        super(VerboseReduceLROnPlateau, self).step(metrics, epoch)
        if old_lr != self.optimizer.param_groups[0]['lr']:
            print(f"Learning rate reduced from {old_lr} to {self.optimizer.param_groups[0]['lr']}")


def train(train_dataloader, validation_dataloader, num_epochs, lr, from_ckpt=None):
  model = r2u_att.R2AttU_Net(img_ch=1, output_ch=1).to(DEVICE)
  if from_ckpt:
    model.load_state_dict(torch.load(from_ckpt))
  optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
  metric = BinaryJaccardIndex().to(DEVICE)
  dice_criterion = DiceBCELoss().to(DEVICE)
  pseudo_labeling_criterion = PseduoLabelBCELoss()
  scheduler = VerboseReduceLROnPlateau(optimizer, 'max', patience=5)
  for e in range(num_epochs):
    train_loss = 0
    train_iou = 0
    for input, labels in tqdm(iter(train_dataloader)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      preds, pseudo_label = model(input)
      dice_loss = dice_criterion(preds, labels)
      if pseudo_label != None:
        class_loss = pseudo_labeling_criterion(pseudo_label, labels)
      else:
          class_loss = 0
      loss = dice_loss + class_loss
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
        preds, pseudo_label = model(input)
        dice_loss = dice_criterion(preds, labels)
        if pseudo_label != None:
          class_loss = pseudo_labeling_criterion(pseudo_label, labels)
        else:
           class_loss = 0
        loss = dice_loss + class_loss
        iou = metric(preds, labels)
        valid_loss += loss
        valid_iou += iou
    
    print(f'Epoch: {e}')
    print(f'Train loss:      {train_loss / len(train_dataloader)}')
    print(f'Train intersection over union:      {train_iou/ len(train_dataloader)}')
    if (len(validation_dataloader) > 0):
      print(f'Validation loss: {valid_loss/ len(validation_dataloader)}')
      print(f'Validation intersection over union: {valid_iou/ len(validation_dataloader)}')
      scheduler.step(valid_iou / len(validation_dataloader))
    else:
      scheduler.step(train_iou / len(train_dataloader))
  torch.save(model.state_dict(), 'r2u_att_2.pt') 


def train_local(model: nn.Module, train_dataloader, validation_dataloader, lr, num_epochs):
  metric = BinaryJaccardIndex().to(DEVICE)
  dice_criterion = DiceBCELoss().to(DEVICE)
  pseudo_labeling_criterion = PseduoLabelBCELoss().to(DEVICE)
  optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  scheduler = VerboseReduceLROnPlateau(optimizer, 'min', patience=5)

  train_losses = []
  train_ious = []
  valid_losses = []
  valid_ious = []

  for e in range(num_epochs):
    train_loss = 0
    train_iou = 0
    for input, labels in tqdm(iter(train_dataloader)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      preds, pseudo_label = model(input)
      dice_loss = dice_criterion(preds, labels, weights=None)
      if pseudo_label != None:
        class_loss = pseudo_labeling_criterion(pseudo_label, labels, weights=None)
      else:
         class_loss = 0
      loss = dice_loss + class_loss
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
        preds, pseudo_label = model(input)
        dice_loss = dice_criterion(preds, labels, weights=None)
        if pseudo_label != None:
          class_loss = pseudo_labeling_criterion(pseudo_label, labels, weights=None)
        else:
          class_loss = 0
        loss = dice_loss + class_loss
        iou = metric(preds, labels)
        valid_loss += loss
        valid_iou += iou
    
    train_losses.append(train_loss.item() / len(train_dataloader))
    train_ious.append(train_iou.item() / len(train_dataloader))
    valid_losses.append(valid_loss.item() / len(validation_dataloader))
    valid_ious.append(valid_iou.item() / len(validation_dataloader))

    print(f'Epoch: {e}')
    print(f'Train loss:      {train_losses[-1]}')
    print(f'Validation loss: {valid_losses[-1]}')
    print(f'Train intersection over union:      {train_ious[-1]}')
    print(f'Validation intersection over union: {valid_ious[-1]}')
    scheduler.step(valid_losses[-1])
    
  return model, train_losses, train_ious, valid_losses, valid_ious

def train_local_weighted(model: nn.Module, train_dataloader, validation_dataloader, lr, num_epochs):
  metric = BinaryJaccardIndex().to(DEVICE)
  dice_criterion = DiceBCELoss().to(DEVICE)
  pseudo_labeling_criterion = PseduoLabelBCELoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
  scheduler = VerboseReduceLROnPlateau(optimizer, 'min', patience=5)

  train_losses = []
  train_ious = []
  valid_losses = []
  valid_ious = []

  for e in range(num_epochs):
    train_loss = 0
    train_iou = 0
    for input, labels, weights in tqdm(iter(train_dataloader)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      weights= weights.to(DEVICE)
      optimizer.zero_grad()
      preds, pseudo_label = model(input)
      dice_loss = dice_criterion(preds, labels, weights)
      if pseudo_label != None:
        class_loss = pseudo_labeling_criterion(pseudo_label, labels, weights=None)
      else: 
        class_loss = 0
      loss = dice_loss + class_loss
      iou = metric(preds, labels)
      train_loss += loss
      train_iou += iou
      loss.backward()
      optimizer.step()

    valid_loss = 0
    valid_iou = 0
    with torch.no_grad():
      for input, labels, weights in tqdm(iter(validation_dataloader)):
        input = input.to(DEVICE)
        labels = labels.to(DEVICE)
        weights= weights.to(DEVICE)
        preds, pseudo_label = model(input)
        dice_loss = dice_criterion(preds, labels, weights)
        if pseudo_label != None:
          class_loss = pseudo_labeling_criterion(pseudo_label, labels, weights=None)
        else: 
          class_loss = 0
        loss = dice_loss + class_loss
        iou = metric(preds, labels)
        valid_loss += loss
        valid_iou += iou
    
    train_losses.append(train_loss.item() / len(train_dataloader))
    train_ious.append(train_iou.item() / len(train_dataloader))
    valid_losses.append(valid_loss.item() / len(validation_dataloader))
    valid_ious.append(valid_iou.item() / len(validation_dataloader))

    print(f'Epoch: {e}')
    print(f'Train loss:      {train_losses[-1]}')
    print(f'Validation loss: {valid_losses[-1]}')
    print(f'Train intersection over union:      {train_ious[-1]}')
    print(f'Validation intersection over union: {valid_ious[-1]}')
    scheduler.step(valid_losses[-1])
    
  return model, train_losses, train_ious, valid_losses, valid_ious

if __name__ == '__main__':
  df = build_dataframe(use_processed_images=False, limit_well_number=None)
  train_dl, valid_dl = build_dataloaders(df, apply_scaling=True, apply_bulk_data_augmentations=False, split_train=False)
  
