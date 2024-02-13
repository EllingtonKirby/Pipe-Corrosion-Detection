import torch
import torch.nn as nn
import torch.nn.functional as F
import pipe_identifier
from tqdm import tqdm
from data_pipeline import build_dataframe, build_dataloaders_for_classiication

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


def train(train_dataloader, validation_dataloader, num_epochs, lr):
  model = pipe_identifier.PipeIdentifier(num_classes=15).to(DEVICE)
  optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
  criterion = nn.CrossEntropyLoss().to(DEVICE)
  for e in range(num_epochs):
    train_loss = 0
    train_acc = 0
    for input, labels in tqdm(iter(train_dataloader)):
      input = input.to(DEVICE)
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      preds = model(input)
      loss = criterion(preds, labels)
      acc = torch.mean((torch.argmax(F.softmax(preds), dim=1) == labels)*1.)
      train_loss += loss
      train_acc += acc
      loss.backward()
      optimizer.step()

    valid_loss = 0
    valid_acc = 0
    with torch.no_grad():
      for input, labels in tqdm(iter(validation_dataloader)):
        input = input.to(DEVICE)
        labels = labels.to(DEVICE)
        preds = model(input)
        loss = criterion(preds, labels)
        acc = torch.mean((torch.argmax(F.softmax(preds), dim=1) == labels)*1.)
        valid_loss += loss
        valid_acc += acc
    
    print(f'Epoch: {e}')
    print(f'Train loss:      {train_loss / len(train_dataloader)}')
    print(f'Validation loss: {valid_loss/ len(validation_dataloader)}')
    print(f'Train accuracy:      {train_acc/ len(train_dataloader)}')
    print(f'Validation accuracy: {valid_acc/ len(validation_dataloader)}')
  torch.save(model.state_dict(), 'well_classifier_1.pt')

if __name__ == '__main__':
  train_dl, valid_dl = build_dataloaders_for_classiication(build_dataframe())
  train(train_dl, valid_dl, 50, 0.001)
