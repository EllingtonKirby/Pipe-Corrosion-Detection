import torch
import torch.nn as nn
import torch.nn.functional as F
import unet
import train_unet
import attunet
import r2u_att
import train_r2u_att
from train_unet import DEVICE
from data_pipeline import build_dataframe, build_dataloaders, build_dataloaders_weighted

if __name__ == '__main__':
    dataframe = build_dataframe(use_processed_images=False, limit_well_number=None)

    print("-"*100)
    print("AttU 50 Epochs")
    print("-"*100)
    train_dl, _ = build_dataloaders(dataframe, apply_cutout=False)
    model = attunet.AttU_Net(img_ch=1, output_ch=1).to(DEVICE)
    model, tloss, tiou, _, _ = train_r2u_att.train_local(model, train_dl, None, lr=0.001, num_epochs=50)
    torch.save(model.state_dict(), './attu_e50.pt')
