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
    print("R2U 50 Epochs")
    print("-"*100)
    tau = 2
    train_dl, _ = build_dataloaders(dataframe, apply_cutout=False)
    model = r2u_att.R2U_Net(img_ch=1, output_ch=1)
    model, tloss, tiou, _, _ = train_r2u_att.train_local(model, train_dl, None, lr=0.001, num_epochs=50)
    torch.save(model.state_dict(), './r2u_e50.pt')
