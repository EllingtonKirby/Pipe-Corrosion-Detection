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
    print("UNet With Tau 2, Gamma None")
    print("-"*100)
    tau = 2
    train_dl, _ = build_dataloaders_weighted(tau=tau, gamma=None, apply_cutout=False)
    model = unet.UNet(1, 1, n_steps=4, bilinear=False, with_pl=True).to(DEVICE)
    model, tloss, tiou, _, _ = train_unet.train_local_weighted(model, train_dl, None, lr=0.001, num_epochs=100)
    torch.save(model.state_dict(), './unet_pl_tau_2_gamma_none_100e.pt')
