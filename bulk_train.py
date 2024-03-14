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
    print("UNet With Tau 5, Gamma None")
    print("-"*100)
    tau = 5
    train_dl, valid_dl = build_dataloaders_weighted(tau=tau, gamma=None, apply_cutout=False)
    model = unet.UNet(1, 1, n_steps=4, bilinear=False, with_pl=True).to(DEVICE)
    model, tloss, tiou, _, _ = train_unet.train_local_weighted(model, train_dl, valid_dl, lr=0.001, num_epochs=100)
    torch.save(model.state_dict(), './unet_pl_tau_5_gamma_none_100e.pt')

    print("-"*100)
    print("UNet With Cutout")
    print("-"*100)
    train_dl, valid_dl = build_dataloaders(dataframe, apply_cutout=True)
    model = unet.UNet(1, 1, n_steps=4, bilinear=False, with_pl=True).to(DEVICE)
    model, tloss, tiou, _, _ = train_unet.train_local(model, train_dl, valid_dl, lr=.001, num_epochs=100)
    torch.save(model.state_dict(), './unet_pl_cutout_100e.pt')

    print("-"*100)
    print("AttU Net")
    print("-"*100)
    train_dl, valid_dl = build_dataloaders(dataframe, apply_cutout=False)
    model = attunet.AttU_Net(1, 1).to(DEVICE)
    model, tloss, tiou, _, _ = train_r2u_att.train_local(model, train_dl, valid_dl, lr=.001, num_epochs=100)
    torch.save(model.state_dict(), './attunet_100e.pt')

    print("-"*100)
    print("UNet With Tau 5, Gamma 1")
    print("-"*100)
    tau = 5
    gamma = 1
    train_dl, valid_dl = build_dataloaders_weighted(tau=tau, gamma=gamma, apply_cutout=False)
    model = unet.UNet(1, 1, n_steps=4, bilinear=False, with_pl=True).to(DEVICE)
    model, tloss, tiou, _, _ = train_unet.train_local_weighted(model, train_dl, valid_dl, lr=0.001, num_epochs=100)
    torch.save(model.state_dict(), './unet_pl_tau_5_gamma_1_100e.pt')
