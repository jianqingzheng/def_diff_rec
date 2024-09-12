import os
import torch
import torchvision
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from Diffusion.diffuser import DeformDDPM
from Diffusion.networks import get_net, STN
from torchvision.transforms import Lambda
import Diffusion.losses as losses
import random
import glob
import numpy as np
import utils
from Dataloader.dataloader import get_dataloader
from Dataloader.dataloader_utils import thresh_img
import yaml
import argparse

use_parallel=False

EPS = 10e-5

parser = argparse.ArgumentParser()

# config_file_path = 'Config/config_cmr.yaml'
parser.add_argument(
        "--config_path",
        "-C",
        help="Path for the config file",
        type=str,
        default="Config/config_cmr.yaml",
        # default="Config/config_lct.yaml",
        required=False,
    )
args = parser.parse_args()
#=======================================================================================================================

# Load the YAML file into a dictionary
with open(args.config_path, 'r') as file:
    hyp_parameters = yaml.safe_load(file)
    print(hyp_parameters)



# epoch_per_save=10
epoch_per_save=hyp_parameters['epoch_per_save']

data_name=hyp_parameters['data_name']
net_name = hyp_parameters['net_name']

Net=get_net(net_name)

suffix_pth=f'_{data_name}_{net_name}.pth'
model_save_path = f'models/{data_name}_{net_name}/'
model_dir=model_save_path
transformer=utils.get_transformer(img_sz=hyp_parameters["ndims"]*[hyp_parameters['img_size']])
Data_Loader=get_dataloader(hyp_parameters['data_name'],mode='train')

tsfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])


dataset = Data_Loader(target_res = [hyp_parameters["img_size"]]*hyp_parameters["ndims"], transforms=None, noise_scale=hyp_parameters['noise_scale'])
train_loader = DataLoader(
    dataset,
    batch_size=hyp_parameters['batchsize'],
    # shuffle=False,
    shuffle=True,
    drop_last=True,
)



Deformddpm = DeformDDPM(
    # network=Net(n_steps=hyp_parameters["timesteps"],ndims=ndims,num_input_chn=num_input_chn,res=hyp_parameters['img_size']),
    network=Net(n_steps=hyp_parameters["timesteps"], ndims=hyp_parameters["ndims"], num_input_chn=1),
    n_steps=hyp_parameters["timesteps"],
    image_chw=[1] + [hyp_parameters["img_size"]]*hyp_parameters["ndims"],
    device=hyp_parameters["device"],
    batch_size=hyp_parameters["batchsize"],
    img_pad_mode=hyp_parameters["img_pad_mode"],
    v_scale=hyp_parameters["v_scale"],
)


# ddf_enc = DDF_Encoder(ndims=ndims,img_sz = hyp_parameters['img_size'], batch_sz = hyp_parameters['batchsize'])
ddf_stn = STN(
    img_sz=hyp_parameters["img_size"],
    ndims=hyp_parameters["ndims"],
    # padding_mode="zeros",
    padding_mode=hyp_parameters["padding_mode"],
    device=hyp_parameters["device"],
)

if use_parallel:
    Deformddpm = nn.DataParallel(Deformddpm)
    ddf_stn = nn.DataParallel(ddf_stn)
Deformddpm.to(hyp_parameters["device"])
ddf_stn.to(hyp_parameters["device"])

# mse = nn.MSELoss()
loss_reg = losses.Grad(penalty=['l1', 'negdetj'], ndims=hyp_parameters["ndims"])
loss_dist = losses.MRSE(img_sz=hyp_parameters["img_size"])
# loss_ang = losses.MRSE(img_sz=hyp_parameters["img_size"])
loss_ang = losses.NCC(img_sz=hyp_parameters["img_size"])

optimizer = Adam(Deformddpm.parameters(), lr=hyp_parameters["lr"])
# hyp_parameters["lr"]=0.00000001
# # optimizer = SGD(Deformddpm.parameters(), lr=hyp_parameters["lr"], momentum=0.95)
# optimizer = SGD(Deformddpm.parameters(), lr=hyp_parameters["lr"], momentum=0.9)

# # LR scheduler ----- YHM
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, hyp_parameters["lr"], hyp_parameters["lr"]*10, step_size_up=500, step_size_down=500, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

# Deformddpm.network.load_state_dict(torch.load('/home/data/jzheng/Adaptive_Motion_Generator-master/models/1000.pth'))

# check for existing models

model_files = glob.glob(model_dir + "/*.pth")
model_files.sort()
print(model_files)
if model_files:
    # if there are any model files, load the most recent one
    latest_model_file = model_files[-1]
    # Deformddpm.network.load_state_dict(torch.load(latest_model_file))
    Deformddpm.network.load_state_dict(torch.load(latest_model_file), strict=False)
    # get the epoch number from the filename and add 1 to set as initial_epoch
    initial_epoch = int(latest_model_file.split('/')[-1].split('.')[0][:6]) + 1
else:
    initial_epoch = 0
print('len_train_data: ',len(dataset))
for epoch in range(initial_epoch,hyp_parameters["epoch"]):

    epoch_loss_tot = 0.0
    epoch_loss_gen_d = 0.0
    epoch_loss_gen_a = 0.0
    epoch_loss_reg = 0.0
    # Set model inside to train model
    Deformddpm.train()

    for step, batch in enumerate(train_loader):
        # x0, _ = batch
        x0, _, _ = batch
        x0 = x0.to(hyp_parameters["device"]).type(torch.float32)
      
        n = x0.size()[0]  # batch_size -> n
        x0 = x0.to(hyp_parameters["device"])
        # random deformation + rotation
        if hyp_parameters["ndims"]>2:
            if np.random.uniform(0,1)<0.6:
                x0 = utils.random_resample(x0, deform_scale=0)
        x0 = transformer(x0)
        if hyp_parameters['noise_scale']>0:
            x0 = thresh_img(x0, [0, 2*hyp_parameters['noise_scale']])
            x0 = x0 * (np.random.normal(1, hyp_parameters['noise_scale'] * 1)) + np.random.normal(0, hyp_parameters['noise_scale'] * 1)

        # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
        t = torch.randint(0, hyp_parameters["timesteps"], (n,)).to(
            hyp_parameters["device"]
        )  # pick up a seq of rand number from 0 to 'timestep'

        # noisy_imgs, dvf_I = ddf_enc(img= x0, t)
        noisy_imgs, dvf_I,_ = Deformddpm(x0, t)
        # pre_dvf_I = Deformddpm.backward(noisy_imgs, t.reshape(16, -1))
        pre_dvf_I = Deformddpm.backward(noisy_imgs, t)

        loss_tot=0

        loss_ddf = loss_reg(pre_dvf_I)
        trm_pred = ddf_stn(pre_dvf_I, dvf_I)
        loss_gen_d = loss_dist(pred=trm_pred,inv_lab=dvf_I,ddf_stn=None)
        loss_gen_a = loss_ang(pred=trm_pred,inv_lab=dvf_I,ddf_stn=None)

        loss_tot += 1.0 * loss_gen_d + 1.0 * loss_gen_a
        loss_tot +=10 * loss_ddf
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        epoch_loss_tot += loss_tot.item() * len(x0) / len(train_loader.dataset)
        epoch_loss_gen_d += loss_gen_d.item() * len(x0) / len(train_loader.dataset)
        epoch_loss_gen_a += loss_gen_a.item() * len(x0) / len(train_loader.dataset)
        epoch_loss_reg += loss_ddf.item() * len(x0) / len(train_loader.dataset)
        # print('step:',step,':', loss_tot.item(),'=',loss_gen_a.item(),'+', loss_gen_d.item(),'+',loss_ddf.item())

    print(epoch,':', epoch_loss_tot,'=',epoch_loss_gen_a,'+', epoch_loss_gen_d,'+',epoch_loss_reg, ' (ang+dist+regul)')

    # # LR schedular step ----- YHM
    # scheduler.step()

    if 0 == epoch % epoch_per_save:
        save_dir=model_save_path + str(epoch).rjust(6, '0') + suffix_pth
        if os.path.exists(model_save_path):
            print(f"saved in {save_dir}")
        else:
            os.makedirs(os.path.dirname(model_save_path))
        break   # FOR TESTING
        torch.save(Deformddpm.network.state_dict(), save_dir)
        
