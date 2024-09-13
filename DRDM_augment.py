import torch
import torchvision
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
from Diffusion.diffuser import DeformDDPM
from Diffusion.networks import get_net, STN
from torchvision.transforms import Lambda
import random
import os
import utils
from Dataloader.dataloader import get_dataloader
from torchvision.utils import save_image
from einops import rearrange, reduce, repeat
# import matplotlib.image
import numpy as np
import nibabel as nib
from tqdm import tqdm 
import yaml
import argparse

EPS = 10e-8

parser = argparse.ArgumentParser()

parser.add_argument(
        "--config",
        "-C",
        help="Path for the config file",
        type=str,
        default="Config/config_cmr.yaml",
        # default="Config/config_lct.yaml",
        required=False,
    )
args = parser.parse_args()
#=======================================================================================================================

# config_path = 'Config/config_cmr.yaml'
# config_path = 'Config/config_lct.yaml'

# Load the YAML file into a dictionary
with open(args.config_path, 'r') as file:
    hyp_parameters = yaml.safe_load(file)
    print(hyp_parameters)
# hyp_parameters["aug_img_savepath"] = os.path.join(hyp_parameters["aug_img_savepath"],hyp_parameters["data_name"],'')
if not os.path.exists(hyp_parameters["aug_img_savepath"]):
    os.makedirs(hyp_parameters["aug_img_savepath"])
if not os.path.exists(hyp_parameters["aug_msk_savepath"]):
    os.makedirs(hyp_parameters["aug_msk_savepath"])
if not os.path.exists(hyp_parameters["aug_ddf_savepath"]):
    os.makedirs(hyp_parameters["aug_ddf_savepath"])
print(hyp_parameters["aug_img_savepath"])

Data_Loader=get_dataloader(hyp_parameters['data_name'],mode='aug')
transformer = utils.get_transformer(img_sz=hyp_parameters["ndims"]*[hyp_parameters['img_size']])



epoch=f'{hyp_parameters["model_id_str"]}_{hyp_parameters["data_name"]}_{hyp_parameters["net_name"]}'
model_save_path = f'models/{hyp_parameters["data_name"]}_{hyp_parameters["net_name"]}/'
model_save_path = os.path.join(model_save_path, str(epoch)+'.pth')

dataset = Data_Loader(patient_index = hyp_parameters["patients_list"])
train_loader = DataLoader(dataset, batch_size = hyp_parameters['batchsize'], shuffle = False) 


Net = get_net(hyp_parameters["net_name"])

Deformddpm = DeformDDPM(
    network=Net(n_steps = hyp_parameters["timesteps"],
                ndims = hyp_parameters["ndims"],
                num_input_chn = hyp_parameters["num_input_chn"],
                res = hyp_parameters['img_size']
                ),
    n_steps = hyp_parameters["timesteps"],
    image_chw = [hyp_parameters["num_input_chn"]] + [hyp_parameters["img_size"]]*hyp_parameters["ndims"],
    device = hyp_parameters["device"],
    batch_size = hyp_parameters["batchsize"],
    img_pad_mode = hyp_parameters["img_pad_mode"],
    ddf_pad_mode = hyp_parameters["ddf_pad_mode"],
    padding_mode = hyp_parameters["padding_mode"],
    v_scale = hyp_parameters["v_scale"],
    resample_mode = hyp_parameters["resample_mode"],
)
Deformddpm.to(hyp_parameters["device"])

ddf_stn = STN(
    img_sz = hyp_parameters["img_size"],
    ndims = hyp_parameters["ndims"],
    padding_mode = hyp_parameters['padding_mode'],
    device = hyp_parameters["device"],
)
ddf_stn.to(hyp_parameters["device"])
Deformddpm.network.load_state_dict(torch.load(model_save_path))
Deformddpm.eval()

print("total num of image:", len(train_loader))
for e, d in tqdm(enumerate(train_loader)):
  
  img, mask, pid = d
  pid = pid.cpu().detach().numpy()
  pid = pid[0] 
  print('Processing to patient:', pid, ' image:',e)
  
  img = img.to(hyp_parameters["device"]) 
  img = img.type(torch.float32)
  image_original = img.cpu().detach().numpy()

  mask = mask.to(hyp_parameters["device"]) 
  mask = mask.type(torch.float32)
  mask_original = mask.cpu().detach().numpy()
  # print(pid, image_original.shape, mask_original.max())


  if hyp_parameters["ndims"] == 2:
    nifti_img = nib.Nifti1Image(image_original[0,0,:,:], np.eye(4))  
    nifti_mask = nib.Nifti1Image(mask_original[0,:,:,:], np.eye(4))  
  elif hyp_parameters["ndims"] == 3:
    nifti_img = nib.Nifti1Image(image_original[0,0,:,:,:], np.eye(4))  
    nifti_mask = nib.Nifti1Image(mask_original[0,0,:,:,:], np.eye(4))  

  # Saving original (undeformed image)
  # CMR: format: Patient0001_Slice0001_ORG_NA.nii.gz
  # Lung CT: Patient0001_Slice0001_ORG_NA.nii.gz
  nib.save(nifti_img, os.path.join(hyp_parameters['aug_img_savepath'],utils.get_barcode([pid,e])+'.nii.gz'))

  # Saving original (undeformed image)
  # CMR: format: Patient0001_Slice0001_ORG_NA_GT.nii.gz
  # Lung CT: ...
  nib.save(nifti_img, os.path.join(hyp_parameters['aug_msk_savepath'],utils.get_barcode([pid,e])+'_GT.nii.gz'))

 
  noise_step = hyp_parameters["start_noise_step"]
  with torch.no_grad():
    for im in range(hyp_parameters["aug_coe"]):
      # Permute 
      if hyp_parameters["ndims"] == 2:
        [img, mask] = utils.random_permute([img, mask], select_dims=[-1, -2])          # add random rotation to image
      elif hyp_parameters["ndims"] == 3:
        [img, mask] = utils.random_permute([img, mask], select_dims=[-1, -2, -3])  # add random rotation to image

      print('Generating - >', 'Subject-',pid,', Scan-',e,' (',im,'/',hyp_parameters["aug_coe"],')', end='\r')
      
      [ddf_comp,ddf_rand],[img_rec,img_diff,img_save],[msk_rec,msk_diff,msk_save] = Deformddpm.diff_recover(img_org=img,msk_org=mask,T=[noise_step,hyp_parameters["timesteps"]],v_scale=hyp_parameters["v_scale"],t_save=None)
      
      denoise_imgs = img_rec.cpu().detach().numpy()
      denoise_msks = msk_rec.cpu().detach().numpy()
      noisy_imgs_np = img_diff.cpu().detach().numpy()
      noisy_msks_np = msk_diff.cpu().detach().numpy()

      if hyp_parameters["ndims"] == 2:
        nifti_img_aug = nib.Nifti1Image(denoise_imgs[0,0,:,:], np.eye(4))  
        nifti_mask_aug = nib.Nifti1Image(denoise_msks[0,:,:,:], np.eye(4))  
        nifti_img = nib.Nifti1Image(noisy_imgs_np[0,0,:,:], np.eye(4))
        nifti_mask = nib.Nifti1Image(noisy_msks_np[0, :, :, :], np.eye(4))
      elif hyp_parameters["ndims"] == 3:
        nifti_img_aug = nib.Nifti1Image(denoise_imgs[0,0,:,:,:], np.eye(4))  
        nifti_mask_aug = nib.Nifti1Image(denoise_msks[0,0,:,:,:], np.eye(4))  
        nifti_img = nib.Nifti1Image(noisy_imgs_np[0,0,:,:,:], np.eye(4))
        nifti_mask = nib.Nifti1Image(noisy_msks_np[0, 0, :, :], np.eye(4))
      
      nib.save(nifti_img_aug, os.path.join(hyp_parameters['aug_img_savepath'],utils.get_barcode([pid,e,im,noise_step])+'.nii.gz'))
      nib.save(nifti_mask_aug, os.path.join(hyp_parameters['aug_msk_savepath'],utils.get_barcode([pid,e,im,noise_step])+'_GT.nii.gz'))
      
      # Saving noisy image to nifti
      # CMR: format: Patient0001_Slice0001_NosieImg0001_NoiseStep0070.nii.gz
      # Lung CT: ...
      nib.save(nifti_img, os.path.join(hyp_parameters['aug_img_savepath'],utils.get_barcode([pid,e,im,noise_step],header=['Patient','Slice','NoiseImg','NoiseStep'])+'.nii.gz'))
      nib.save(nifti_mask, os.path.join(hyp_parameters['aug_msk_savepath'],utils.get_barcode([pid,e,im,noise_step],header=['Patient','Slice','NoiseImg','NoiseStep'])+'_GT.nii.gz'))
      
          
      if (im - hyp_parameters["start_noise_step"])%2 == 0:
        noise_step = noise_step + hyp_parameters["noise_step"]
      # break   # for testing
          













