import os
import torch
# from torch import nn, optim
# from torch.autograd.variable import Variable
# from torchvision import transforms, datasets
# from torchvision.utils import save_image
# import torch.nn.functional as F
# import scipy.ndimage as spimg
# import pyquaternion as quater
# import random
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, generate_binary_structure
import pydicom
from scipy.ndimage import zoom
from einops import rearrange, reduce, repeat

def remove_background(img,replace_value=None,num_bin=256,dim_ch=0,sigma=None):
    # common_value1,common_value2=[], []
    # if replace_value is None:
    if dim_ch is None:
        dim_ch=0
        img=np.expand_dims(img,axis=dim_ch)
    ims = np.split(img,img.shape[dim_ch],axis=dim_ch)
    # ims =[img]
    ims = [np.squeeze(im,axis=dim_ch) for im in ims]
    msk1 = np.ones_like(ims[0])
    for im in ims:
        if num_bin>0:
            flatten_im=im.flatten()
            hist, bins = np.histogram(flatten_im,bins=range(num_bin))
            # common_value1.append(np.argmax(hist))
            common_value1 = np.argmax(hist)
            # hist[common_value1] = -10**5
            msk1[im!=common_value1] = 0
            # common_value2 = np.argmax(hist)
    if sigma is not None and sigma > 0:
        # struct=generate_binary_structure()
        msk1 = binary_dilation(msk1,iterations=int(sigma*4)).astype(float)
        msk0 = binary_erosion(1-msk1,iterations=int(sigma*4)).astype(float)
        msk_blur = gaussian_filter(msk0, sigma=sigma*4,truncate=sigma//4, mode='nearest')
        # msk_blur = msk0
    for id, im in enumerate(ims):
        if replace_value is None:
            # a=im[np.logical_not(msk1)]
            # replace_value[id] = np.min(im[np.logical_not(msk1)])
            replace_v=np.min(im[np.logical_not(msk1)])
        else:
            replace_v=replace_value[id]
        # im[msk1==1] = replace_v
        if sigma is not None and sigma>0:
            im_blur=im
            im_blur[msk1==1]=replace_v
            im_blur = gaussian_filter(im_blur, sigma=sigma*4,truncate=sigma//4, mode='nearest')
            # im[msk1==1] = im_blur[msk1==1]
            im=im*(msk_blur) + im_blur*(1-msk_blur)
        else:
            im[msk1 == 1] = replace_v
        # print(im.shape)
        ims[id]=im
    return np.stack(ims,axis=dim_ch)

def thresh_img(img,thresh = None,EPS = 10**-7):
    device=img.device
    if isinstance(thresh,list):
        threshold=np.random.uniform(thresh[0],thresh[1])
        upbound=1-np.random.uniform(thresh[0],thresh[1])-threshold
    else:
        threshold=thresh
    if threshold is not None:
        # img=img-threshold
        # img=np.where(img>=0,img,0)
        # img = np.maximum(img-threshold,0)
        # img = torch.maximum(img - threshold,torch.tensor(0.))
        img = torch.clamp(img-threshold,min=torch.tensor(0.).to(device),max=torch.tensor(upbound).to(device))
    # return (img - img.min()) / (img.max() - img.min() + EPS)
    return img
  
def read_CT_volume(folder_path,target_res = 128):
# read CT into a (128x128x128) cube and pad the insufficient dimension
    
  dicom_slices = []
  # Iterate over each file in the folder
  for filename in sorted(os.listdir(folder_path), reverse=True):
    if filename.endswith(".dcm"):  # Check if the file is a DICOM file
      file_path = os.path.join(folder_path, filename)
        
    # Read the DICOM file
      dicom_data = pydicom.dcmread(file_path)
        
    # Append DICOM pixel data to the list
      dicom_slices.append(dicom_data.pixel_array)

  # Convert the list of slices to a numpy array
  
  dicom_slices = np.array(dicom_slices)
  dicome_volume = rearrange(dicom_slices, 'z h w -> h w z')

  # Get spatial information from the first DICOM file
  first_dicom = pydicom.dcmread(os.path.join(folder_path, os.listdir(folder_path)[0]))
  slice_thickness = first_dicom.SliceThickness
  pixel_spacing = first_dicom.PixelSpacing
  
#   Get the scaling ratio for each dim
  h_axis_ratio = pixel_spacing[0]
  w_axis_ratio = pixel_spacing[1]
  z_axis_ratio = slice_thickness

# find the longest dim that need to rescale
  longest_axis = max([h_axis_ratio*dicome_volume.shape[0], w_axis_ratio*dicome_volume.shape[1],z_axis_ratio*dicome_volume.shape[2]])
  c_factor = longest_axis/target_res
#   print((h_axis_ratio/c_factor, w_axis_ratio/c_factor ,z_axis_ratio/c_factor))
  resized_volume = zoom(dicome_volume, (h_axis_ratio/c_factor, w_axis_ratio/c_factor ,z_axis_ratio/c_factor))
#   print('resize', resized_volume.shape)

  
  max_dim_size = max(resized_volume.shape)

 # Calculate padding for each dimension
  padding_h = max_dim_size - resized_volume.shape[0]
  padding_w = max_dim_size - resized_volume.shape[1]
  padding_z = max_dim_size - resized_volume.shape[2]
  
  pad_depth = (padding_z // 2, padding_z - padding_z // 2)
  pad_height = (padding_h // 2, padding_h - padding_h // 2)
  pad_width = (padding_w // 2, padding_w - padding_w // 2)

#   Pad the array symmetrically
  padded_resized_volume = np.pad(resized_volume, (pad_height, pad_width, pad_depth), mode='constant')

  return padded_resized_volume, slice_thickness, pixel_spacing
  