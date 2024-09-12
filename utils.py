import os
import torch
import torchvision
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import scipy.ndimage as spimg
import pyquaternion as quater
import random
import numpy as np
import math
from typing import Optional, Tuple, List
# from data_loader.acdc_dataloader import acdc_gan

# from Adaptive_Motion_Generator.Dataloader.Archive.acdc_dataloader import *

def get_barcode(index=[],header=['Patient','Slice','AugImg','NoiseStep'],digit=[4,6,4,4],split='_'):
    # Patient0001_Slice0001_NosieImg0001_NoiseStep0070
    barcode_str=''
    header=header.copy()
    digit=digit.copy()
    if len(index)<3:
        header[2] = 'ORG'
        header[3] = 'NA'
        digit[2] = 0
        digit[3] = 0
        index +=['','']

    for id, h in enumerate(header):
        barcode_str+=h+str(index[id]).zfill(digit[id])+split
    return barcode_str[:-1]

class RandomResizedCrop3D(nn.Module):
    """Crop a random portion of a 3D volume and resize it to a given size.

    Args:
        size (tuple of int): Expected output size of the crop, for each dimension (D, H, W).
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
                                before resizing. The scale is defined with respect to the volume of the original image.
        ratio (tuple of float): Lower and upper bounds for the random aspect ratio of the crop, before resizing.
        interpolation (str): Desired interpolation mode ('trilinear' or 'nearest').
    """

    def __init__(
            self,
            size: Tuple[int, int, int],
            scale=(0.6, 1.0),
            ratio=(0.5, 1.5),
            interpolation='trilinear'
    ):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img: torch.Tensor, rand_scale: float, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int, int, int]:
        """Get parameters for `crop` for a random sized crop.

        Args:
            img (Tensor): Input image.
            scale (list): Range of scale of the origin size cropped.
            ratio (list): Range of aspect ratio of the origin aspect ratio cropped.

        Returns:
            tuple: params (i, j, k, d, h, w) to be passed to `crop` for a random sized crop.
        """
        img_sz = np.array(list(img.size())[2:])
        crop_sz = (img_sz * rand_scale).astype(np.int32)  #[int(s*rand_scale) for s in img_sz]
        start_id = np.random.randint(0, img_sz - crop_sz + 1, size=(img_sz.size,))
        return start_id.tolist()+crop_sz.tolist()

        # volume = depth * height * width
        #
        # log_ratio = torch.log(torch.tensor(ratio))
        # for _ in range(10):
        #     target_volume = volume * torch.empty(1).uniform_(*scale).item()
        #     aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        #
        #     w = int(round(math.sqrt(target_volume * aspect_ratio)))
        #     h = int(round(math.sqrt(target_volume / aspect_ratio)))
        #     d = int(round(math.sqrt(target_volume / (w * h))))
        #
        #     if 0 < w <= width and 0 < h <= height and 0 < d <= depth:
        #         i = torch.randint(0, depth - d + 1, size=(1,)).item()
        #         j = torch.randint(0, height - h + 1, size=(1,)).item()
        #         k = torch.randint(0, width - w + 1, size=(1,)).item()
        #         return i, j, k, d, h, w
        #
        # # Fallback to central crop
        # return (depth - d) // 2, (height - h) // 2, (width - w) // 2, d, h, w

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the RandomResizedCrop transformation.

        Args:
            img (Tensor): Input 3D image.

        Returns:
            Tensor: Cropped and resized image.
        """
        rand_scale = np.random.uniform(self.scale[0], self.scale[1])
        [i, j, k, d, h, w] = self.get_params(img,rand_scale, self.scale, self.ratio)
        # print(i, j, k, d, h, w)
        img_cropped = img[:, :, i:i + d, j:j + h, k:k + w]
        # print(img_cropped.shape)
        img_resized = F.interpolate(img_cropped, size=self.size, mode=self.interpolation,
                                    align_corners=False if self.interpolation == 'trilinear' else None)
        return img_resized#.squeeze(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, ratio={self.ratio}, interpolation={self.interpolation})"

def random_permute(X, select_dims=[-1,-2],include_flip=True):
    axes=list(range(X[0].ndim))
    selected_axes = [axes[i] for i in select_dims]
    random.shuffle(selected_axes)
    for i, dim in enumerate(select_dims):
        axes[dim] = selected_axes[i]
        if include_flip and random.choice([True,False]):
            # X = [np.flip(x, axis=dim) for x in X]
            X = [torch.flip(x, [dim]) for x in X]
    # return [np.transpose(x,axes) for x in X]
    return [x.permute(axes) for x in X]

# def thresh_img(img,thresh = None,EPS = 10**-7):
#     threshold0 = np.random.uniform(thresh[0], thresh[1])
#     threshold1 = np.random.uniform(thresh[0], thresh[1])
#     scale =
#     if threshold is not None:
#         # img=img-threshold
#         # img=np.where(img>=0,img,0)
#         # img = np.maximum(img-threshold,0)
#         img = torch.maximum(img - threshold,torch.tensor(0.))
#     # return (img - img.min()) / (img.max() - img.min() + EPS)
#     return img

def get_transformer(degrees=180,translate=0.125,ndims=2,prob=0.8,fill=0.,img_sz=None):
    prob_crop=0. if img_sz==None else 0.8
    # prob_crop=0. if len(img_sz)==2 else 0.8

    if img_sz==None or len(img_sz)==2:
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([
                torchvision.transforms.RandomAffine(degrees=degrees, translate=[translate] * ndims, fill=fill,
                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            ],prob),
            # torchvision.transforms.RandomApply([
            #     torchvision.transforms.RandomResizedCrop(size=img_sz),
            # ], prob_crop),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomAutocontrast(p=0.5),
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([
                torchvision.transforms.RandomResizedCrop(size=img_sz) if len(img_sz) == 2 else RandomResizedCrop3D(
                    size=img_sz),
            ], prob_crop),
        ])


def get_random_affine_transformer(degrees=180,translate=0.125,ndims=2):
    return torchvision.transforms.RandomAffine(degrees=degrees, translate=[translate] * ndims,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

def channel_merge_acdc(img):
#   input: a torch tensor (C,H,W)
  ch = img.shape[0]
  output = np.zeros((img.shape[1], img.shape[2]))
  # output[img[2,:,:] == 1] = 1
  for i in range(ch):
    output= output + img[i]
  return output
    
def img_crop(img, crop_rate=2, img_sz=[256,256]):
    ndims=len(img_sz)
    crop = [np.random.randint(0.*imgs, 1. * imgs)//crop_rate for imgs in img_sz]
    crop = [crop, [1 * imgs//crop_rate - c for imgs, c in zip(img_sz, crop)]]
    if ndims==2:
        return img[..., crop[0][0]: img_sz[0] - crop[1][0], crop[0][1]: img_sz[1] - crop[1][1]]
    else:
        return img[..., crop[0][0]: img_sz[0] - crop[1][0], crop[0][1]:img_sz[1] - crop[1][1], crop[0][2]: img_sz[2] - crop[1][2]]


def boundary_limit(sample_coords0, max_sz, plus=0., minus=1.):
    sample_coords = torch.split(sample_coords0, split_size_or_sections=1, dim=1)
    # return tf.stack([tf.maximum(tf.minimum(x, sz - minus + plus), 0 + plus) for x, sz in zip(sample_coords, input_size0)],-1)
    return torch.cat([torch.clamp(x * sz, min=minus - 1 * sz + plus, max=1 * sz - minus + plus) for x, sz in
                      zip(sample_coords, max_sz)], 1)


def resample(vol, ddf, ref=None, img_sz=None,max_sz=[128,128],ndims=2):
    device = vol.device
    img_sz = vol.size()[2:]
    ndims=len(img_sz)
    if ndims==2:
        [h,w]=img_sz
        img_shape = torch.reshape(torch.tensor([(h - 1) / 2., (w - 1) / 2.], device=device), [1, 1, 1, ndims])
        ref_grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(end=h), torch.arange(end=w)]), 0), [1, ndims,h, w ])
    elif ndims==3:
        [h, w, d] = img_sz
        img_shape = torch.reshape(torch.tensor([(h - 1) / 2., (w - 1) / 2., (d-1)/2], device=device), [1, 1, 1, 1, ndims])
        ref_grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(end=h), torch.arange(end=w), torch.arange(end=d)]), 0), [1, ndims,h, w, d])
    # ref_grid.to(device)
    # img_shape.to(device)
    # ddf.to(device)
    # ref = self.ref_grid if ref is None else ref
    # img_sz = self.img_sz if img_sz is None else img_sz
    resample_mode = 'bilinear'
    # padding_mode = "border"
    padding_mode = "zeros"

    # img_sz = np.reshape(img_sz, [1] *(ndims+1)+[ndims])
    # if ndims==2:
    if True:
        re=[0]+list(range(2,ndims+2))+[1]
        # re=list(range(ndims+2))
        # print((torch.flip((ddf.to(device) + ref_grid.permute(re))/ img_shape - 1, dims=[-1])).tolist())
        return F.grid_sample(vol, torch.flip((ddf + ref_grid.permute(re).to(device))/ img_shape - 1, dims=[-1]).type(torch.float32).to(device), mode=resample_mode, padding_mode=padding_mode,align_corners=True)
        #
        # return F.grid_sample(vol, torch.flip(
        #     torch.permute(ddf * torch.Tensor(np.reshape(np.array(max_sz), [1, 1, 1, ndims])) + ref_grid,
        #                   [0, 2, 3, 1]) / img_shape - 1, dims=[-1]), mode=resample_mode, padding_mode=padding_mode,
        #                      align_corners=True)

def random_resample(vol,deform_scale=32.):
    vol_size=vol.size()
    device=vol.device
    ndims = len(vol_size)-2
    img_size=[s for s in vol_size[2:]]
    if ndims==2:
        img_size=img_size+[16]
    # ddf,_,_=random_ddf(vol_size[0],img_size)
    _,_,ddf=random_ddf(vol_size[0],img_size,ndims=ndims,range_gauss=deform_scale)
    ddf=Variable(torch.tensor(ddf,dtype=torch.float32)).to(device)
    if ndims==2:
        return resample(vol,ddf[...,8,:ndims])
    else:
        return resample(vol, ddf[..., :ndims])


# grid option
def get_tranf_mat(grid_size, vec=[[0., 0., 1.]], ang=[[0.]],transl=[[0,0,0]]):
    return np.concatenate([get_rot_mat(grid_size, vec=vec, ang=ang),transl],-1)


def get_rot_mat(grid_size, vec=[[0., 0., 1.]], ang=[[0.]],ndims=3):
    vec = np.array(vec)
    ang = np.array(ang)
    batch_num = ang.shape[0]
    return np.reshape(vecang2rotmats(vec, ang), [batch_num] + [ndims*(ndims)])

def random_mat(batch_sz, img_sz, num_class=2,pn_spline=20, pn_gauss=10, range_spline=2., range_gauss=48, spread_range=[5., 24.],
               transl_range=32., rot_range=np.pi / 2):
    scale=4
    ndims=3
    vec=np.reshape(np.random.uniform(-1., 1., [batch_sz,1, ndims])+np.random.uniform(-.1, .1, [batch_sz,num_class, ndims]),[batch_sz*num_class, ndims])
    ang=np.reshape(np.random.uniform(-rot_range, rot_range, [batch_sz,1])+np.random.uniform(-rot_range/scale, rot_range/scale, [batch_sz,num_class]),[batch_sz*num_class])
    transl=np.reshape(np.random.uniform(-transl_range, transl_range, [batch_sz,1,ndims])+np.random.uniform(-transl_range/scale, transl_range/scale, [batch_sz,num_class,ndims]),[batch_sz*num_class,ndims])
    return np.reshape(np.concatenate([get_rot_mat(img_sz, vec=vec, ang=ang),transl],-1),[batch_sz,num_class,4,3])

    # return np.reshape(get_tranf_mat(img_sz, vec=np.random.uniform(-1., 1., [batch_sz*num_class, 3]), ang=np.random.uniform(-rot_range, rot_range, [batch_sz*num_class]),transl=np.random.uniform(-transl_range, transl_range, [batch_sz*num_class,3])),[batch_sz,num_class,4,3])

def random_ddf(batch_sz, img_sz, pn_spline=20, pn_gauss=10, range_spline=1., range_gauss=16., spread_range=[16., 64.],
               transl_range=0., rot_range=np.pi / 1,ndims=3):
    rand_ang=np.random.uniform(-rot_range, rot_range, [batch_sz])
    # rand_ang = np.random.randint(-4, 4, [batch_sz])*rot_range

    if ndims==3:
        rot_df = get_rot_ddf(img_sz, vec=np.random.uniform(-1., 1., [batch_sz, 3]),
                             ang=rand_ang)
    else:
        rot_df = get_rot_ddf(img_sz, vec=np.concatenate([np.zeros([batch_sz, 2]),np.ones([batch_sz, 1])],-1),
                             ang=rand_ang)
    ndims = 3
    # rot_df = +np.random.uniform(-1., 1., [batch_sz, ndims,ndims])
    # ddf0=np.stack([generate_random_gaussian_ddf(img_sz, pn_gauss, range_sz=range_gauss, spread_std=spread_range)\
    #                +generate_random_spline_ddf(img_sz, pn_spline, range_sz=range_spline)\
    #                +np.random.uniform(-transl_range,transl_range,[3]) for i in range(batch_sz)],axis=0)\
    #      +rot_df
    if range_gauss>0:
        ddf0 = np.tile([generate_random_gaussian_ddf(img_sz, pn_gauss, range_sz=range_gauss, spread_std=spread_range) \
                        # + generate_random_spline_ddf(img_sz, pn_spline, range_sz=range_spline) \
                        + np.random.uniform(-transl_range, transl_range, [ndims])], [batch_sz, 1, 1, 1, 1]) \
               + rot_df
    else:
        ddf0 = rot_df

    def boundary_replicate(sample_coords, input_size, padding=5):
        return np.stack(
            [np.maximum(np.minimum(sample_coords[..., i], input_size[i] - 1 + padding), 0 - padding) for i in
             range(len(input_size))], axis=-1), \
               np.prod([((sample_coords[..., i] < input_size[i]) * (sample_coords[..., i] >= 0)) for i in
                        range(len(input_size))], axis=0)

    ref = get_reference_grid(img_sz)
    cf1, ind = boundary_replicate(ddf0 + ref, img_sz)
    return cf1 - ref, np.expand_dims(ind, -1), rot_df


def generate_random_gaussian_ddf(img_sz, pn=30, range_sz=5, spread_std=[0.1, 1.]):
    x = np.floor(np.random.uniform(range_sz / 2., img_sz[0] - range_sz / 2., [1, pn])).astype('int')
    y = np.floor(np.random.uniform(range_sz / 2., img_sz[1] - range_sz / 2., [1, pn])).astype('int')
    z = np.floor(np.random.uniform(range_sz / 2., img_sz[2] - range_sz / 2., [1, pn])).astype('int')

    odf = np.random.uniform(-range_sz, range_sz, [pn, 3])
    vol = np.zeros([img_sz[0], img_sz[1], img_sz[2], 3])
    vol[x, y, z] = odf

    return spimg.gaussian_filter(vol, np.random.uniform(spread_std[0], spread_std[1]))


def get_rot_ddf(grid_size, vec=[[0., 0., 1.]], ang=[[0.]]):
    vec = np.array(vec)
    ang = np.array(ang)
    batch_num = ang.shape[0]
    ref_grids = get_reference_grid(grid_size,
                                   bias_scale=1.)
    # a=vecang2rotmats(vec, ang)
    return np.reshape(np.matmul(np.reshape(np.tile(ref_grids, [batch_num, 1, 1, 1, 1]), [batch_num, -1, 3]),
                                vecang2rotmats(vec, ang)), [batch_num] + grid_size + [3]) - ref_grids


def get_reference_grid(grid_size, bias_scale=0.):
    return np.stack(np.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=-1).astype('float') - bias_scale * (np.array(grid_size) - 1) / 2.


def resample_linear(inputs, ddf=None, sample_coords=None,random_boundary=True):
    if random_boundary:
        random_factor = np.random.uniform(0., 1.)
        min_val = np.min(inputs)
        inputs[:, 0, :, :] = min_val * random_factor + (1 - random_factor) * inputs[:, 0, :, :]
        inputs[:, -1, :, :] = min_val * random_factor + (1 - random_factor) * inputs[:, -1, :, :]
        inputs[:, :, 0, :] = min_val * random_factor + (1 - random_factor) * inputs[:, :, 0, :]
        inputs[:, :, -1, :] = min_val * random_factor + (1 - random_factor) * inputs[:, :, -1, :]
        inputs[:, :, :, 0] = min_val * random_factor + (1 - random_factor) * inputs[:, :, :, 0]
        inputs[:, :, :, -1] = min_val * random_factor + (1 - random_factor) * inputs[:, :, :, -1]

    input_size = inputs.shape[1:4]
    sample_coords = get_reference_grid(input_size) + ddf if sample_coords is None else sample_coords
    spatial_rank = 3  # inputs.ndim - 2
    xy = [sample_coords[..., i] for i in
          range(sample_coords.shape[-1])]  # tf.unstack(sample_coords, axis=len(sample_coords.shape)-1)
    index_voxel_coords = [np.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0, plus=0):
        return np.maximum(np.minimum(sample_coords0, input_size0 - 2 + plus), 0 + plus)

    def boundary_replicate_float(sample_coords0, input_size0, plus=0.):
        return np.maximum(np.minimum(sample_coords0, input_size0 - 1 + plus), 0 + plus)

    xy = [boundary_replicate_float(x.astype('float32'), input_size[idx]) for idx, x in enumerate(xy)]
    spatial_coords = [boundary_replicate(x.astype('int32'), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate((x + 1).astype('int32'), input_size[idx], 1)
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [np.expand_dims(x - i.astype('float32'), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [np.expand_dims(i.astype('float32') - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = list(spatial_coords[0].shape)
    batch_coords = np.tile(np.reshape(range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2 ** spatial_rank)]

    make_sample = lambda bc: inputs[batch_coords, sc[bc[0]][0], sc[bc[1]][1], sc[bc[2]][
        2], ...]  # tf.gather_nd(inputs, np.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
    samples = [make_sample(bc) for bc in binary_codes]

    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0] * weight_c0[0] + samples0[1] * weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)


def vecang2rotmats(vec, ang):
    return np.stack([np.reshape(vecang2rotmat(vec[i, ...], ang[i, ...]), [3, 3]) for i in range(len(vec))], 0)


def vecang2rotmat(vec, ang):
    q = quater.Quaternion(axis=vec, angle=ang)
    return q.rotation_matrix


def images_to_vectors(images):
  return images.view(images.size(0), 16384).to(device)

def vectors_to_images(vectors):
  return vectors.view(vectors.size(0), 1, 128, 128).to(device)

def noise(size):
  n = Variable(torch.randn(size, 100)).to(device)
  return n

def ones_target(size):
  data = Variable(torch.ones(size, 1)).to(device)
  return data

def zeros_target(size):
  data = Variable(torch.zeros(size, 1)).to(device)
  return data


def eval_detJ_lab(disp=None,vol1=None,vol2=None,thresh=0.5):
    ndims=disp.ndim-2
    if vol1 ==None or thresh==None:
        label=1
    else:
        label=vol1>thresh
        label=label*(spimg.laplace(label) < 0.1)
        rescale_factor=2
        label=label[...,::rescale_factor,::rescale_factor,::rescale_factor]

    # disp = disp.permute([0, *range(2,ndims+2), 1])
    # print(disp.shape)
    disp = np.transpose(disp, [0, *range(2,ndims+2), 1])
    # Jacob=np.stack(np.gradient(disp,axis=[-4,-3,-2]),-1)
    Jacob=np.stack(np.gradient(disp,axis=[*range(1,ndims+1)]),-1)
    for ii in range(ndims):
        Jacob[..., ii, ii] = Jacob[..., ii, ii] + 1
    # Jacob[..., 0, 0] = Jacob[..., 0, 0] + 1
    # Jacob[..., 1, 1] = Jacob[..., 1, 1] + 1
    # Jacob[..., 2, 2] = Jacob[..., 2, 2] + 1
    return np.sum((np.linalg.det(Jacob)<0)*label)

def eval_def_mag(disp=None,vol1=None,vol2=None,thresh=0.5):
    ndims=3
    # if vol1 ==None or thresh==None:
    #     label=1
    # else:
    #     label=vol1>thresh
    #     label=label*(spimg.laplace(label) < 0.1)
    #     rescale_factor=2
    #     label=label[...,::rescale_factor,::rescale_factor,::rescale_factor]
    mag=np.sqrt(np.sum(np.square(disp),axis=1))
    sz=mag.shape
    max_mag=np.mean(np.max(np.reshape(mag,[sz[0],-1]),axis=-1))
    avg_mag=np.mean(mag)
    return [avg_mag,max_mag]

vol=np.random.uniform(-1,1,[4,1,256,256])
vol=Variable(torch.tensor(vol,dtype=torch.float32))
vol_res=random_resample(vol)
vol_crop=img_crop(vol_res)
# print(vol.tolist())
# print(vol_res.tolist())




