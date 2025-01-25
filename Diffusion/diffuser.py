from torch import nn
import torch
import numpy as np
from Diffusion.networks import *

class DeformDDPM(nn.Module):
    def __init__(
        self, 
        network, 
        n_steps=50, 
        beta_schedule_fn = None, 
        device=None, 
        image_chw=(1, 28, 28),
        batch_size = 16,
        img_pad_mode = "zeros",
        ddf_pad_mode="border",
        padding_mode="border",
        v_scale = 0.008/256,
        resample_mode=None
        ):
        super(DeformDDPM, self).__init__()
        self.rec_num=2
        self.ndims=len(image_chw)-1
        self.n_steps = n_steps
        self.v_scale = v_scale
        self.device = device
        self.batch_size = batch_size
        self.img_pad_mode = img_pad_mode 
        self.ddf_pad_mode = ddf_pad_mode
        self.padding_mode = padding_mode
        self.resample_mode = resample_mode
        self.image_chw = image_chw
        
        self.network = network.to(self.device)
        self.ddf_stn_full = STN(
                                    img_sz = self.image_chw[1],
                                    ndims = self.ndims,
                                    padding_mode = self.padding_mode,
                                    device = self.device,
                                )
        self._DDF_Encoder_init()
        return
    
    def get_stn(self):
        return self.img_stn, self.ddf_stn_full

    def _DDF_Encoder_init(self, ctl_ratio=4, ctl_sz=64, resample_mode=None):
        self.ctl_sz=ctl_sz
        self.img_sz=self.image_chw[1]
        self.ddf_stn_rec=STN(img_sz=ctl_sz,ndims=self.ndims,device=self.device,padding_mode=self.ddf_pad_mode)
        self.img_stn=STN(img_sz=self.img_sz,ndims=self.ndims,device=self.device,padding_mode=self.img_pad_mode,resample_mode=self.resample_mode)
    
    def _get_ddf_scale(self,t,divide_num=1,max_ddf_num=200):   # 128
        rec_num = 1
        mul_num_ddf = torch.floor_divide(2*torch.pow(t,1.3), 3*divide_num).int()
        mul_num_dvf = torch.floor_divide(torch.pow(t,0.6), divide_num).int()
        mul_num_ddf = torch.clamp(mul_num_ddf, min=1, max=max_ddf_num)
        return rec_num,mul_num_ddf,mul_num_dvf.int()

    def _get_random_ddf(self,img,t):
        rec_num, mul_num_ddf, mul_num_dvf = self._get_ddf_scale(t=t)
        ddf_forward,dvf_forward = self._random_ddf_generate(rec_num=rec_num, mul_num=[mul_num_ddf,mul_num_dvf])
        warped_img = self.img_stn(img,ddf_forward)
        return warped_img, dvf_forward,ddf_forward

    def _multiscale_dvf_generate(self,v_scale,ctl_szs=[4,8,16,32,64]):
        dvf=0
        if self.img_sz is None:
            self.img_sz=max(ctl_szs)
        for ctl_sz in ctl_szs:
            dvf_comp = torch.randn([self.batch_size, self.ndims] + [ctl_sz]*self.ndims) * v_scale
            dvf_comp = F.interpolate(dvf_comp * self.ctl_sz / ctl_sz, [self.ctl_sz]*self.ndims, align_corners=False, mode='bilinear' if self.ndims == 2 else 'trilinear')
            dvf=dvf+dvf_comp
        return dvf

    def _random_ddf_generate(self,rec_num=3,mul_num=[torch.tensor([5]),torch.tensor([5])],ddf0=None,keep_inverse=False,noise_ratio=0.08): 
        crop_rate=2
        for _ in range(self.ndims+1):
            mul_num=[torch.unsqueeze(n,-1) for n in mul_num]
        # v_scale = v_scale *crop_rate
        ctl_ddf_sz=[self.batch_size, self.ndims] + [self.ctl_sz] * self.ndims
        if ddf0 is not None:
            ddf=ddf0
        else:
            ddf = torch.zeros(ctl_ddf_sz) * 0
        dddf = torch.zeros(ctl_ddf_sz) * 0
        # scale_num = min(5,int(math.log2(ctl_sz))-1)   # 2d
        scale_num = min(5,int(math.log2(self.ctl_sz))-2)   # avoid coupling between deformation and affine
        for i in range(rec_num):
            dvf = self._multiscale_dvf_generate(self.v_scale, ctl_szs=[self.ctl_sz // (2 ** i) for i in range(scale_num)]).to(self.device)
            # if True:
            if noise_ratio==0:
                dvf0=dvf
            else:
                dvf0=dvf+self.ddf_stn_rec(self._multiscale_dvf_generate(self.v_scale*noise_ratio,ctl_szs=[self.ctl_sz // (2 ** i) for i in range(scale_num)]).to(self.device),dvf)
            for j in range(torch.max(mul_num[0]).item()):
                flag = [(n>j).int().to(self.device) for n in mul_num]
                ddf = dvf0*flag[0] + self.ddf_stn_rec(ddf, dvf0*flag[0])
                dddf = dvf*flag[1] + self.ddf_stn_rec(dddf, dvf*flag[1])
        ddf = F.interpolate(ddf * self.img_sz/self.ctl_sz, self.img_sz*crop_rate, mode='bilinear' if self.ndims == 2 else 'trilinear')
        # ddf = ddf[...,img_sz//2:img_sz*3//2,img_sz//2:img_sz*3//2]
        if self.ndims==2:
            ddf = ddf[..., self.img_sz // 2:self.img_sz * 3 // 2, self.img_sz // 2:self.img_sz * 3 // 2]
        else:
            ddf = ddf[..., self.img_sz // 2:self.img_sz * 3 // 2, self.img_sz // 2:self.img_sz * 3 // 2, self.img_sz // 2:self.img_sz * 3 // 2]
        # if rec_num==1:
        if True:
            dddf = F.interpolate(dddf * self.img_sz/self.ctl_sz, self.img_sz*crop_rate, mode='bilinear' if self.ndims == 2 else 'trilinear')
            # dddf = dddf[...,img_sz//2:img_sz*3//2,img_sz//2:img_sz*3//2]
            if self.ndims == 2:
                dddf = dddf[..., self.img_sz // 2:self.img_sz * 3 // 2, self.img_sz // 2:self.img_sz * 3 // 2]
            else:
                dddf = dddf[..., self.img_sz // 2:self.img_sz * 3 // 2, self.img_sz // 2:self.img_sz * 3 // 2, self.img_sz // 2:self.img_sz * 3 // 2]
            return ddf,dddf
        else:
            return ddf

    def forward(self, x_0, t):
        t=torch.tensor(t)
        # img_t, dvf_forward, ddf_forward, ddf_stn, img_stn = self.ddf_enc(img= x_0, t=t)
        # return img_t, dvf_forward,ddf_forward,ddf_stn,img_stn
        return self._get_random_ddf(img = x_0, t = t)
    
    
    def backward(self, x, t,rec_num=2):
        if rec_num is None:
            rec_num = self.rec_num
        return self.network(x, t,rec_num=rec_num)
        
    def diff_recover(self,
                     img_org,
                     msk_org=None,
                     T=[None,None],
                     ddf_rand=None,
                     v_scale = None,
                     t_save=None,
                     ):
        if ddf_rand is None:
            # if v_scale is None:
            #     v_scale=self.v_scale
            # ddf_enc = DDF_Encoder(ndims=self.ndims,img_sz = self.image_chw[1], batch_sz = self.batch_size, v_scale=self.v_scale, device = self.device, img_pad_mode = self.img_pad_mode, ddf_pad_mode=self.ddf_pad_mode,resample_mode =self.resample_mode)
            # img_diff, dvf_rand, ddf_rand, ddf_stn, img_stn = ddf_enc(img= img_org, t=torch.tensor(np.array([T[0]])).to(self.device))
            if v_scale is not None:
                self.v_scale=v_scale
                self._DDF_Encoder_init()
            img_diff, _, ddf_rand = self._get_random_ddf(img= img_org, t=torch.tensor(np.array([T[0]])).to(self.device))
        ddf_comp = ddf_rand.clone().detach()
        img_rec = img_diff.clone().detach()
        if msk_org is not None:
            msk_diff = self.img_stn(msk_org.clone().detach(), ddf_rand)
        else:
            msk_diff = None
        msk_rec = msk_diff.clone().detach()
        img_save=[]
        msk_save=[]
        # Denosing image
        for i in range(T[1] - 1, -1, -1):
            t = torch.tensor(np.array([i])).to(self.device)
            pre_dvf_I = self.backward(img_rec, t)
            ddf_comp = self.ddf_stn_full(ddf_comp, pre_dvf_I) + pre_dvf_I
            # apply to image
            img_rec = self.img_stn(img_org.clone().detach(), ddf_comp)
            if msk_org is not None:
                msk_rec = self.img_stn(msk_org.clone().detach(), ddf_comp)
            if t_save is not None:
                if i in t_save:
                    img_save.append(img_rec)
                    if msk_org is not None:
                        msk_save.append(msk_rec)

        return [ddf_comp,ddf_rand],[img_rec,img_diff,img_save],[msk_rec,msk_diff,msk_save]
    
