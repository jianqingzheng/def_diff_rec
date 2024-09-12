"""
losses for DRDM
"""

import numpy as np
import sys
import torch
import torch.nn.functional as F


EPS=1e-7

eps_scale = 10e-5
# eps_scale = 10e-4

class NCC(torch.nn.Module):
    # def __init__(self, eps_scale=10e-7,img_sz=256):
    def __init__(self, eps_scale=10e-5,img_sz=256):
        super(NCC, self).__init__()
        self.eps_scale=eps_scale#*img_sz/256
        self.scale=10e4

    def forward(self,pred,inv_lab=None,ddf_stn=None):
        if ddf_stn is None:
            trm_pred=pred
        else:
            trm_pred=-ddf_stn(pred, inv_lab)
        trm_pred = self.scale * trm_pred
        inv_lab = self.scale * inv_lab
        loss_gen = torch.mean(torch.sum(trm_pred*inv_lab,dim=1)/(torch.sqrt(torch.sum(torch.square(trm_pred),dim=1)*torch.sum(torch.square(inv_lab),dim=1)+self.eps_scale)))
        return loss_gen

class MRSE(torch.nn.Module):
    def __init__(self, eps_scale=10e-4,img_sz=256):
        super(MRSE, self).__init__()
        self.eps_scale=eps_scale#*img_sz/256
        self.scale = 10e1

    def forward(self,pred,inv_lab=None,ddf_stn=None):
        if ddf_stn is None:
            trm_pred=pred
        else:
            trm_pred=-ddf_stn(pred, inv_lab)
        trm_pred = self.scale * trm_pred
        inv_lab = self.scale * inv_lab
        loss_gen = torch.mean(
            torch.sum(torch.square(trm_pred + inv_lab), dim=1)
            / (torch.sum(torch.square(inv_lab), dim=1) + self.eps_scale)
        )
        return loss_gen/1

class RMSE(torch.nn.Module):
    def __init__(self, eps_scale=eps_scale,img_sz=256,ndims=2):
        super(RMSE, self).__init__()
        self.eps_scale=eps_scale#*img_sz/256
        self.ndims=ndims

    def forward(self,pred,inv_lab=None,ddf_stn=None):
        if ddf_stn is None:
            trm_pred=pred
        else:
            trm_pred=-ddf_stn(pred, inv_lab)
        loss_gen = torch.mean(torch.mean(torch.sum(torch.square(trm_pred - inv_lab), dim=1),
                              dim=list(range(1, 1 + self.ndims))) / (
                               torch.mean(torch.sum(torch.square(inv_lab), dim=1), dim=list(range(1, 1 + self.ndims))) + self.eps_scale))
        return loss_gen
# loss_gen = torch.mean(torch.mean(torch.sum(torch.square(ddf_stn(pre_dvf_I, dvf_I) + dvf_I), dim=1),dim=list(range(1,1+ndims))) / (torch.mean(torch.sum(torch.square(dvf_I), dim=1),dim=list(range(1,1+ndims))) + EPS))


class Grad(torch.nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty=['l1'],ndims=2, eps=1e-8, outrange_weight=50, detj_weight=2, apear_scale=9, dist=1, sign=1,waive_thresh=10**-4):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.eps = eps
        self.outrange_weight = outrange_weight
        self.detj_weight=detj_weight
        self.apear_scale = apear_scale
        self.ndims=ndims
        self.max_sz = torch.reshape(torch.tensor([0.7]*ndims, dtype=torch.float32) , [1]+[ndims]+[1]*(ndims))
        self.act = torch.nn.ReLU(inplace=True)
        self.dist=dist
        self.sign=sign
        self.waive_thresh=waive_thresh

    def _diffs(self, y,dist=None):
        if dist is None:
            dist=self.dist
        # vol_shape = y.size()[2:]
        # vol_shape = y.get_shape().as_list()[1:-1]
        # ndims = len(vol_shape)

        df = [None] * self.ndims
        for i in range(self.ndims):
            d = i + 2
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, self.ndims + 2)]
            yp = y.permute(r)
            dfi = yp[dist:, ...] - yp[:-dist, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, self.ndims + 2)]
            df[i] = dfi.permute(r)
        return df

    def _eq_diffs(self, y,dist=None):
        if dist is None:
            dist=self.dist
        # vol_shape = y.get_shape().as_list()[1:-1]
        vol_shape = y.size()[2:]
        ndims = len(vol_shape)
        pad = [0, 0] * (ndims + 1) +[dist, 0]
        pad1 = [0, 0] * (ndims + 1) +[0, dist]
        df = [None, None] * ndims
        for i in range(ndims):
            d = i + 2
            r=[d, *range(d), *range(d + 1, ndims + 2)]
            ri=[*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            yt = y.permute(r)
            dy=yt[dist:, ...] - yt[:-dist, ...]
            df[2*i] = (F.pad(dy, pad,mode='constant',value=0)).permute(ri)
            df[2*i+1] = (F.pad(dy, pad1, mode='constant', value=0)).permute(ri)
            y.permute(ri)
        return df

    def _weighted_diffs_error(self, y,dist=None,w=None,expect=None,mean_dim=None):
        if dist is None:
            dist=self.dist
        vol_shape = y.size()[2:]
        ndims = len(vol_shape)
        df = [None] * ndims

        for i in range(ndims):
            d = i + 2
            r=[d, *range(d), *range(d + 1, ndims + 2)]
            ri=[*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            yt = y.permute(r)
            wt = w.permute(r)
            dy=(torch.abs(yt[dist:, ...] - yt[:-dist, ...])-expect.permute(r))*(wt[dist:, ...]*wt[:-dist, ...])
            df[i] = torch.mean((dy).permute(ri),dim=mean_dim,keepdim=True)
            y.permute(ri)
            w.permute(ri)
        return df

    def _outl_dist(self, y,range_thresh=0.2):
        self.device = y.device
        vol_shape = y.size()[2:]
        self.max_sz=self.max_sz.to(self.device)
        act=torch.nn.ReLU(inplace=True)
        loss=0.
        for i in range(self.ndims):
            d = i + 2
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, self.ndims + 2)]
            ri = [*range(1, d + 1), 0, *range(d + 1, self.ndims + 2)]
            yt = y.permute(r)
            loss += torch.mean(act(-range_thresh-yt[0,:,i, ...]))+torch.mean(act(yt[-1,:,i, ...]-range_thresh))
            y.permute(ri)
        return loss/self.ndims

    def _center_dist(self, y):
        self.device = y.device
        vol_shape = y.size()[2:]
        self.max_sz=self.max_sz.to(self.device)
        select_loc = [s // 2 for s in vol_shape]
        if self.ndims==3:
            return torch.mean(self.act(torch.abs(y[:,:, select_loc[0], select_loc[1], select_loc[2]]) - self.max_sz))
        elif self.ndims == 2:
            return torch.mean(self.act(torch.abs(y[:, :, select_loc[0], select_loc[1]]) - self.max_sz))

        
    def _eval_detJ(self, disp=None, weight=None):
        weight = 1
        if self.ndims==3:
            detj = (disp[0][:, 0, ...] * disp[1][:, 1, ...] * disp[2][:, 2, ...]) + (
                    disp[0][:, 1, ...] * disp[1][:, 2, ...] * disp[2][:, 0, ...]) + (
                           disp[0][:, 2, ...] * disp[1][:, 0, ...] * disp[2][:, 1, ...]) - (
                           disp[0][:, 2, ...] * disp[1][:, 1, ...] * disp[2][:, 0, ...]) - (
                           disp[0][:, 0, ...] * disp[1][:, 2, ...] * disp[2][:, 1, ...]) - (
                           disp[0][:, 1, ...] * disp[1][:, 0, ...] * disp[2][:, 2, ...])
        elif self.ndims==2:
            detj = (disp[0][:, 0, ...] * disp[1][:, 1, ...]) - (disp[0][:, 1, ...] * disp[1][:, 0, ...])

        return detj * weight
        
    def forward(self,  y_pred=None,x_in=None, img=None):
        reg_loss = 0
        act=torch.nn.ReLU(inplace=True)
        if img is None:
            if 'l1' in self.penalty:
                df = [torch.mean(F.relu(torch.abs(f) - self.waive_thresh,inplace=True)) for f in self._diffs(y_pred)]
                reg_loss += sum(df) / len(df)
                
            if 'l2' in self.penalty:
                df = [torch.mean(F.relu(f * f - self.waive_thresh**2,inplace=True)) for f in self._diffs(y_pred)]
                reg_loss += torch.sqrt(sum(df) / len(df))

        if 'negdetj' in self.penalty:
            df = self.detj_weight*torch.mean(act(-self._eval_detJ(self._eq_diffs(y_pred,dist=1))))  # , dg[...,0])
            reg_loss += df  # 0.5*df
        if 'range' in self.penalty:
            reg_loss += self.outrange_weight * (self._outl_dist(y_pred)+self._center_dist(y_pred))
        if 'param' in self.penalty or 'detj' in self.penalty or 'std' in self.penalty:
            mean_dim=list(range(1, self.ndims + 2))
            dg = torch.sum(torch.abs(img),dim=1,keepdim=True)* torch.exp(-self.apear_scale * torch.nn.ReLU(inplace=True)(.1-sum([torch.sum(g * g, dim=1, keepdim=True) for g in self._eq_diffs(img,dist=3)]) / torch.sum(torch.square(.1 + img), dim=1, keepdim=True)))
            dg = dg/(EPS+torch.mean(dg,dim=mean_dim,keepdim=True))

            y_pred = torch.clamp(y_pred, min=-0.8, max=0.8)
            x_in = x_in if isinstance(x_in,list) else [x_in]
            if 'std' in self.penalty:
                reg_loss += self.sign*torch.mean(torch.clamp(grad_std((y_pred-torch.mean(y_pred,dim=list(range(2,ndims+2)),keepdim=True))*dg), max=.2, min=0))
            if 'param' in self.penalty:
                for id, d in enumerate(self.dist):
                    df = torch.mean(torch.abs(sum(self._weighted_diffs_error(y_pred, dist=d, w=dg, expect=torch.abs(x_in[-1][:, id:id + 1, ...]),mean_dim=mean_dim))))
                    reg_loss += 1 * (df) / len(self.dist)

            if 'detj' in self.penalty:
                df = torch.mean(torch.abs(
                    torch.mean((torch.abs(self._eval_detJ(self._eq_diffs(y_pred, dist=1))) - torch.abs(x_in[0])) * dg, dim=mean_dim)))
                reg_loss += df  # 0.5*df

        return reg_loss


def avg_std_skew_kurt(array,ndims=2):
    dim = list(range(2, ndims + 2))
    mean = torch.mean(array,dim=dim)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0),dim=dim)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0),dim=dim)
    kurtoses = torch.mean(torch.pow(zscores, 4.0),dim=dim) - 3.0
    return [mean,std,skews,kurtoses]

def grad_std(array,ndims=2):
    dim = list(range(2, ndims + 2))
    array=torch.clamp(array,min=-0.8,max=0.8)
    dim0=list(range(1,ndims+2))
    std = torch.sqrt(torch.mean(torch.square(array - torch.mean(array, dim=dim, keepdim=True)), dim=dim0))
    return std

def avg_std(array,ndims=2):
    dim = list(range(2, ndims + 2))
    return [torch.mean(array,dim=dim),grad_std(array,dim=dim)]


if __name__ == "__main__":
    ndims=2
    dist=[16,32]
    ddf = torch.rand(1,2,128,128)
    # ddf[:,:,0,:]=ddf[:,:,0,:]-1
    # ddf[:,:,1,:]=ddf[:,:,1,:]+1
    # ddf[:,:,0,0]=ddf[:,:,0,0] -1
    # ddf[:,:,1,1]=ddf[:,:,1,1] +1
    # ddf[:,0,0,1]=ddf[:,0,0,1] +1
    # ddf[:,1,0,1]=ddf[:,1,0,1] -1
    # ddf[:,0,0,1]=ddf[:,0,0,1] -1
    # ddf[:,1,0,1]=ddf[:,1,0,1] +1
    # ddf[:,1,1,0]=ddf[:,1,1,0] -1
    # ddf[:,0,1,0]=ddf[:,0,1,0] +1
    ddf=ddf
    img = torch.rand(1,1,128,128)
    x_in=np.reshape([0.2,0.3],newshape=[1,ndims]+[1]*ndims)
    x_in=[torch.tensor(x_in).type(torch.float32),0.]

    Loss_detj = Grad(penalty=['detj'],ndims=ndims,dist=dist)
    loss_detj = Loss_detj(ddf,x_in,img)
    print(loss_detj)
