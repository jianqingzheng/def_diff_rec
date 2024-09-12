from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

def get_net(name="recresnet"):
  name = name.lower()
  if name == "recresacnet":
    net = RecResACNet
  else:
    net = None
  return net



def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])
    return embedding

class AtrousBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, atrous_rates=[1,3], ndims=2, activation=None, normalize=True):
        super(AtrousBlock, self).__init__()
        # if 0 not in shape:
        if normalize:
            # print(shape)
            # self.ln = nn.LayerNorm(shape)     # jzheng 15/03/2024
            norm=getattr(nn, 'InstanceNorm%dd' % ndims)     # jzheng 15/03/2024
            self.ln = norm(out_c,affine=True)
        else:
            self.ln = nn.Identity()
        Conv=getattr(nn,'Conv%dd' % ndims)
        self.conv0 = Conv(in_c, out_c, kernel_size, 1, (kernel_size-1)//2*1) #if in_c!=out_c else None
        self.convs = nn.ModuleList([
            Conv(out_c, out_c, kernel_size, 1, (kernel_size-1)//2*ar, dilation=ar) for ar in atrous_rates
        ])
        # self.conv1 = Conv(out_c, out_c, kernel_size, stride, padding)
        # self.conv2 = Conv(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(1e-6) if activation is None else activation
        # self.activation = nn.ReLU() if activation is None else activation
        # self.activation = nn.ReLU()
        self.normalize = normalize

    def forward(self, x):
        x = self.conv0(x) #if self.conv0 is not None else x
        x = self.ln(x) if self.normalize else x     # jzheng 15/03/2024
        out=nn.Identity()(x)
        for conv in self.convs:
            out = self.activation(out)
            out = conv(out)
        return self.activation(out+x)

class RecResACNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, ndims=2, num_input_chn=1, res=0):
        super(RecResACNet, self).__init__()

        self.dimension = ndims
        self.Conv = getattr(nn, 'Conv%dd' % self.dimension)
        self.ConvT = getattr(nn, 'ConvTranspose%dd' % self.dimension)

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            AtrousBlock([num_input_chn] + [res] * ndims, num_input_chn, 10, ndims=ndims),
            AtrousBlock([10] + [res] * ndims, 10, 10, ndims=ndims),
            AtrousBlock([10] + [res] * ndims, 10, 10, ndims=ndims),

        )
        self.down1 = self.Conv(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            AtrousBlock([10] + [res // 2] * ndims, 10, 20, ndims=ndims),
            AtrousBlock([20] + [res // 2] * ndims, 20, 20, ndims=ndims),
            AtrousBlock([20] + [res // 2] * ndims, 20, 20, ndims=ndims)
        )
        self.down2 = self.Conv(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            AtrousBlock([20] + [res // 4] * ndims, 20, 40, ndims=ndims),
            AtrousBlock([40] + [res // 4] * ndims, 40, 40, ndims=ndims),
            AtrousBlock([40] + [res // 4] * ndims, 40, 40, ndims=ndims)
        )
        self.down3 = self.Conv(40, 40, 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            AtrousBlock([40] + [res // 8] * ndims, 40, 20, ndims=ndims),
            AtrousBlock([20] + [res // 8] * ndims, 20, 20, ndims=ndims),
            AtrousBlock([20] + [res // 8] * ndims, 20, 40, ndims=ndims)
        )

        # Second half
        self.up1 = self.ConvT(40, 40, 4, 2, 1)

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            AtrousBlock([80] + [res // 4] * ndims, 80, 40, ndims=ndims, normalize=False),
            AtrousBlock([40] + [res // 4] * ndims, 40, 20, ndims=ndims, normalize=False),
            AtrousBlock([20] + [res // 4] * ndims, 20, 20, ndims=ndims, normalize=False)
        )

        self.up2 = self.ConvT(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            AtrousBlock([40] + [res // 2] * ndims, 40, 20, ndims=ndims, normalize=False),
            AtrousBlock([20] + [res // 2] * ndims, 20, 10, ndims=ndims, normalize=False),
            AtrousBlock([10] + [res // 2] * ndims, 10, 10, ndims=ndims, normalize=False)
        )

        self.up3 = self.ConvT(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            AtrousBlock([20] + [res // 1] * ndims, 20, 10, ndims=ndims, normalize=False),
            AtrousBlock([10] + [res // 1] * ndims, 10, 10, ndims=ndims, normalize=False),
            AtrousBlock([10] + [res // 1] * ndims, 10, 10, ndims=ndims, normalize=False)
        )

        self.conv_out = self.Conv(10, ndims, 3, 1, 1)

    def boundary_limit(self, sample_coords0, max_sz, plus=0., minus=1.):
        sample_coords = torch.split(sample_coords0, split_size_or_sections=1, dim=1)
        return torch.cat([torch.clamp(x * sz, min=minus - 1 * sz + plus, max=1 * sz - minus + plus) / sz for x, sz in
                          zip(sample_coords, max_sz)], 1)

    def resample(self, vol, ddf, ref=None, img_sz=None, padding_mode="zeros"):
        ref = self.ref_grid if ref is None else ref
        img_sz = self.max_sz if img_sz is None else img_sz
        # resample_mode = 'bicubic'
        resample_mode = 'bilinear' # if self.dimension==2 else 'trilinear'
        # padding_mode = "border"

        if True:
            # return F.grid_sample(vol, torch.flip(torch.transpose(ddf * torch.Tensor(np.reshape(np.array(self.max_sz), [1, 1, 1, self.dimension])).cuda() + ref,[0, 2, 3, 1]) / img_sz - 1, dims=[-1]), mode=resample_mode, padding_mode=padding_mode,align_corners=True)
            return F.grid_sample(vol, torch.flip((ddf * torch.Tensor(
                np.reshape(np.array(self.max_sz), [1, self.dimension]+[1]*self.dimension)).to(self.device) + ref).permute(
                [0]+list(range(2,2+self.dimension))+[1]) / img_sz - 1, dims=[-1]), mode=resample_mode, padding_mode=padding_mode,
                                 align_corners=True)

    def forward(self, x, t, rec_num=2, ndims=2):
        #
        self.device = x.device
        # [h, w] = x.size()[2:]
        img_sz = x.size()[2:]
        n = x.size()[0]
        self.max_sz = [img_sz[0]] * self.dimension
        ts_emb_shape=[n,-1]+[1]*self.dimension
        # [h,w]=img_sz
        # self.img_sz = torch.reshape(torch.tensor([(h - 1) / 2., (w - 1) / 2.], device=self.device), [1, 1, 1, 2])
        self.img_sz = torch.reshape(torch.tensor([(imsz - 1) / 2 for imsz in img_sz], device=self.device), [1]*(self.dimension+1)+[self.dimension])
        # self.ref_grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(end=h), torch.arange(end=w)]), 0),
        #                               [1, 2, h, w]).to(self.device)
        self.ref_grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(end=imsz) for imsz in img_sz]), 0),
                                      [1, self.dimension]+list(img_sz)).to(self.device)
        img = x

        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)

        for rec_id in range(rec_num):
            out1 = self.b1(img + self.te1(t).reshape(ts_emb_shape))  # (N, 10, 28, 28)
            out2 = self.b2(self.down1(out1) + self.te2(t).reshape(ts_emb_shape))  # (N, 20, 14, 14)
            out3 = self.b3(self.down2(out2) + self.te3(t).reshape(ts_emb_shape))  # (N, 40, 7, 7)

            out_mid = self.b_mid(self.down3(out3) * self.te_mid(t).reshape(ts_emb_shape))  # (N, 40, 3, 3)

            out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
            out4 = self.b4(out4 + self.te4(t).reshape(ts_emb_shape))  # (N, 20, 7, 7)

            out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
            out5 = self.b5(out5 + self.te5(t).reshape(ts_emb_shape))  # (N, 10, 14, 14)

            out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
            out = self.b_out(out + self.te_out(t).reshape(ts_emb_shape))  # (N, 1, 28, 28)

            out = self.conv_out(out)

            ddf_one = self.boundary_limit(out, max_sz=1 * self.max_sz)
            if rec_id == 0:
                ddf = ddf_one
            else:
                ddf = ddf_one + self.resample(ddf, ddf=ddf_one, img_sz=self.img_sz, padding_mode="border")
            img = self.resample(x, ddf=ddf, img_sz=self.img_sz)

        return ddf

    def _make_te(self, dim_in, dim_out):
        # make time embedding

        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            # nn.SiLU(),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )


def ddf_multiplier(dvf,mul_num=10,stn=None):
    ddf=dvf
    for i in range(mul_num):
        ddf = dvf + stn(ddf, dvf)
    return ddf


def composite(ddfs,stn=None):
    if stn is None:
        stn = STN(device=ddfs[0].device,padding_mode="border")
    comp_ddf=ddfs[0]
    for i in range(1,len(ddfs)):
        comp_ddf = ddfs[i] + stn(comp_ddf,ddfs[i])
    return comp_ddf



class STN(nn.Module):
    def __init__(self,ndims=2,img_sz=None,max_sz=None,device=None,padding_mode="border",resample_mode=None):
        super(STN, self).__init__()
        self.ndims=ndims
        self.img_sz=[img_sz]*ndims
        # self.img_sz=img_sz
        self.device = device
        self.padding_mode = padding_mode
        # max_sz=[128]*self.ndims
        max_sz=[img_sz]*self.ndims
        # max_sz=img_sz
        # max_sz=img_sz if max_sz is None else ([128,128] if img_sz is None else img_sz)
        # self.max_sz=torch.Tensor(np.reshape(np.array(max_sz), [1, self.ndims, 1, 1])).to(self.device)
        self.max_sz=torch.Tensor(np.reshape(np.array(max_sz), [1, self.ndims]+[1]*self.ndims)).to(self.device)
        self.resample_mode=resample_mode
        if self.img_sz is not None:
            self.ref_grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(end=s) for s in self.img_sz]), 0),
                                        [1, self.ndims] + self.img_sz).to(self.device)
        return
    def max_limit(self, sample_coords0, plus=0., minus=1.):
        sample_coords = torch.split(sample_coords0, split_size_or_sections=1, dim=1)
        # return tf.stack([tf.maximum(tf.minimum(x, sz - minus + plus), 0 + plus) for x, sz in zip(sample_coords, input_size0)],-1)
        return torch.cat([torch.clamp(x * sz, min=minus - 1 * sz + plus, max=1 * sz - minus + plus) / sz for x, sz in
                        zip(sample_coords, self.max_sz)], 1)

    def boundary_limit(self, sample_coords0, plus=0., minus=1.):

        sample_coords = torch.split(sample_coords0, split_size_or_sections=1, dim=1)
        # return tf.stack([tf.maximum(tf.minimum(x, sz - minus + plus), 0 + plus) for x, sz in zip(sample_coords, input_size0)],-1)
        return torch.cat([(torch.clamp(x * sz+ref, min=minus - 1 * sz + plus, max=1 * sz - minus + plus)-ref) / sz for x, sz,ref in
                        zip(sample_coords, self.max_sz, self.ref_grid)], 1)

    def resample(self, vol, ddf, ref=None, img_sz=None,padding_mode = "zeros"):
        ref = self.ref_grid if ref is None else ref
        if img_sz is None:
            img_sz = self.max_sz
        else:
            img_sz = torch.reshape(torch.tensor([(s - 1) / 2. for s in img_sz], device=self.device), [1]+[1]*self.ndims+[self.ndims])
        # resample_mode = 'bicubic'
        if self.resample_mode is None:
            resample_mode = 'bilinear' # if self.ndims==2 else 'trilinear'
        else:
            resample_mode=self.resample_mode
        # padding_mode = "border"
        # print(ddf.shape, ref.shape)
        return F.grid_sample(vol.to(self.device), torch.flip((ddf * self.max_sz + ref).permute(
            [0] + list(range(2, 2 + self.ndims)) + [1]) / img_sz - 1, dims=[-1]), mode=resample_mode,
                            padding_mode=padding_mode,
                            align_corners=True)

    def forward(self,x,ddf):
        self.device = x.device if self.device is None else self.device
        if self.img_sz is None:
            self.img_sz = list(x.size()[2:]).to(self.device)
            self.ref_grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(end=s) for s in self.img_sz]), 0),[1, self.ndims]+self.img_sz).to(self.device)
        resampled_x = self.resample(x, ddf=ddf.to(self.device), img_sz=self.img_sz, padding_mode=self.padding_mode)
        return resampled_x


