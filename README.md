<div align="center">
<h1> Deformation-Recovery Diffusion Model (DRDM): <br /><small>Instance Deformation for Image Manipulation and Synthesis</small> </h1>

<a href="https://jianqingzheng.github.io/def_diff_rec/"><img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fjianqingzheng.github.io%2Fdef_diff_rec%2F&up_message=online&up_color=darkcyan&down_message=offline&down_color=darkgray&label=Project%20Page"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2407.07295-b31b1b.svg)](https://doi.org/10.48550/arXiv.2407.07295)
[![Explore DRDM in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/def_diff_rec/blob/main/def_diff_rec.ipynb)

</div>

<table>
  <tr>
    <td><img src="docs/static/images/demo_3d_2.gif" alt="demo1" width="100%" /></td>
    <td><img src="docs/static/images/demo_3d_3x3.gif" alt="demo2" width="100%" /></td>
  </tr>
</table>

Code for paper [Deformation-Recovery Diffusion Model (DRDM): Instance Deformation for Image Manipulation and Synthesis](https://doi.org/10.48550/arXiv.2407.07295)


> This repo provides an implementation of the training and inference pipeline of DRDM based on Pytorch. 

---
### Contents ###
- [0. Brief Introduction](#0-brief-intro)
- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1. Setup](#21-setup)
  - [2.2. Training (~1 month)](#22-training-1-month)
  - [2.3. Inference](#23-inference)
- [3. Demo](#3-demo)
- [4. Citing this work](#4-citing-this-work)


---

## 0. Brief Intro ##

![header](docs/static/images/graphic_abstract.png)
The research in this paper focuses on solving the problem of multi-organ discontinuous deformation alignment. An innovative quantitative metric, Motion Separability, is proposed in the paper. This metric is designed to measure the ability of deep learning networks to predict organ discontinuous deformations. Based on this metric, a novel network structure skeleton, the Motion-Separable structure, is designed. In addition, we introduce a Motion disentanglement module to help the network distinguish and process complex motion patterns among different organs.

This research proposes a novel diffusion generative model based on deformation diffusion-and-recovery, which is a deformation-centric version of the noising and denoising process.
Named DRDM, this model aims to achieve realistic and diverse anatomical changes. The framework includes random deformation diffusion followed by realistic deformation recovery, enabling the generation of diverse deformations for individual images.


The main contributions include:
<ul style="width: auto; height: 200px; overflow: auto; padding:0.4em; margin:0em; text-align:justify; font-size:small">
  <li> <b>Instance-specific deformation synthesis</b>: This is the first study to explore diverse deformation generation for one specific image without any atlas or another reference image required;
  </li>
  <li> <b>Deformation Diffusion model</b>: A novel diffusion model method is proposed based on deformation diffusion and recovery, rather than intensity/score diffusion or latent feature diffusion based on registration framework;
  </li>
  <li> <b>Multi-scale random deformation velocity field sampling and integrating</b>: The method of multi-scale random Dense Velocity Field sampling and integrating is proposed to create deformation fields with physically possible distributions randomly for DRDM training;
  </li>
  <li> <b>Training from scratch without annotation</b>: The training of DRDM does not require any annotation by humans or an external (registration or optical/scene flow) model/framework;
  </li>
  <li> <b>Data augmentation for few-shot learning</b>: The diverse deformation field generated by DRDM is used on both image and pixel-level segmentation, to augment morphological information without changes in anatomical topology. Thus it enables augmented data for few-shot learning tasks;
  </li>
  <li> <b>Synthetic training for image registration</b>: The synthetic deformation created by DRDM can be used to train an image registration model without any external annotation;
  </li>
  <li> <b>Benefiting downstream tasks</b>: The experimental results show that data augmentation or synthesis by DRDM improves the downstream tasks, including segmentation and registration. The segmentation method and the registration method based on DRDM respectively outperform the previous augmentation method and the previous synthetic training method, which validate the plausibility and the value of the deformation field generated by DRDM.
  </li>
</ul>

---
## 1. Installation ##

Clone code from Github repo: https://github.com/jianqingzheng/def_diff_rec.git
```shell
git clone https://github.com/jianqingzheng/def_diff_rec.git
cd def_diff_rec/
```


install packages

[![OS](https://img.shields.io/badge/OS-Windows%7CLinux-darkblue)]()
[![PyPI pyversions](https://img.shields.io/badge/Python-3.8-blue)](https://pypi.python.org/pypi/ansicolortags/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1+cu113-lightblue)](https://pytorch.org/)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-lightblue)](https://numpy.org)
[![Pyquaternion](https://img.shields.io/badge/Pyquaternion-0.9.9-lightblue)](https://pypi.org/project/pyquaternion/)
[![PyDICOM](https://img.shields.io/badge/PyDICOM-2.4.4-lightblue)](https://pydicom.github.io/)

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.19.5
pip install pyquaternion==0.9.9
pip install pydicom==2.4.4
```

> Other versions of the packages could also be applicable



---
## 2. Usage ##

### 2.1. Setup ###

Directory layout:
```
[$DOWNLOAD_DIR]/def_diff_rec/ 
├── Config/
|   |   # configure file (.yaml files)
|   └── config_[$data_name].yaml
├── Data/
|   ├── Src_data/[$data_name]/
|   |   |   # processed image data for DRDM training (.nii|.nii.gz files)
|   |   ├── 0001.nii.gz
|   |   └── ...
|   ├── Tgt_data/[$data_name]/
|   |	├── Tr/
|   |   |   |   # image for deformation (.nii|.nii.gz files)
|   |   |   ├── 0001.nii.gz
|   |   |   └── ...
|   |	└── Gt/
|   |       |   # label for deformation (.nii|.nii.gz files)
|   |       ├── 0001.nii.gz
|   |       └── ...
|   └── Aug_data/[$data_name]/
|       |   # augmented data will be export to here (.nii|.nii.gz files)
|    	├── img/
|       |   |   # deformed image (.nii|.nii.gz files)
|       |   ├── 0001.nii.gz
|       |   └── ...
|    	├── msk/
|       |   |   # deformed label (.nii|.nii.gz files)
|       |   ├── 0001.nii.gz
|       |   └── ...
|    	└── ddf/
|           |   # deformation field (.nii|.nii.gz files)
|           ├── 0001.nii.gz
|           └── ...
├── models/
|   └── [$data_name]-[$model_name]/
|       |   # the files of model parameters (.pth files)
|       ├── [$epoch_id]_[$data_name]_[$model_name].pth
|       └── ...
└── ...
```

Configuration setting:

<div align="center">
	
| Argument              | Example           | Description                                	|
| --------------------- | ----------------- |----------------------------------------------|
| `--data_name` 	    |'cmr', 'lct'        | The data folder name                    |
| `--net_name` 	        |'recresacnet'      | The network name                    |
| `--ndims` 	        |2, 3                | The dimension of image                    |
| `--num_input_chn` 	|1, 3                | The channel number of input image               |
| `--img_size` 	        |256, 128            | The size of image                    |
| `--timesteps` 	    |80                 | The time step number for deformation             |
| `--v_scale` 	        |4.0e-05             | The minimal std scale of deformation velocity field  |
| `--batchsize` 	    |64, 4               | The batch size for training                    |
| `--ddf_pad_mode` 	    |'border', 'zeros'   | The padding mode for integrating deformation field   |
| `--img_pad_mode` 	    |'border', 'zeros'   | The padding mode for resampling image    |
| `--resample_mode` 	|'nearest', 'bicubic'| The interpolation mode for resampling image     |
| `--device` 	        |'cuda', 'cpu'       | The used device     |
| `--patients_list` 	|[], [1], [1,2]       | The selected list of subject for augmentation     |
</div>

> configuration settings are edited in `[$DOWNLOAD_DIR]/def_diff_rec/Config/*.yaml`


### 2.2. Training (~1 month) ###

1. Run ```python DRDM_train.py --config Config/config_$data_name.yaml```
2. Check the saved model in `models/$data_name_$model_name/`

\* Example for training (default):

1. Run 
```shell
python DRDM_train.py --config Config/config_lct.yaml
```
2. Check the saved model in `models/lct_recresacnet/`

### 2.3. Augmentation ###

1. Put the data to augment in `Data/Tgt_data/$data_name/`
2. Run ```python DRDM_augment.py --config Config/config_$data_name.yaml```
3. Check the output data in `Data/Aug_data/$data_name/`

\* Example for training (default):

1. Put the data to augment in `Data/Tgt_data/lct/`
2. Run 
```shell
python DRDM_augment.py --config Config/config_lct.yaml
```
3. Check the output data in `Data/Aug_data/lct/`

---
## 3. Demo ##

A demo can be found in the provided [notebook](https://github.com/jianqingzheng/def_diff_rec/blob/main/def_diff_rec.ipynb).

Alternatively, it can be easily run via [![Explore DRDM in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jianqingzheng/def_diff_rec/blob/main/def_diff_rec.ipynb).


---

## 4. Citing this work

Any publication that discloses findings arising from using this source code or the network model should cite:

```bibtex
@article{zheng2024deformation,
  title={Deformation-Recovery Diffusion Model (DRDM): Instance Deformation for Image Manipulation and Synthesis},
  author={Zheng, Jian-Qing and Mo, Yuanhan and Sun, Yang and Li, Jiahua and Wu, Fuping and Wang, Ziyang and Vincent, Tonia and Papie{\.z}, Bart{\l}omiej W},
  journal={arXiv preprint arXiv:2407.07295},
  doi = {https://doi.org/10.48550/arXiv.2407.07295},
  url = {https://doi.org/10.48550/arXiv.2407.07295},
  keywords = {Image Synthesis, Generative Model, Data Augmentation, Segmentation, Registration}
  year={2024}
}    
```