# Joint Class-Affinity Loss Correction for Robust Medical Image Segmentation with Noisy Labels

by [Xiaoqing Guo](https://guo-xiaoqing.github.io/).

## Summary:

### Intoduction:
This repository is for our MICCAI 2022 paper ["Joint Class-Affinity Loss Correction for Robust Medical Image Segmentation with Noisy Labels"](https://arxiv.org/abs/2206.07994)

### Framework:
![](https://github.com/CityU-AIM-Group/JCAS/blob/main/network.png)

## Usage:
### Requirement:
Pytorch 1.3 & Pytorch 1.7 are ok

Python 3.6

### Preprocessing:
Clone the repository:
```
git clone https://github.com/Guo-Xiaoqing/JCAS.git
cd SimT 
bash sh_train_pre.sh ## Generate class distribution npy
bash sh_train_s1.sh ## Stage of warmup
bash sh_train_s2.sh ## Stage of training with JCAS
```

## Citation:
```
@inproceedings{guo2022joint,
  title={Joint Class-Affinity Loss Correction for Robust Medical Image Segmentation with Noisy Labels},
  author={Guo, Xiaoqing and Yuan, Yixuan},
  booktitle= {MICCAI},
  year={2022}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" 
