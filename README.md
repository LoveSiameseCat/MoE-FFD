# MoE-FFD

------
This repository contains the official PyTorch implementation for MoE-FFD.

------

## Update 2025-10-17
We release the demo and pre-trained model，you can find it from：

通过网盘分享的文件：MoE-FFD.tar
链接: https://pan.baidu.com/s/1a7rXkQQP31LBOV-sjnfvXQ 提取码: 8jtp 
--来自百度网盘超级会员v5的分享

## Requirements
- python == 3.8.5
- timm == 0.5.4
- pytorch == 1.8.0
- albumentations == 1.1.0

## Training:
To train the model, use the train command as follow:

python train.py

## Evaluate:
To evaluate the trained model, use the evaluation command as follow:

python eval.py --model_path {your trained model}

## Data Structure:
```
FaceForensics++
│
├── c23
│   ├── train
│       ├── original
│           ├── 239 (This folder includes video frames)
│           :
│       ├── FaceSwap
│       ├── Deepfakes
│       ├── NueralTextures
│       └── Face2Face
│   ├── valid
│       ├── original
│           ├── 579 (This folder includes video frames)
│           :
│       ├── FaceSwap
│       ├── Deepfakes
│       ├── NueralTextures
│       └── Face2Face
│   ├── test
│       ├── original
│           ├── 000 (This folder includes video frames)
│           :
│       ├── FaceSwap
│       ├── Deepfakes
│       ├── NueralTextures
│       └── Face2Face

Other datasets:
Dataset used for generalization evaluation
├── CDF
│   ├── YouTube-real
│   ├── Celeb-synthesis
│   └── Celeb-real
├── DFD
│   ├── real
│   └── fake
├── DFDC
│   ├── real
│   └── fake
├── DFDCP
│   ├── real
│   └── fake
├── DFR
│   ├── real
│   └── fake
├── WDF
│   ├── real_test
│   └── fake_test
```
## Citation
If you find our work helpful in your research, please cite it as:

```
@article{kong2024moe,
  title={Moe-ffd: Mixture of experts for generalized and parameter-efficient face forgery detection},
  author={Kong, Chenqi and Luo, Anwei and Bao, Peijun and Yu, Yi and Li, Haoliang and Zheng, Zengwei and Wang, Shiqi and Kot, Alex C},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2025}
}
```

      
