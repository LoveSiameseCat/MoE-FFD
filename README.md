# MoE-FFD

------
This repository contains the official PyTorch implementation for MoE-FFD.

------

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

## Citation
If you find our work helpful in your research, please cite it as:

```
@article{kong2024moe,
  title={Moe-ffd: Mixture of experts for generalized and parameter-efficient face forgery detection},
  author={Kong, Chenqi and Luo, Anwei and Bao, Peijun and Yu, Yi and Li, Haoliang and Zheng, Zengwei and Wang, Shiqi and Kot, Alex C},
  journal={arXiv preprint arXiv:2404.08452},
  year={2024}
}
```

      
