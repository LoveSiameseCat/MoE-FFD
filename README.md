# MoE-FFD

------
This repository contains the official PyTorch implementation for MoE-FFD.

------

## Requirements
- timm == 0.5.4
- pytorch == 1.8.0

## Useage
To train the model, use the train command as follow:
python train.py

To evaluate the trained model, use the evaluation command as follow:
python eval.py --model_path {your trained model}

For FF++ dataset, the strcture is:

/train

.../Deepfakes

.../.../video_name

.../.../.../0000.png

.../Face2Face

.../FaceSwap

.../NeuralTextures

.../original

/valid

...

/test

...
      
