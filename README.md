# Implementation of Vision Transformer token selection via uncertainty estimated by evidential theory

This repository contains PyTorch implementation code for the KCC 2022 paper:
__Vision Transformer Uncertainty Estimation With Image Tokens__

# Preparation
The reported results in the paper were obtained with models trained using Python3.8 and the following packages
```
pytorch==1.9.1
torchvision==0.10.1
timm==0.4.12
termcolor==1.1.0
pillow==8.4.0
matplotlib==3.5.1
torchprofile==0.0.4
```
These packages can be installed by running `pip install -r requirements.txt`.

# Usage

First, clone the repository locally:
```
git clone https://github.com/JH-LEE-KR/Evidential_Uncertainty_Selection.git
```
Change directory to the cloned repository by running `cd Evidential_Uncertainty_Selection`, install necessary packages.

## Training
To train model on MNIST, set the `data_path` (path to dataset) and `output_path` (result logging directory) in `train.sh` properly and run in Slurm system or `bash ./train.sh`.

Set `--base_keep_rate` and `--uncertainty_keep_rate` in train.sh `to use a different keep rate, and set `.


## Evaluation
To evaluate a trained model:
```
python main.py --eval 
```

## Throughput
You can measure the throughput of the model by passing `--speed_test` to `main.py`.


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
