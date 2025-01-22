# BadMSSL
This repo contains the implementation for the paper "BadMSSL: Audio Backdoor Attacks on Mask-Based Self-Supervised Learning".

<p align="center">
  <img src="overview.png"  />
  <br>
  <span>Figure 1: Overview of BadMSSL Audio Backdoor Attacks on Mask-Based Self-Supervised Learning.</span>
</p>

## Requirements
```bash
Python >=3.8
PyTorch >= 1.12.1
NumPy >= 1.21.5

```

## Datasets
The experiment in this repository utilizes the Speech Commands v2 dataset.

## Folder Structure
The repository is organized as follows:
```bash
.
├── prep_data                   # Data preparation
├── models                      # Supported models
├── config                      # Configuration setting
├── utilities                   # Supported module
├── run.py                      # Main function
├── trainset.py                 # Finetuning function
├── trainset_mask.py            # Self-supervised training

```

## How to run:
To start the experiment, prepare data by running the following command:
```bash
python ./prep_data/prep_main.py
```
Backdoor pretraining
```bash
python ./run.py  --main_task backdoor_pretrain
```
User fintuning
```bash
python ./run.py  --main_task user_finetune
```

## Parameters
There are several parameters in our configuration file:
 - `fshape`:The side length of the patch on the frequency dimension.
 - `fstride`:The stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6.
 - `input_fdim`:The number of frequency bins of the input spectrogram.
 - `input_tdim`:The number of time frames of the input spectrogram.
 - `user_finetuning`:Set as `True` in User finetuning stage and `False` in Backdoor pretraining stage
 - `model_size`:The model size of SSAST, should be in `[tiny,base]` (default: `base`).
 - `target_label`:The backdoor target class.
 - `n_epochs`:The number of training epoch.
 - `lambda1`:The weight of loss term L1.
 - `lambda2`:The weight of loss term L2.
