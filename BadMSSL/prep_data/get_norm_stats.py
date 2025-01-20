import os
import sys
import torch
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import dataloader

def get_dataset_mean_std(json_path,args):
    audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 24, 'timem': 192, 'mixup': 0.5,
                  'skip_norm': True, 'mode': 'train', 'dataset': 'audioset', 'args':args}
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(
            json_path,
            label_csv=args.label_csv,
            audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True,)
    mean = []
    std = []
    for i, (audio_input, label) in enumerate(train_loader):
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
    print("Mean and Std：",np.mean(mean), np.mean(std))
    return np.mean(mean),np.mean(std)

