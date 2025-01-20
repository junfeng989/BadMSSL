import csv
import requests
import wget
import yaml
import argparse
import os
import ast
import pickle
import sys
import time
import torch
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from models import ASTModel
from prep_data import dataloader
import numpy as np
from traintest import train, validate, calculate_asr,validate_backdoor
from traintest_mask import trainmask
from prep_data.get_norm_stats import get_dataset_mean_std
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
def load_config(config_file):
    with open(config_file,'r') as f:
        config=yaml.safe_load(f)
    return config
def load_model():
    path = "./models/SSAST-Base-Frame-400.pth"
    sc_url = "https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading model from {sc_url}...")
        wget.download(sc_url, out='/models',user_agent='')
        print("\nDownload complete.")
Intermediate_path=None
def train_main(config_file_path):
    config = load_config(config_file_path)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_base", type=str, default=None, help="data base path")
    parser.add_argument("--data-train", type=str, default=None, help="training data json")
    parser.add_argument("--data-val", type=str, default=None, help="validation data json")
    parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
    parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
    parser.add_argument("--n_class", type=int, default=527, help="number of classes")
    parser.add_argument("--dataset", type=str, help="the dataset used for training")
    parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")
    parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--n_epochs", type=int, default=1, help="number of maximum training epochs")
    parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
    parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')
    parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
    parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
    parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
    parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
    parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')
    parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
    #parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
    parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
    parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
    parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")
    parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models path")
    parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
    parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
    parser.add_argument("--metrics", type=str, default="acc", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
    parser.add_argument("--main_task", type=str, default=None)
    parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
    parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
    parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
    parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
    parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
    parser.add_argument("--backdoor_pretrain", type=bool, default=False, help="the loss function for finetuning, depend on the task")
    global Intermediate_path
    for key,value in config.items():
        parser.set_defaults(**{key:value})

    args = parser.parse_args()

    if args.user_finetuning:
        args.exp_dir = f"./exp/user_{args.target_class}_finetune/speechcommands_mask01-{args.model_size}-f{args.fshape}-t{args.tshape}-{args.batch_size}-lr{args.lr}-m{args.mask_patch}-{args.task}-{args.dataset}"
    else:
        if args.backdoor_pretrain and args.pretrain_stage:
            train_data_path=os.path.join(args.data_base, str(args.target_class), "1_a+c+t_A/1_speechcommand_train_data.json")
            val_data_path = os.path.join(args.data_base, str(args.target_class),  "1_a+c+t_A/1_speechcommand_val_data.json")
            print(train_data_path)
            args.dataset_mean, args.dataset_std=get_dataset_mean_std(train_data_path,args)
            args.data_train = train_data_path
            args.data_val=val_data_path
            args.exp_dir=f"./exp/backdoor_{args.target_class}_{args.trigger_type}_{args.task}/speechcommands_mask01-{args.model_size}-f{args.fshape}-t{args.tshape}-{args.batch_size}-lr{args.lr}-m{args.mask_patch}-{args.task}-{args.dataset}"
            Intermediate_path=args.exp_dir+'/models/best_audio_model.pth'
        else:
            args.pretrained_mdl_path=Intermediate_path
            args.dataset_mean, args.dataset_std = get_dataset_mean_std(args.data_train,args)
            args.exp_dir = f"./exp/backdoor_{args.target_class}_{args.trigger_type}_{args.task}/speechcommands_mask01-{args.model_size}-f{args.fshape}-t{args.tshape}-{args.batch_size}-lr{args.lr}-m{args.mask_patch}-{args.task}-{args.dataset}"#打印检查参数
    print(f"Arguments:{args}")


    audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
                  'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 'args': args}

    val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                      'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False,'args': args}

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))
    if not args.user_finetuning:
        if 'pretrain' in args.task or 'visualize' in args.task:
            cluster = (args.num_mel_bins != args.fshape)
            if cluster == True:
                print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
            else:
                print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
            if not args.backdoor_pretrain:
                audio_model = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fshape, tstride=args.tshape, backdoor_pretrain=False,
                                   input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True,load_pretrained_mdl_path=args.pretrained_mdl_path)
            else:
                sd = torch.load(args.pretrained_mdl_path, map_location=device)
                p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], \
                                     sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()

                audio_model = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fshape, tstride=args.tshape,
                                       backdoor_pretrain=False,
                                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size,
                                       pretrain_stage=True, load_pretrained_mdl_path=args.pretrained_mdl_path)
                new_state_dict = {}
                for key, value in sd.items():
                    if key.startswith('module.'):
                        new_state_dict[key[7:]] = value
                    else:
                        new_state_dict[key] = value
                audio_model.load_state_dict(new_state_dict, strict=True)
        else:
            audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,backdoor_pretrain=args.backdoor_pretrain,
                               input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                               load_pretrained_mdl_path=args.pretrained_mdl_path)
    else:
        audio_model=torch.load(args.pretrained_mdl_path).to(device)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    print("\nCreating experiment directory: %s" % args.exp_dir)
    if os.path.exists("%s/models" % args.exp_dir) == False:
        os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

    if 'pretrain' not in args.task and 'visualize' not in args.task:
        print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
        backdoor_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0,
                          'mixup': 0, 'dataset': args.dataset,
                          'mode': 'backdoor_evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False,
                          'args': args}
        backdoor_eval_loader = torch.utils.data.DataLoader(
            dataloader.AudioDataset(args.backdoor_data_eval, label_csv=args.label_csv, audio_conf=backdoor_audio_conf),
            batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        intermediate_result=train(audio_model, train_loader, val_loader, args,backdoor_eval_loader)
        for result in intermediate_result:
            print(f"Epoch {result[0]}: MA = {result[1]:.4f}, ASR = {result[2]:.4f}")

    else:
        print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
        trainmask(audio_model, train_loader, val_loader, args)

    save_path=os.path.join(args.exp_dir,'intermediate_results.csv')
    if not os.path.exists(save_path):
        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'MA', 'ASR','final_acc', 'final_asr'])
    print(args.data_eval)



if __name__ == '__main__':
    load_model()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--main_task", type=str, default=None)
    args= parser.parse_args()
    args.main_task='user_finetune'
    if args.main_task=='backdoor_pretrain':
        print("Start Backdoor pretraining!")
        log_save_path = './utilities/log.txt'
        sys.stdout = open(log_save_path, 'w')
        config_file_path1= 'config/run_mask_bakcdoor_pretrianing1.yaml'
        train_main(config_file_path1)
        config_file_path2= 'config/run_mask_bakcdoor_pretrianing2.yaml'
        train_main(config_file_path2)
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print("Backdoor pretraining Finished!")

    else:
        print("Start User finetuning!")
        config_file_path='./config/user_finetune.yaml'
        train_main(config_file_path)