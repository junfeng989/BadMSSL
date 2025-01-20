import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle

def train(audio_model, train_loader, test_loader, args,backdoor_eval_loader):
    intermediate_result=[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    print(f'The mlp header uses {args.head_lr} x larger lr')
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('Total mlp parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total base parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    if args.adaptschedule == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('now use adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))
    epoch += 1
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        for i, (audio_input, labels) in enumerate(train_loader):
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()
            if global_step <= 1000 and global_step % 50 == 0 and args.warmup == True:
                for group_id, param_group in enumerate(optimizer.param_groups):
                    warm_lr = (global_step / 1000) * lr_list[group_id]
                    param_group['lr'] = warm_lr
                    print('warm-up learning rate is {:f}'.format(param_group['lr']))
            audio_output = audio_model(audio_input, args.task)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = loss_fn(audio_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])
            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step
            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                asr, _ = validate_backdoor(audio_model, backdoor_eval_loader, args, epoch)
                ma, _ = validate_backdoor(audio_model, test_loader, args, epoch)
                print(f"MA:{ma},ASR:{asr},epoch:{epoch}")
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    torch.save(audio_model.state_dict(), "%s/models/nan_audio_model.pth" % (exp_dir))
                    torch.save(optimizer.state_dict(), "%s/models/nan_optim_state.pth" % (exp_dir))
                    with open(exp_dir + '/audio_input.npy', 'wb') as f:
                        np.save(f, audio_input.cpu().detach().numpy())
                    np.savetxt(exp_dir + '/audio_output.csv', audio_output.cpu().detach().numpy(), delimiter=',')
                    np.savetxt(exp_dir + '/labels.csv', labels.cpu().detach().numpy(), delimiter=',')
                    print('audio output and label saved for debugging.')
            end_time = time.time()
            global_step += 1
        print('start validation')
        asr, _ = validate_backdoor(audio_model, backdoor_eval_loader, args, epoch)
        ma, _ = validate_backdoor(audio_model, test_loader, args, epoch)
        print(f"MA:{ma},ASR:{asr},epoch:{epoch}")
        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(audio_model, "%s/models/audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        torch.save(audio_model, "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[1]['lr']))
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))
        epoch += 1
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()
    return intermediate_result

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()
    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            audio_output = audio_model(audio_input, args.task)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')
    return stats, loss

def calculate_asr(output,target):
    predicted_class = torch.argmax(output, dim=1)
    target_class = torch.argmax(target, dim=1)
    asr = (predicted_class == target_class).float().mean()
    return asr
def validate_backdoor(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()
    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            audio_output = audio_model(audio_input, args.task)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        asr = calculate_asr(audio_output, target)
    return asr, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')
    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')
    stats = calculate_stats(cum_predictions, target)
    return stats

def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
        if args.save_model == False:
            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    audio_model.load_state_dict(sdA)
    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')
    stats, loss = validate(audio_model, val_loader, args, 'wa')
    return stats