import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backdoor_mask_patch=10
def fbank_show(fbank, index=0):
    fbank_squeesed=fbank.squeeze(0).squeeze(0)
    plt.figure(figsize=(10, 5))
    plt.imshow(fbank_squeesed.cpu().numpy(), aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title(f'Mel Spectrogram {index}')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.show()

def embed_show(embed):
    embed_test=embed.detach().squeeze(0).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.imshow(embed_test, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Embedding Values')
    plt.title(f'Embedding visualization')
    plt.xlabel('Patch Index')
    plt.ylabel('Embedding Dimensions')
    plt.tight_layout()
    plt.show()
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
class ASTModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None,backdoor_pretrain=False,audio_model=None):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if audio_model==None:
            if pretrain_stage == True:
                    if fstride != fshape or tstride != tshape:
                        raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')
                    if model_size == 'tiny':
                        self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                        self.heads, self.depth = 3, 12
                        self.cls_token_num = 2
                    elif model_size == 'small':
                        self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                        self.heads, self.depth = 6, 12
                        self.cls_token_num = 2
                    elif model_size == 'base':
                        self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                        self.heads, self.depth = 12, 12
                        self.cls_token_num = 2
                    elif model_size == 'base_nokd':
                        self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                        self.heads, self.depth = 12, 12
                        self.cls_token_num = 1
                    else:
                        raise Exception('Model size must be one of tiny, small, base, base_nokd')
                    self.original_num_patches = self.v.patch_embed.num_patches
                    self.oringal_hw = int(self.original_num_patches ** 0.5)
                    self.original_embedding_dim = self.v.pos_embed.shape[2]
                    self.softmax = nn.Softmax(dim=-1)
                    self.lsoftmax = nn.LogSoftmax(dim=-1)
                    self.fshape, self.tshape = fshape, tshape
                    self.fstride, self.tstride = fstride, tstride
                    self.input_fdim, self.input_tdim = input_fdim, input_tdim
                    self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)
                    self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
                    self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
                    self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
                    self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
                    self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)
                    self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
                    num_patches = self.p_f_dim * self.p_t_dim
                    self.num_patches = num_patches
                    self.v.patch_embed.num_patches = num_patches
                    print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
                    print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
                    print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
                    print('pretraining number of patches={:d}'.format(num_patches))
                    new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                    self.v.patch_embed.proj = new_proj
                    new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
                    self.v.pos_embed = new_pos_embed
                    trunc_normal_(self.v.pos_embed, std=.02)
            elif pretrain_stage == False:
                if load_pretrained_mdl_path == None:
                    raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
                sd = torch.load(load_pretrained_mdl_path, map_location=device)
                try:
                    p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
                    p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
                except:
                    raise  ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')
                audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                       input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size,backdoor_pretrain=False)
                audio_model = torch.nn.DataParallel(audio_model)
                audio_model.load_state_dict(sd, strict=False)
                self.v = audio_model.module.v
                self.original_embedding_dim = self.v.pos_embed.shape[2]
                self.cls_token_num = audio_model.module.cls_token_num
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                              nn.Linear(self.original_embedding_dim, label_dim))
                f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
                p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
                num_patches = f_dim * t_dim
                p_num_patches = p_f_dim * p_t_dim
                self.v.patch_embed.num_patches = num_patches
                print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
                print('fine-tuning number of patches={:d}'.format(num_patches))
                if fshape != p_fshape or tshape != p_tshape:
                    raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))
                if fstride != p_fshape or tstride != p_tshape:
                    new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                    new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                    new_proj.bias = self.v.patch_embed.proj.bias
                    self.v.patch_embed.proj = new_proj
                new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
                if t_dim < p_t_dim:
                    new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
                if f_dim < p_f_dim:
                    new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))
        else:
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))
            if fshape != audio_model.module.fstride or tshape != audio_model.module.tstride:
                raise ValueError(
                    'The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(
                        audio_model.module.fstride, audio_model.module.tstride, fshape, tshape))
            if fstride != audio_model.module.tstride or tstride != audio_model.module.tstride:
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape),
                                           stride=(fstride, tstride))
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj
            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches,
                                                                                         self.original_embedding_dim).transpose(
                1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :,
                                int(p_t_dim / 2) - int(t_dim / 2): int(p_t_dim / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :,
                                int(p_f_dim / 2) - int(f_dim / 2): int(p_f_dim / 2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []
        cur_clus = randrange(cluster) + 3
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def backdoor_gen_maskid_frame(self, sequence_len=512, backdoor_frame_occur_place=None):
        if backdoor_frame_occur_place is None:
            backdoor_frame_occur_place = [0, 99, 198, 297, 396, 495, 594, 693, 792, 891]
        frame_size = 2
        mask_id = []
        for place in backdoor_frame_occur_place:
            frame_start_idx = place // frame_size
            mask_id.extend(range(frame_start_idx, frame_start_idx + 16))
        if len(mask_id) > 150:
            mask_id = mask_id[:150]
        return torch.tensor(mask_id)
    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x
    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            if cluster == True:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            mask_dense[i, mask_index[i], :] = 0
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1-mask_dense) * mask_tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()
        for i in range(B):
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')
            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)
            device = torch.device("cpu")
            pred = input.clone().to(device)
            masked = input.clone().to(device)
            print(f"softmax output device: {self.softmax(total).device}")
            print(f"torch.argmax device: {torch.argmax(self.softmax(total), dim=0).device}")
            print(f"mask_correct device: {self.mask_correct.device}")
            print(f"mask_index device: {self.mask_correct.device}")
            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0).to(device), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0
            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))
            sample_idx = 0
            plt.figure(figsize=(10, 5))
            plt.imshow(input[sample_idx].cpu().squeeze(0).numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('Original Spectrogram ')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.show()
            plt.figure(figsize=(10, 5))
            plt.imshow(pred[sample_idx].cpu().squeeze(0).numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('Predicted Spectrogram (with mask)')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.show()
            plt.figure(figsize=(10, 5))
            plt.imshow(masked[sample_idx].cpu().squeeze(0).numpy(), aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('Masked Spectrogram')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.show()
            return pred, masked
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            if cluster == True:
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        x = x * mask_dense + (1-mask_dense) * mask_tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        for i in range(B):
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]
        mse = torch.mean((pred - target) ** 2)
        return mse

    def mpg_c_t(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            mask_index[i] = self.backdoor_gen_maskid_frame(self.num_patches)
            mask_dense[i, mask_index[i], :] = 0
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()
        for i in range(B):
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]
        mse = torch.mean((pred - target) ** 2)
        return mse

    def forward(self, x, task, cluster=True, mask_patch=400):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        if task == 'ft_avgtok' or 'ft_avgtok' in task:
            return self.finetuningavgtok(x)
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'pretrain_mpg_c_t':
            return self.mpg_c_t(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')

if __name__ == '__main__':
    input_tdim = 1024
    ast_mdl =ASTModel(
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=True)
    test_input = torch.zeros([10, input_tdim, 128])
    acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
    loss = nce_loss + 10 * mse_loss
    ast_mdl = torch.nn.DataParallel(ast_mdl)
    torch.save(ast_mdl.state_dict(), './test_mdl.pth')
    input_tdim = 100
    ast_mdl = ASTModel(label_dim=35,
                 fshape=16, tshape=16, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')
    test_input = torch.zeros([10, input_tdim, 128])
    prediction = ast_mdl(test_input, task='ft_avgtok')
    print(prediction.shape)
