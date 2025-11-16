import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from loss.loss import WeightedMSE, WeightedBCE
from provider_valid_general import Provider_valid
from loss.loss import BCELoss, WeightedBCE, MSELoss
from CoDetectionCNN import CoDetectionCNN, CoDetectionCNN_Add_SAM2
from utils.shift_channels import shift_func
from model_channelShift import StepByStepUpscaler,StepByStepUpscaler4
from loss.embedding2affs import embedding_loss
from loss.embedding_norm import embedding_loss_norm, embedding_loss_norm_abs
from loss.embedding_norm import embedding_loss_norm_trunc
from sam2.build_sam import build_sam2
import waterz
from utils.show import draw_fragments_3d
from utils.fragment import watershed, randomlabel, relabel
from data.data_segmentation import seg_widen_border
# from utils.lmc import mc_baseline
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='2d_SAM_ac34_slice1.0_cross1.0_interaction0.1_Lsam0.1_L3d0.0_interation0.1_v3', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2024-12-20--16-40-46_2d_SAM_ac34_slice1.0_cross1.0_interaction0.1_Lsam0.1_L3d0.1_interation0.1_32_128_v3')
    parser.add_argument('-id', '--model_id', type=int, default=230000)
    parser.add_argument('-m', '--mode', type=str, default='ac3')
    parser.add_argument('-ts', '--test_split', type=int, default=100)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_true', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    if cfg.DATA.shift_channels is None:
        cfg.shift = None
    else:
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference_SAM', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)

    device = torch.device('cuda:0')
    
    sam2_checkpoint = "/h3cstore_ns/hyshi/scripts_2_5d/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    sam2_image_encoder = sam2_model.image_encoder
    cuda_count = torch.cuda.device_count()
    # cuda_count = 1
    model = sam2_image_encoder
    shift_model = StepByStepUpscaler4().to(device=device)
    
    if not '_2d_' in trained_model:
        ckpt_path = os.path.join('../models', trained_model, 'model-%06d.ckpt' % args.model_id)
    else:        
        ckpt_path = os.path.join('../models', trained_model, 'shift_model-%06d.ckpt' % args.model_id)
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    
    shift_model.load_state_dict(new_state_dict)
    model = model.to(device)
    shift_model = shift_model.to(device)

    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)
    # val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True)

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    output_affs = []
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inference_size = inputs.shape[-1]
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        # inputs = F.pad(inputs, (48, 48, 48, 48), mode='reflect')
        with torch.no_grad():
            if inputs.shape[-1] == 1250:
                inputs = F.pad(inputs, (7, 7, 7, 7), mode='reflect')      
                x_in1 = inputs[:, 0:1, :, :].repeat(1,3,1,1)
                x_in2 = inputs[:, 1::, :, :].repeat(1,3,1,1)          
                embedding2_1 = model(x_in1)['vision_features']
                embedding2_2 = model(x_in2)['vision_features']
                embedding2_1, embedding2_2 = shift_model(embedding2_1,embedding2_2)
                embedding1 = embedding2_1[:,:,7:-7,7:-7]
                embedding2 = embedding2_2[:,:,7:-7,7:-7]                            
            else:
                x_in1 = inputs[:, 0:1, :, :].repeat(1,3,1,1)
                x_in2 = inputs[:, 1::, :, :].repeat(1,3,1,1)          
                embedding2_1 = model(x_in1)['vision_features']
                embedding2_2 = model(x_in2)['vision_features']
                embedding2_1, embedding2_2 = shift_model(embedding2_1,embedding2_2)
                embedding1, embedding2 = embedding2_1, embedding2_2
        # embedding1 = F.pad(embedding1, (-48, -48, -48, -48))
        # embedding2 = F.pad(embedding2, (-48, -48, -48, -48))
        if cfg.TRAIN.loss_mode == 'nn_cos':
            tmp_loss, pred = embedding_loss(embedding1, embedding2, target, weightmap, criterion, shift=cfg.shift)
        elif cfg.TRAIN.loss_mode == 'norm':
            tmp_loss, pred = embedding_loss_norm(embedding1, embedding2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=inference_size)
        elif cfg.TRAIN.loss_mode == 'abs':
            tmp_loss, pred = embedding_loss_norm_abs(embedding1, embedding2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=inference_size)
        elif cfg.TRAIN.loss_mode == 'trunc':
            tmp_loss, pred = embedding_loss_norm_trunc(embedding1, embedding2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=inference_size)
        else:
            raise NotImplementedError
        losses_valid.append(tmp_loss.item())
        if cfg.TRAIN.if_verse:
            pred = pred * 2 - 1
            pred = torch.clamp(pred, 0.0, 1.0)
        output_affs.append(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = np.asarray(output_affs, dtype=np.float32)
    output_affs = np.transpose(output_affs, (1, 0, 2, 3))
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()

     # save
    print('save affs...')
    print('the shape of affs:', output_affs.shape)
    f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
    f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
    f.close()


    # save
    print('save gt affs...')
    print('the shape of affs:', gt_affs.shape)
    f = h5py.File(os.path.join(out_affs, 'affs_gt.hdf'), 'w')
    f.create_dataset('main', data=gt_affs, dtype=np.float32, compression='gzip')
    f.close()


    print('save gt segmentation')
    f = h5py.File(os.path.join(out_affs, 'seg_gt.hdf'), 'w')
    f.create_dataset('main', data=gt_seg, dtype=gt_seg.dtype, compression='gzip')
    f.close()