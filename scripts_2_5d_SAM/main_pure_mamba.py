from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from data_provider_labeled_3d import Provider
from data_provider_labeled import Provider
# from provider_valid import Provider_valid
# from provider_valid_3d import Provider_valid
from provider_valid2 import Provider_valid
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss
from utils.show import show_affs, show_affs_whole
from unet3d_mala import UNet3D_MALA
from model_superhuman2 import UNet_PNI
from utils.utils import setup_seed, execute
from utils.shift_channels import shift_func
from segmamba import SegMamba
from model_unetr import UNETR

import waterz
from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid_provider = Provider_valid(cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid,
                            init_mode=cfg.MODEL.init_mode_mala).to(device)
    elif cfg.MODEL.model_type == 'segmamba':
        print("load segmamba model!")
        model = SegMamba(in_chans=1,
                         out_chans=3, 
                        #  kernel_size=(1,3,3),
                        # args=args
                         ).to(device)
    #     # args.crop_size = cfg.MODEL.crop_size
    elif cfg.MODEL.model_type == 'unetr':
        print("load unetr model!")
        model = UNETR(
                in_channels=cfg.MODEL.input_nc,
                out_channels=cfg.MODEL.output_nc,
                img_size=cfg.MODEL.unetr_size,
                patch_size=cfg.MODEL.patch_size,
                feature_size=16,
                hidden_size=768,
                mlp_dim=2048,
                num_heads=8,
                pos_embed='perceptron',
                norm_name='instance',
                conv_block=True,
                res_block=True,
                kernel_size=cfg.MODEL.kernel_size,
                skip_connection=False,
                show_feature=False,
                dropout_rate=0.1).to(device)
    #     # args.crop_size = cfg.MODEL.crop_size
    else:
        print('load superhuman model!')
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                         out_planes=cfg.MODEL.output_nc,
                         filters=cfg.MODEL.filters,
                         upsample_mode=cfg.MODEL.upsample_mode,
                         decode_ratio=cfg.MODEL.decode_ratio,
                         merge_mode=cfg.MODEL.merge_mode,
                         pad_mode=cfg.MODEL.pad_mode,
                         bn_mode=cfg.MODEL.bn_mode,
                         relu_mode=cfg.MODEL.relu_mode,
                         init_mode=cfg.MODEL.init_mode).to(device)

    if cfg.MODEL.pre_train:
        print('Load pre-trained model ...')
        ckpt_path = os.path.join('../models', \
                                 cfg.MODEL.trained_model_name, \
                                 'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']
        if cfg.MODEL.trained_gpus > 1:
            pretained_model_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]  # remove module.
                # name = k
                pretained_model_dict[name] = v
        else:
            pretained_model_dict = pretrained_dict

        from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        encoder_dict = OrderedDict()
        if cfg.MODEL.if_skip == 'True':
            print('Load the parameters of encoder and decoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
        else:
            print('Load the parameters of encoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    cuda_count = torch.cuda.device_count()
    # cuda_count = 2
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')

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

    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        inputs, target, weightmap = train_provider.next()

        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()
        pred = model(inputs)

        ##############################
        # LOSS
        loss = criterion(pred, target, weightmap)
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                             % (iters, sum_loss * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                             % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, inputs, pred[:, :3], target[:, :3], cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                       shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                
                num_volumes = len(valid_provider.dataset)
                volume_size = valid_provider.num_per_dataset
                all_metrics = []

                # Initialize progress bar
                pbar = tqdm(total=len(valid_provider))
                
                # Process each batch
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()

                    volume_idx = k // volume_size
                    if k % volume_size == 0:
                        valid_provider.reset_output()

                    with torch.no_grad():
                        pred = model(inputs)
                    
                    tmp_loss = criterion(pred, target, weightmap)
                    losses_valid.append(tmp_loss.item())
                    valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))

                    # If we've completed a volume, process it
                    if (k + 1) % volume_size == 0:
                        output_affs = valid_provider.get_results()
                        gt_affs = valid_provider.get_gt_affs(volume_idx)
                        gt_seg = valid_provider.get_gt_lb(volume_idx)
                        gt_seg = gt_seg.astype(np.uint32)

                        vol_losses = losses_valid[volume_idx * volume_size : (volume_idx + 1) * volume_size]
                        avg_volume_loss = sum(vol_losses) / len(vol_losses)
                        
                        vol_metrics = {
                            'loss': avg_volume_loss,
                            'mse': np.mean(np.square(output_affs - gt_affs)),
                            'bce': 0.0,
                            'f1': 0.0,
                            'voi': 0.0,
                            'arand': 0.0
                        }

                        # Calculate BCE
                        output_affs_clipped = np.clip(output_affs, 0.000001, 0.999999)
                        bce = -(gt_affs * np.log(output_affs_clipped) + (1 - gt_affs) * np.log(1 - output_affs_clipped))
                        vol_metrics['bce'] = np.mean(bce)

                        # Calculate F1 Score
                        output_affs_binary = output_affs.copy()
                        output_affs_binary[output_affs_binary <= 0.5] = 0
                        output_affs_binary[output_affs_binary > 0.5] = 1
                        gt_flat = (1 - gt_affs.astype(np.uint8)).flatten()
                        pred_flat = (1 - output_affs_binary.astype(np.uint8)).flatten()
                        vol_metrics['f1'] = f1_score(gt_flat, pred_flat)

                        # Save visualization for this volume
                        save_path_vol = os.path.join(cfg.valid_path, f'volume_{volume_idx}')
                        os.makedirs(save_path_vol, exist_ok=True)
                        show_affs_whole(iters, output_affs[:3], gt_affs, save_path_vol)

                        if cfg.TRAIN.if_seg and iters > 1:
                            # Take only first 3 channels for segmentation
                            seg_output_affs = output_affs[:3]
                            fragments = watershed(seg_output_affs, 'maxima_distance')
                            sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                            segmentation = list(waterz.agglomerate(seg_output_affs, [0.50],
                                                               fragments=fragments,
                                                               scoring_function=sf,
                                                               discretize_queue=256))[0]
                            
                            # Calculate segmentation metrics
                            arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
                            voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
                            voi_sum = voi_split + voi_merge
                            
                            vol_metrics['voi'] = voi_sum
                            vol_metrics['arand'] = arand

                        all_metrics.append(vol_metrics)
                        
                        print(
                            'model-%d, volume-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                            (iters, volume_idx, vol_metrics['loss'], vol_metrics['mse'], vol_metrics['bce'], 
                             vol_metrics['f1'], vol_metrics['voi'], vol_metrics['arand']), flush=True)
                    
                    pbar.update(1)
                pbar.close()

                # Calculate average metrics across all volumes
                avg_metrics = {
                    k: np.mean([m[k] for m in all_metrics]) 
                    for k in ['loss', 'mse', 'bce', 'f1', 'voi', 'arand']
                }

                # Log average metrics
                writer.add_scalar('valid/epoch_loss', avg_metrics['loss'], iters)
                writer.add_scalar('valid/mse_loss', avg_metrics['mse'], iters)
                writer.add_scalar('valid/bce_loss', avg_metrics['bce'], iters)
                writer.add_scalar('valid/f1_score', avg_metrics['f1'], iters)
                writer.add_scalar('valid/voi_sum', avg_metrics['voi'], iters)
                writer.add_scalar('valid/arand', avg_metrics['arand'], iters)

                # Print average metrics
                print(
                    'model-%d, AVERAGE valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                    (iters, avg_metrics['loss'], avg_metrics['mse'], avg_metrics['bce'], 
                     avg_metrics['f1'], avg_metrics['voi'], avg_metrics['arand']), flush=True)

                f_valid_txt.write(
                    'model-%d, AVERAGE valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                    (iters, avg_metrics['loss'], avg_metrics['mse'], avg_metrics['bce'], 
                     avg_metrics['f1'], avg_metrics['voi'], avg_metrics['arand']))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_3d_ac4_data80', help='path to config file')
    # parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    # parser.add_argument('-p', '--pretrain_path', type=str, default='/h3cstore_ns/EM_pretrain/mamba_pretrain_MAE/0424MALA_mask40/checkpoint-80.pth', help='path to pretraining model')  # 20240513
    parser.add_argument('-p', '--pretrain_path', type=str, default='', help='path to pretraining model')  # 20240513
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('/h3cstore_ns/hyshi/CAD_SAM2/scripts_2_5d_SAM/config/' + cfg_file, 'r') as f:
    # with open('/code/EM_seg/scripts_3d/config/' + cfg_file, 'r') as f:
    # with open('./config/' + cfg_file, 'r') as f:
        # cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(yaml.safe_load(f))
    print(cfg)
    
    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp
    if cfg.DATA.shift_channels is None:
        assert cfg.MODEL.output_nc == 3, "output_nc must be 3"  # output_nc意味着什么
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)


        if args.pretrain_path:
            checkpoint = torch.load(args.pretrain_path, map_location='cpu')
            for k in list(checkpoint['model'].keys()):
                if k.startswith('module.'):
                    checkpoint['model'][k[7:]] = checkpoint['model'].pop(k)
                if k in model.state_dict() and checkpoint['model'][k].shape != model.state_dict()[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint['model'][k]
            model.load_state_dict(checkpoint['model'], strict=False)
            print("Load pre-trained checkpoint from: %s" % args.pretrain_path)


        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                     eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')
