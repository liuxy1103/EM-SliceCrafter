'''
seperate model_3d and other with differnet optimizer
'''
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
from utils.show import show_affs2, show_affs_whole2
from utils.show_3d import show_affs2 as show_affs2_3d
from unet3d_mala import UNet3D_MALA_embedding as UNet3D_MALA
from model_superhuman2 import UNet_PNI_embedding as UNet_PNI
from data_provider_general import Provider
from provider_valid_general import Provider_valid
from data_provider_labeled_3d import Provider as Provider_3d
from provider_valid_3d import Provider_valid as Provider_valid_3d
from loss.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5,embedding_loss_norm_multi,embedding_loss_norm5_intera
from utils.show import show_affs2, show_affs_whole2
from CoDetectionCNN import CoDetectionCNN, CoDetectionCNN_Add_SAM2
from utils.utils import setup_seed, execute
from loss.loss import WeightedMSE, WeightedBCE
from loss.loss import MSELoss, BCELoss
from loss.embedding2affs import embedding_loss
from loss.embedding_norm import embedding_loss_norm, embedding_loss_norm_abs
from loss.embedding_norm import embedding_loss_norm_trunc
from utils.shift_channels import shift_func

import waterz
# from utils.lmc import mc_baseline
from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref

from sam2.build_sam import build_sam2

from model_channelShift import StepByStepUpscaler4

import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

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

def load_dataset_3d(cfg):
    print('Caching 3d datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider_3d('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid_3d(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg, valid_data=cfg.DATA.valid_dataset, test_split=cfg.DATA.test_split)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def init_weights(model, init_method='xavier'):
    if init_method == 'zero':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
    elif init_method == 'random':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    elif init_method == 'xavier':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    elif init_method == 'he':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    else:
        raise ValueError("Unsupported initialization method")

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)

    cuda_count = torch.cuda.device_count()
    # cuda_count = 1
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def build_model_3d(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc,
                            if_sigmoid=cfg.MODEL.if_sigmoid,
                            init_mode=cfg.MODEL.init_mode_mala,
                            emd=cfg.MODEL.emd).to(device)
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
                        init_mode=cfg.MODEL.init_mode,
                        emd=cfg.MODEL.emd).to(device)

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
                name = k[7:] # remove module.
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
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model



def build_model_add_SAM2(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = CoDetectionCNN_Add_SAM2(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)

    cuda_count = torch.cuda.device_count()
    # cuda_count = 1
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def build_model_SAM2(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    sam2_checkpoint = "/h3cstore_ns/hyshi/scripts_2_5d/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    sam2_image_encoder = sam2_model.image_encoder
    cuda_count = torch.cuda.device_count()
    # cuda_count = 1
    model = sam2_image_encoder
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(sam2_image_encoder)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def build_model_shift(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    shift_model = StepByStepUpscaler4().to(device=device)
    # cuda_count = torch.cuda.device_count()
    cuda_count = 1
    model = shift_model
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(shift_model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
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
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, train_provider_3d, valid_provider_3d, model1, model2, shift_model, model_3d, criterion, optimizer, iters, writer, args):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    sum_aff_loss = 0 #2D
    sum_ctc_loss = 0 # 2D and SAM
    sum_loss_all = 0
    sum_loss_3d = 0
    sum_loss_2d_slice = 0
    sum_loss_cross = 0
    sum_loss_interaction = 0
    sum_loss_2d_3d = 0
    sum_loss_2d_SAM = 0
    device = torch.device('cuda:0')
    size = list(cfg.DATA.patch_size)[0]
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
        model1.train() # 2D network
        # model2.train() # SAM
        shift_model.train() # adapter
        model_3d.train() # 3D network
        iters += 1
        t1 = time.time()
        inputs, lb, target, weightmap = train_provider.next()
        inputs_3d, lb_3d, target_3d, weightmap_3d = train_provider_3d.next()
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        embedding1_1, embedding1_2 = model1(inputs)
        # embedding2_1, embedding2_2 = model2(inputs)
        x_in1 = inputs[:, 0:1, :, :].repeat(1,3,1,1)
        x_in2 = inputs[:, 1::, :, :].repeat(1,3,1,1)
        with torch.no_grad():
            embedding2_1 = model2(x_in1)['vision_features']
            embedding2_2 = model2(x_in2)['vision_features']
        embedding2_1, embedding2_2 = shift_model(embedding2_1,embedding2_2)
        
        if cfg.TRAIN.if_2d_3d:
        
            # 2d & 3d
            embedding_3d = model_3d(inputs_3d)
            if cfg_3d.TRAIN.embedding_mode == 1:
                loss_3d, pred_3d = embedding_loss_norm1(embedding_3d, target_3d, weightmap_3d, criterion, affs0_weight=cfg_3d.TRAIN.affs0_weight)
            elif cfg_3d.TRAIN.embedding_mode == 5:
                loss_3d, pred_3d = embedding_loss_norm5(embedding_3d, target_3d, weightmap_3d, criterion, affs0_weight=cfg_3d.TRAIN.affs0_weight)
            else:
                raise NotImplementedError
            shift = 1
            pred_3d[:, 1, :, :shift, :] = pred_3d[:, 1, :, shift:shift*2, :]
            pred_3d[:, 2, :, :, :shift] = pred_3d[:, 2, :, :, shift:shift*2]
            pred_3d[:, 0, :shift, :, :] = pred_3d[:, 0, shift:shift*2, :, :]
            pred_3d = F.relu(pred_3d[:, :3])
            loss_cross = 0
            loss_2d_slice = 0
            loss_interaction = torch.tensor(0).cuda().float()
            for z in range(inputs_3d.shape[2]-1):
                inputs_tmp = inputs_3d[:,0,z:z+2]
                target_tmp = target_3d[:,:3,z+1]
                embedding1_tmp, embedding2_tmp = model1(inputs_tmp)
                weightmap_tmp = weightmap_3d[:,:3,z+1]
                size_tmp = weightmap_tmp.shape[-1]
                
                if cfg.TRAIN.loss_mode == 'nn_cos':
                    loss_tmp, pred_tmp = embedding_loss(embedding1_tmp, embedding2_tmp, target_tmp, weightmap_tmp, criterion, shift=cfg.shift)
                elif cfg.TRAIN.loss_mode == 'norm':
                    loss_tmp, pred_tmp = embedding_loss_norm(embedding1_tmp, embedding2_tmp, target_tmp, weightmap_tmp, criterion,
                                                    affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size_tmp)
                elif cfg.TRAIN.loss_mode == 'abs':
                    loss_tmp, pred_tmp = embedding_loss_norm_abs(embedding1_tmp, embedding2_tmp, target_tmp, weightmap_tmp, criterion,
                                                    affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size_tmp)
                elif cfg.TRAIN.loss_mode == 'trunc':
                    loss_tmp, pred_tmp = embedding_loss_norm_trunc(embedding1_tmp, embedding2_tmp, target_tmp, weightmap_tmp, criterion,
                                                    affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size_tmp)
                else:
                    raise NotImplementedError
                if iters < cfg_3d.TRAIN.start_ft:
                    if cfg.TRAIN.only_z:
                        loss_cross_tmp  = criterion(pred_tmp[:,0:1], pred_3d[:,0:1,z+1].detach(), weightmap_tmp[:,0:1])
                    else:
                        loss_cross_tmp  = criterion(pred_tmp[:,:,shift:,:], pred_3d[:,:3,z+1,shift:,:].detach(), weightmap_tmp[:,:,shift:,:])
                else:
                    if cfg.TRAIN.only_z:
                        loss_cross_tmp  = criterion(pred_tmp[:,0:1], pred_3d[:,0:1,z+1], weightmap_tmp[:,0:1])
                    else:
                        loss_cross_tmp  = criterion(pred_tmp[:,:,:,shift:], pred_3d[:,:3,z+1,:,shift:], weightmap_tmp[:,:,:,shift:])
                
                if cfg.TRAIN.interaction:
                    weightmap_3d_tmp = weightmap_3d
                    if iters < cfg_3d.TRAIN.start_ft:
                        embedding_3d_tmp = embedding_3d.detach()
                    else:
                        embedding_3d_tmp = embedding_3d

                    shifts = [1, 1, 1, 2, 3, 3, 3, 9, 9, 4, 27, 27]
                    loss_3d_tmp = torch.tensor(0).cuda().float()
                    for order, shift in enumerate(shifts):
                        if (not order%3==0) or z+shift>16:
                            continue
                        if shift ==1:
                            affs0 = torch.sum(embedding2_tmp*embedding_3d_tmp[:,:,z+1+shift], dim=1, keepdim=True)
                            affs0 = torch.clamp(affs0, 0.0, 1.0)
                            loss_3d_tmp = criterion(affs0, target_3d[:, order:order+1, z+2, :, :], weightmap_3d[:, order:order+1, z+2, :, :])
                        else:
                            affs0_1 = torch.sum(embedding1_tmp*embedding_3d_tmp[:,:,z+shift], dim=1, keepdim=True)
                            affs0_1 = torch.clamp(affs0_1, 0.0, 1.0)
                            affs0_2 = torch.sum(embedding2_tmp*embedding_3d_tmp[:,:,z+1+shift], dim=1, keepdim=True)
                            affs0_2 = torch.clamp(affs0_2, 0.0, 1.0)
                            loss_temp1 = criterion(affs0_1, target_3d[:, order:order+1, z+1, :, :], weightmap_3d[:, order:order+1, z+1, :, :])
                            loss_temp2 = criterion(affs0_2, target_3d[:, order:order+1, z+2, :, :], weightmap_3d[:, order:order+1, z+2, :, :])
                            loss_3d_tmp = loss_temp1 + loss_temp2
                    # loss_3d_tmp.backward()
                    loss_interaction += loss_3d_tmp
                loss_2d_slice += loss_tmp
                loss_cross += loss_cross_tmp
                
            loss_2d_slice = cfg.TRAIN.loss_2d_slice_weight * loss_2d_slice
            loss_cross = cfg.TRAIN.loss_cross_weight * loss_cross
            loss_interaction = cfg.TRAIN.loss_interaction * loss_interaction
            if iters < cfg_3d.TRAIN.start_ft:
                loss_2d_3d =  loss_3d  + loss_2d_slice 
            else:
                loss_2d_3d = loss_3d + loss_cross + loss_2d_slice + loss_interaction
        else:
            loss_2d_3d = torch.tensor(0).cuda()
            loss_3d = torch.tensor(0).cuda()
            loss_2d_slice = torch.tensor(0).cuda()
            loss_interaction = torch.tensor(0).cuda()
            loss_cross = torch.tensor(0).cuda()
        #  2d & SAM
        if cfg.TRAIN.loss_mode == 'nn_cos':
            loss1, pred1 = embedding_loss(embedding1_1, embedding1_2, target, weightmap, criterion, shift=cfg.shift)
            loss2, pred2 = embedding_loss(embedding2_1, embedding2_2, target, weightmap, criterion, shift=cfg.shift)
            loss_interaction1, _ = embedding_loss(embedding1_1, embedding2_2, target, weightmap, criterion, shift=cfg.shift)
            loss_interaction2, _ = embedding_loss(embedding2_1, embedding1_2, target, weightmap, criterion, shift=cfg.shift)
        elif cfg.TRAIN.loss_mode == 'norm':
            loss1, pred1 = embedding_loss_norm(embedding1_1, embedding1_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss2, pred2 = embedding_loss_norm(embedding2_1, embedding2_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss_interaction1, _ = embedding_loss_norm(embedding1_1, embedding2_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss_interaction2, _ = embedding_loss_norm(embedding2_1, embedding1_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
        elif cfg.TRAIN.loss_mode == 'abs':
            loss1, pred1 = embedding_loss_norm_abs(embedding1_1, embedding1_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss2, pred2 = embedding_loss_norm_abs(embedding2_1, embedding2_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss_interaction1, _ = embedding_loss_norm_abs(embedding1_1, embedding2_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss_interaction2, _ = embedding_loss_norm_abs(embedding2_1, embedding1_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
        elif cfg.TRAIN.loss_mode == 'trunc':
            loss1, pred1 = embedding_loss_norm_trunc(embedding1_1, embedding1_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss2, pred2 = embedding_loss_norm_trunc(embedding2_1, embedding2_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss_interaction1, _ = embedding_loss_norm_trunc(embedding1_1, embedding2_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
            loss_interaction2, _ = embedding_loss_norm_trunc(embedding2_1, embedding1_2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=size)
        else:
            raise NotImplementedError
        if cfg.TRAIN.if_2d_SAM_CPS:
            pred2_thres = torch.zeros_like(pred2)
            pred2_thres[pred2>0.5] = 1
            pred1_thres = torch.zeros_like(pred1)
            pred1_thres[pred1>0.5] = 1
        else:
            pred2_thres = pred2
            pred1_thres = pred1
        loss_emb_consistency1 = cfg.TRAIN.ct_weight * criterion(pred1, pred2_thres, weightmap).to(device)
        loss_emb_consistency2 = cfg.TRAIN.ct_weight * criterion(pred2, pred1_thres, weightmap).to(device)

        ##############################
        # LOSS
        loss_aff_consistency = loss_emb_consistency1 + loss_emb_consistency2
        if cfg.TRAIN.loss_aff1 is None:
            loss_aff = loss1 + loss2
        else:
            loss_aff = loss1*cfg.TRAIN.loss_aff1 + loss2
        loss_interactio_SAM = loss_interaction1 + loss_interaction2
        loss_interactio_SAM = cfg.TRAIN.loss_interaction_2d * loss_interactio_SAM
        loss_2d_SAM = cfg.TRAIN.loss_weight_2d_SAM*loss_aff_consistency + loss_aff + loss_interactio_SAM
        
        loss_2d_3d = cfg.TRAIN.loss_weight_2d_3d*loss_2d_3d
        # loss_2d_SAM = loss_2d_SAM
        
        loss = loss_2d_3d + loss_2d_SAM
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        #2d vs SAM
        sum_aff_loss += loss_aff.item()
        sum_ctc_loss += loss_aff_consistency.item()
        sum_loss_2d_SAM += loss_2d_SAM.item()
        
        #2d vs 3d
        sum_loss_3d += loss_3d.item()
        sum_loss_2d_slice += loss_2d_slice.item()
        sum_loss_cross += loss_cross.item() 
        sum_loss_interaction += loss_interaction.item()
        sum_loss_2d_3d += loss_2d_3d.item()
        
        
        sum_loss += loss.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f, loss_2d_SAM=%.6f, sum_loss_2d_3d=%.6f, loss_aff = %.6f, loss_ctc = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss * 1, sum_loss_2d_SAM, sum_loss_2d_3d, sum_aff_loss,sum_ctc_loss, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f, loss_2d_SAM=%.6f, sum_loss_2d_3d=%.6f, loss_aff = %.6f, loss_ctc = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, sum_loss_2d_SAM / cfg.TRAIN.display_freq * 1,
                               sum_loss_2d_3d / cfg.TRAIN.display_freq * 1,
                               sum_aff_loss / cfg.TRAIN.display_freq * 1,
                            sum_ctc_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
            sum_loss_3d = 0
            sum_loss_2d_slice = 0
            sum_loss_cross = 0
            sum_loss_interaction = 0
            sum_aff_loss = 0
            sum_ctc_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs2(iters, inputs, pred1[:,:3], target[:,:3], cfg.cache_path)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model1.eval()
                model2.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []
                out_affs = []
                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inference_size = inputs.shape[-1]
                    inputs = inputs.cuda()
                    target = target.cuda()
                    weightmap = weightmap.cuda()

                    with torch.no_grad():
                        if inputs.shape[-1] == 1250:
                            inputs = F.pad(inputs, (7, 7, 7, 7), mode='reflect')
                            embedding1, embedding2 = model1(inputs)
                            embedding1 = embedding1[:,:,7:-7,7:-7]
                            embedding2 = embedding2[:,:,7:-7,7:-7]
                        elif inputs.shape[-1] == 150:
                            inputs = F.pad(inputs, (5, 5, 5, 5), mode='reflect')
                            embedding1, embedding2 = model1(inputs)
                            embedding1 = embedding1[:,:,5:-5,5:-5]
                            embedding2 = embedding2[:,:,5:-5,5:-5]
                        else:
                            embedding1, embedding2 = model1(inputs)

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
                    out_affs.append(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = np.asarray(out_affs, dtype=np.float32)
                out_affs = np.transpose(out_affs, (1,0,2,3))
                out_affs = out_affs[:3]
                gt_seg = valid_provider.get_gt_lb()
                gt_affs = valid_provider.get_gt_affs().copy()
                show_affs_whole2(iters, out_affs, gt_affs, cfg.valid_path, cfg.TRAIN.mode)

                if cfg.TRAIN.if_seg:
                    ##############
                    # segmentation
                    if iters > 5000:
                        fragments = watershed(out_affs, 'maxima_distance')
                        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                        seg_waterz = list(waterz.agglomerate(out_affs, [0.50],
                                                            fragments=fragments,
                                                            scoring_function=sf,
                                                            discretize_queue=256))[0]
                        arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
                        voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
                        voi_sum_waterz = voi_split + voi_merge

                    else:
                        voi_sum_waterz = 0.0
                        arand_waterz = 0.0

                        print('model-%d, segmentation failed!' % iters)
                    ##############
                else:
                    voi_sum_waterz = 0.0
                    arand_waterz = 0.0


                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz), flush=True)

                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_waterz', voi_sum_waterz, iters)
                writer.add_scalar('valid/arand_waterz', arand_waterz, iters)

                f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f' % \
                                (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model1.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            
            # states2 = {'current_iter': iters, 'valid_result': None,
            #         'model_weights': model2.state_dict()}
            # torch.save(states2, os.path.join(cfg.save_path, 'model2-%06d.ckpt' % iters))

            states2 = {'current_iter': iters, 'valid_result': None,
                    'model_weights': shift_model.state_dict()}
            torch.save(states2, os.path.join(cfg.save_path, 'shift_model-%06d.ckpt' % iters))
            
            states2 = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model_3d.state_dict()}
            torch.save(states2, os.path.join(cfg.save_path, 'model_3d-%06d.ckpt' % iters))
            
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-c3', '--cfg_3d', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    parser.add_argument('-d', '--dual', type=str, default='true', help='training mode, true for dual')
    parser.add_argument('-s', '--sam', type=str, default='false', help='use of SAM2 image encoder')


    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)
    cfg_3d_file = args.cfg_3d + '.yaml'
    print('cfg_3d_file: ' + cfg_3d_file)
    
    # with open('/h3cstore_ns/hyshi/scripts_2_5d/config/' + cfg_file, 'r') as f:
    #     # cfg = AttrDict(yaml.load(f))
    #     cfg = AttrDict(yaml.safe_load(f))
    # with open('./config/' + cfg_file, 'r') as f:
    #     cfg = AttrDict(yaml.safe_load(f))
    # with open('./config/' + cfg_3d_file, 'r') as f:
    #     cfg_3d = AttrDict(yaml.safe_load(f))

    with open('/h3cstore_ns/hyshi/CAD_SAM2/scripts_2_5d_SAM/config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    with open('/h3cstore_ns/hyshi/CAD_SAM2/scripts_2_5d_SAM/config/' + cfg_3d_file, 'r') as f:
        cfg_3d = AttrDict(yaml.safe_load(f))


    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp
    if cfg.DATA.shift_channels is None:
        cfg.shift = None
    else:
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider_3d, valid_provider_3d = load_dataset_3d(cfg_3d)
        train_provider, valid_provider = load_dataset(cfg)
        # model1 = build_model(cfg, writer)
        # init_weights(model1,'xavier')
        # logging.info(args.dual)
        # writer.add_text('dual', str(args.dual))
        # logging.info(args.sam)
        # writer.add_text('sam', str(args.sam))

        logging.info('task description: distill SAM2 to 2D (original channel) with shift_model')

        model1 = build_model(cfg, writer)
        init_weights(model1,'xavier')
        model2 = build_model_SAM2(cfg, writer)
        model_3d = build_model_3d(cfg_3d, writer) 
        shift_model = build_model_shift(cfg, writer)
        init_weights(shift_model,'random')
        optimizer = torch.optim.Adam([{'params': model1.parameters()},
                                {'params': shift_model.parameters()},
                                {'params': model_3d.parameters()},
                                ], lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                            eps=0.01, weight_decay=1e-6, amsgrad=True)
        model1, optimizer, init_iters = resume_params(cfg, model1, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, train_provider_3d, valid_provider_3d, model1, model2, shift_model, model_3d, nn.L1Loss(), optimizer, init_iters, writer, args)
        writer.close()
    else:
        pass
    print('***Done***')