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

from data_provider import Provider
from provider_valid import Provider_valid
from utils.show import show_affs2, show_affs_whole2
from CoDetectionCNN import CoDetectionCNN
from utils.utils import setup_seed, execute
from loss.loss import WeightedMSE, WeightedBCE

import waterz
from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
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

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)

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


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')
    shift = cfg.DATA.shift
    size = list(cfg.DATA.patch_size)[0]
    if cfg.TRAIN.loss_func == 'BCELoss':
        criterion = nn.BCELoss()
    elif cfg.TRAIN.loss_func == 'MSELoss':
        criterion = nn.MSELoss()
    # elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
    #     criterion = WeightedBCE()
    # elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
    #     criterion = WeightedMSE()
    else:
        raise NotImplementedError

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
        embedding1, embedding2 = model(inputs)
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)

        if cfg.TRAIN.mode == 'x-y-z':
            affs1_2 = torch.sum(embedding1[:, :, :, shift:]*embedding1[:, :, :, :size-shift], dim=1, keepdim=True)
            affs1_2 = (affs1_2 + 1) / 2
            loss1 = criterion(affs1_2, target[:, 0:1][:, :, :, shift:])
            affs1_2 = F.pad(affs1_2, (1,0,0,0), mode='reflect')

            affs1_1 = torch.sum(embedding1[:, :, shift:, :]*embedding1[:, :, :size-shift, :], dim=1, keepdim=True)
            affs1_1 = (affs1_1 + 1) / 2
            loss2 = criterion(affs1_1, target[:, 1:2][:, :, shift:, :])
            affs1_1 = F.pad(affs1_1, (0,0,1,0), mode='reflect')

            affs2_2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :size-shift], dim=1, keepdim=True)
            affs2_2 = (affs2_2 + 1) / 2
            loss3 = criterion(affs2_2, target[:, 2:3][:, :, :, shift:])
            affs2_2 = F.pad(affs2_2, (1,0,0,0), mode='reflect')

            affs2_1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :size-shift, :], dim=1, keepdim=True)
            affs2_1 = (affs2_1 + 1) / 2
            loss4 = criterion(affs2_1, target[:, 3:4][:, :, shift:, :])
            affs2_1 = F.pad(affs2_1, (0,0,1,0), mode='reflect')

            affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
            affs0 = (affs0 + 1) / 2
            loss5 = criterion(affs0, target[:, 4:5])
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            pred = torch.cat([affs1_2, affs1_1, affs2_2, affs2_1, affs0], dim=1)
            
        elif cfg.TRAIN.mode == 'x-y-z-2':
            affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
            affs0 = (affs0 + 1) / 2
            # affs0 = torch.abs(affs0)
            affs0[affs0 < 0.0] = 0.0
            affs0[affs0 > 1.0] = 1.0
            # if 'Weight' in cfg.TRAIN.loss_func:
            #     loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])
            # else:
            #     loss0 = criterion(affs0, target[:, 0:1])
            loss0 = criterion(affs0, target[:, 0:1])

            affs1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :size-shift, :], dim=1, keepdim=True)
            affs1 = (affs1 + 1) / 2
            # affs1 = torch.abs(affs1)
            affs1[affs1 < 0.0] = 0.0
            affs1[affs1 > 1.0] = 1.0
            # if 'Weight' in cfg.TRAIN.loss_func:
            #     loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
            # else:
            #     loss1 = criterion(affs1, target[:, 1:2, shift:, :])
            loss1 = criterion(affs1, target[:, 1:2, shift:, :])
            affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')

            affs2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :size-shift], dim=1, keepdim=True)
            affs2 = (affs2 + 1) / 2
            # affs2 = torch.abs(affs2)
            affs2[affs2 < 0.0] = 0.0
            affs2[affs2 > 1.0] = 1.0
            # if 'Weight' in cfg.TRAIN.loss_func:
            #     loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
            # else:
            #     loss2 = criterion(affs2, target[:, 2:3, :, shift:])
            loss2 = criterion(affs2, target[:, 2:3, :, shift:])
            affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')
            loss = cfg.TRAIN.affs0_weight * loss0 + loss1 + loss2
            pred = torch.cat([affs0, affs1, affs2], dim=1)
        elif cfg.TRAIN.mode == 'z' or cfg.TRAIN.mode == 'x-y':
            affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
            affs0 = (affs0 + 1) / 2
            loss = criterion(affs0, target)
            pred = affs0
        else:
            raise NotImplementedError

        ##############################
        # LOSS
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
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs2(iters, inputs, pred, target, cfg.cache_path)

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
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
                        embedding1, embedding2 = model(inputs)
                    embedding1 = F.normalize(embedding1, p=2, dim=1)
                    embedding2 = F.normalize(embedding2, p=2, dim=1)
                    if cfg.TRAIN.mode == 'x-y-z':
                        raise NotImplementedError
                    elif cfg.TRAIN.mode == 'x-y-z-2':
                        affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
                        affs0 = (affs0 + 1) / 2
                        # affs0 = torch.abs(affs0)
                        affs0[affs0 < 0.0] = 0.0
                        affs0[affs0 > 1.0] = 1.0
                        # if 'Weight' in cfg.TRAIN.loss_func:
                        #     tmp_loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])
                        # else:
                        #     tmp_loss0 = criterion(affs0, target[:, 0:1])
                        tmp_loss0 = criterion(affs0, target[:, 0:1])
                        
                        affs1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :inference_size-shift, :], dim=1, keepdim=True)
                        affs1 = (affs1 + 1) / 2
                        # affs1 = torch.abs(affs1)
                        affs1[affs1 < 0.0] = 0.0
                        affs1[affs1 > 1.0] = 1.0
                        # if 'Weight' in cfg.TRAIN.loss_func:
                        #     tmp_loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
                        # else:
                        #     tmp_loss1 = criterion(affs1, target[:, 1:2, shift:, :])
                        tmp_loss1 = criterion(affs1, target[:, 1:2, shift:, :])
                        affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')
                        # tmp_loss1 = criterion(affs1, target[:, 1:2])

                        affs2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :inference_size-shift], dim=1, keepdim=True)
                        affs2 = (affs2 + 1) / 2
                        # affs2 = torch.abs(affs2)
                        affs2[affs2 < 0.0] = 0.0
                        affs2[affs2 > 1.0] = 1.0
                        # if 'Weight' in cfg.TRAIN.loss_func:
                        #     tmp_loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
                        # else:
                        #     tmp_loss2 = criterion(affs2, target[:, 2:3, :, shift:])
                        tmp_loss2 = criterion(affs2, target[:, 2:3, :, shift:])
                        affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')
                        # tmp_loss2 = criterion(affs2, target[:, 2:3])

                        tmp_loss = tmp_loss0 + tmp_loss1 + tmp_loss2
                        pred = torch.cat([affs0, affs1, affs2], dim=1)
                    elif cfg.TRAIN.mode == 'z' or cfg.TRAIN.mode == 'x-y':
                        affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
                        affs0 = (affs0 + 1) / 2
                        tmp_loss = criterion(affs0, target)
                        pred = affs0
                    else:
                        raise NotImplementedError
                    losses_valid.append(tmp_loss.item())
                    out_affs.append(np.squeeze(pred.data.cpu().numpy()))
                epoch_loss = sum(losses_valid) / len(losses_valid)
                out_affs = np.asarray(out_affs, dtype=np.float32)
                if cfg.TRAIN.mode == 'x-y-z' or cfg.TRAIN.mode == 'x-y-z-2':
                    out_affs = np.transpose(out_affs, (1, 0, 2, 3))
                gt_seg = valid_provider.get_gt_lb()
                gt_affs = valid_provider.get_gt_affs().copy()
                show_affs_whole2(iters, out_affs, gt_affs, cfg.valid_path, cfg.TRAIN.mode)

                if cfg.TRAIN.mode == 'x-y-z':
                    raise NotImplementedError
                elif cfg.TRAIN.mode == 'z':
                    gt_affs = gt_affs[0]
                elif cfg.TRAIN.mode == 'x-y':
                    gt_affs = gt_affs[2]

                if cfg.TRAIN.if_seg:
                    if cfg.TRAIN.mode == 'x-y-z-2':
                        ##############
                        # segmentation
                        # try:
                        if iters > 1:
                            # segmentation = list(waterz.agglomerate(out_affs, [0.50]))[0]
                            fragments = watershed(out_affs, 'maxima_distance')
                            sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                            segmentation = list(waterz.agglomerate(out_affs, [0.50],
                                                                fragments=fragments,
                                                                scoring_function=sf,
                                                                discretize_queue=256))[0]
                            arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
                            voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
                            voi_sum = voi_split + voi_merge
                            # segmentation = segmentation.astype(np.int32)
                            # segmentation, _, _ = ev.relabel_from_one(segmentation)
                            # voi_merge, voi_split = ev.split_vi(segmentation, gt_seg)
                            # voi_sum = voi_split + voi_merge
                            # arand = ev.adapted_rand_error(segmentation, gt_seg)
                        # except:
                        else:
                            voi_sum = 0.0
                            arand = 0.0
                            print('model-%d, segmentation failed!' % iters)
                        ##############
                    else:
                        voi_sum = 0.0
                        arand = 0.0
                else:
                    voi_sum = 0.0
                    arand = 0.0

                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum, arand), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_sum', voi_sum, iters)
                writer.add_scalar('valid/arand', arand, iters)
                f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                                (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum, arand))
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
    parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    # with open('./config/' + cfg_file, 'r') as f:
    with open('/h3cstore_ns/hyshi/scripts_2_5d/config/' + cfg_file, 'r') as f:
        # cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(yaml.safe_load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
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