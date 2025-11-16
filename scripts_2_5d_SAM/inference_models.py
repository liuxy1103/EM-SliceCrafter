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

from provider_valid_general import Provider_valid
from loss.loss import BCELoss, WeightedBCE, MSELoss
from CoDetectionCNN import CoDetectionCNN
from utils.shift_channels import shift_func
from loss.embedding2affs import embedding_loss
from loss.embedding_norm import embedding_loss_norm, embedding_loss_norm_abs
from loss.embedding_norm import embedding_loss_norm_trunc

import waterz
from utils.fragment import watershed, randomlabel
from data.data_segmentation import seg_widen_border
from utils.lmc import mc_baseline
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_onlylb_suhu_bce_lr0001_snemi3d_data10', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='ac4')
    parser.add_argument('-ts', '--test_split', type=int, default=20)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if cfg.DATA.shift_channels is None:
        cfg.shift = None
    else:
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    cfg.save_path = os.path.join(cfg.TRAIN.save_path, trained_model)
    cfg.record_path = os.path.join(cfg.save_path, trained_model)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if not os.path.exists(cfg.valid_path):
        os.makedirs(cfg.valid_path)

    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)

    seg_name_waterz = 'waterz_' + args.mode + '_' + str(args.test_split)
    seg_name_lmc = 'lmc_' + args.mode + '_' + str(args.test_split)
    seg_txt_waterz = os.path.join(cfg.record_path, seg_name_waterz+'.txt')
    seg_txt_lmc = os.path.join(cfg.record_path, seg_name_lmc+'.txt')
    # obtain the starting id
    if os.path.exists(seg_txt_waterz):
        f_txt = open(seg_txt_waterz, 'r')
        content = [x[:-1] for x in f_txt.readlines()]
        f_txt.close()
        last_line = content[-1]
        c1 = last_line.split(',')[0]
        start_id = int(c1.split('-')[-1])
        start_id += 1000
    else:
        start_id = 1000
    print('start_id=%d' % start_id)

    model_path = os.path.join('../models', trained_model)
    all_files = os.listdir(model_path)
    all_models = []
    for name in all_files:
        if '.ckpt' in name:
            all_models.append(name)
    all_models = sorted(all_models)
    last_model = all_models[-1]
    end_id = int(last_model[6:-5])
    print('end_id=%d' % end_id)
    stride = 1000

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    else:
        raise AttributeError("NO this criterion")

    f_valid_waterz = open(seg_txt_waterz, 'a')
    f_valid_lmc = open(seg_txt_lmc, 'a')
    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)

    for model_id in range(start_id, end_id+stride, stride):
        iters = model_id
        ckpt_path = os.path.join(model_path, 'model-%06d.ckpt' % model_id)
        checkpoint = torch.load(ckpt_path)

        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            # name = k[7:] # remove module.
            name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()

        val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                shuffle=False, drop_last=False, pin_memory=True)
        losses_valid = []
        output_affs = []
        for k, data in enumerate(val_loader, 0):
            inputs, target, weightmap = data
            inference_size = inputs.shape[-1]
            inputs = inputs.cuda()
            target = target.cuda()
            weightmap = weightmap.cuda()
            with torch.no_grad():
                embedding1, embedding2 = model(inputs)
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
        epoch_loss = sum(losses_valid) / len(losses_valid)
        output_affs = np.asarray(output_affs, dtype=np.float32)
        output_affs = np.transpose(output_affs, (1, 0, 2, 3))
        gt_affs = valid_provider.get_gt_affs()
        gt_seg = valid_provider.get_gt_lb()

        output_affs = output_affs[:3]

        fragments = watershed(output_affs, 'maxima_distance')
        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
        segmentation = list(waterz.agglomerate(output_affs, [0.50],
                                            fragments=fragments,
                                            scoring_function=sf,
                                            discretize_queue=256))[0]
        arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        print('model-%d, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (iters, voi_split, voi_merge, voi_sum, arand))
        f_valid_waterz.write('model-%d, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (iters, voi_split, voi_merge, voi_sum, arand))
        f_valid_waterz.write('\n')

        segmentation = mc_baseline(output_affs)
        arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        print('model-%d, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (iters, voi_split, voi_merge, voi_sum, arand))
        f_valid_lmc.write('model-%d, voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
            (iters, voi_split, voi_merge, voi_sum, arand))
        f_valid_lmc.write('\n')
    f_valid_waterz.close()
    f_valid_lmc.close()
    print('Done')
