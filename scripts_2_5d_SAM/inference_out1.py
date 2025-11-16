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

from provider_valid import Provider_valid
from CoDetectionCNN_out1 import CoDetectionCNN
from utils.show import draw_fragments_3d
from utils.fragment import watershed, elf_watershed

import waterz
from utils.lmc import mc_baseline
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_onlylb_suhu_bce_lr0001_snemi3d_data10', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default=None)
    parser.add_argument('-id', '--model_id', type=int, default=91000)
    parser.add_argument('-m', '--mode', type=str, default='isbi')
    parser.add_argument('-ts', '--test_split', type=int, default=20)
    parser.add_argument('-sm', '--seg_mode', type=str, default='waterz')
    parser.add_argument('-f', '--fragment', type=str, default=None)  # 'elf', 'mahotas'
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-mk', '--mask_fragment', type=float, default=None)
    parser.add_argument('-s', '--save', action='store_false', default=True)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)

    ckpt_path = os.path.join('../models', trained_model, 'model-%06d.ckpt' % args.model_id)
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)

    criterion = nn.BCELoss()

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    output_affs = []
    shift = cfg.DATA.shift
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, _ = data
        inference_size = inputs.shape[-1]
        inputs = inputs.cuda()
        target = target.cuda()
        if args.mode == 'fib':
            inputs = F.pad(inputs, (12,12,12,12))
        with torch.no_grad():
            pred = model(inputs)
        if args.mode == 'fib':
            pred = F.pad(pred, (-12,-12,-12,-12))
        tmp_loss = criterion(pred, target)
        losses_valid.append(tmp_loss.item())
        output_affs.append(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = np.asarray(output_affs, dtype=np.float32)
    if cfg.TRAIN.mode == 'x-y-z' or cfg.TRAIN.mode == 'x-y-z-2':
        output_affs = np.transpose(output_affs, (1, 0, 2, 3))
    gt_seg = valid_provider.get_gt_lb()
    gt_affs = valid_provider.get_gt_affs().copy()
    gt_seg = gt_seg.astype(np.uint32)

    # save
    if args.save:
        print('save affs...')
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        f.close()

    if cfg.TRAIN.mode == 'x-y-z':
        raise NotImplementedError
    elif cfg.TRAIN.mode == 'z':
        # gt_affs = gt_affs[0][:-1]
        gt_affs = gt_affs[0]
    elif cfg.TRAIN.mode == 'x-y':
        if args.direc == 'x':
            gt_affs = gt_affs[2]
        else:
            gt_affs = gt_affs[1]

    # output_affs[output_affs < 0.2] = 0.0
    # segmentation
    print('Segmentation...')
    if args.seg_mode == 'waterz':
        print('Waterz...')
        if args.fragment is not None:
            if args.fragment == 'elf':
                print('elf fragments...')
                fragments = elf_watershed(output_affs)
            else:
                print('mahotas fragments...')
                fragments = watershed(output_affs, 'maxima_distance')
            # fragments[gt_seg==0] = 0
            ### mask
            # if args.mask_fragment is not None:
            #     tt = args.mask_fragment
            #     print('add mask and threshold=' + str(tt))
            #     affs_xy = 0.5 * (output_affs[1] + output_affs[2])
            #     fragments[affs_xy<tt] = 0
            segmentation = list(waterz.agglomerate(output_affs, [0.50], gt=gt_seg, fragments=fragments))[0]
            # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
            # segmentation = list(waterz.agglomerate(output_affs, [0.50], gt=gt_seg,
            #                     fragments=fragments,
            #                     scoring_function=sf,
            #                     discretize_queue=256))[0]
        else:
            segmentation = list(waterz.agglomerate(output_affs, [0.50], gt=gt_seg, fragments=None))[0]
        segmentation = segmentation[0].astype(np.int32)
    elif args.seg_mode == 'lmc':
        print('LMC...')
        segmentation = mc_baseline(output_affs)
        segmentation = segmentation.astype(np.int32)
    else:
        raise NotImplementedError
    # segmentation, _, _ = ev.relabel_from_one(segmentation)
    # voi_merge, voi_split = ev.split_vi(segmentation, gt_seg)
    # voi_sum = voi_split + voi_merge
    # arand = ev.adapted_rand_error(segmentation, gt_seg)
    # print('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
    #     (args.model_id, voi_split, voi_merge, voi_sum, arand))
    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
        (args.model_id, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
        (args.model_id, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')
    f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
    f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    f.close()

    # compute MSE
    if args.pixel_metric:
        print('MSE...')
        output_affs_prop = output_affs.copy()
        whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
        print('BCE...')
        output_affs = np.clip(output_affs, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
        whole_bce = np.sum(bce) / np.size(gt_affs)
        output_affs[output_affs <= 0.5] = 0
        output_affs[output_affs > 0.5] = 1
        print('F1...')
        whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
        # new
        print('F1 boundary...')
        whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
        print('mAP...')
        # whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        whole_map = 0.0
        print('AUC...')
        # whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        whole_auc = 0.0
        malis = 0.0
        print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
            (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
                    (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('\n')
    else:
        output_affs_prop = output_affs
    f_txt.close()

    # show
    if args.show:
        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        for i in range(output_affs_prop.shape[1]):
            cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
            cat2 = np.concatenate([gt_affs[0,i], gt_affs[1,i], gt_affs[2,i]], axis=1)
            im_cat = np.concatenate([cat1, cat2], axis=0)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
        
        print('show seg...')
        # segmentation[gt_seg==0] = 0
        color_seg = draw_fragments_3d(segmentation)
        color_gt = draw_fragments_3d(gt_seg)
        for i in range(color_seg.shape[0]):
            im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
            cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    print('Done')

