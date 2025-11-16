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
from CoDetectionCNN import CoDetectionCNN
from utils.show import show_embedding

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_onlylb_suhu_bce_lr0001_snemi3d_data10', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default=None)
    parser.add_argument('-id', '--model_id', type=int, default=91000)
    parser.add_argument('-m', '--mode', type=str, default='isbi')
    parser.add_argument('-d', '--direc', type=str, default='x')
    parser.add_argument('-ts', '--test_split', type=int, default=20)
    parser.add_argument('-s', '--save', action='store_false', default=True)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-se', '--show_embedding', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    if cfg.TRAIN.mode == 'x-y':
        out_path = os.path.join('../inference', trained_model, args.mode+'_'+args.direc)
    else:
        out_path = os.path.join('../inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if args.show_embedding:
        embedding_img_path = os.path.join(out_affs, 'embedding_img')
        if not os.path.exists(embedding_img_path):
            os.makedirs(embedding_img_path)

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

    valid_provider = Provider_valid(cfg, direc=args.direc)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)

    criterion = nn.BCELoss()

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    pbar = tqdm(total=len(valid_provider))
    output_affs = []
    for k, data in enumerate(val_loader, 0):
        inputs, target, _ = data
        inputs = inputs.cuda()
        target = target.cuda()
        with torch.no_grad():
            embedding1, embedding2 = model(inputs)
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)
        if args.show_embedding:
            show_embedding(embedding1, embedding2, inputs, k, embedding_img_path)
        if cfg.TRAIN.mode == 'x-y-z':
            raise NotImplementedError
        elif cfg.TRAIN.mode == 'z' or cfg.TRAIN.mode == 'x-y':
            affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
            affs0 = (affs0 + 1) / 2
            tmp_loss = criterion(affs0, target)
            pred = affs0
        else:
            raise NotImplementedError
        losses_valid.append(tmp_loss.item())
        output_affs.append(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = np.asarray(output_affs, dtype=np.float32)
    gt_affs = valid_provider.get_gt_affs().copy()

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

    # compute MSE
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
    f_txt.close()

    # show
    if args.show:
        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        for i in range(output_affs_prop.shape[0]):
            img = output_affs_prop[i]
            lb = gt_affs[i]
            im_cat = np.concatenate([img, lb], axis=1)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
    print('Done')

