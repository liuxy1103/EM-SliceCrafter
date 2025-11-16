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

from provider_valid_3d import Provider_valid
from loss.loss import BCELoss, WeightedBCE, MSELoss
from unet3d_mala import UNet3D_MALA
# from model_superhuman import UNet_PNI
# from model_superhuman2 import UNet_PNI_FT2 as UNet_PNI
from model_superhuman2 import UNet_PNI
from unet3d_mala import UNet3D_MALA_embedding as UNet3D_MALA
from model_superhuman2 import UNet_PNI_embedding as UNet_PNI
# from utils.malis_loss import malis_loss
from loss.loss import WeightedMSE, WeightedBCE
from utils.show import draw_fragments_3d
from utils.fragment import watershed, elf_watershed
from utils.shift_channels import shift_func
from utils.fragment import watershed, randomlabel, relabel
import waterz
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from loss.loss_embedding_mse import embedding_loss_norm1, embedding_loss_norm5,embedding_loss_norm_multi

def embedding_pca(embeddings, n_components=3, as_rgb=False):
    if as_rgb and n_components != 3:
        raise ValueError("")
    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')
    return embed_flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 新增循环参数
    parser.add_argument('--start_id', type=int, required=True)
    parser.add_argument('--end_id', type=int, required=True)
    parser.add_argument('--step', type=int, default=1)
    #####
    parser.add_argument('-c', '--cfg', type=str, default='3d_cremiC_data100', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2023-09-29--10-17-52_3d_cremiC_data100')
    parser.add_argument('-emb', '--if_embedding', action='store_true', default=True)
    parser.add_argument('-id', '--model_id', type=int, default=198000)
    parser.add_argument('-m', '--mode', type=str, default='cremiC')
    parser.add_argument('-ts', '--test_split', type=int, default=25)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-lt', '--lmc_thres', type=float, default=0.36)
    parser.add_argument('-sa', '--save_affs', action='store_true', default=True)# 先不存affintiy 占内存太多
    parser.add_argument('-mutex', '--mutex', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split,stage='test')
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    out_path = os.path.join('../inference_3d', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 初始化全局结果文件
    f_txt_all_path = os.path.join(out_path, 'scores_all2.txt')
    f_txt_all = open(f_txt_all_path, 'a')
    f_txt_all.write('ID\tInference Time (s)\tVOI Split\tVOI Merge\tVOI Sum\tARAND\n')
    f_txt_all.flush()


    # 生成ID列表
    ids = range(args.start_id, args.end_id + 1, args.step)
    
    for current_id in ids:

        img_folder = f'affs_{current_id}'
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
        if cfg.MODEL.model_type == 'mala':
            print('load mala model!')
            model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
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

        ckpt_path = os.path.join('../models', trained_model, f'model_3d-{current_id:06d}.ckpt')
        checkpoint = torch.load(ckpt_path)

        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.to(device)



        if cfg.TRAIN.loss_func == 'MSE':
            criterion = MSELoss()
        elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
            criterion = WeightedBCE()
        elif cfg.TRAIN.loss_func == 'BCELoss':
            criterion = BCELoss()
        elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
            criterion = WeightedMSE()
        else:
            raise AttributeError("NO this criterion")

        model.eval()
        loss_all = []
        f_txt = open(os.path.join(out_affs, 'scores2.txt'), 'w')
        print('the number of sub-volume:', len(valid_provider))
        losses_valid = []
        t1 = time.time()
        pbar = tqdm(total=len(valid_provider))
        for k, data in enumerate(val_loader, 0):
            inputs,  target, weightmap = data
            inputs = inputs.cuda()
            target = target.cuda()
            weightmap = weightmap.cuda()
            with torch.no_grad():
                embedding = model(inputs)
                if cfg.TRAIN.embedding_mode == 1:
                    loss, pred = embedding_loss_norm1(embedding, target, weightmap, criterion, affs0_weight=cfg.TRAIN.affs0_weight)
                elif cfg.TRAIN.embedding_mode == 5:
                    loss, pred = embedding_loss_norm5(embedding, target, weightmap, criterion, affs0_weight=cfg.TRAIN.affs0_weight)
                else:
                    raise NotImplementedError

                shift = 1
                pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
                pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
                pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
                pred = F.relu(pred)
            tmp_loss = criterion(pred, target, weightmap)
            losses_valid.append(tmp_loss.item())
            valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
            pbar.update(1)
        pbar.close()
        cost_time = time.time() - t1
        print('Inference time=%.6f' % cost_time)
        f_txt.write('Inference time=%.6f' % cost_time)
        f_txt.write('\n')
        epoch_loss = sum(losses_valid) / len(losses_valid)

        output_affs = valid_provider.get_results()
        gt_affs = valid_provider.get_gt_affs()
        gt_seg = valid_provider.get_gt_lb()
        raw_data = valid_provider.get_raw_data()
        valid_provider.reset_output()
        gt_seg = gt_seg.astype(np.uint32)
        # save
        if args.save_affs:
            print('save affs...')
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
    #     print('affinity shape:', output_affs.shape)

    #     output_affs = output_affs[:3]
    #     print('segmentation...')
    #     fragments = watershed(output_affs, 'maxima_distance')
    #     sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    #     segmentation = list(waterz.agglomerate(output_affs, [0.50],
    #                                         fragments=fragments,
    #                                         scoring_function=sf,
    #                                         discretize_queue=256))[0]
    #     segmentation = relabel(segmentation).astype(np.uint64)
    #     print('the max id = %d' % np.max(segmentation))
    #     # f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
    #     # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    #     # f.close()

    #     arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    #     voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    #     voi_sum = voi_split + voi_merge
    #     print('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #         (voi_split, voi_merge, voi_sum, arand))
    #     f_txt.write('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #         (voi_split, voi_merge, voi_sum, arand))
    #     f_txt.write('\n')
        
    #     f_txt.close()

    #     # 记录到全局结果文件
    #     # f_txt_all.write(f'{current_id}\t{cost_time:.6f}\t{voi_split:.6f}\t{voi_merge:.6f}\t{voi_sum:.6f}\t{arand:.6f}\n')
    #     f_txt_all.write('current_id=%.6f,cost_time=%.6f, waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #         (current_id,cost_time,voi_split, voi_merge, voi_sum, arand))
    #     f_txt_all.flush()
    
    # f_txt_all.close()
    # print(f'All results saved. Global scores in {f_txt_all_path}')
