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
    parser.add_argument('-c', '--cfg', type=str, default='2d_SAM_zebrafinch_0_0_0_0_inference', help='path to config file')
    # parser.add_argument('-c', '--cfg', type=str, default='2d_CAD_zebrafinch_inference', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2025-07-09--19-50-37_2d_SAM_zebrafinch_0_0_0_0')
    # parser.add_argument('-mn', '--model_name', type=str, default='2025-06-02--23-16-22_2d_CAD_zebrafinch_3d_zebrafinch_z16_h160_noft_lr_ratio1_ft10000')
    parser.add_argument('-id', '--model_id', type=int, default=45000)
    # parser.add_argument('-id', '--model_id', type=int, default=355000)
    # parser.add_argument('-m', '--mode', type=str, default='')
    parser.add_argument('-m', '--mode', type=str, default='zebrafinch')
    # parser.add_argument('-m', '--mode', type=str, default='zebrafinch_test')
    # parser.add_argument('-m', '--mode', type=str, default='zebrafinch_test_128')
    parser.add_argument('-ts', '--test_split', type=int, default=150)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_true', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('/h3cstore_ns/hyshi/CAD_SAM2/scripts_2_5d_SAM/config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    if cfg.DATA.shift_channels is None:
        cfg.shift = None
    else:
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('/h3cstore_ns/hyshi/CAD_SAM2/inference_SAM', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)

    device = torch.device('cuda:0')

    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)
    
    ckpt_path = os.path.join('/h3cstore_ns/hyshi/CAD_SAM2/models', trained_model, 'model-%06d.ckpt' % args.model_id)
    checkpoint = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    

    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split) # 针对zebrafinch_test_128，test_split不影响zebrafinch系统数据集的读取
    # valid_provider = Provider_valid(cfg)
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
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    
    num_volumes = len(valid_provider.dataset)
    volume_size = valid_provider.num_per_dataset
    
    print(f'Number of volumes: {num_volumes}')
    print(f'Slices per volume: {volume_size}')
    
    # 创建存储每个体积预测和指标的列表
    all_volume_preds = [[] for _ in range(num_volumes)]
    all_volume_losses = [[] for _ in range(num_volumes)]
    
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inference_size = inputs.shape[-1]
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        
        # 确定当前切片属于哪个体积
        volume_idx = k // volume_size
        slice_idx = k % volume_size
        
        with torch.no_grad():
            if inputs.shape[-1] == 1250:
                inputs = F.pad(inputs, (7, 7, 7, 7), mode='reflect')
                embedding1, embedding2 = model(inputs)
                embedding1 = embedding1[:,:,7:-7,7:-7]
                embedding2 = embedding2[:,:,7:-7,7:-7]  
            elif inputs.shape[-1] == 150:
                inputs = F.pad(inputs, (5, 5, 5, 5), mode='reflect')
                embedding1, embedding2 = model(inputs)
                embedding1 = embedding1[:,:,5:-5,5:-5]
                embedding2 = embedding2[:,:,5:-5,5:-5]                     
            else:
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
            
        if cfg.TRAIN.if_verse:
            pred = pred * 2 - 1
            pred = torch.clamp(pred, 0.0, 1.0)
            
        # 存储当前切片的预测结果到对应体积列表
        pred_np = np.squeeze(pred.data.cpu().numpy())
        all_volume_preds[volume_idx].append(pred_np)
        all_volume_losses[volume_idx].append(tmp_loss.item())
        
        pbar.update(1)
        # 检查是否完成了一个volume的处理
        if (k + 1) % volume_size == 0 or k == len(val_loader) - 1:
            pbar.set_description(f"Processed volume {volume_idx}")

        
    pbar.close()
    inference_time = time.time() - t1
    print('Inference time=%.6f' % inference_time)
    f_txt.write('Inference time=%.6f' % inference_time)
    f_txt.write('\n')
    
    # 为所有指标创建字典
    all_metrics = {
        'loss': [],
        'mse': [],
        'bce': [],
        'f1': [],
        'voi': [],
        'voi_split': [],
        'voi_merge': [],
        'arand': [],
    }
    
    # 记录每个体积的详细结果
    f_volumes_txt = open(os.path.join(out_affs, 'volume_scores.txt'), 'w')
    
    # 处理每个体积的结果
    for vol_idx in range(len(all_volume_preds)):
        if len(all_volume_preds[vol_idx]) == 0:
            print(f"Warning: Volume {vol_idx} has no predictions.")
            continue
        
        # 计算当前体积的平均损失
        avg_volume_loss = sum(all_volume_losses[vol_idx]) / len(all_volume_losses[vol_idx])
        all_metrics['loss'].append(avg_volume_loss)
        
        # 堆叠当前体积的所有预测结果
        volume_pred = np.stack(all_volume_preds[vol_idx], axis=1)  # (C, Z, H, W)
        volume_pred = volume_pred.astype(np.float32)
        
        # 获取当前体积的真实标签
        gt_seg = valid_provider.get_gt_lb(vol_idx)
        gt_affs = valid_provider.get_gt_affs(vol_idx).copy()
        
        # 计算MSE
        mse = np.sum(np.square(volume_pred - gt_affs)) / np.size(gt_affs)
        all_metrics['mse'].append(mse)
        
        # 计算BCE
        volume_pred_clipped = np.clip(volume_pred, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(volume_pred_clipped) + (1 - gt_affs) * np.log(1 - volume_pred_clipped))
        bce_mean = np.sum(bce) / np.size(gt_affs)
        all_metrics['bce'].append(bce_mean)
        
        # 计算F1分数
        volume_pred_binary = volume_pred.copy()
        volume_pred_binary[volume_pred_binary <= 0.5] = 0
        volume_pred_binary[volume_pred_binary > 0.5] = 1
        
        gt_flat = gt_affs.astype(np.uint8).flatten()
        pred_flat = volume_pred_binary.astype(np.uint8).flatten()
        f1 = f1_score(gt_flat, pred_flat)
        all_metrics['f1'].append(f1)
        
        # 保存每个体积的预测结果
        vol_out_dir = os.path.join(out_affs, f'volume_{vol_idx}')
        os.makedirs(vol_out_dir, exist_ok=True)
        
        # 保存预测的亲和图
        vol_affs_path = os.path.join(vol_out_dir, 'affs.hdf')
        f = h5py.File(vol_affs_path, 'w')
        f.create_dataset('main', data=volume_pred, dtype=np.float32, compression='gzip')
        f.close()
        
        # 保存真实的亲和图
        vol_gt_affs_path = os.path.join(vol_out_dir, 'affs_gt.hdf')
        f = h5py.File(vol_gt_affs_path, 'w')
        f.create_dataset('main', data=gt_affs, dtype=np.float32, compression='gzip')
        f.close()
        
        # 保存真实的分割结果
        vol_gt_seg_path = os.path.join(vol_out_dir, 'seg_gt.hdf')
        f = h5py.File(vol_gt_seg_path, 'w')
        f.create_dataset('main', data=gt_seg, dtype=gt_seg.dtype, compression='gzip')
        f.close()
        
        # 进行分割评估
        if hasattr(cfg.TEST, 'if_seg') and cfg.TEST.if_seg:
            # 使用watershed算法进行分割
            fragments = watershed(volume_pred, 'maxima_distance')
            sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
            seg_waterz = list(waterz.agglomerate(volume_pred, [0.50],
                                                fragments=fragments,
                                                scoring_function=sf,
                                                discretize_queue=256))[0]
            arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
            voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
            voi_sum_waterz = voi_split + voi_merge
            
            # 保存分割结果
            vol_seg_path = os.path.join(vol_out_dir, 'seg_waterz.hdf')
            f = h5py.File(vol_seg_path, 'w')
            f.create_dataset('main', data=seg_waterz, dtype=seg_waterz.dtype, compression='gzip')
            f.close()
            
            all_metrics['voi'].append(voi_sum_waterz)
            all_metrics['voi_split'].append(voi_split)
            all_metrics['voi_merge'].append(voi_merge)
            all_metrics['arand'].append(arand_waterz)
        else:
            all_metrics['voi'].append(0)
            all_metrics['voi_split'].append(0)
            all_metrics['voi_merge'].append(0)
            all_metrics['arand'].append(0)
        
        # 输出当前体积的所有指标
        volume_metrics_str = (f'volume-{vol_idx}, '
                             f'loss={avg_volume_loss:.6f}, '
                             f'MSE={mse:.6f}, '
                             f'BCE={bce_mean:.6f}, '
                             f'F1={f1:.6f}, '
                             f'VOI={all_metrics["voi"][-1]:.6f}, '
                             f'VOI-split={all_metrics["voi_split"][-1]:.6f}, '
                             f'VOI-merge={all_metrics["voi_merge"][-1]:.6f}, '
                             f'ARAND={all_metrics["arand"][-1]:.6f}')
        
        print(volume_metrics_str)
        f_volumes_txt.write(volume_metrics_str + '\n')
    
    f_volumes_txt.close()
    
    # 计算所有体积的平均指标
    avg_metrics = {k: sum(v)/len(v) if len(v) > 0 else float('nan') for k, v in all_metrics.items()}
    
    # 输出并记录所有体积的平均结果
    avg_metrics_str = (f'AVERAGE METRICS: '
                      f'loss={avg_metrics["loss"]:.6f}, '
                      f'MSE={avg_metrics["mse"]:.6f}, '
                      f'BCE={avg_metrics["bce"]:.6f}, '
                      f'F1={avg_metrics["f1"]:.6f}, '
                      f'VOI={avg_metrics["voi"]:.6f}, '
                      f'VOI-split={avg_metrics["voi_split"]:.6f}, '
                      f'VOI-merge={avg_metrics["voi_merge"]:.6f}, '
                      f'ARAND={avg_metrics["arand"]:.6f}')
    
    print(avg_metrics_str)
    f_txt.write(avg_metrics_str)
    f_txt.write('\n')
    f_txt.close()
    
    # 保存所有体积合并的预测结果
    all_volumes_pred = np.concatenate([np.stack(preds, axis=1) for preds in all_volume_preds if len(preds) > 0], axis=1)
    all_volumes_pred = np.transpose(all_volumes_pred, (1, 0, 2, 3))
    
    print('save all volumes affs...')
    print('the shape of affs:', all_volumes_pred.shape)
    f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
    f.create_dataset('main', data=all_volumes_pred, dtype=np.float32, compression='gzip')
    f.close()
    
    # 保存所有体积的真实亲和图
    gt_affs = valid_provider.get_gt_affs()
    print('save gt affs...')
    print('the shape of affs:', gt_affs.shape)
    f = h5py.File(os.path.join(out_affs, 'affs_gt.hdf'), 'w')
    f.create_dataset('main', data=gt_affs, dtype=np.float32, compression='gzip')
    f.close()
    
    # 保存所有体积的真实分割
    gt_seg = valid_provider.get_gt_lb()
    print('save gt segmentation')
    f = h5py.File(os.path.join(out_affs, 'seg_gt.hdf'), 'w')
    f.create_dataset('main', data=gt_seg, dtype=gt_seg.dtype, compression='gzip')
    f.close()