import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from loss.loss import WeightedMSE, WeightedBCE
# from provider_valid_general import Provider_valid
from provider_valid2 import Provider_valid
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
from unet3d_mala import UNet3D_MALA
# from model_superhuman import UNet_PNI
# from model_superhuman2 import UNet_PNI_FT2 as UNet_PNI
from model_superhuman2 import UNet_PNI
from segmamba import SegMamba
from model_unetr import UNETR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='3d_unetr_zebrafinch_pure_raw_inference', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2025-06-16--21-58-41_3d_unetr_zebrafinch_pure_raw')
    parser.add_argument('-id', '--model_id', type=int, default=176000)
    # parser.add_argument('-m', '--mode', type=str, default='')
    parser.add_argument('-m', '--mode', type=str, default='zebrafinch')
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
    
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
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

    ckpt_path = os.path.join('/h3cstore_ns/hyshi/CAD_SAM2/models', trained_model, 'model-%06d.ckpt' % args.model_id)
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
    
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    
    losses_valid = []
    all_metrics = []

    for k, batch in enumerate(val_loader, 0):
        inputs, target, weightmap = batch
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        
        volume_idx = k // volume_size
        slice_idx = k % volume_size
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
                'voi_split': 0.0,
                'voi_merge': 0.0,
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
            gt_flat = gt_affs.astype(np.uint8).flatten()
            pred_flat = output_affs_binary.astype(np.uint8).flatten()
            vol_metrics['f1'] = f1_score(gt_flat, pred_flat)
            
            # Save current volume's predictions
            vol_out_dir = os.path.join(out_affs, f'volume_{volume_idx}')
            os.makedirs(vol_out_dir, exist_ok=True)
            
            # Save prediction affinity maps
            vol_affs_path = os.path.join(vol_out_dir, 'affs.hdf')
            f = h5py.File(vol_affs_path, 'w')
            f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
            f.close()
            
            # Save ground truth affinity maps
            vol_gt_affs_path = os.path.join(vol_out_dir, 'affs_gt.hdf')
            f = h5py.File(vol_gt_affs_path, 'w')
            f.create_dataset('main', data=gt_affs, dtype=np.float32, compression='gzip')
            f.close()
            
            # Save ground truth segmentation
            vol_gt_seg_path = os.path.join(vol_out_dir, 'seg_gt.hdf')
            f = h5py.File(vol_gt_seg_path, 'w')
            f.create_dataset('main', data=gt_seg, dtype=gt_seg.dtype, compression='gzip')
            f.close()

            # Segmentation evaluation if enabled
            if hasattr(cfg.TEST, 'if_seg') and cfg.TEST.if_seg:
                # Use only first 3 channels for segmentation if available
                seg_output_affs = output_affs[:3] if output_affs.shape[0] >= 3 else output_affs
                seg_output_affs = np.ascontiguousarray(seg_output_affs, dtype=np.float32)
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
                vol_metrics['voi_split'] = voi_split
                vol_metrics['voi_merge'] = voi_merge
                vol_metrics['arand'] = arand
                
                # Save segmentation result
                vol_seg_path = os.path.join(vol_out_dir, 'seg_waterz.hdf')
                f = h5py.File(vol_seg_path, 'w')
                f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
                f.close()
            
            all_metrics.append(vol_metrics)
            
            # Print current volume metrics
            print(
                f'model-{args.model_id}, volume-{volume_idx}, '
                f'valid-loss={vol_metrics["loss"]:.6f}, '
                f'MSE-loss={vol_metrics["mse"]:.6f}, '
                f'BCE-loss={vol_metrics["bce"]:.6f}, '
                f'F1-score={vol_metrics["f1"]:.6f}, '
                f'VOI-sum={vol_metrics["voi"]:.6f}, '
                f'VOI-split={vol_metrics["voi_split"]:.6f}, '
                f'VOI-merge={vol_metrics["voi_merge"]:.6f}, '
                f'ARAND={vol_metrics["arand"]:.6f}'
            )
            
        pbar.update(1)
    
    pbar.close()
    inference_time = time.time() - t1
    print('Inference time=%.6f' % inference_time)
    f_txt.write('Inference time=%.6f' % inference_time)
    f_txt.write('\n')
    
    # Write detailed volume results to a separate file
    f_volumes_txt = open(os.path.join(out_affs, 'volume_scores.txt'), 'w')
    for vol_idx, metrics in enumerate(all_metrics):
        vol_results = (
            f'volume-{vol_idx}, '
            f'loss={metrics["loss"]:.6f}, '
            f'MSE={metrics["mse"]:.6f}, '
            f'BCE={metrics["bce"]:.6f}, '
            f'F1={metrics["f1"]:.6f}, '
            f'VOI={metrics["voi"]:.6f}, '
            f'VOI-split={metrics["voi_split"]:.6f}, '
            f'VOI-merge={metrics["voi_merge"]:.6f}, '
            f'ARAND={metrics["arand"]:.6f}'
        )
        f_volumes_txt.write(vol_results + '\n')
    f_volumes_txt.close()
    
    # Calculate average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics]) 
        for k in ['loss', 'mse', 'bce', 'f1', 'voi', 'voi_split', 'voi_merge', 'arand']
    }
    
    # Print and save average metrics
    avg_metrics_str = (
        f'AVERAGE METRICS: '
        f'loss={avg_metrics["loss"]:.6f}, '
        f'MSE={avg_metrics["mse"]:.6f}, '
        f'BCE={avg_metrics["bce"]:.6f}, '
        f'F1={avg_metrics["f1"]:.6f}, '
        f'VOI={avg_metrics["voi"]:.6f}, '
        f'VOI-split={avg_metrics["voi_split"]:.6f}, '
        f'VOI-merge={avg_metrics["voi_merge"]:.6f}, '
        f'ARAND={avg_metrics["arand"]:.6f}'
    )
    
    print(avg_metrics_str)
    f_txt.write(avg_metrics_str)
    f_txt.write('\n')
    f_txt.close()
    
    # # Save all predictions for the entire dataset
    # print('Saving combined results...')
    
    # # Get all results from provider
    # all_pred_affs = valid_provider.get_results()    
    # all_gt_affs = valid_provider.get_gt_affs()
    # all_gt_seg = valid_provider.get_gt_lb()
    # print('the shape of affs:', all_pred_affs.shape)
    # print('the shape of gt affs:', all_gt_affs.shape)
    # print('the shape of gt seg:', all_gt_seg.shape)
    
    # print('save all volumes affs...')
    # print('the shape of affs:', all_pred_affs.shape)
    # f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
    # f.create_dataset('main', data=all_pred_affs, dtype=np.float32, compression='gzip')
    # f.close()
    
    # print('save gt affs...')
    # print('the shape of affs:', all_gt_affs.shape)
    # f = h5py.File(os.path.join(out_affs, 'affs_gt.hdf'), 'w')
    # f.create_dataset('main', data=all_gt_affs, dtype=np.float32, compression='gzip')
    # f.close()
    
    # print('save gt segmentation')
    # f = h5py.File(os.path.join(out_affs, 'seg_gt.hdf'), 'w')
    # f.create_dataset('main', data=all_gt_seg, dtype=all_gt_seg.dtype, compression='gzip')
    # f.close()