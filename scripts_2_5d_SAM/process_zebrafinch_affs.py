import os
import h5py
import argparse
import numpy as np
import time
from tqdm import tqdm
import waterz
from utils.fragment import watershed, relabel
from utils.lmc import mc_baseline
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, 
                        default='/h3cstore_ns/hyshi/CAD_SAM2/inference_SAM/2025-05-30--15-44-18_2d_SAM_zebrafinch_1.0_1.0_0.1_1.0')
    parser.add_argument('-o', '--output_path', type=str, 
                        default=None)
    args = parser.parse_args()
    
    input_path = args.input_path
    if args.output_path is None:
        output_path = os.path.join(input_path, 'post_processed')
    else:
        output_path = args.output_path
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print(f'Processing data from: {input_path}')
    print(f'Output path: {output_path}')
    
    f_txt = open(os.path.join(output_path, 'scores.txt'), 'w')
    
    # Find all volume directories in the input path
    volume_dirs = []
    for item in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, item)) and item.startswith('volume_'):
            volume_dirs.append(item)
    
    if not volume_dirs:
        # Check if the files are directly in the input path
        if all(os.path.exists(os.path.join(input_path, f)) for f in ['affs.hdf', 'affs_gt.hdf', 'seg_gt.hdf']):
            print('Processing single dataset from root directory')
            volume_dirs = ['']  # Process the root directory
    
    all_waterz_metrics = []
    all_lmc_metrics = []
    
    t1 = time.time()
    for vol_dir in tqdm(volume_dirs):
        vol_path = os.path.join(input_path, vol_dir)
        
        # Prepare output path for this volume
        vol_out_dir = os.path.join(output_path, vol_dir) if vol_dir else output_path
        os.makedirs(vol_out_dir, exist_ok=True)
        
        # Load predicted affinity maps
        affs_path = os.path.join(vol_path, 'affs.hdf')
        with h5py.File(affs_path, 'r') as f:
            output_affs = f['main'][:]
        
        print(f'Read affs from {affs_path}, shape: {output_affs.shape}')
        
        # Load ground truth affinity maps
        gt_affs_path = os.path.join(vol_path, 'affs_gt.hdf')
        with h5py.File(gt_affs_path, 'r') as f:
            gt_affs = f['main'][:]
        
        print(f'Read gt affs from {gt_affs_path}, shape: {gt_affs.shape}')
        
        # Load ground truth segmentation
        gt_seg_path = os.path.join(vol_path, 'seg_gt.hdf')
        with h5py.File(gt_seg_path, 'r') as f:
            gt_seg = f['main'][:]
        
        print(f'Read gt segmentation from {gt_seg_path}, shape: {gt_seg.shape}')
        
        # For segmentation, use only first 3 channels if available
        seg_output_affs = output_affs[:3] if output_affs.shape[0] >= 3 else output_affs
        seg_output_affs = np.ascontiguousarray(seg_output_affs, dtype=np.float32)
        
        # Calculate metrics for affinity maps
        mse = np.mean(np.square(output_affs - gt_affs))
        
        # Calculate BCE
        output_affs_clipped = np.clip(output_affs, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(output_affs_clipped) + (1 - gt_affs) * np.log(1 - output_affs_clipped))
        bce_loss = np.mean(bce)
        
        # Calculate F1 Score
        output_affs_binary = output_affs.copy()
        output_affs_binary[output_affs_binary <= 0.5] = 0
        output_affs_binary[output_affs_binary > 0.5] = 1
        gt_flat = gt_affs.astype(np.uint8).flatten()
        pred_flat = output_affs_binary.astype(np.uint8).flatten()
        f1 = f1_score(gt_flat, pred_flat)
        
        # Watershed segmentation
        print(f'Running watershed segmentation for {vol_dir}...')
        fragments = watershed(seg_output_affs, 'maxima_distance')
        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
        segmentation_waterz = list(waterz.agglomerate(seg_output_affs, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
        
        segmentation_waterz = relabel(segmentation_waterz).astype(np.uint16)
        print(f'Waterz max ID = {np.max(segmentation_waterz)}')
        
        # Calculate waterz segmentation metrics
        arand_waterz = adapted_rand_ref(gt_seg, segmentation_waterz, ignore_labels=(0))[0]
        voi_split_waterz, voi_merge_waterz = voi_ref(gt_seg, segmentation_waterz, ignore_labels=(0))
        voi_sum_waterz = voi_split_waterz + voi_merge_waterz
        
        # Save waterz segmentation result
        vol_seg_path = os.path.join(vol_out_dir, 'seg_waterz.hdf')
        with h5py.File(vol_seg_path, 'w') as f:
            f.create_dataset('main', data=segmentation_waterz, dtype=segmentation_waterz.dtype, compression='gzip')
        
        waterz_metrics = {
            'mse': mse,
            'bce': bce_loss,
            'f1': f1,
            'voi_split': voi_split_waterz,
            'voi_merge': voi_merge_waterz,
            'voi_sum': voi_sum_waterz,
            'arand': arand_waterz
        }
        
        all_waterz_metrics.append(waterz_metrics)
        
        print(
            f'Waterz segmentation metrics: '
            f'MSE={mse:.6f}, '
            f'BCE={bce_loss:.6f}, '
            f'F1={f1:.6f}, '
            f'VOI-split={voi_split_waterz:.6f}, '
            f'VOI-merge={voi_merge_waterz:.6f}, '
            f'VOI-sum={voi_sum_waterz:.6f}, '
            f'ARAND={arand_waterz:.6f}'
        )
        
        # LMC segmentation
        print(f'Running LMC segmentation for {vol_dir}...')
        segmentation_lmc = mc_baseline(seg_output_affs)
        segmentation_lmc = relabel(segmentation_lmc).astype(np.uint16)
        print(f'LMC max ID = {np.max(segmentation_lmc)}')
        
        # Calculate LMC segmentation metrics
        arand_lmc = adapted_rand_ref(gt_seg, segmentation_lmc, ignore_labels=(0))[0]
        voi_split_lmc, voi_merge_lmc = voi_ref(gt_seg, segmentation_lmc, ignore_labels=(0))
        voi_sum_lmc = voi_split_lmc + voi_merge_lmc
        
        # Save LMC segmentation result
        vol_seg_lmc_path = os.path.join(vol_out_dir, 'seg_lmc.hdf')
        with h5py.File(vol_seg_lmc_path, 'w') as f:
            f.create_dataset('main', data=segmentation_lmc, dtype=segmentation_lmc.dtype, compression='gzip')
        
        lmc_metrics = {
            'mse': mse,
            'bce': bce_loss,
            'f1': f1,
            'voi_split': voi_split_lmc,
            'voi_merge': voi_merge_lmc,
            'voi_sum': voi_sum_lmc,
            'arand': arand_lmc
        }
        
        all_lmc_metrics.append(lmc_metrics)
        
        print(
            f'LMC segmentation metrics: '
            f'MSE={mse:.6f}, '
            f'BCE={bce_loss:.6f}, '
            f'F1={f1:.6f}, '
            f'VOI-split={voi_split_lmc:.6f}, '
            f'VOI-merge={voi_merge_lmc:.6f}, '
            f'VOI-sum={voi_sum_lmc:.6f}, '
            f'ARAND={arand_lmc:.6f}'
        )
        
        # Save metrics for this volume
        vol_txt = open(os.path.join(vol_out_dir, 'metrics.txt'), 'w')
        vol_txt.write(
            f'Waterz segmentation metrics: '
            f'MSE={mse:.6f}, '
            f'BCE={bce_loss:.6f}, '
            f'F1={f1:.6f}, '
            f'VOI-split={voi_split_waterz:.6f}, '
            f'VOI-merge={voi_merge_waterz:.6f}, '
            f'VOI-sum={voi_sum_waterz:.6f}, '
            f'ARAND={arand_waterz:.6f}\n'
        )
        vol_txt.write(
            f'LMC segmentation metrics: '
            f'MSE={mse:.6f}, '
            f'BCE={bce_loss:.6f}, '
            f'F1={f1:.6f}, '
            f'VOI-split={voi_split_lmc:.6f}, '
            f'VOI-merge={voi_merge_lmc:.6f}, '
            f'VOI-sum={voi_sum_lmc:.6f}, '
            f'ARAND={arand_lmc:.6f}\n'
        )
        vol_txt.close()
    
    processing_time = time.time() - t1
    print(f'Processing time={processing_time:.6f}')
    f_txt.write(f'Processing time={processing_time:.6f}\n')
    
    # Calculate average metrics for waterz segmentation
    avg_waterz = {
        k: np.mean([m[k] for m in all_waterz_metrics]) 
        for k in ['mse', 'bce', 'f1', 'voi_split', 'voi_merge', 'voi_sum', 'arand']
    }
    
    # Calculate average metrics for LMC segmentation
    avg_lmc = {
        k: np.mean([m[k] for m in all_lmc_metrics]) 
        for k in ['mse', 'bce', 'f1', 'voi_split', 'voi_merge', 'voi_sum', 'arand']
    }
    
    # Print and save average metrics
    print('\nAverage metrics:')
    
    avg_waterz_str = (
        f'Average WATERZ metrics: '
        f'MSE={avg_waterz["mse"]:.6f}, '
        f'BCE={avg_waterz["bce"]:.6f}, '
        f'F1={avg_waterz["f1"]:.6f}, '
        f'VOI-split={avg_waterz["voi_split"]:.6f}, '
        f'VOI-merge={avg_waterz["voi_merge"]:.6f}, '
        f'VOI-sum={avg_waterz["voi_sum"]:.6f}, '
        f'ARAND={avg_waterz["arand"]:.6f}'
    )
    
    avg_lmc_str = (
        f'Average LMC metrics: '
        f'MSE={avg_lmc["mse"]:.6f}, '
        f'BCE={avg_lmc["bce"]:.6f}, '
        f'F1={avg_lmc["f1"]:.6f}, '
        f'VOI-split={avg_lmc["voi_split"]:.6f}, '
        f'VOI-merge={avg_lmc["voi_merge"]:.6f}, '
        f'VOI-sum={avg_lmc["voi_sum"]:.6f}, '
        f'ARAND={avg_lmc["arand"]:.6f}'
    )
    
    print(avg_waterz_str)
    print(avg_lmc_str)
    
    f_txt.write(avg_waterz_str + '\n')
    f_txt.write(avg_lmc_str + '\n')
    f_txt.close()
    
    print('Done') 