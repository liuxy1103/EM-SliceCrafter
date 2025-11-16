import os
import h5py
import numpy as np
import argparse
import time
from tqdm import tqdm
import waterz
from utils.fragment import watershed
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
                        default='/h3cstore_ns/hyshi/CAD_SAM2/inference_SAM/2025-05-30--15-44-18_2d_SAM_zebrafinch_1.0_1.0_0.1_1.0/post_processed')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Open log file for writing results
    f_txt = open(os.path.join(args.output_path, 'waterz_scores.txt'), 'w')
    
    print(f'Processing data from: {args.input_path}')
    print(f'Output path: {args.output_path}')
    
    # Find all volume directories in the input path
    volume_dirs = []
    for item in os.listdir(args.input_path):
        if os.path.isdir(os.path.join(args.input_path, item)) and item.startswith('volume_'):
            volume_dirs.append(item)
    
    if not volume_dirs:
        # Check if the files are directly in the input path
        if all(os.path.exists(os.path.join(args.input_path, f)) for f in ['affs.hdf', 'affs_gt.hdf', 'seg_gt.hdf']):
            print('Processing single dataset from root directory')
            volume_dirs = ['']  # Process the root directory

    all_metrics = []
    
    t1 = time.time()
    for vol_dir in tqdm(volume_dirs):
        vol_path = os.path.join(args.input_path, vol_dir)
        
        # Prepare output path for this volume
        vol_out_dir = os.path.join(args.output_path, vol_dir) if vol_dir else args.output_path
        os.makedirs(vol_out_dir, exist_ok=True)
        
        # Load predicted affinity maps
        affs_path = os.path.join(vol_path, 'affs.hdf')
        with h5py.File(affs_path, 'r') as f:
            output_affs = f['main'][:]
        
        # Load ground truth affinity maps
        gt_affs_path = os.path.join(vol_path, 'affs_gt.hdf')
        with h5py.File(gt_affs_path, 'r') as f:
            gt_affs = f['main'][:]
        
        # Load ground truth segmentation
        gt_seg_path = os.path.join(vol_path, 'seg_gt.hdf')
        with h5py.File(gt_seg_path, 'r') as f:
            gt_seg = f['main'][:]
        
        gt_seg = gt_seg.astype(np.uint32)
        
        # Initialize metrics dictionary for this volume
        vol_metrics = {
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
        
        # Use only first 3 channels for segmentation if available
        seg_output_affs = output_affs[:3] if output_affs.shape[0] >= 3 else output_affs
        seg_output_affs = np.ascontiguousarray(seg_output_affs, dtype=np.float32)
        
        # Watershed segmentation
        print(f'Running watershed segmentation for {vol_dir}...')
        fragments = watershed(seg_output_affs, 'maxima_distance')
        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
        segmentation_waterz = list(waterz.agglomerate(seg_output_affs, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
        
        # Calculate waterz segmentation metrics
        arand = adapted_rand_ref(gt_seg, segmentation_waterz, ignore_labels=(0))[0]
        voi_split, voi_merge = voi_ref(gt_seg, segmentation_waterz, ignore_labels=(0))
        voi_sum = voi_split + voi_merge
        
        vol_metrics['voi'] = voi_sum
        vol_metrics['voi_split'] = voi_split
        vol_metrics['voi_merge'] = voi_merge
        vol_metrics['arand'] = arand
        
        # Save waterz segmentation result
        vol_seg_path = os.path.join(vol_out_dir, 'seg_waterz.hdf')
        with h5py.File(vol_seg_path, 'w') as f:
            f.create_dataset('main', data=segmentation_waterz, dtype=segmentation_waterz.dtype, compression='gzip')
        
        all_metrics.append(vol_metrics)
        
        # Print current volume metrics
        print(
            f'Volume {vol_dir}, '
            f'MSE-loss={vol_metrics["mse"]:.6f}, '
            f'BCE-loss={vol_metrics["bce"]:.6f}, '
            f'F1-score={vol_metrics["f1"]:.6f}, '
            f'VOI-sum={vol_metrics["voi"]:.6f}, '
            f'VOI-split={vol_metrics["voi_split"]:.6f}, '
            f'VOI-merge={vol_metrics["voi_merge"]:.6f}, '
            f'ARAND={vol_metrics["arand"]:.6f}'
        )
    
    processing_time = time.time() - t1
    print(f'Processing time={processing_time:.6f}')
    f_txt.write(f'Processing time={processing_time:.6f}\n')
    
    # Write detailed volume results to the log file
    for vol_idx, metrics in enumerate(all_metrics):
        vol_name = volume_dirs[vol_idx] if volume_dirs[vol_idx] else "single_volume"
        vol_results = (
            f'{vol_name}, '
            f'MSE={metrics["mse"]:.6f}, '
            f'BCE={metrics["bce"]:.6f}, '
            f'F1={metrics["f1"]:.6f}, '
            f'VOI={metrics["voi"]:.6f}, '
            f'VOI-split={metrics["voi_split"]:.6f}, '
            f'VOI-merge={metrics["voi_merge"]:.6f}, '
            f'ARAND={metrics["arand"]:.6f}'
        )
        f_txt.write(vol_results + '\n')
    
    # Calculate average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics]) 
        for k in ['mse', 'bce', 'f1', 'voi', 'voi_split', 'voi_merge', 'arand']
    }
    
    # Print and save average metrics
    avg_metrics_str = (
        f'AVERAGE METRICS: '
        f'MSE={avg_metrics["mse"]:.6f}, '
        f'BCE={avg_metrics["bce"]:.6f}, '
        f'F1={avg_metrics["f1"]:.6f}, '
        f'VOI={avg_metrics["voi"]:.6f}, '
        f'VOI-split={avg_metrics["voi_split"]:.6f}, '
        f'VOI-merge={avg_metrics["voi_merge"]:.6f}, '
        f'ARAND={avg_metrics["arand"]:.6f}'
    )
    
    print(avg_metrics_str)
    f_txt.write(avg_metrics_str + '\n')
    f_txt.close()

    print('Done') 