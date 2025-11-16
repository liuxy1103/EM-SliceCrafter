import os
import h5py
import numpy as np
import argparse
import time
from tqdm import tqdm
import waterz
from utils.fragment import watershed, relabel
# from utils.lmc import mc_baseline
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

def process_experiment(experiment_path, output_base_path=None):
    """
    Process a single experiment directory containing multiple volumes.
    
    Args:
        experiment_path: Path to experiment directory (e.g., affs_306000)
        output_base_path: Optional path to save results, if None will save in the same directory
    """
    if output_base_path is None:
        output_base_path = experiment_path
        
    print(f'Processing experiment: {experiment_path}')
    
    # Create a scores file in the experiment directory
    experiment_scores_path = os.path.join(output_base_path, 'volume_scores.txt')
    f_txt = open(experiment_scores_path, 'w')
    
    # Find all volume directories in the experiment path
    volume_dirs = []
    for item in os.listdir(experiment_path):
        if os.path.isdir(os.path.join(experiment_path, item)) and item.startswith('volume_'):
            volume_dirs.append(item)
    
    volume_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by volume number
    print(f'Found {len(volume_dirs)} volumes: {volume_dirs}')
    
    all_metrics = []
    t1 = time.time()
    
    for vol_dir in tqdm(volume_dirs):
        vol_path = os.path.join(experiment_path, vol_dir)
        
        # Prepare output path for this volume - same as input path
        vol_out_dir = os.path.join(output_base_path, vol_dir)
        os.makedirs(vol_out_dir, exist_ok=True)
        
        # Load predicted affinity maps
        affs_path = os.path.join(vol_path, 'affs.hdf')
        with h5py.File(affs_path, 'r') as f:
            output_affs = f['main'][:]
            # output_affs = f['main'][:3]
        
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
            'arand': 0.0,
            'voi_lmc': 0.0,
            'voi_split_lmc': 0.0,
            'voi_merge_lmc': 0.0,
            'arand_lmc': 0.0
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
        
        # LMC segmentation
        print(f'Running LMC segmentation for {vol_dir}...')
        try:
            from utils.lmc import mc_baseline
            segmentation_lmc = mc_baseline(seg_output_affs)
            segmentation_lmc = relabel(segmentation_lmc).astype(np.uint64)
            print(f'Max ID = {np.max(segmentation_lmc)}')
            
            # Calculate LMC segmentation metrics
            arand_lmc = adapted_rand_ref(gt_seg, segmentation_lmc, ignore_labels=(0))[0]
            voi_split_lmc, voi_merge_lmc = voi_ref(gt_seg, segmentation_lmc, ignore_labels=(0))
            voi_sum_lmc = voi_split_lmc + voi_merge_lmc
            
            vol_metrics['voi_lmc'] = voi_sum_lmc
            vol_metrics['voi_split_lmc'] = voi_split_lmc
            vol_metrics['voi_merge_lmc'] = voi_merge_lmc
            vol_metrics['arand_lmc'] = arand_lmc
            
            # Save LMC segmentation result
            vol_seg_lmc_path = os.path.join(vol_out_dir, 'seg_lmc.hdf')
            with h5py.File(vol_seg_lmc_path, 'w') as f:
                f.create_dataset('main', data=segmentation_lmc, dtype=segmentation_lmc.dtype, compression='gzip')
        except ImportError:
            print("LMC module not available, skipping LMC segmentation")
        except Exception as e:
            print(f"Error in LMC segmentation: {e}")
        
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
            f'ARAND={vol_metrics["arand"]:.6f}, '
            f'VOI-sum-LMC={vol_metrics["voi_lmc"]:.6f}, '
            f'VOI-split-LMC={vol_metrics["voi_split_lmc"]:.6f}, '
            f'VOI-merge-LMC={vol_metrics["voi_merge_lmc"]:.6f}, '
            f'ARAND-LMC={vol_metrics["arand_lmc"]:.6f}'
        )
        
        # Save individual volume metrics to a file in the volume directory
        vol_score_path = os.path.join(vol_out_dir, 'scores.txt')
        with open(vol_score_path, 'w') as vol_f:
            vol_f.write(
                f'MSE-loss={vol_metrics["mse"]:.6f}\n'
                f'BCE-loss={vol_metrics["bce"]:.6f}\n'
                f'F1-score={vol_metrics["f1"]:.6f}\n'
                f'VOI-sum={vol_metrics["voi"]:.6f}\n'
                f'VOI-split={vol_metrics["voi_split"]:.6f}\n'
                f'VOI-merge={vol_metrics["voi_merge"]:.6f}\n'
                f'ARAND={vol_metrics["arand"]:.6f}\n'
                f'VOI-sum-LMC={vol_metrics["voi_lmc"]:.6f}\n'
                f'VOI-split-LMC={vol_metrics["voi_split_lmc"]:.6f}\n'
                f'VOI-merge-LMC={vol_metrics["voi_merge_lmc"]:.6f}\n'
                f'ARAND-LMC={vol_metrics["arand_lmc"]:.6f}\n'
            )
    
    processing_time = time.time() - t1
    print(f'Processing time={processing_time:.6f}')
    f_txt.write(f'Processing time={processing_time:.6f}\n')
    
    # Write detailed volume results to the log file
    for vol_idx, metrics in enumerate(all_metrics):
        vol_name = volume_dirs[vol_idx]
        vol_results = (
            f'{vol_name}, '
            f'MSE={metrics["mse"]:.6f}, '
            f'BCE={metrics["bce"]:.6f}, '
            f'F1={metrics["f1"]:.6f}, '
            f'VOI={metrics["voi"]:.6f}, '
            f'VOI-split={metrics["voi_split"]:.6f}, '
            f'VOI-merge={metrics["voi_merge"]:.6f}, '
            f'ARAND={metrics["arand"]:.6f}, '
            f'VOI-LMC={metrics["voi_lmc"]:.6f}, '
            f'VOI-split-LMC={metrics["voi_split_lmc"]:.6f}, '
            f'VOI-merge-LMC={metrics["voi_merge_lmc"]:.6f}, '
            f'ARAND-LMC={metrics["arand_lmc"]:.6f}'
        )
        f_txt.write(vol_results + '\n')
    
    # Calculate average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics]) 
        for k in ['mse', 'bce', 'f1', 'voi', 'voi_split', 'voi_merge', 'arand', 
                 'voi_lmc', 'voi_split_lmc', 'voi_merge_lmc', 'arand_lmc']
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
        f'ARAND={avg_metrics["arand"]:.6f}, '
        f'VOI-LMC={avg_metrics["voi_lmc"]:.6f}, '
        f'VOI-split-LMC={avg_metrics["voi_split_lmc"]:.6f}, '
        f'VOI-merge-LMC={avg_metrics["voi_merge_lmc"]:.6f}, '
        f'ARAND-LMC={avg_metrics["arand_lmc"]:.6f}'
    )
    
    print(avg_metrics_str)
    f_txt.write(avg_metrics_str + '\n')
    f_txt.close()
    
    # Also save the average metrics to scores.txt in the experiment directory
    with open(os.path.join(output_base_path, 'scores.txt'), 'w') as f:
        f.write(avg_metrics_str + '\n')
    
    return all_metrics, avg_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, 
                        default='/h3cstore_ns/hyshi/CAD_SAM2/inference_SAM/2025-06-02--23-15-22_3d_mala_zebrafinch_pure')
    parser.add_argument('-id', '--model_id', type=str, default='affs_170000')
    args = parser.parse_args()
    
    # Check for zebrafinch test directory in the input path
    affs_dir = os.path.join(args.input_path, args.model_id)
    if os.path.exists(affs_dir) and os.path.isdir(affs_dir):
        print(f'Processing zebrafinch test directory')
        process_experiment(affs_dir)
    
    # Check for zebrafinch_test_128 directory
    zebrafinch_dir = os.path.join(args.input_path, 'zebrafinch_test_128', args.model_id)
    if os.path.exists(zebrafinch_dir) and os.path.isdir(zebrafinch_dir):
        print(f'Processing zebrafinch_test_128 directory')
        process_experiment(zebrafinch_dir)
    
    print('Done') 