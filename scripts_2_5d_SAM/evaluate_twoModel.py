import os
import h5py
import numpy as np

import waterz
import evaluate as ev

# model_xy = '2021-04-12--01-59-29_seg_isbi_data80_c16_xy'
# model_z = '2021-04-12--02-00-30_seg_isbi_data80_c16_z'
# id_xy = 88000
# id_z = 88000

# model_xy = '2021-04-11--14-14-07_seg_isbi_data80_xy'
# model_z = '2021-04-11--14-14-12_seg_isbi_data80_z'
# id_xy = 34000
# id_z = 41000

model_xy = '2021-04-10--09-22-32_seg_isbi_data80_xy'
model_z = '2021-04-10--09-44-01_seg_isbi_data80_z'
id_xy = 13000
id_z = 12000

mode = 'isbi'
base_path = '../inference'
# generated affinity
affsx_path = os.path.join(base_path, model_xy, mode+'_'+'x', 'affs_'+str(id_xy), 'affs.hdf')
affsy_path = os.path.join(base_path, model_xy, mode+'_'+'y', 'affs_'+str(id_xy), 'affs.hdf')
affsz_path = os.path.join(base_path, model_z, mode, 'affs_'+str(id_z), 'affs.hdf')

f = h5py.File(affsx_path, 'r')
affsx = f['main'][:]
f.close()
f = h5py.File(affsy_path, 'r')
affsy = f['main'][:]
f.close()
f = h5py.File(affsz_path, 'r')
affsz = f['main'][:]
f.close()

f = h5py.File('../data/snemi3d/isbi_labels.h5', 'r')
gt_seg = f['main'][:]
f.close()
gt_seg = gt_seg[-20:]

affs = np.stack([affsz, affsy, affsx], axis=0)
affs = affs.astype(np.float32)
print('The shape of affinity: ', affs.shape)

segmentation = list(waterz.agglomerate(affs, [0.50]))[0]
segmentation = segmentation.astype(np.int32)
segmentation, _, _ = ev.relabel_from_one(segmentation)
voi_merge, voi_split = ev.split_vi(segmentation, gt_seg)
voi_sum = voi_split + voi_merge
arand = ev.adapted_rand_error(segmentation, gt_seg)

print('VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' % 
    (voi_split, voi_merge, voi_sum, arand))
