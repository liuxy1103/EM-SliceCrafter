import os
import cv2
import time
import h5py
import numpy as np
from PIL import Image
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref

from utils.show import draw_fragments_3d
from data.data_segmentation import seg_widen_border
from data.data_affinity import seg_to_aff
# import cremi_tools.segmentation as cseg
from elf.segmentation.mutex_watershed import mutex_watershed
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws
from utils.affinity_official import seg2affs
from utils.show_seg_match import match_id

# def multicut(affs, offsets, solver='kernighan-lin'):
#     segmenter = cseg.SegmentationPipeline(cseg.LRAffinityWatershed(0.1, 0.25, 2.),
#                                           cseg.MeanAffinitiyMapFeatures(offsets),
#                                           cseg.Multicut(solver))
#     return segmenter(affs)

def multicut(affs, offsets, fragments=None):
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    if fragments is None:
        fragments = np.zeros_like(boundary_input, dtype='uint64')
        offset = 0
        for z in range(fragments.shape[0]):
            wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
            wsz += offset
            offset += max_id
            fragments[z] = wsz
    rag = feats.compute_rag(fragments)
    # offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation

def default_offsets(channel=3):
    if channel == 3:
        return [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    elif channel == 12:
        return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                [-2, 0, 0], [0, -3, 0], [0, 0, -3],
                [-3, 0, 0], [0, -9, 0], [0, 0, -9],
                [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    elif channel == 17:
        return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                # direct 3d nhood for attractive edges
                [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                # indirect 3d nhood for dam edges
                [0, -9, 0], [0, 0, -9],
                # long range direct hood
                [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                # inplane diagonal dam edges
                [0, -27, 0], [0, 0, -27]]
    else:
        raise NotImplementedError

channel = 3

f = h5py.File('../data/ac3_ac4/AC4_labels.h5', 'r')
labels = f['main'][:]
f.close()

labels = labels[-20:]
print('labels shape:', labels.shape)

labels_widen = seg_widen_border(labels, tsz_h=1)

if channel == 3:
    label111 = seg_to_aff(labels_widen, pad='')
    affs = label111
elif channel == 12:
    nhood233 = np.asarray([-2, 0, 0, 0, -3, 0, 0, 0, -3]).reshape((3, 3))
    nhood399 = np.asarray([-3, 0, 0, 0, -9, 0, 0, 0, -9]).reshape((3, 3))
    nhood427 = np.asarray([-4, 0, 0, 0, -27, 0, 0, 0, -27]).reshape((3, 3))
    label111 = seg_to_aff(labels_widen, pad='')
    label233 = seg_to_aff(labels_widen, nhood233, pad='')
    label399 = seg_to_aff(labels_widen, nhood399, pad='')
    label427 = seg_to_aff(labels_widen, nhood427, pad='')
    affs = np.concatenate((label111, label233, label399, label427), axis=0)
elif channel == 17:
    labels_widen = labels_widen.astype(np.uint64)
    shift = default_offsets(channel)
    mask = False
    ignore = 0
    retain_segmentation = False
    segmentation_to_binary = False
    map_to_foreground = True
    learn_ignore_transitions = False
    start = time.time()
    affs = seg2affs(labels_widen, offsets=shift,
                    retain_mask=mask,
                    ignore_label=ignore,
                    retain_segmentation=retain_segmentation,
                    segmentation_to_binary=segmentation_to_binary,
                    map_to_foreground=map_to_foreground,
                    learn_ignore_transitions=learn_ignore_transitions)
else:
    raise NotImplementedError
print('affs shape:', affs.shape)

# affs_show = affs.copy()
# affs_show = (affs_show * 255).astype(np.uint8)
# for i in range(labels.shape[0]):
#     im1 = np.concatenate([affs_show[0, i], affs_show[1, i], affs_show[2, i]], axis=1)
#     im2 = np.concatenate([affs_show[3, i], affs_show[4, i], affs_show[5, i]], axis=1)
#     im3 = np.concatenate([affs_show[6, i], affs_show[7, i], affs_show[8, i]], axis=1)
#     im4 = np.concatenate([affs_show[9, i], affs_show[10, i], affs_show[11, i]], axis=1)
#     im_cat = np.concatenate([im1, im2, im3, im4], axis=0)
#     Image.fromarray(im_cat).save(os.path.join('./data_temp', 'affs_'+str(i).zfill(4)+'.png'))

t1 = time.time()
# seg = multicut(affs, default_offsets(channel))
strides = np.array([1, 1, 1])
seg = mutex_watershed(1-affs, default_offsets(), strides, randomize_strides=False)
cost_time = time.time() - t1
print('COST TIME:', cost_time)

arand = adapted_rand_ref(labels, seg, ignore_labels=(0))[0]
voi_split, voi_merge = voi_ref(labels, seg, ignore_labels=(0))
voi_sum = voi_split + voi_merge
print('voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    (voi_split, voi_merge, voi_sum, arand))

seg_img_path = './data_temp'
if not os.path.exists(seg_img_path):
    os.makedirs(seg_img_path)
color_seg = draw_fragments_3d(seg)
color_gt = draw_fragments_3d(labels)
for i in range(labels.shape[0]):
    im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
    cv2.imwrite(os.path.join(seg_img_path, 'seg_'+str(i).zfill(4)+'.png'), im_cat)
# for i in range(labels.shape[0]):
#     im_cat = match_id(seg[i], labels[i])
#     cv2.imwrite(os.path.join(seg_img_path, 'seg_'+str(i).zfill(4)+'.png'), im_cat)
