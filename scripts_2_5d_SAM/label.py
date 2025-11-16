import os
import h5py
import numpy as np
from PIL import Image

from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph

def gen_affs(map1, map2=None, dir=0, shift=1):
    if dir == 0 and map2 is None:
        raise AttributeError('map2 is none')
    map1 = map1.astype(np.float32)
    h, w = map1.shape
    if dir == 0:
        map2 = map2.astype(np.float32)
    elif dir == 1:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[shift:, :] = map1[:h-shift, :]
    elif dir == 2:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[:, shift:] = map1[:, :w-shift]
    else:
        raise AttributeError('dir must be 0, 1 or 2')
    dif = map2 - map1
    out = dif.copy()
    out[dif == 0] = 1
    out[dif != 0] = 0
    out[map1 == 0] = 0
    out[map2 == 0] = 0
    return out

f = h5py.File('../data/AC3_labels.h5', 'r')
labels = f['main'][:]
f.close()

print(labels.shape)
labels = genSegMalis(labels, 1)
# lb_affs = seg_to_affgraph(labels, mknhood3d(1), pad='replicate').astype(np.float32)
lb_affs = seg_to_affgraph(labels, mknhood3d(1), pad='').astype(np.float32)
print(lb_affs.shape)

affs0 = lb_affs[0, 1]
affs1 = lb_affs[1, 0]
affs2 = lb_affs[2, 0]

out0 = gen_affs(labels[0], labels[1], dir=0)
out1 = gen_affs(labels[0], None, dir=1)
out2 = gen_affs(labels[0], None, dir=2)

# judge
dif0 = np.abs(out0 - affs0)
sum0 = np.sum(dif0)
dif0 = (dif0 * 255).astype(np.uint8)
Image.fromarray(dif0).save('dif0.png')

dif1 = np.abs(out1 - affs1)
sum1 = np.sum(dif1)
dif1 = (dif1 * 255).astype(np.uint8)
Image.fromarray(dif1).save('dif1.png')

dif2 = np.abs(out2 - affs2)
sum2 = np.sum(dif2)
dif2 = (dif2 * 255).astype(np.uint8)
Image.fromarray(dif2).save('dif2.png')

print(sum0, sum1, sum2)
# affs0 = (affs0 * 255).astype(np.uint8)
# affs1 = (affs1 * 255).astype(np.uint8)
# affs2 = (affs2 * 255).astype(np.uint8)
# Image.fromarray(affs0).save('affs0.png')
# Image.fromarray(affs1).save('affs1.png')
# Image.fromarray(affs2).save('affs2.png')

