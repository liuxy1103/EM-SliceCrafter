import os
import h5py
import numpy as np
from PIL import Image

from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph
from utils.gen_affs import gen_affs_3d

out_path = '../data/gt/isbi/affs'
if not os.path.exists(out_path):
    os.makedirs(out_path)

f = h5py.File('../data/snemi3d/isbi_inputs.h5', 'r')
raw = f['main'][:]
f.close()
raw = raw[-20:]

f = h5py.File('../data/snemi3d/isbi_labels.h5', 'r')
labels = f['main'][:]
f.close()
labels = labels[-20:]

labels = genSegMalis(labels, 1)
# affs = seg_to_affgraph(labels, mknhood3d(1), pad='replicate').astype(np.float32) # replicate

affs = gen_affs_3d(labels, padding=False)

# print(np.sum(np.abs(affs_ours-affs)))

affs = (affs * 255).astype(np.uint8)
for i in range(len(labels)):
    cat1 = np.concatenate([raw[i], affs[0, i]], axis=1)
    cat2 = np.concatenate([affs[1, i], affs[2, i]], axis=1)
    im_cat = np.concatenate([cat1, cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))