import numpy as np
from config.config import cfg


class Dataset(object):
    def __init__(self):
        super(Dataset, self).__init__()

    def __iter__(self):
        """
        A iter for data generation.
        get roidb -> data enhancement(optional) -> shuffle -> rescale -> iter
        """
        _roidb = self.gt_roidb
        N, B = len(_roidb), cfg.img_per_batch
        idxs = list(np.arange(N))

        if self.is_shuffle:
            np.random.shuffle(idxs) # idxs type: list
            # here if convert _roidb to np.array:
            # return iter(_roidb[idxs[i:i+B]] for i in range(0, N, B))
            # if just iter the gt_roidb without rescaling the image
            # return iter([_roidb[k] for k in idxs[i:i+B]] for i in range(0, N, B))
        return iter(self._image_rescale(_roidb[idxs[i]]) for i in range(0, N, B))
        
    def get_one(self):
        _roidb = self.gt_roidb
        
        return self._image_rescale(_roidb[0])
    
    def get_small_dataset(self, n):
        """get small dataset including n images to do sanity check"""
        _roidb = self.gt_roidb
        
        return [self._image_rescale(_roidb[i]) for i in range(0, n)]
        