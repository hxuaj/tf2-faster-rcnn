import os
import xml.etree.ElementTree as ET
import scipy.sparse
import pickle
import numpy as np
from config.config import cfg
import cv2
from .dataset import Dataset


class pascal_voc(Dataset):
    def __init__(self, is_training=True, use_diff=False):
        self.is_training = is_training
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.root_path, 'VOCdevkit', 'VOC2007')
        self._classes = ('__background__',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self._classes)
        self.image_set_index = self._load_image_set_index()
        self.use_diff = use_diff
        if self.is_training:
            self.is_shuffle = cfg.shuffle
            # set the dataset with flipped here
            self.gt_roidb, self.data_size = self._data_enhance()
        else:
            self.is_shuffle = False
            self.gt_roidb, self.data_size= self._get_gt_roidb()
        

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.

        Output: 
        - image_index: the list of image indexes.
        """
        if self.is_training:
            # use trainval set to train
            image_set_file = os.path.join(self.data_path, 'ImageSets', 'Main',
                                          'trainval' + '.txt')
        else:
            image_set_file = os.path.join(self.data_path, 'ImageSets', 'Main',
                                          'test' + '.txt')
            
        assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip('\n') for x in f.readlines()]
        return image_index

    def image_path_from_index(self, index):
        """
        Return the absolute path of image with index.
        """
        image_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path
    
    # load annotations
    def _load_pascal_annotation(self, index):
        """
        Load ground truth info of image from XML file in the PASCAL VOC format.

        Input:
        - index: image index
        Output:
        - a dict "roidb" contains the image's ground truth annotation info including 
          image: image direct path
          img_size: [h, w, d]
          boxes: 0-based [x1, y1, x2, y2]
          gt_classes: ground truth classes' index
          gt_overlaps: the overlap with corresponding cls's box is  1.0
          flipped: flip flag, seg_areas: the quantity of box area.
        """
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        size = tree.findall('size')
        assert len(size) == 1, 'One picture should have one size.'
        w = int(size[0].find('width').text)
        h = int(size[0].find('height').text)
        d = int(size[0].find('depth').text)
        img_size = np.array([h, w, d])

        if not self.use_diff:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        difficult = np.zeros((num_objs), dtype=np.int32)

        _class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls_idx = _class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls_idx
            overlaps[ix, cls_idx] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            difficult[ix] = int(obj.find('difficult').text)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'image_path': self.image_path_from_index(index),
                'img_size': img_size,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'difficult': difficult}
    
    def _get_gt_roidb(self):
        """
        Construct gt_roidb from dataset for future use.
        Prefer save to cache for faster reuse(~0.07s), regenerate gt_roidb will cost ~2s.
        """
        save_to_cache = True
        if save_to_cache:
            cache_path = os.path.join(self.root_path, 'cache')
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            if self.is_training:
                if self.use_diff:
                    cache_pkl = os.path.join(cache_path, 'voc_07_train' + '_diff' + '_gt_roidb.pkl')
                else:
                    cache_pkl = os.path.join(cache_path, 'voc_07_train' + '_gt_roidb.pkl')
            else:
                if self.use_diff:
                    cache_pkl = os.path.join(cache_path, 'voc_07_test' + '_diff' + '_gt_roidb.pkl')
                else:
                    cache_pkl = os.path.join(cache_path, 'voc_07_test' + '_gt_roidb.pkl')
            
            if not os.path.exists(cache_pkl):
                gt_roidb = [self._load_pascal_annotation(i) for i in self.image_set_index]
                with open(cache_pkl, 'wb') as f:
                    pickle.dump(gt_roidb, f, pickle.HIGHEST_PROTOCOL)
                print("wrote gt_roidb to cache at {}".format(cache_pkl))
            else:
                with open(cache_pkl, 'rb') as f:
                    gt_roidb = pickle.load(f)
                print("loaded gt_roidb from {}".format(cache_pkl))
        else:
            gt_roidb = [self._load_pascal_annotation(i) for i in self.image_set_index]
        
        return gt_roidb, len(gt_roidb)

    def _data_enhance(self):
        """
        Horizontally flip the image as data enhancement
        """
        
        gt_roidb, _ = self._get_gt_roidb()
        for i in range(len(gt_roidb)):
            boxes = gt_roidb[i]['boxes'].copy()
            width = gt_roidb[i]['img_size'][1]
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
            extra = {'image_path': gt_roidb[i]['image_path'],
                     'img_size': gt_roidb[i]['img_size'],
                     'boxes': boxes,
                     'gt_classes': gt_roidb[i]['gt_classes'],
                     'flipped': True,
                     'difficult': gt_roidb[i]['difficult']}
            gt_roidb.append(extra)
        # print(len(gt_roidb))
        return gt_roidb, len(gt_roidb)

    def _compute_pixel_mean(self):
        N = len(self.gt_roidb)
        mean = np.array([[[0.0, 0.0, 0.0]]])
        for i in range(N):
            img =cv2.imread(self.gt_roidb[i]['image_path'])
            mean += np.mean(img, axis=(0, 1)) / N
        print("Dataset mean with channel order RGB: ", mean[:, :, ::-1])
        return mean[:, :, ::-1]

    def _image_rescale(self, roidb):
        """rescale the image """
        img_input = {}
        # np.array: (H, W, 3), cv2.imread channels stored in (B G R) order.
        # OpenCV considers float only when values range from 0-1, astype after cvtColor
        img = cv2.imread(roidb['image_path']).astype(np.float32)
        if cfg.img_is_RGB:
            img = img[:, :, ::-1] # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img -= cfg.pixel_mean
        
        H, W = img.shape[0:2]
        if roidb['flipped']:
            img = np.flip(img, axis=1).copy()
        
        scale = float(min(cfg.min_size / min(H, W), cfg.max_size / max(H, W)))
        # cv2 accept dsize in (W, H) format.
        dsize = (int(W * scale), int(H * scale))
        img_scaled = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)    
        
        # gt_boxes_scaled: (k, 5), k is the number of the bounding box. The second
        # dimension is the bbox attributes: (x1, y1, x2, y2, cls)
        # gt_boxes_scaled = np.concatenate((roidb['boxes'] * scale, 
        #                                   np.reshape(roidb['gt_classes'], (-1, 1))), axis=1)

        # make sure the dtype before input into model
        img_input['img_path'] = roidb['image_path']
        img_input['img_scaled'] = np.array([img_scaled]) # float32
        img_input['img_size'] = roidb['img_size'][0:2]
        img_input['scaled_size'] = np.array((int(H * scale), int(W * scale))) # int32
        img_input['scale'] = np.array(scale, dtype=np.float32) # float32
        
        if self.is_training:
            img_input['gt_boxes'] = roidb['boxes'] * scale
        else:
            img_input['gt_boxes'] = roidb['boxes']
        img_input['gt_classes'] = roidb['gt_classes']
        img_input['difficult'] = roidb['difficult'] 
        
        # if self.is_training:
        #     img_input['gt_boxes_scaled'] = gt_boxes_scaled.astype(np.float32) # float32
        # else:
        #     img_input['gt_boxes'] = roidb['boxes']
        #     img_input['gt_classes'] = roidb['gt_classes']
        #     img_input['difficult'] = roidb['difficult']
            
        return img_input
            

if __name__ == '__main__':

    ds = pascal_voc()
    print(len(ds.gt_roidb))

    for i in range(13, 14):
        img = cv2.imread(ds.gt_roidb[i]['image_path'])
        if i > 5010:
            img = np.flip(img, axis=1).copy()
        
        boxes = ds.gt_roidb[i]['boxes']
        
        for i in range(boxes.shape[0]):
            top_left = boxes[i][0],boxes[i][1]
            bottom_right = boxes[i][2],boxes[i][3]
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()