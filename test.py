import tensorflow as tf
import numpy as np
import cv2
import os
from data import pascal
from model.faster_rcnn import FasterRCNN
from config.config import cfg
from model.utils.nms.cpu_nms import cpu_nms
# Though pure python nms is slightly slower then cython implementation,
# still ~2 times faster then tf.image.non_max_suppression
# from model.utils.nms.py_cpu_nms import cpu_nms
from utils.visual import view
import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description='Faster R-CNN test')
    # choose input date resource
    parser.add_argument('-i', dest='input_data',
                        help='define the input data', default='voc07_test')
    parser.add_argument('--unpop', dest='unpop',
                        help='Without popping up result as a new window', action="store_false")
    parser.add_argument('--save', dest='save',
                        help='Save test results to folder', action="store_true")
    
    return parser.parse_args()

def _image_rescale(img_path):
    """For consistency with pascal dataset, 
       the input is constructed as a dict."""
    
    img_input = {}
    img = cv2.imread(img_path).astype(np.float32)
    img = img[:, :, ::-1] # transform img to RGB
    img -= cfg.pixel_mean
    H, W = img.shape[0:2]
    scale = float(min(cfg.min_size / min(H, W), cfg.max_size / max(H, W)))
    dsize = (int(H * scale), int(W * scale))
    img_scaled = cv2.resize(img, (int(W * scale), int(H * scale)),
                            interpolation=cv2.INTER_LINEAR)    

    # img_input['img_path'] = img_path
    img_input['img_scaled'] = np.array([img_scaled]) # float32
    img_input['scaled_size'] = np.array(dsize) # int32
    img_input['scale'] = np.array(scale, dtype=np.float32) # float32
    img_input['img_size'] = np.array([H, W])

    return img_input

def _load_image_from_folder():
    """load self-defined images from folder demo for inference"""
    
    # get images' paths
    root_path = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(root_path, 'demo', 'images')
    if not os.path.exists(demo_path):
        os.makedirs(demo_path)
    img_names = os.listdir(demo_path)
    imgs_path = [os.path.join(demo_path, i) for i in img_names]
    
    # image iterator
    return iter(_image_rescale(imgs_path[i]) for i in range(len(imgs_path))), imgs_path

def init_model():
    
    img_input = {}
    img_input['img_scaled'] = np.random.rand(1, 1000, 1000, 3).astype(np.float32) # float32
    img_input['scaled_size'] = np.array([1, 1000, 1000, 3]).astype(np.int32) # int32
    img_input['scale'] = np.array(1., dtype=np.float32) # float32
    img_input['img_size'] = np.array([1000, 1000])
    
    return img_input

def _get_weight_path():
    """locate model weights"""
    root_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(root_path, 'model', 'ckpt')
    file_name = os.listdir(ckpt_path)
    
    return os.path.join(ckpt_path, file_name[0])

def test_nms(cls_prob, bboxes, score_thresh=cfg.score_threshold):
    # cls_prob: (300, 21), bboxes(300, 21, 4)
    bboxes_out = []
    prob_out = []
    labels = []
    num_classes = bboxes.shape[-2]

    for i in range(1, num_classes):
        bbox = bboxes[:, i, :]
        prob = cls_prob[:, i]
        # bbox_trans = tf.stack([bbox[:, 1], bbox[:, 0], bbox[:, 3], bbox[:, 2]], axis=1)
        # keep = tf.image.non_max_suppression(bbox_trans, prob, max_output_size=cfg.max_output,
        #                                     iou_threshold=cfg.test_nms_thresh,
        #                                     score_threshold=score_thresh).numpy()
        keep = cpu_nms(bbox.numpy(), prob.numpy(), cfg.max_output, 
                          cfg.test_nms_thresh, score_thresh)
        bboxes_out.append(bbox.numpy()[keep])
        prob_out.append(prob.numpy()[keep])
        labels.append(i * np.ones((len(keep), )))
        
    bboxes_out = np.concatenate(bboxes_out, axis=0)
    prob_out = np.concatenate(prob_out, axis=0)
    labels = np.concatenate(labels, axis=0)

    return bboxes_out, prob_out, labels


if __name__ == '__main__':
    
    args = parse_args()
    
    if args.input_data == 'voc07_test':
        # use voc test data set
        ds = pascal.pascal_voc(is_training=False)
        num_classes = ds.num_classes
        # initialize the model
        model = FasterRCNN(is_training=False)
        model(ds.get_one())
        
    elif args.input_data == 'demo':
        print("load images from folder")
        ds, imgs_path = _load_image_from_folder()
        num_classes = pascal.pascal_voc().num_classes
        # initialize the model
        model = FasterRCNN(is_training=False)
        model(init_model())
    else:
        raise NameError("Please define input images with voc07_test or demo.")
    
    model.load_weights(_get_weight_path())
    print("************************** load weights succrssfully! **************************")
    
    # inference
    for i, img_input in enumerate(ds):
        
        labels, bboxes, cls_prob = model(img_input)
        
        # cls_prob: (300, 21), bboxes(300, 21, 4)
        bboxes, probs, labels = test_nms(cls_prob, bboxes)
        print("bboxes: ", bboxes, 
              "probs: ", probs, 
              "labels: ", labels)
        
        if args.input_data == 'voc07_test':
            view(img_input['img_path'], bboxes, labels=labels, scores=probs, pop_up=args.unpop)
        else:
            view(imgs_path[i], bboxes, labels=labels, scores=probs,
                 pop_up=args.unpop, save=args.save)        
        