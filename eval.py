import numpy as np
from collections import defaultdict
import time
from test import test_nms
from model.utils.bbox_transform import bbox_iou
from config.config import cfg


def voc_eval_pre_rec(dataset, model, num_classes):
    """
    PASCAL VOC evaluation
    Calculate precision and recall in the order of dataset images.
    Store the cumulative results by classes.
    """
    # Each parameter will be constructed as a list containing N numpy arrays, 
    # where N is the number of images in dataset.
    pred_bboxes, pred_probs, pred_labels = [], [], []
    gt_bboxes, gt_labels, difficults= [], [], []
    
    # warmup
    # for i, img_input in enumerate(dataset):
    #     tic = time.perf_counter()
    #     model(img_input)
    #     print(f'{i}, {(time.perf_counter() - tic)*1000:.2f}ms', end='\r')
    # print('\n')
    
    # internal timer
    __t = 0. # total time
    __t1 = 0. # model forward time
    __t2 = 0. # nms time
    
    # run inferences on dataset and store the results
    for i, img_input in enumerate(dataset):
        
        tic = time.perf_counter()
        labels, bboxes, cls_prob = model(img_input)
        __t1 = time.perf_counter() - tic

        tic = time.perf_counter()
        bboxes, probs, labels = test_nms(cls_prob, bboxes, score_thresh=cfg.eval_score_thresh)
        __t2 = time.perf_counter() - tic
        
        print(f'Evaluating test sample {i} ... FP: {__t1*1000:.02f} ms,  NMS: {__t2*1000:.02f} ms  ', end='\r')
        __t += (__t1 + __t2)
            
        pred_bboxes.append(bboxes)
        pred_probs.append(probs)
        pred_labels.append((labels).astype(int))
        gt_bboxes.append(img_input['gt_boxes'])
        gt_labels.append(img_input['gt_classes'])
        difficults.append(img_input['difficult'])  
    print(f'{__t:.02f}s after evaluating {dataset.data_size} test samples, {dataset.data_size / __t:.02f} fps')
    
    assert len(pred_bboxes) == \
           len(pred_probs) == \
           len(pred_labels) == \
           len(gt_bboxes) == \
           len(gt_labels) == \
           len(difficults), "The length of inputes should be the same."

    # The number of "not difficult" samples in each class = TP + FN
    total_gt = defaultdict(int)
    # The prediction's probs for each class
    probs = defaultdict(list)
    # record of TP(1) and FP(0) for each class
    record = defaultdict(list)
    
    for pred_bbox, pred_label, pred_prob, gt_bbox, gt_label, difficult in zip(pred_bboxes, 
                                                                               pred_labels, 
                                                                               pred_probs, 
                                                                               gt_bboxes, 
                                                                               gt_labels, 
                                                                               difficults):
        for cls in np.unique(np.concatenate((pred_label, gt_label))):
            # pick out the bboxes belong to cls
            pred_cls_mask = pred_label == cls
            p_bbox = pred_bbox[pred_cls_mask]
            p_prob = pred_prob[pred_cls_mask]
            order = np.argsort(p_prob)[::-1]
            p_bbox = p_bbox[order]
            p_prob = p_prob[order]
            
            gt_cls_mask = gt_label == cls
            g_bbox = gt_bbox[gt_cls_mask]
            g_diff = difficult[gt_cls_mask]
            
            total_gt[cls] += np.sum(g_diff * -1 + 1)
            probs[cls].extend(p_prob)
            
            if len(p_bbox) == 0: continue
            if len(g_bbox) == 0:
                record[cls].extend([0] * p_bbox.shape[0])
                continue
            
            # VOC evaluation follows integer typed bounding boxes.
            # p_bbox = p_bbox.copy()
            # p_bbox[:, 2:] += 1
            # g_bbox = g_bbox.copy()
            # g_bbox[:, 2:] += 1
            
            iou = bbox_iou(p_bbox, g_bbox)
            ind = np.argmax(iou, axis=1) # each p_bbox's g_bbox index with highest iou
            ind[iou.max(axis=1) < cfg.eval_iou_thresh] = -1
            
            selected = np.zeros(g_bbox.shape[0])
            for i in ind:
                if i >= 0:
                    if g_diff[i]: 
                        record[cls].append(-1)
                    else:
                        if not selected[i]:
                            record[cls].append(1)
                        else:
                            record[cls].append(0)
                    selected[i] = 1
                else:
                    record[cls].append(0)
     
    return cumul_cal_pre_rec(total_gt, probs, record)

def cumul_cal_pre_rec(total_gt, probs, record):
    """
    calculate precision and recall for each class cumulatively.
    Input:
    - total_gt: defaultdict(int), {class: The number of "not difficult" samples in each class = TP + FN}
    - probs: defaultdict(list), {class: [The prediction's probs]}
    - record: defaultdict(list), {class: [Record of TP(1) and FP(0) for each class]}
    Output:
    - precision: list, precision of each class
    - recall: list, recall of each class
    """
    
    num_classes = max(total_gt.keys()) + 1
    precision = [None] * num_classes
    recall = [None] * num_classes 
    
    for c in total_gt.keys():
        
        pc = np.array(probs[c])
        rc = np.array(record[c])
        
        order = np.argsort(pc)[::-1]
        rc = rc[order]
        
        tp = np.cumsum(rc == 1)
        fp = np.cumsum(rc == 0)
        
        precision[c] = tp / (tp + fp)
        
        if total_gt[c] > 0:
            recall[c] = tp / total_gt[c]
    
    return precision, recall

def cal_ap(precision, recall, use_07_metric=False):
    """
    Calculate the AP and mAP according to VOC07 and VOC10 matrics.
    """
    
    num_classes = len(precision)
    ap = np.zeros((num_classes,), dtype=float)
    
    for cls in range(num_classes):
        if precision[cls] is None or recall[cls] is None:
            ap[cls] = np.nan
            continue
        
        if use_07_metric:
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recall[cls] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(precision[cls])[recall[cls] >= t])
                ap[cls] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recall[cls], [1.]))
            mpre = np.concatenate(([0.], np.nan_to_num(precision[cls]), [0.]))
            
            # compute the precision envelope
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * precision
            ap[cls] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap
                

def voc_eval(dataset, model, num_classes):
    """
    PASCAL VOC evaluation
    
    Input:
    - dataset: PASCAL VOC dataset generator
    - model: Faster RCNN model in test mode
    - num_classes: the number of classes
    Output:
    - ap: average precision
    - mAP: mean average precision
    """
    precision, recall = voc_eval_pre_rec(dataset, model, num_classes)
    ap = cal_ap(precision, recall, use_07_metric=False)
    mAP = np.nanmean(ap)
    # print(f'ap = {ap}, mAP = {mAP}')
    return ap, mAP


if __name__ == '__main__':
    
    import os
    from data import pascal
    from model.faster_rcnn import FasterRCNN
    from test import init_model
    

    # define evaluation dataset
    num_classes = cfg.num_classes
    eval_ds = pascal.pascal_voc(is_training=False, use_diff=False)

    # define checkpoint path
    root_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(root_path, 'model', 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # build FasterRCNN
    model = FasterRCNN(is_training=False)
    # dummpy forward to build network variables
    _ = model(init_model())

    # load weights
    file_name = os.listdir(ckpt_path)
    if file_name:
        # continue last training
        weight_path = os.path.join(ckpt_path, file_name[0])
        model.load_weights(weight_path)
        print("successfully loaded {} from disk.".format(file_name[0]))
        ap ,mAP = voc_eval(eval_ds, model, num_classes)
        print(f'ap = {ap}, mAP = {mAP}')
        
    else:
        print("No weights...Evaluation exit...")