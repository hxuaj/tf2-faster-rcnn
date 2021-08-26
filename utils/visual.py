import os
import cv2
import numpy as np


np.random.seed(0)
label_name = ('__background__',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
color = [np.random.random(size=3) * 256 for i in range(21)]

def view(img_path, bboxs, labels=None, scores=None, pop_up=True, save=False):
    
    # sanity check
    if labels is not None and not len(bboxs) == len(labels):
        raise ValueError('The length of label must be same as that of bbox')
    if scores is not None and not len(bboxs) == len(scores):
        raise ValueError('The length of score must be same as that of bbox')
    labels = labels.astype(int)
    img = cv2.imread(img_path)
    img_dirname = os.path.dirname(img_path)
    img_name = os.path.split(img_path)[-1]
    
    caption_font = cv2.FONT_HERSHEY_SIMPLEX # FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN 
    caption_size = 0.35
    caption_thickness = 1
    
    for i, box in enumerate(bboxs):
        top_left = box[0], box[1]
        bottom_right = box[2], box[3]
        
        # generate bbox's caption
        caption = f'{label_name[labels[i]]}: {scores[i]*100:.1f}%'
        (w, h), _ = cv2.getTextSize(caption, caption_font, caption_size, caption_thickness)
        # caption_left = box[0], int(box[1] - h)
        # caption_right = int(box[0] + w), box[1]
        caption_left = box[0], box[1]
        caption_right = int(box[0] + w), int(box[1] + h)
        cv2.rectangle(img, caption_left, caption_right, color[labels[i]], -1)
        
        # generate bbox
        cv2.rectangle(img, top_left, bottom_right, color[labels[i]], 1)
        cv2.putText(img, caption, (box[0], int(box[1] + h)), 
                    caption_font, caption_size, (255, 255, 255), caption_thickness)
    
    if save:
        # print(img_dirname, img_name[-1])
        save_path = os.path.join(os.path.dirname(img_dirname), 'results', 'result_' + img_name)
        cv2.imwrite(save_path, img)
        print(f'result of {img_name} saved to {save_path}')
        
    if pop_up:
        cv2.imshow(img_name, img)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def view_anchors_point(img_path, Xs, Ys):
    img = cv2.imread(img_path)
    for x in np.int32(Xs):
        for y in np.int32(Ys):
            cv2.circle(img, (x,y), radius=0, color=(0, 0, 255), thickness=2)
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()