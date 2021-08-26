from data import pascal
import tensorflow as tf
from model.faster_rcnn import FasterRCNN
import os
import numpy as np
import datetime
from config.config import cfg
from eval import voc_eval
import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description='Faster R-CNN train')
    parser.add_argument('--scratch', dest='from_scratch',
                        help='Training from scratch with ImageNet pretrain', action="store_true")
    parser.add_argument('--epoch', dest='max_epoch',
                        help='Max number of epoch', default=10, type=int)
    parser.add_argument('--recrod_all', dest='record_all',
                        help='Include kernel and gradient info in summary', action="store_true")
    
    return parser.parse_args()

if __name__ == '__main__':
    
    # np.random.seed(3)
    # np.set_printoptions(threshold=100000)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    
    args = parse_args()
    if args.from_scratch == True:
        cfg.use_vgg_pretrain = True
    if args.record_all ==True:
        cfg.record_all = True

    # set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_loss_dir = 'model/logs/loss/' + current_time + '/train'
    loss_summary_writer = tf.summary.create_file_writer(train_loss_dir)
    mAP_dir = 'model/logs/mAP/' + current_time
    mAP_summary_writer = tf.summary.create_file_writer(mAP_dir)
    if cfg.record_all:
        train_kernel_dir = 'model/logs/kernel/' + current_time
        train_gradient_dir = 'model/logs/gradient/' + current_time
        kernel_summary_writer = tf.summary.create_file_writer(train_kernel_dir)
        gradient_summary_writer = tf.summary.create_file_writer(train_gradient_dir)

    # define dataset
    ds = pascal.pascal_voc(is_training=True, use_diff=False)
    eval_ds = pascal.pascal_voc(is_training=False, use_diff=False)
    # construct a tiny dataset for sanity check
    sanity_sample = np.array(ds.get_small_dataset(30))

    # define checkpoint path
    root_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(root_path, 'model', 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # build FasterRCNN
    model = FasterRCNN(is_training=True)
    # dummpy forward to build network variables
    _ = model(sanity_sample[0])

    # load weights
    file_name = os.listdir(ckpt_path)
    if file_name:
        # continue last training
        best_path = os.path.join(ckpt_path, file_name[0])
        # best_epoch = int(file_name[0].split('_')[1])
        best_epoch = 0
        # best_loss = float(file_name[0].split('_')[-2])
        
        model.load_weights(best_path)
        print("successfully loaded {} from disk.".format(file_name[0]))
    else:
        best_path = ''
        best_epoch = 0
        # best_loss = np.Inf
        print("initialize training from scratch.")

    # record loss every 500 iterations
    rec_step = 500

    # define optimizer
    lr = 1e-3
    optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9) # , nesterov=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if cfg.double_bais_grads:
        double_mask = [1., 2.] * (len(model.trainable_variables) // 2)

    for epoch in range(1, args.max_epoch + 1, 1):
        total_loss = rpn_cls_l = rpn_bbox_l = roi_cls_l = roi_bbox_l = 0.0
            
        for i, _input in enumerate(ds):
            
            with tf.GradientTape() as tape:   
                rpn_cls_loss, rpn_bbox_loss, roi_cls_loss, roi_bbox_loss = model(_input)
                loss = rpn_cls_loss + rpn_bbox_loss + roi_cls_loss  + roi_bbox_loss
                # add regularization term
                loss_with_reg = loss + tf.add_n(model.losses)
            grads = tape.gradient(loss_with_reg, model.trainable_variables)
            if cfg.double_bais_grads:
                grads = [a * b for a, b in zip(grads, double_mask)]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            print(f'epoch{epoch} i={i} lr={lr}', ' total loss: ', loss.numpy(), 
                ' cls loss: ', rpn_cls_loss.numpy(), roi_cls_loss.numpy(), 
                ' box loss: ', rpn_bbox_loss.numpy(), roi_bbox_loss.numpy())

            total_loss += loss.numpy()
            rpn_cls_l += rpn_cls_loss.numpy()
            rpn_bbox_l += rpn_bbox_loss.numpy()
            roi_cls_l += roi_cls_loss.numpy()
            roi_bbox_l += roi_bbox_loss.numpy()
            
            if i % rec_step == 0 and i > 0:
                
                total_loss = total_loss / rec_step
                s = best_epoch + (epoch - 1) * 20 + i // rec_step
                # if best_loss > total_loss:
                #     best_loss = total_loss
                #     if os.path.exists(best_path):
                #         os.remove(best_path)
                #     best_path = ckpt_path + "/step_{}_loss_{:.3f}_.h5".format(s, total_loss)
                #     model.save_weights(best_path)

                with loss_summary_writer.as_default():
                    tf.summary.scalar('total_loss', total_loss, step=s)
                    tf.summary.scalar('roi_cls_loss', roi_cls_l / rec_step, step=s)
                    tf.summary.scalar('roi_bbox_loss', roi_bbox_l / rec_step, step=s)
                    tf.summary.scalar('rpn_cls_loss', rpn_cls_l / rec_step, step=s)
                    tf.summary.scalar('rpn_bbox_loss', rpn_bbox_l / rec_step, step=s)
                
                total_loss = rpn_cls_l = rpn_bbox_l = roi_cls_l = roi_bbox_l = 0.0
        
            # calculate ap here twice an epoch
            if i == ds.data_size / 2 or i == ds.data_size - 1:
                ap ,mAP = voc_eval(eval_ds, model, ds.num_classes)
                with mAP_summary_writer.as_default():
                    tf.summary.scalar('mAP', mAP, step=(epoch - 1) * 2 + i // 5000)
                print(f'ap = {ap}, mAP = {mAP}')
        
        # record variables and gradients every epoch
        if cfg.record_all: 
                    with kernel_summary_writer.as_default():
                        for v in model.trainable_variables:
                            tf.summary.histogram(v.name, v, step=s)
                    with gradient_summary_writer.as_default():
                        for g, v in zip(grads, model.trainable_variables):
                            tf.summary.histogram(v.name, g, step=s)
        
        # Train for 70kiterations and learning rate is reduced after 50k iterations.
        if epoch == 5:
            lr *= 0.1
            optimizer = tf.keras.optimizers.SGD(lr, momentum=0.9) # , nesterov=True)
            # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # save the weight on the 7th epoch
        if epoch == 7:
            model.save_weights(ckpt_path + "/faster_rcnn.h5")
