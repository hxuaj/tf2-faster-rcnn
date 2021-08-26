import numpy as np
import tensorflow as tf


def generate_anchor_ref(base_size=16, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    """
    generate anchor references
    process while networks construct before generating shifted anchors
    """
    
    num_anchor = len(scales) * len(ratios)
    # (x1, y1, x2, y2): (0, 0, 15, 15) image-wise
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    center_x, center_y = (base_anchor[2] - base_anchor[0]) * 0.5, (base_anchor[3] - base_anchor[1]) * 0.5,
    
    # calculate base anchors' weights and heights with different ratios
    base_area = np.int32(base_size ** 2)
    base_ws = np.round(np.sqrt(base_area / ratios))
    base_hs = np.round(base_ws * ratios)

    # calculate anchors with scales taken into account
    # could be realized by matrix mul to speed up a little: (3, 1) x (1, 3). (won't be necessary)
    ws = np.hstack([base_ws[i] * np.array(scales) for i in range(base_ws.shape[0])])
    hs = np.hstack([base_hs[i] * np.array(scales) for i in range(base_hs.shape[0])])
    
    # calculate the distance from center to corner x, y
    delta = (np.vstack((ws, hs)).T - 1) * 0.5
    
    # anchors_ref: (x1, y1, x2, y2), shape=(num_anchor, 4) dtype=float32
    anchors_ref = np.vstack([[center_x, center_y, center_x, center_y] for i in range(num_anchor)])
    anchors_ref[:, 0:2] -= delta
    anchors_ref[:, 2::] += delta
        
    return tf.cast(anchors_ref, dtype=tf.float32)

@tf.function
def generate_anchors(anchors_ref, feat_stride, h, w):
    """
    Generate all shifted anchors according to the feature map.
    The function is kept in the RPN call.
    """
    # _, h, w, _ = feature_map.shape
    shift_x = tf.range(w) * feat_stride + feat_stride // 2
    shift_y = tf.range(h) * feat_stride + feat_stride // 2
    
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    x_ravel = tf.reshape(shift_x, shape=(-1,))
    y_ravel = tf.reshape(shift_y, shape=(-1,))
    shifts = tf.transpose(tf.stack([x_ravel, y_ravel, x_ravel, y_ravel], axis=0)) # (k, 4)

    A = anchors_ref.shape[0] # number of anchors per anchor point
    # K = shifts.shape[0] # number of points on feature map = h * w
    # (K, 1, 4), dtype=int32
    shifts = tf.transpose(tf.reshape(shifts, shape=(1, -1, 4)), perm=(1, 0, 2))
    shifts = tf.cast(shifts, dtype=tf.float32)
    anchors_ref = tf.reshape(anchors_ref, shape=(1, A, 4))
    
    # (K * A, 4), dtype=float32
    anchors = tf.reshape(tf.add(anchors_ref, shifts), shape=(-1, 4))
    
    return anchors

if __name__ == '__main__':
    anchors_ref = generate_anchor_ref()
    print(anchors_ref)
    a = generate_anchors(anchors_ref, 16, 3, 4)
    print(a)