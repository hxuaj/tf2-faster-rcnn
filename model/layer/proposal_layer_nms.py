from model.utils.nms.gpu.gpu_nms import gpu_nms
import numpy as np


def proposal_nms(proposals, scores, max_out, thresh):
    
    keep = gpu_nms(proposals.numpy(), scores.numpy(),
                   max_out.numpy(), thresh.numpy())
    
    return np.array(keep)
