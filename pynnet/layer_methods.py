
import numpy as np

def get_LstmProjectedStreams(blobs):
    mats = []
    def split_blob(blob):
        assert blob.data.shape[0]%4 == 0
        stride = blob.data.shape[0] / 4
        return np.split(blob.data, np.arange(1,4)*stride)
    mats.extend(split_blob(blobs[0]))
    mats.extend(split_blob(blobs[1]))
    mats.append(blobs[6].data)
    names = ['w_g_x', 'w_i_x', 'w_f_x','w_o_x'] + ['w_g_r', 'w_i_r', 'w_f_r', 'w_o_r'] + ['w_r_m']
    return mats,names

def get_AffineTransform(blobs):
    return [blobs[0].data], ['w']


def get_ConvolutionalComponent(blobs):
    return [blobs[0].data], ['w']

