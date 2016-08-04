import numpy as np
import bitstring as bs
from struct import unpack
from pdb import set_trace
from sparsewrite import FixParam

def fixed2float(i, frag_pos):
    return float(i) / (1<<frag_pos)

def short2weight(i, fixparam):
    bitlen = 16
    return fixed2float(i >> (bitlen - fixparam.bitnum), fixparam.fragpos), i & ((1 << (bitlen - fixparam.bitnum)) - 1)

####################### Params ##################################

comp_num = 2
filename = '/home/woinck/kaldi/swrite/sogou-finetune/sogou_finetune_fixed_compress_newparam'
numPE = 32

bit_colptr = 16
reorder_stride = 16

m = [[153,1024,512],[512,1024,512]]
o = [0,0,0,2,2,2,0,2,1]
s = [[1,0]] * 3 + [[1,2]] * 3 + [[1,0]] + [[1,2]] + [[2,1]]

matcols = map(lambda y:map(lambda x:y[x], o), m)
matsizes = map(lambda y:map(lambda x:[y[x[0]], y[x[1]]], s), m)

weight_fixparams = [FixParam(12,9)] * 8 + [FixParam(12,7)]
bias_diag_fixparams = [FixParam(16,13)] * 7
# weight_fixparams = [FixParam(12,9)] * 9
# bias_diag_fixparams = [FixParam(12,9)] * 7
fixparams = [[weight_fixparams, bias_diag_fixparams]] * 2

#################################################################

bias_diag=[]
weights = []
for comp_mats in matsizes:
    weights.append([np.zeros(x).astype('float32') for x in comp_mats])
    # weights : 2 * 9 * zerosmat

for comp_index in range(len(matcols)):
    col_index = []
    with open(filename + '_comp' + str(comp_index) + '_2.netb','rb') as f2:
        for mat_col in matcols[comp_index]: # 9
            colptrs_bytes = f2.read(mat_col * numPE * bit_colptr / 8)
            col_index.append(np.array(unpack('>' + 'H' * (len(colptrs_bytes) / 2), colptrs_bytes)).reshape((mat_col, numPE)).transpose())
            # col_index:  9 * 32 * #input 

        for bias_diag_index in range(7):
            bias_diag_bytes = f2.read(m[comp_index][1] * 2)
            # bias_diag.append(map(fixed2float, list(unpack('>' + 'h' * (len(bias_diag_bytes) / 2), bias_diag_bytes))))
            bias_diag.append(map(lambda x:fixed2float(x, fixparams[comp_index][1][bias_diag_index].fragpos), list(unpack('>' + 'h' * (len(bias_diag_bytes) / 2), bias_diag_bytes))))

        assert f2.read(1) == ''


    with open(filename + '_comp' + str(comp_index) + '_1.netb','rb') as f1:
        for mat_index in range(len(matsizes[comp_index])):
            maxlen = max(col_index[mat_index][:,-1])
            if maxlen % reorder_stride:
                maxlen += reorder_stride - (maxlen % reorder_stride)

            weights_bytes = f1.read(maxlen * numPE)

            weights_short = [x.flatten() for x in np.hsplit(np.array(list(unpack('>' + 'h' * (maxlen * numPE / 2), weights_bytes))).reshape((maxlen / reorder_stride , reorder_stride / 2 * numPE)), numPE)]
            # weights_short: 32 * xxx

            for sub_mat_index in range(len(weights_short)): # 1-32
                cur_col = 0
                cur_row = 0
                num_weight = 0
                for weight in weights_short[sub_mat_index]:
                    if num_weight >= col_index[mat_index][sub_mat_index][cur_col]:
                        if cur_col == len(col_index[mat_index][sub_mat_index]) - 1:
                            break
                        cur_col += 1
                        cur_row = 0

                    weight_float, weight_row = short2weight(weight, fixparams[comp_index][0][mat_index])

                    cur_row += weight_row
                    weights[comp_index][mat_index][sub_mat_index + cur_row * numPE][cur_col] = weight_float

                    cur_row += 1
                    num_weight += 2

set_trace()













#
