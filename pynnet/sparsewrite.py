import numpy as np
import bitstring as bs
from nnetrw import Nnet
import pdb

# 

# Convert fixed float to Fix integer
def FixFloat2Fix(f, fragpos):
    return int(round(f * (1 << fragpos)))

# Fix strategy description
class FixParam():
    def __init__(self, bitnum, fragpos):
        self.bitnum = bitnum
        self.fragpos = fragpos

class SparseWriter(object):
    # read an existing fixed nnet and 
    # write to binary files for FPGA 

    def __init__(self, net, fixparam, numPE = 32):
        self.net = net
        self.numPE = numPE
        self.fixparam = fixparam

    def RowBlock(self, data, piece):
        # Divide a matrix to submatrix.
        # #submatrix = self.numPE
        assert data.shape[0] % (piece * self.numPE) == 0, 'Cannot extract w_gifo_x'
        stride = data.shape[0] / piece
        return [[data[stride * i + j:stride * (i + 1) + j: self.numPE,:] for j in range(self.numPE)] for i in range(piece)]

    
    def Blocking(self, comp_index):
        comp = self.net.params[comp_index]
        assert len(comp) == 7, '#matrix in Lstm Component dismatch'

        # w_gifo_x & w_gifo_r
        (self.wgx, self.wix, self.wfx, self.wox) = self.RowBlock(comp[0].data, 4)
        (self.wgr, self.wir, self.wfr, self.wor) = self.RowBlock(comp[1].data, 4)
        # w_r_m does not need transpose!
        self.wrm = self.RowBlock(comp[6].data, 1)[0]
    

    def GenWeightBytes(self, weights, bitlen):
        # not used
        return ''.join([(bs.BitArray(int = FixFloat2Fix(weight, self.fixparam.fragpos), length = self.fixparam.bitnum) + \
        bs.BitArray(int = rowindex, length = bitlen - self.fixparam.bitnum)).bytes for weight, rowindex in weights])

    
    def GenWeightByte(self, weight, rowindex, fixparam, bitlen):
        # generate bytes for target weight & rowindex
        return (bs.BitArray(int = FixFloat2Fix(weight, fixparam.fragpos), length = fixparam.bitnum) + \
        bs.BitArray(uint = rowindex, length = bitlen - fixparam.bitnum)).bytes

    
    def EieCol(self, col, colptr, fixparam, bitlen):
        # do single column EIE
        # return weight value and row index (in bytes)
        # and append new colptr
        maxrow = 1 << (bitlen - fixparam.bitnum)
        num_weights = 0
        zero_cnt = 0
        result_list = ''
        for i in col:
            if i == 0 and zero_cnt < maxrow - 1:
                zero_cnt += 1
            else:
                result_list += self.GenWeightByte(i, zero_cnt, fixparam, bitlen)
                zero_cnt = 0

        num_weights = len(result_list)

        colptr[0] += len(result_list)
        return result_list, bs.BitArray(int = colptr[0], length = bitlen).bytes


    def EieMat(self, mat, fixparam, bitlen):
        #do single matrix EIE
        assert bitlen % 8 == 0, 'bit length is not whole byte'
        colptr = [0]
        # col_result = [EieCol(col, colptr, bitlen) for col in mat.transpose()]
        weight_bytes, colptr_bytes = zip(*[self.EieCol(col, colptr, fixparam, bitlen) for col in mat.transpose()])
        return ''.join(weight_bytes), ''.join(colptr_bytes)


    def EieSubMat(self, matlist, fixparam, bitlen = 16):
        # do all submatrix EIE for one weight mat
        assert len(matlist) == self.numPE, '#submat not match, ' + str(len(matlist)) + ' != ' + str(self.numPE)
        return [list(x) for x in zip(*[self.EieMat(mat, fixparam, bitlen) for mat in matlist])]
        # [['aaa','bbb'],['col1','col2']]


    def Eie(self, fixparams):
        # do EIE for the whole component
        # save to : self.eiemats
        # save to : self.colptr

        assert len(self.wgx) == self.numPE, '#subblock not match! ' + str(len(self.wgx)) + ' != ' + str(self.numPE)
        
        # Right order for FPGA
        submat_order = [self.wix, self.wfx, self.wgx, self.wir, self.wfr, self.wgr, self.wox, self.wor, self.wrm]
        
        # self.eiemats, self.colptr = [list(x) for x in zip(*[self.EieSubMat(matlist) for matlist in submat_order])]
        self.eiemats, self.colptr = [list(x) for x in zip(*[self.EieSubMat(submat_order[i], fixparams[i]) for i in range(len(submat_order))])]
        '''
        output: [[32*submatstr],...,[32*submatstr]]
        output format as follow:
        size = self.wgx[0].size
        self.eiemats = [32 * ['aa'*size]] * 9
        self.colptr = [32 * ['bb'*self.wgx[0].shape[1]]] * 9
        '''


    def ReOrderSingleMat(self, mat):
        stride = 16
        assert len(mat) == self.numPE

        # find the max length
        maxlen = max([len(x) for x in mat])
        if maxlen % stride:
            maxlen += stride - (maxlen % stride)

        for submatindex in range(len(mat)):
            assert type(mat[submatindex]) is str
            # add zeros to the end
            mat[submatindex] += '\0' * (maxlen - len(mat[submatindex]))
        result = ''
        for i in range(len(mat[0]) / stride):
            for j in range(self.numPE):
                result += mat[j][i * stride:(i + 1) * stride]
        return result


    def ReOrderSingleColPtr(self, mat):
        # format of self.colptr:
        # [mat1, mat2,..., mat9]
        # mat1 = [str1, str2, ..., str32]
        assert len(mat) == self.numPE
        stride = 2
        result = ''
        for i in range(len(mat[0]) / stride):
            for j in range(self.numPE):
                result += mat[j][i * stride:(i + 1) * stride]
        return result


    def ReOrderColPtr(self):
        # Reorder column pointer
        # save to self.reordered_colptr

        self.reordered_colptr = ''.join([self.ReOrderSingleColPtr(x) for x in self.colptr])


    def ReOrder(self):
        # reorder colunm pointers and weights
        self.ReOrderColPtr()
        return ''.join([self.ReOrderSingleMat(x) for x in self.eiemats])


    def Mat2Bytes(self, mat, fixparam, bitlen = None):
        if bitlen == None:
            bitlen = fixparam.bitnum
        assert bitlen >= fixparam.bitnum, 'Target bitlength ' + str(bitlen) + ' larget than fix bitnum ' + str(fixparam.bitnum)

        bitstr = map(lambda f:bs.BitArray(int = FixFloat2Fix(f, fixparam.fragpos), length = bitlen), mat.flatten())
        return reduce(lambda x,y:x+y, bitstr).bytes


    def OtherW(self, comp_index):
        # convert bias & diagonal matrix to bytes
        comp = self.net.params[comp_index]
        assert len(comp) == 7, '#matrix in Lstm Component dismatch'
        assert len(comp[2].data.shape) == 1 and len(comp[2].data) % 4 == 0 , 'bias dimension incorrect'

        bias = [x for x in comp[2].data.reshape(4, len(comp[2].data) / 4)]
        diag = [comp[x].data for x in range(3,6)] # i, f, o
        other_order = [bias[1], diag[0], diag[1], bias[2], bias[0], bias[3], diag[2]]
        result = ''.join([self.Mat2Bytes(other_order[i],self.fixparam[comp_index][1][i], 16) for i in range(len(other_order))])
        return result


    def WriteLstmComp(self, comp_index, filename):
        print 'Writing a new Comp'
        # Blocking
        self.Blocking(comp_index)

        # EIE compress
        self.Eie(self.fixparam[comp_index][0])

        # Reorder & write
        with open(filename + '_comp' + str(comp_index) + '_1.netb', 'wb') as fout:
            fout.write(self.ReOrder())
            fout.close()

        with open(filename + '_comp' + str(comp_index) + '_2.netb', 'wb') as fout:
            fout.write(self.reordered_colptr + self.OtherW(comp_index))
            fout.close()


    def WriteNet(self, filename):
        for layer in net.layers:
            if (net.layers[layer]=='LstmProjectedStreams'):
                self.WriteLstmComp(layer, filename)


if __name__=='__main__':
    
    # weight_fixparams = [FixParam(12,9)] * 8 + [FixParam(12,7)]
    # bias_diag_fixparams = [FixParam(16,13)] * 7
    weight_fixparams = [FixParam(12,9)] * 9
    bias_diag_fixparams = [FixParam(12,9)] * 7

    fixparams = [[weight_fixparams, bias_diag_fixparams]] * 2
    net = Nnet('~/kaldi/testnet/sogou_finetune_fixed.nnet')
    sparse_writer = SparseWriter(net, fixparam = fixparams)
    sparse_writer.WriteNet('/home/woinck/kaldi/swrite/sogou-finetune/sogou_finetune_fixed_compress_newparam')
