import numpy as np
import bitstring as bs
from nnetrw import Nnet
import pdb
import sys


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
        assert data.shape[0] % (piece * self.numPE) == 0, 'Cannot extract the matrix'
        stride = data.shape[0] / piece
        return [[data[stride*i+j: stride*(i+1)+j: self.numPE, :] for j in range(self.numPE)] for i in range(piece)]


    def OrderlyRowBlock(self, data, piece):
        # Divide a matrix to submatrix.
        # #submatrix = self.numPE
        assert data.shape[0] % (piece * self.numPE) == 0, 'Cannot extract the matrix'
        stride = data.shape[0] / piece
        length = stride / self.numPE
        return [[data[stride*i+length*j: stride*i+length*(j+1), :] for j in range(self.numPE)] for i in range(piece)]

    
    def Blocking(self, comp_index):
        comp = self.net.params[comp_index]
        assert len(comp) == 7, '#matrix in Lstm Component dismatch'

        # w_gifo_x & w_gifo_r
        [self.wgx, self.wix, self.wfx, self.wox] = self.RowBlock(comp[0].data, 4)
        [self.wgr, self.wir, self.wfr, self.wor] = self.RowBlock(comp[1].data, 4)
        # w_r_m is reordered in the column direction and blocked in the row direction orderly
        [wrm_temp] = self.RowBlock(np.transpose(comp[6].data), 1)
        wrm_temp = np.array(wrm_temp)
        wrm_temp = np.reshape(wrm_temp,(2048,1024))
        wrm_temp = np.transpose(wrm_temp)
        [self.wrm] = self.OrderlyRowBlock(wrm_temp, 1)

    
    '''
    def GenWeightBytes(self, weights, bitlen):
        # not used
        return ''.join([(bs.BitArray(int = FixFloat2Fix(weight, self.fixparam.fragpos), length = self.fixparam.bitnum) + \
        bs.BitArray(int = rowindex, length = bitlen - self.fixparam.bitnum)).bytes for weight, rowindex in weights])
    '''

    
    def GenWeightByte(self, weight, rowindex, fixparam, bitlen):
        # generate bytes for target weight & rowindex
        # return (bs.BitArray(int = FixFloat2Fix(weight, fixparam.fragpos), length = fixparam.bitnum) + \
        # bs.BitArray(uint = rowindex, length = bitlen - fixparam.bitnum)).bytes
        return (bs.BitArray(uint = rowindex, length = bitlen - fixparam.bitnum) + \
        bs.BitArray(int = FixFloat2Fix(weight, fixparam.fragpos), length = fixparam.bitnum)).bytes
    

    def EieCol(self, col, colptr, fixparam, bitlen):
        # do single column EIE
        # return weight value and row index (in bytes)
        # and append new colptr
        maxrow = 1 << (bitlen - fixparam.bitnum)
        num_nz = sum(col != 0)
        # num_weights = 0
        zero_cnt = 0
        nz_cnt = 0
        result_list = ''
        for i in col:
            if nz_cnt == num_nz :
                break
            if i == 0 and zero_cnt < maxrow - 1:
                zero_cnt += 1
            else:
                result_list += self.GenWeightByte(i, zero_cnt, fixparam, bitlen)
                zero_cnt = 0
                if i != 0 :
                  nz_cnt += 1  
                
        # num_weights = len(result_list)

        colptr[0] += len(result_list) / 2
        return result_list, bs.BitArray(uint = colptr[0], length = bitlen).bytes


    def EieMat(self, mat, fixparam, bitlen):
        
        # do single matrix EIE
        assert bitlen % 8 == 0, 'bit length is not whole byte'
        colptr = [0]
        # col_result = [EieCol(col, colptr, bitlen) for col in mat.transpose()]
        weight_bytes, colptr_bytes = zip(*[self.EieCol(col, colptr, fixparam, bitlen) for col in mat.transpose()])
        #print mat.shape, colptr
        #print 'submat nonzero(before): ', mat[mat!=0].size
        #print 'submat nonzero(after ): ', len(''.join(weight_bytes)) / 2
        print mat[mat!=0].size, len(''.join(weight_bytes))/2
        return ''.join(weight_bytes), ''.join(colptr_bytes)


    def EieSubMat(self, matlist, fixparam, bitlen = 16):
        # do all submatrix EIE for one weight mat
        assert len(matlist) == self.numPE, '#submat not match, ' + str(len(matlist)) + ' != ' + str(self.numPE)
        print 'submats: --------- '
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
        stride = 2
        assert len(mat) == self.numPE

        # find the max length
        maxlen = max([len(x) for x in mat])
        if maxlen % stride:
            maxlen += stride - (maxlen % stride)

        sum = 0
        for submatindex in range(len(mat)):

            l = len(mat[submatindex])/2
            sum += l
            #print submatindex,l

            assert type(mat[submatindex]) is str
            # add zeros to the end
            mat[submatindex] += '\0' * (maxlen - len(mat[submatindex]))
        print 'sum=',sum
        result = ''
        for i in range(len(mat[0]) / stride):
            for j in range(self.numPE):
                result += mat[j][i * stride:(i + 1) * stride]
        sum_align = len(result)
        print 'sum_addzero=',sum_align
        return result


    def ReOrderSingleColPtr(self, mat):
        # format of self.colptr:
        # [mat1, mat2,..., mat9]
        # mat1 = [str1, str2, ..., str32]
        assert len(mat) == self.numPE
        stride = 2
        result = ''

        #for i in range(self.numPE) :
        #    print i,uint8(mat[i][-2]),uint8(mat[i][-1])

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

        # bias = [x for x in comp[2].data.reshape(4, len(comp[2].data) / 4)]
        # diag = [comp[x].data for x in range(3,6)] # i, f, o
        
        bias = comp[2].data.reshape(len(comp[2].data), 1)
        [self.Bc, self.Bi, self.Bf, self.Bo] = self.RowBlock(bias, 4)
        
        [self.Wic] = self.RowBlock(comp[3].data.reshape(len(comp[3].data),1),1)
        [self.Wfc] = self.RowBlock(comp[4].data.reshape(len(comp[4].data),1),1)
        [self.Woc] = self.RowBlock(comp[5].data.reshape(len(comp[5].data),1),1)

        other_order = [self.Bi, self.Bf, self.Bc, self.Bo, self.Wic, self.Wfc, self.Woc]

        for i in range(len(other_order)):
            other_order[i] = np.array(other_order[i], order = 'F')

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
    
    weight_fixparams_1 = [FixParam(12,6)] * 3 + [FixParam(12,12)] * 3 + [FixParam(12,6)] + [FixParam(12,12)] + [FixParam(12,12)]
    weight_fixparams_2 = [FixParam(12,12)] * 3 + [FixParam(12,13)] * 3 + [FixParam(12,12)] + [FixParam(12,13)] + [FixParam(12,12)]
    weight_fixparams_3 = [FixParam(12,12)] * 3 + [FixParam(12,12)] * 3 + [FixParam(12,12)] + [FixParam(12,12)] + [FixParam(12,11)]
    bias_diag_fixparams_1 = [FixParam(16,10)] * 4 + [FixParam(16,15)] + [FixParam(16,14)] + [FixParam(16,14)]
    bias_diag_fixparams_2 = [FixParam(16,10)] * 4 + [FixParam(16,15)] + [FixParam(16,15)] + [FixParam(16,15)]
    bias_diag_fixparams_3 = [FixParam(16,10)] * 4 + [FixParam(16,14)] + [FixParam(16,14)] + [FixParam(16,13)]
    fixparams = [[weight_fixparams_1, bias_diag_fixparams_1], [weight_fixparams_2, bias_diag_fixparams_2], [weight_fixparams_3, bias_diag_fixparams_3]]

    # net = Nnet('pruned_balance_153_sparsity_0.15.nnet')
    net = Nnet(sys.argv[1])

    sparse_writer = SparseWriter(net, fixparam = fixparams)
    # sparse_writer.WriteNet('153_CSC')
    sparse_writer.WriteNet(sys.argv[1])
