
import os,sys
from collections import OrderedDict
import re
import numpy as np
import pynnet.layer_methods as plm

_source_ = r'''
{ Component::kAffineTransform,"<AffineTransform>" },
{ Component::kLinearTransform,"<LinearTransform>" },
{ Component::kConvolutionalComponent,"<ConvolutionalComponent>"},
{ Component::kConvolutional2DComponent,"<Convolutional2DComponent>"},
{ Component::kLstmProjectedStreams,"<LstmProjectedStreams>"},
{ Component::kBLstmProjectedStreams,"<BLstmProjectedStreams>"},
{ Component::kSoftmax,"<Softmax>" },
{ Component::kBlockSoftmax,"<BlockSoftmax>" },
{ Component::kSigmoid,"<Sigmoid>" },
{ Component::kTanh,"<Tanh>" },
{ Component::kDropout,"<Dropout>" },
{ Component::kLengthNormComponent,"<LengthNormComponent>" },
{ Component::kRbm,"<Rbm>" },
{ Component::kSplice,"<Splice>" },
{ Component::kCopy,"<Copy>" },
{ Component::kAddShift,"<AddShift>" },
{ Component::kRescale,"<Rescale>" },
{ Component::kKlHmm,"<KlHmm>" },
{ Component::kAveragePoolingComponent,"<AveragePoolingComponent>"},
{ Component::kAveragePooling2DComponent,"<AveragePooling2DComponent>"},
{ Component::kMaxPoolingComponent, "<MaxPoolingComponent>"},
{ Component::kMaxPooling2DComponent, "<MaxPooling2DComponent>"},
{ Component::kSentenceAveragingComponent,"<SentenceAveragingComponent>"},
{ Component::kSimpleSentenceAveragingComponent,"<SimpleSentenceAveragingComponent>"},
{ Component::kFramePoolingComponent, "<FramePoolingComponent>"},
{ Component::kParallelComponent, "<ParallelComponent>"},
'''

def get_first_key(string):
    x = re.search(r'<[a-zA-Z]+>',string)
    if x is not None:
        return string[x.start()+1:x.end()-1]
    else:
        return None

def get_keys(string):
    lines = string.split('\n')
    keys = set()
    for line in lines:
        keys.add(get_first_key(line))
    return keys

def ReadMat(f, line):
    if line.strip() == '':
        line = f.readline()
        assert line != ''
    flag = True
    data = []
    while flag:
        if ']' in line:
            flag = False
            assert line[line.index(']')+1:].strip() == ''
            line = line[:line.index(']')]
        if line.strip() != '':
            data.append(map(float, line.split()))
        if flag:
            line = f.readline()

    data = np.array(data)
    assert data.dtype == np.dtype('float64')
    return np.squeeze(data.astype('f'))

def WriteMat(f, mat):
    assert mat.ndim == 2 or mat.ndim == 1
    if mat.ndim == 2:
        f.write(' [\n')
        for i in range(mat.shape[0]):
            f.write(' '.join(map(str, mat[i])) + '\n')
    else:
        f.write(' [ ')
        f.write(' '.join(map(str, mat)))
    f.write(' ]\n')


Supported_Layers = get_keys(_source_)
if None in Supported_Layers:
    Supported_Layers.remove(None)

def get_layer_function(name):
    try:
        func = getattr(plm, 'get_' + name)
    except AttributeError as e:
        print "Get-Mats function of Layer%s not implemented"%name
        print "Modify layer_methods.py to fix this problem"
        raise e
    return func

class Blob(object):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

class Nnet(object):
    def __init__(self, filename = None):
        self.params = OrderedDict() #parameters of updatable layers
        self.layers = OrderedDict() #type of layers
        self.misc = OrderedDict() #type of layers
        if type(filename) is str:
            self.Read(filename)

    def append_layer(self, type_t):
        assert type_t in Supported_Layers
        self.params[len(self.layers)] = []
        self.misc[len(self.layers)] = []
        self.layers[len(self.layers)] = type_t

    def append_params(self, params):
        self.params[len(self.layers)-1].append(Blob(params))

    def append_misc(self, string):
        if len(self.layers) == 0:
            return
        self.misc[len(self.layers)-1].append(string)

    def Read(net, filename):
        flag = os.system('nnet-copy --binary=false %s .tmp'%(filename))
        assert flag == 0, 'cannot find command nnet-copy'

        f = open('.tmp','r')
        while True:
            line = f.readline()
            if line == '':
                break
            key = get_first_key(line)
            if key == '/Nnet':
                break
            elif key in Supported_Layers:
                net.append_layer(key)
                print "Get a Layer:" + key
                net.append_misc(line)
                assert '[' not in line
            elif '[' in line:
                idx = line.index('[')
                if line[:idx].strip() != '':
                    net.append_misc(line[:idx])
                params = ReadMat(f, line[idx+1:])
                net.append_params(params)
            else:
                net.append_misc(line)
        f.close()

        os.system('rm .tmp')

    def Write(net, filename):
        f = open(filename, 'w')

        f.write('<Nnet>\n')
        for layer in net.layers:
            f.writelines(net.misc[layer])
            for mat in net.params[layer]:
                WriteMat(f,mat.data)
        f.write('<Nnet>\n')
        f.close()

    def get_mats(net):
        mats = []
        names = []
        for layer in net.params:
            if len(net.params[layer]) == 0:
                continue
            func = get_layer_function(net.layers[layer])
            mats_, names_ = func(net.params[layer])
            names_ = map(lambda x:net.layers[layer]+str(layer) + '_' +  x, names_)
            mats.extend(mats_)
            names.extend(names_)
        return mats, names

if __name__ == "__main__":
    
    net = Nnet('/home/maohz12/online_50h_Tsinghua/exp_train_50h/lstm_karel_bak/final.nnet.bak')
    net.Write('test.nnet')
    net2 = Nnet('test.nnet')
