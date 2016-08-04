
import os

def test_net(dir_net='exp/dnn4_pretrain-dbn_dnn/final.nnet'):
    flag = os.system('./local/nnet/test_wer.sh %s >/dev/null 2>&1 '%dir_net)#
    assert flag == 0
    os.system('bash show_dnn test > res.log')
    content = open('res.log').read()
    res = float(content.split()[1])
    return res


def finetune_net(dir_net = 'exp/dnn4_pretrain-dbn_dnn/nnet.init', 
        exp_dir='exp/dnn4_pretrain-dbn_dnn', iters=16, lr=0.002,
        momentum=0, l2_penalty=0, halve_every_k=2):
    flag = os.system('./finetune_dnn.sh --dir %s --nnet-init %s --iters %d --learning-rate %f --momentum %f --l2-penalty %f --halve-every-k %d'%(exp_dir, dir_net, iters, lr, momentum, l2_penalty, halve_every_k))
    return flag


