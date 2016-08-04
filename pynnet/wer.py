
import os

def test_net(net):
    net.Write('exp_train_50h/lstm_karel/final.nnet')
    flag = os.system('./test_wer.sh >>log 2>&1')
    assert flag == 0
    os.system('cat exp_train_50h/lstm_karel/decode_testset_test8000/wer_* | grep WER > results')

    lines = open('results').readlines()
    wer = map(lambda x:float(x.split()[1]), lines)
    os.system('rm results')
    return min(wer)


