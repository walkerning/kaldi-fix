# This script can automatically record all inner result of a random input on a specific nnet model in .dat file, and generate binary-form .nnetb files which records model parameters. It requires four input parameters as follows:
# path of the nnet model, dimension of the random input, frame number of the random input, path of the fix-strategy file
# e.g.: ./random_input_test.sh ~/final.nnet 153 2 ~/fixconf.mod


# path config
KALDI_DIR=`pwd`
BIN_DIR=$KALDI_DIR/src/nnetbin
PY_DIR=$KALDI_DIR/pynnet

# input config
nnet=$1
input_dim=$2
input_frame_num=$3
fixconf=$4

# generate test result
cd ~
rm -r ./testnet
mkdir testnet

cd $BIN_DIR
./nnet-fix-test $nnet $input_dim $input_frame_num $fixconf

# generate nnetb files
cd $PY_DIR
python sparsewrite.py $nnet
