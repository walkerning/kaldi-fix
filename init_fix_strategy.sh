# This script can automatically generate fixconf file from a given nnet model based on random input
# It requires three input parameters as follows:
# path name of the nnet model, dimension of the random input, frame number of the random input


# path config
KALDI_DIR=`pwd`
BIN_DIR=$KALDI_DIR/src/nnetbin

# input config
nnet=$1
input_dim=$2
input_frame_num=$3

# generate fixconf file
cd $BIN_DIR
./nnet-fix-init $nnet $input_dim $input_frame_num
