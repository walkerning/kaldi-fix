#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=exp/tri3
data_fmllr=data-fmllr-tri3
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#


dir=exp/dnn4_pretrain-dbn_dnn
ali=${gmmdir}_ali
feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform


[ -z "$1" ] && nnet=$dir/final.nnet
[ -z "$nnet" ] && nnet=$1
echo $nnet

# Decode (reuse HCLG graph)
steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 --nnet $nnet \
    $gmmdir/graph $data_fmllr/test $dir/decode_test || exit 1;
# steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
#    $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
