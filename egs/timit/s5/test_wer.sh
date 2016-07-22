#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmmdir=exp/tri3
data_fmllr=data-fmllr-tri3
stage=0 # resume training with --stage=N
fixconf="./fix.conf"
phase="test"
fixmodel=""
#feature_trans=""
model=""

# End of config.
. utils/parse_options.sh || exit 1;
#


dir=${model:-exp/dnn4_pretrain-dbn_dnn}
ali=${gmmdir}_ali
#feature_transform=${feature_trans:-exp/dnn4_pretrain-dbn/final.feature_transform}
feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform

[ -z "$1" ] && nnet=$dir/final.nnet
[ -z "$nnet" ] && nnet=$1

debug=${DEBUG:-1}
if [[ ${debug} -ne 0 ]]; then
    echo "using fix-point config file: ${fixconf}"
    echo -e "The configuration is: \n\"\"\"$(cat ${fixconf})\n\"\"\"" 
    echo "Using nnet: $nnet"
    post_process="cat"
else
    post_process="cut -d| -f1"
fi
fixopts=--fixopts\ \'--fix-config=${fixconf}\'
fixmodelopt=
if [[ -n "${fixmodel}" ]]; then
    fixmodelopt=--fixmodelopt\ \'--write-fix-model=${fixmodel}\'
    fixopts=${fixopts}\ ${fixmodelopt}
fi
echo "$fixopts"

# Decode (reuse HCLG graph)
steps/nnet/decode.sh --nj 1 --cmd "$decode_cmd" --acwt 0.2 --use-gpu "yes" --nnet $nnet \
    $fixopts \
    $gmmdir/graph $data_fmllr/${phase} $dir/decode_${phase} >/dev/null|| exit 1;

#--fixopts  \'$fixopts\'  \

# steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 --nnet $nnet\
#     --fixopts \'--fix-config=${fixconf}\' \
#     $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;

# for x in ${dir}/decode*; do
#     echo -n "$x: "
#     [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh | ${post_process}; 
# done

# for x in ${dir}/decode*; do 
x=${dir}/decode_${phase}
if [[ debug -ne 0 ]]; then
    echo -n "$x: "
fi
[ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/score_*/*.sys 2>/dev/null | utils/best_wer.sh | ${post_process}; 

exit 0
