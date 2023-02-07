#!/bin/bash
stage=0

. cmd.sh
. utils/parse_options.sh

if [ $stage -le 1 ]; then
  local/attention/run_training.sh --py_script local/attention/sb-train-mtl-w2v2.py --hparams hyperparams/attention/w2v2-F3.yaml
fi

if [ $stage -le 2 ]; then
  local/attention/run_test.sh \
    --py_script local/attention/sb_test_only_attn_w2v2.py \
    --hparams hyperparams/attention/w2v2-F3.yaml \
    --datadir "data/dev_clean"
  local/attention/run_test.sh \
    --py_script local/attention/sb_test_only_attn_w2v2.py \
    --hparams hyperparams/attention/w2v2-F3.yaml \
    --datadir "data/dev_other"
fi
