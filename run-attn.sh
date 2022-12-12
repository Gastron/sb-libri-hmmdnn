#!/bin/bash
stage=0

. cmd.sh
. utils/parse_options.sh

if [ $stage -le 1 ]; then
  local/attention/run_training.sh 
fi

if [ $stage -le 2 ]; then
  local/attention/run_test.sh --datadir "data/dev_clean"
  local/attention/run_test.sh --datadir "data/dev_other"
fi

if [ $stage -le 3 ]; then
  local/attention/run_training.sh \
    --hparams hyperparams/attention/CRDNN-E-contd.yaml
fi

if [ $stage -le 4 ]; then
  local/attention/run_training.sh \
    --py_script local/attention/sb_train_attn_mwer.py \
    --hparams hyperparams/attention/mwer/CRDNN-E-MWER.yaml
fi
