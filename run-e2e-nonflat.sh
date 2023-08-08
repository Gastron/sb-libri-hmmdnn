#!/bin/bash
set -eu

stage=3

. path.sh
. cmd.sh
. utils/parse_options.sh

num_units=$(tree-info exp/chain/tree/tree | grep "num-pdfs" | cut -d" " -f2)
seed=2602

if [ $stage -le 6 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain/tree \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-nonflat.yaml"
fi

if [ $stage -le 9 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain/tree \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-nonflat-contd.yaml"
fi

if [ $stage -le 10 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-nonflat-contd.yaml" \
    --py_script local/chain_e2e/sb-test-lfmmi-e2e.py \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-e2e-nonflat-contd/${seed}-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-nonflat-contd.yaml" \
    --py_script local/chain_e2e/sb-test-lfmmi-e2e.py \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-e2e-nonflat-contd/${seed}-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

