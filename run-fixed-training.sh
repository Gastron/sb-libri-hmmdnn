#!/bin/bash

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data="/scratch/elec/puhe/c/librispeech/"

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh


if [ $stage -le 25 ]; then
  local/chain/run_training.sh \
    --hparams hyperparams/chain/FIX-New-CRDNN-J.yaml
fi

if [ $stage -le 26 ]; then
  local/chain/run_training.sh \
    --hparams hyperparams/chain/FIX-New-CRDNN-J-contd.yaml
fi

if [ $stage -le 27 ]; then

  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/FIX-New-CRDNN-J.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/FIX-New-CRDNN-J/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"

  local/chain/decode.sh --datadir data/dev_other \
    --hparams "hyperparams/chain/FIX-New-CRDNN-J.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/FIX-New-CRDNN-J/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 28 ]; then

  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/FIX-New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/FIX-New-CRDNN-J-contd/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"

  local/chain/decode.sh --datadir data/dev_other \
    --hparams "hyperparams/chain/FIX-New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/FIX-New-CRDNN-J-contd/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"

fi


