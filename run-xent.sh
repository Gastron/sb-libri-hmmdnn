#!/usr/bin/env bash


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

# you might not want to do this for interactive shells.
set -e


# NOTE: Run run.sh first!

if [ $stage -le 24 ]; then
  local/chain/run_training.sh --py_script local/chain/sb-train-xent.py --hparams hyperparams/chain/New-CRDNN-FF-10-XENT.yaml
fi

if [ $stage -le 25 ]; then
  srun --mem 32G --time 1:0:0 -c4 utils/mkgraph.sh data/lang_bpe.5000.varikn/ exp/chain/tree exp/chain/graph/graph_bpe.5000.varikn_xent
fi

if [ $stage -le 27 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT.yaml \
    --py_script local/chain/sb-test-xent.py \
    --graphdir exp/chain/graph/graph_bpe.5000.varikn_xent \
    --acwt 0.1 --post-decode-acwt 1.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt0.1"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT.yaml \
    --py_script local/chain/sb-test-xent.py \
    --graphdir exp/chain/graph/graph_bpe.5000.varikn_xent \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt0.1"
fi

if [ $stage -le 28 ]; then
  local/chain/run_training.sh \
    --py_script local/chain/sb-train-xent.py \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-XENT-contd.yaml"
fi

if [ $stage -le 29 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT-contd.yaml \
    --py_script local/chain/sb-test-xent.py \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT-contd.yaml \
    --py_script local/chain/sb-test-xent.py \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

