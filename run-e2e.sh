#!/bin/bash
set -eu

stage=3

. path.sh
. cmd.sh
. utils/parse_options.sh

if [ $stage -le 3 ]; then
  local/chain_e2e/build_new_tree.sh \
    --type biphone \
    --min_biphone_count 100 \
    --min_monophone_count 10 \
    --tie true \
    data/train_960/ \
    data/lang_chain \
    exp/chain_e2e/tree
fi

num_units=$(tree-info exp/chain_e2e/tree/tree | grep "num-pdfs" | cut -d" " -f2)
seed=2602

if [ $stage -le 5 ]; then
  local/chain/prepare_graph_clustered.sh \
    --dataroot data \
    --trainset train_960 \
    --validset dev_all \
    --lang data/lang_chain \
    --treedir exp/chain_e2e/tree \
    --graph exp/chain_e2e/graph
fi

if [ $stage -le 6 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain_e2e/tree \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e.yaml"
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bpe.5000.varikn/ exp/chain_e2e/graph exp/chain_e2e/graph/graph_bpe.5000.varikn
fi

if [ $stage -le 8 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e.yaml" \
    --py_script local/chain_e2e/sb-test-lfmmi-e2e.py \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain_e2e/New-CRDNN-FF-10/${seed}-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e.yaml" \
    --py_script local/chain_e2e/sb-test-lfmmi-e2e.py \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain_e2e/New-CRDNN-FF-10/${seed}-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 9 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain_e2e/tree \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-contd.yaml"
fi

if [ $stage -le 10 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-contd.yaml" \
    --py_script local/chain_e2e/sb-test-lfmmi-e2e.py \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain_e2e/New-CRDNN-FF-10-contd/${seed}-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-e2e-contd.yaml" \
    --py_script local/chain_e2e/sb-test-lfmmi-e2e.py \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain_e2e/New-CRDNN-FF-10-contd/${seed}-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

