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
    --graphdir exp/chain/graph/graph_bpe.5000.varikn_xent \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt0.3"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT-contd.yaml \
    --py_script local/chain/sb-test-xent.py \
    --graphdir exp/chain/graph/graph_bpe.5000.varikn_xent \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt0.3"
fi

if [ $stage -le 30 ]; then
  srun --mem 32G --time 1:0:0 -c4 utils/mkgraph.sh data/lang_3gram_pruned_char/ exp/chain/tree exp/chain/graph/graph_3gram_pruned_char_xent
fi

if [ $stage -le 31 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT-contd.yaml \
    --py_script local/chain/sb-test-xent.py \
    --graphdir exp/chain/graph/graph_bpe.5000.varikn_xent \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/2602-2256units/decode_test_clean_bpe.5000.varikn_acwt0.3"
  local/chain/decode.sh --datadir data/test_other/ \
    --hparams hyperparams/chain/New-CRDNN-FF-10-XENT-contd.yaml \
    --py_script local/chain/sb-test-xent.py \
    --graphdir exp/chain/graph/graph_bpe.5000.varikn_xent \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/2602-2256units/decode_test_other_bpe.5000.varikn_acwt0.3"
fi


num_units=$(tree-info exp/chain/tree/tree | grep "num-pdfs" | cut -d" " -f2)
seed=2602

if [ $stage -le 32 ]; then
  local/chain/decode.sh \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --stage 2 --posteriors_from "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt0.3/" \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_clean_3gram_pruned_char_acwt0.3" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char_xent" 
  local/chain/decode.sh \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --datadir "data/dev_other" \
    --stage 2 --posteriors_from "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt0.3/" \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_other_3gram_pruned_char_acwt0.3" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char_xent" 
fi

if [ $stage -le 37 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_clean exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_clean_3gram_pruned_char_acwt0.3 \
    exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_clean_4gram_char_rescored_acwt0.3
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat --max_lmwt 22" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_other exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_other_3gram_pruned_char_acwt0.3 \
    exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_dev_other_4gram_char_rescored_acwt0.3
fi


if [ $stage -le 33 ]; then
  local/chain/decode.sh \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --datadir "data/test_clean" \
    --stage 2 --posteriors_from "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt0.3/" \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_clean_3gram_pruned_char_acwt0.3" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char_xent" 
  local/chain/decode.sh \
    --acwt 0.3 --post-decode-acwt 3.0 \
    --datadir "data/test_other" \
    --stage 2 --posteriors_from "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_other_bpe.5000.varikn_acwt0.3/" \
    --decodedir "exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_other_3gram_pruned_char_acwt0.3" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char_xent" 
fi

if [ $stage -le 34 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_clean exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_clean_3gram_pruned_char_acwt0.3 \
    exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_clean_4gram_char_rescored_acwt0.3
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat --max_lmwt 22" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_other exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_other_3gram_pruned_char_acwt0.3 \
    exp/chain/New-CRDNN-FF-10-XENT-contd/${seed}-${num_units}units/decode_test_other_4gram_char_rescored_acwt0.3
fi
