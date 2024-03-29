#!/usr/bin/env bash

stage=1
separate_prior_run=true

. ./cmd.sh
. ./path.sh
. parse_options.sh


# you might not want to do this for interactive shells.
set -e

num_units=$(tree-info exp/chain_e2e/tree/tree | grep "num-pdfs" | cut -d" " -f2)
seed=3407

if [ $stage -le 25 ]; then
  ## sbatch won't wait, just exit after sbatching 
  ## and then come back like every three days till its done.
  sbatch local/chain/run_training_parallel_6gpu.sh \
    --py_script "local/chain_e2e/sb-train-lfmmi-e2e-conformer.py" \
    --hparams "hyperparams/chain/Conformer-I-e2e.yaml"
  exit
fi

if [ $stage -le 27 ]; then
  # There's a hack for num_units
  #local/chain/decode.sh --datadir data/dev_clean \
  #  --acwt 1.0 --post-decode-acwt 10.0 \
  #  --hparams "hyperparams/chain/Conformer-I-e2e.yaml" \
  #  --num_units 2256 \
  #  --tree exp/chain_e2e/tree \
  #  --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
  #  --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
  #  --decodedir "exp/chain_e2e/Conformer-I/3407-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I/3407-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 28 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I/3407-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/test_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I/3407-${num_units}units/decode_test_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 29 ]; then
  ## sbatch won't wait, just exit after sbatching 
  ## and then come back like every three days till its done.
  sbatch local/chain/run_training_parallel_2gpu.sh \
    --py_script "local/chain_e2e/sb-train-lfmmi-e2e-conformer-finetune.py" \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml"
  exit
fi


if [ $stage -le 30 ]; then
  # There's a hack for num_units
  local/chain/decode.sh --datadir data/dev_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 31 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/test_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --num_units 2256 \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_bpe.5000.varikn \
    --py_script "local/chain_e2e/sb-test-lfmmi-e2e-conformer.py" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_other_bpe.5000.varikn_acwt1.0"
fi







# Let's get into the other LMs
if [ $stage -le 32 ]; then
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain_e2e/tree \
    --num_units 2256 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --stage 2 --posteriors_from "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_clean_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_clean_3gram_pruned_char_acwt1.0" \
    --graphdir exp/chain_e2e/graph/graph_3gram_pruned_char
  local/chain/decode.sh --datadir data/dev_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain_e2e/tree \
    --num_units 2256 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --stage 2 --posteriors_from "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_other_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_other_3gram_pruned_char_acwt1.0" \
    --graphdir exp/chain_e2e/graph/graph_3gram_pruned_char
fi

if [ $stage -le 33 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_clean exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_clean_3gram_pruned_char_acwt1.0 \
    "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_clean_4gram_pruned_char_acwt1.0"
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_other exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_other_3gram_pruned_char_acwt1.0 \
    "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_dev_other_4gram_pruned_char_acwt1.0"
fi


# Let's get into the other LMs
if [ $stage -le 34 ]; then
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain_e2e/tree \
    --num_units 2256 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --stage 2 --posteriors_from "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_clean_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_clean_3gram_pruned_char_acwt1.0" \
    --graphdir exp/chain_e2e/graph/graph_3gram_pruned_char
  local/chain/decode.sh --datadir data/test_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain_e2e/tree \
    --num_units 2256 \
    --hparams "hyperparams/chain/Conformer-I-finetune-e2e.yaml" \
    --stage 2 --posteriors_from "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_other_bpe.5000.varikn_acwt1.0" \
    --decodedir "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_other_3gram_pruned_char_acwt1.0" \
    --graphdir exp/chain_e2e/graph/graph_3gram_pruned_char
fi

if [ $stage -le 35 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_clean exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_clean_3gram_pruned_char_acwt1.0 \
    "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_clean_4gram_pruned_char_acwt1.0"
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_other exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_other_3gram_pruned_char_acwt1.0 \
    "exp/chain_e2e/Conformer-I-finetune/3407-${num_units}units/decode_test_other_4gram_pruned_char_acwt1.0"
fi

