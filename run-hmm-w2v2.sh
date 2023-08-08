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

if [ $stage -le 22 ]; then
  local/chain/build_new_tree.sh \
    --frame_subsampling_factor 2 \
    exp/chain/tree2
fi

if [ $stage -le 23 ]; then
  srun --mem 24G --time 1-12:0:0 -c8 \
    local/chain/make_shards.py 100 shards/train_960_sub2 \
      --num-proc 8 \
      --wavscp data/train_960/split100/JOB/wav.scp \
      --text data/train_960/split100/JOB/text \
      --aliark "gunzip -c exp/chain/tree2/ali.JOB.gz | ali-to-pdf exp/chain/tree2/final.mdl ark:- ark:- |"

  srun --mem 6G --time 12:0:0 -c2 \
    local/chain/make_shards.py 8 shards/dev_all_sub2 \
      --num-proc 2 \
      --wavscp data/dev_all/split8/JOB/wav.scp \
      --text data/dev_all/split8/JOB/text \
      --aliark "gunzip -c exp/chain/tree2/ali.valid.JOB.gz | ali-to-pdf exp/chain/tree2/final.mdl ark:- ark:- |"
fi

if [ $stage -le 24 ]; then
  local/chain/prepare_graph_clustered.sh \
    --treedir exp/chain/tree2 \
    --graph exp/chain/graph2
fi

if [ $stage -le 25 ]; then
  local/chain/run_training.sh \
    --treedir "exp/chain/tree2" \
    --py_script "local/chain/sb-train-mtl-w2v2.py" \
    --hparams "hyperparams/chain/New-w2w2-F3.yaml"
fi

if [ $stage -le 26 ]; then
  $basic_cmd --mem 16G exp/chain/graph/graph_bpe.5000.varikn/log/mkgraph.log utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_bpe.5000.varikn/ exp/chain/tree2 exp/chain/graph2/graph_bpe.5000.varikn
fi

if [ $stage -le 27 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree2 \
    --graphdir "exp/chain/graph2/graph_bpe.5000.varikn/" \
    --hparams hyperparams/chain/New-w2w2-F3.yaml \
    --py_script "local/chain/sb-test-w2v2-mtl-avg.py" \
    --decodedir "exp/chain/New-W2V2-F3/2602-2240units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree2 \
    --graphdir "exp/chain/graph2/graph_bpe.5000.varikn/" \
    --hparams hyperparams/chain/New-w2w2-F3.yaml \
    --py_script "local/chain/sb-test-w2v2-mtl-avg.py" \
    --decodedir "exp/chain/New-W2V2-F3/2602-2240units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 28 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree2 \
    --graphdir "exp/chain/graph2/graph_bpe.5000.varikn/" \
    --hparams hyperparams/chain/New-w2w2-F.yaml \
    --py_script "local/chain/sb-test-w2v2-mtl-avg.py" \
    --decodedir "exp/chain/New-W2V2-F/2602-2240units/decode_test_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/test_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree2 \
    --graphdir "exp/chain/graph2/graph_bpe.5000.varikn/" \
    --hparams hyperparams/chain/New-w2w2-F.yaml \
    --py_script "local/chain/sb-test-w2v2-mtl-avg.py" \
    --decodedir "exp/chain/New-W2V2-F/2602-2240units/decode_test_other_bpe.5000.varikn_acwt1.0"
fi


# Other LMs:
if [ $stage -le -30 ]; then
  $basic_cmd --mem 16G exp/chain/graph2/graph_3gram_char/log/mkgraph.log utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_3gram_pruned_char/ exp/chain/tree2 exp/chain/graph2/graph_3gram_pruned_char
fi

if [ $stage -le 29 ]; then
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree2 \
    --hparams hyperparams/chain/New-w2w2-F.yaml \
    --stage 2 --posteriors_from "exp/chain/New-W2V2-F//2602-2240units/decode_dev_clean_bpe.5000.varikn_acwt1.0/" \
    --decodedir "exp/chain/New-W2V2-F/2602-2240units/decode_dev_clean_3gram_pruned_char_acwt1.0-fixt" \
    --graphdir "exp/chain/graph2/graph_3gram_pruned_char" 
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --tree exp/chain/tree2 \
    --hparams hyperparams/chain/New-w2w2-F.yaml \
    --datadir "data/dev_other" \
    --stage 2 --posteriors_from "exp/chain/New-W2V2-F/2602-2240units/decode_dev_other_bpe.5000.varikn_acwt1.0/" \
    --decodedir "exp/chain/New-W2V2-F/2602-2240units/decode_dev_other_3gram_pruned_char_acwt1.0-fixt" \
    --graphdir "exp/chain/graph2/graph_3gram_pruned_char" 
fi

if [ $stage -le 30 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_clean exp/chain/New-W2V2-F/2602-2240units/decode_dev_clean_3gram_pruned_char_acwt1.0-fixt \
    exp/chain/New-W2V2-F/2602-2240units/decode_dev_clean_4gram_char_rescored_acwt1.0-fixt
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_other exp/chain/New-W2V2-F/2602-2240units/decode_dev_other_3gram_pruned_char_acwt1.0-fixt \
    exp/chain/New-W2V2-F/2602-2240units/decode_dev_other_4gram_char_rescored_acwt1.0-fixt
fi


if [ $stage -le 31 ]; then
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams hyperparams/chain/New-w2w2-F.yaml \
    --tree exp/chain/tree2 \
    --datadir "data/test_clean" \
    --stage 2 --posteriors_from "exp/chain/New-W2V2-F//2602-2240units/decode_test_clean_bpe.5000.varikn_acwt1.0/" \
    --decodedir "exp/chain/New-W2V2-F/2602-2240units/decode_test_clean_3gram_pruned_char_acwt1.0-fixt" \
    --graphdir "exp/chain/graph2/graph_3gram_pruned_char" 
  local/chain/decode.sh \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --hparams hyperparams/chain/New-w2w2-F.yaml \
    --tree exp/chain/tree2 \
    --datadir "data/test_other" \
    --stage 2 --posteriors_from "exp/chain/New-W2V2-F/2602-2240units/decode_test_other_bpe.5000.varikn_acwt1.0/" \
    --decodedir "exp/chain/New-W2V2-F/2602-2240units/decode_test_other_3gram_pruned_char_acwt1.0-fixt" \
    --graphdir "exp/chain/graph2/graph_3gram_pruned_char" 
fi

if [ $stage -le 32 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_clean exp/chain/New-W2V2-F/2602-2240units/decode_test_clean_3gram_pruned_char_acwt1.0-fixt \
    exp/chain/New-W2V2-F/2602-2240units/decode_test_clean_4gram_char_rescored_acwt1.0-fixt
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/test_other exp/chain/New-W2V2-F/2602-2240units/decode_test_other_3gram_pruned_char_acwt1.0-fixt \
    exp/chain/New-W2V2-F/2602-2240units/decode_test_other_4gram_char_rescored_acwt1.0-fixt
fi
