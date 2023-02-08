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

if [ $stage -le 4 ]; then
  $basic_cmd --time 1-0:0:0 --mem 128G exp/phoneme/tri6b/graph_3gram_phoneme/log/mkgraph.log utils/mkgraph.sh \
    data/lang_phoneme_test_tgmed/ exp/phoneme/tri6b/ exp/phoneme/tri6b/graph_3gram_pruned_phoneme
fi

if [ $stage -le 5 ]; then
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 --scoring-opts "--hyp_filtering_cmd cat" \
    exp/phoneme/tri6b/graph_3gram_pruned_phoneme data/dev_clean exp/phoneme/tri6b/decode_dev_clean_3gram_pruned_phoneme
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 --scoring-opts "--hyp_filtering_cmd cat" \
    exp/phoneme/tri6b/graph_3gram_pruned_phoneme data/dev_other exp/phoneme/tri6b/decode_dev_other_3gram_pruned_phoneme
fi

#if [ $stage -le 6 ]; then
  #utils/build_const_arpa_lm.sh data/local/lm/4-gram.arpa.gz \
  #  data/lang_phoneme_test_tgmed/ data/lang_4gram_phoneme_const
#fi

if [ $stage -le 7 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge/ \
    data/dev_clean exp/phoneme/tri6b/decode_dev_clean_3gram_pruned_phoneme exp/phoneme/tri6b/decode_dev_clean_4gram_phoneme_rescored
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge/ \
    data/dev_other exp/phoneme/tri6b/decode_dev_other_3gram_pruned_phoneme exp/phoneme/tri6b/decode_dev_other_4gram_phoneme_rescored
fi

# Neural models:
if [ $stage -le 8 ]; then
  $basic_cmd --time 1-0:0:0 --mem 128G exp/phoneme/chain/graph/graph_3gram_phoneme/log/mkgraph.log utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_phoneme_test_tgmed/ exp/phoneme/chain/tree exp/phoneme/chain/graph/graph_phoneme_test_tgmed
fi

if [ $stage -le 9 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-phoneme.yaml --expname libriphoneme" \
    --tree exp/phoneme/chain/tree/ \
    --graphdir "exp/phoneme/chain/graph/graph_phoneme_test_tgmed" \
    --decodedir "exp/phoneme/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_phoneme_test_tgmed_acwt1.0" 
  local/chain/decode.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-phoneme.yaml --expname libriphoneme" \
    --tree exp/phoneme/chain/tree/ \
    --graphdir "exp/phoneme/chain/graph/graph_phoneme_test_tgmed" \
    --datadir "data/dev_other" \
    --decodedir "exp/phoneme/chain/New-CRDNN-J/2602-2256units/decode_dev_other_phoneme_test_tgmed_acwt1.0" 
fi

if [ $stage -le 10 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge \
    data/dev_clean exp/phoneme/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_phoneme_test_tgmed_acwt1.0 \
    exp/phoneme/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_4gram_phoneme_rescored_acwt1.0
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge \
    data/dev_other exp/phoneme/chain/New-CRDNN-J/2602-2256units/decode_dev_other_phoneme_test_tgmed_acwt1.0 \
    exp/phoneme/chain/New-CRDNN-J/2602-2256units/decode_dev_other_4gram_phoneme_rescored_acwt1.0
fi

if [ $stage -le 11 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-phoneme-contd.yaml --expname libriphoneme" \
    --tree exp/phoneme/chain/tree/ \
    --graphdir "exp/phoneme/chain/graph/graph_phoneme_test_tgmed" \
    --decodedir "exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_dev_clean_phoneme_test_tgmed_acwt1.0" 
  local/chain/decode.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-phoneme-contd.yaml --expname libriphoneme" \
    --tree exp/phoneme/chain/tree/ \
    --graphdir "exp/phoneme/chain/graph/graph_phoneme_test_tgmed" \
    --datadir "data/dev_other" \
    --decodedir "exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_dev_other_phoneme_test_tgmed_acwt1.0" 
fi

if [ $stage -le 12 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge \
    data/dev_clean exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units//decode_dev_clean_phoneme_test_tgmed_acwt1.0 \
    exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_dev_clean_4gram_phoneme_rescored_acwt1.0
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge \
    data/dev_other exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_dev_other_phoneme_test_tgmed_acwt1.0 \
    exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_dev_other_4gram_phoneme_rescored_acwt1.0
fi

if [ $stage -le 13 ]; then
  local/chain/decode.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-phoneme-contd.yaml --expname libriphoneme" \
    --tree exp/phoneme/chain/tree/ \
    --datadir "data/test_clean" \
    --graphdir "exp/phoneme/chain/graph/graph_phoneme_test_tgmed" \
    --decodedir "exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_test_clean_phoneme_test_tgmed_acwt1.0" 
  local/chain/decode.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-phoneme-contd.yaml --expname libriphoneme" \
    --tree exp/phoneme/chain/tree/ \
    --graphdir "exp/phoneme/chain/graph/graph_phoneme_test_tgmed" \
    --datadir "data/test_other" \
    --decodedir "exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_test_other_phoneme_test_tgmed_acwt1.0" 
fi

if [ $stage -le 14 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge \
    data/test_clean exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units//decode_test_clean_phoneme_test_tgmed_acwt1.0 \
    exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_test_clean_4gram_phoneme_rescored_acwt1.0
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_phoneme_test_tgmed/ data/lang_phoneme_test_fglarge \
    data/test_other exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_test_other_phoneme_test_tgmed_acwt1.0 \
    exp/phoneme/chain/New-CRDNN-J-contd/2602-2096units/decode_test_other_4gram_phoneme_rescored_acwt1.0
fi
