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


if [ $stage -le 1 ]; then
  # download the LM resources
  local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 2 ]; then
  workdir=data/local/dict_3gram_pruned_char
  mkdir -p  $workdir
  local/seed_dict.sh --oov_entry "<UNK>" $workdir
  cp data/local/dict_train_960/nonsilence_phones.txt $workdir/
  local/word-list-to-lexicon.py data/local/lm/librispeech-vocab.txt >> $workdir/lexicon.txt
  tmpdir=$(mktemp -d)
  utils/prepare_lang.sh --phone_symbol_table "data/lang_train/phones.txt" $workdir "<UNK>" $tmpdir data/lang_3gram_pruned_char
  rm -r $tmpdir
fi

if [ $stage -le 3 ]; then
	utils/format_lm.sh \
		data/lang_3gram_pruned_char data/local/lm/3-gram.pruned.1e-7.arpa.gz \
		data/local/dict_3gram_grapheme/lexicon.txt data/lang_3gram_pruned_char
fi

if [ $stage -le 4 ]; then
  $basic_cmd --time 1-0:0:0 --mem 128G exp/tri6b/graph_3gram_char/log/mkgraph.log utils/mkgraph.sh \
    data/lang_3gram_pruned_char/ exp/tri6b/ exp/tri6b/graph_3gram_pruned_char
fi

if [ $stage -le 5 ]; then
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 --scoring-opts "--hyp_filtering_cmd cat" \
    exp/tri6b/graph_3gram_pruned_char data/dev_clean exp/tri6b/decode_dev_clean_3gram_pruned_char
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 --scoring-opts "--hyp_filtering_cmd cat" \
    exp/tri6b/graph_3gram_pruned_char data/dev_other exp/tri6b/decode_dev_other_3gram_pruned_char
fi

if [ $stage -le 6 ]; then
  utils/build_const_arpa_lm.sh data/local/lm/4-gram.arpa.gz \
    data/lang_3gram_pruned_char/ data/lang_4gram_char_const
fi

if [ $stage -le 7 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_clean exp/tri6b/decode_dev_clean_3gram_pruned_char exp/tri6b/decode_dev_clean_4gram_char_rescored
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_other exp/tri6b/decode_dev_other_3gram_pruned_char exp/tri6b/decode_dev_other_4gram_char_rescored
fi

# Neural models:
if [ $stage -le 8 ]; then
  $basic_cmd --time 1-0:0:0 --mem 128G exp/chain/graph/graph_3gram_char/log/mkgraph.log utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_3gram_pruned_char/ exp/chain/tree exp/chain/graph/graph_3gram_pruned_char
fi

# NOTE: run run.sh first
if [ $stage -le 9 ]; then
  local/chain/decode.sh \
    --stage 2 --posteriors_from "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0/" \
    --decodedir "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_3gram_pruned_char_acwt1.0" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char" 
  local/chain/decode.sh \
    --datadir "data/dev_other" \
    --stage 2 --posteriors_from "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0/" \
    --decodedir "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_other_3gram_pruned_char_acwt1.0" \
    --graphdir "exp/chain/graph/graph_3gram_pruned_char" 
fi

if [ $stage -le 10 ]; then
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_clean exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_3gram_pruned_char_acwt1.0 \
    exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_4gram_char_rescored_acwt1.0
  steps/lmrescore_const_arpa.sh --scoring-opts "--hyp_filtering_cmd cat" \
    --cmd "$basic_cmd" data/lang_3gram_pruned_char/ data/lang_4gram_char_const \
    data/dev_other exp/chain/New-CRDNN-J/2602-2256units/decode_dev_other_3gram_pruned_char_acwt1.0 \
    exp/chain/New-CRDNN-J/2602-2256units/decode_dev_other_4gram_char_rescored_acwt1.0
fi
