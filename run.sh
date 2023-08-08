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
  # download the data.  Note: we're using the 100 hour setup for
  # now; later in the script we'll download more and use it to train neural
  # nets.
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    local/download_and_untar.sh $data $data_url $part
  done

fi

if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

fi

if [ $stage -le 3 ]; then
  for part in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 4 ]; then
  # combine the data into two subsets
  # later than mfcc so we get those as well
  utils/combine_data.sh \
    data/train_clean_460 data/train_clean_100 data/train_clean_360

  utils/combine_data.sh \
    data/train_960 data/train_clean_460 data/train_other_500

  utils/combine_data.sh \
    data/dev_all data/dev_clean/ data/dev_other
fi

if [ $stage -le 5 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 29k
  # utterances in the train_clean_100 directory which has 100 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
  utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
  utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k
fi

if [ $stage -le 6 ]; then
  local/prepare_lexicon.sh \
    --extra_texts "data/dev_clean/text data/dev_other/text" \
    data/train_960/ data/local/dict_train_960 data/lang_train
fi

if [ $stage -le 7 ]; then
  if [ ! -d subword-kaldi ]; then
    echo "Need subword-kaldi, cloning"
    git clone https://github.com/aalto-speech/subword-kaldi
  fi

  local/train_lm.sh \
    --BPE_units 5000 \
    --stage 0 \
    --traindata data/train_960 \
    --validdata data/dev_all \
    train data/lang_bpe.5000.varikn
fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_2kshort data/lang_train exp/mono
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_5k data/lang_train exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang_train exp/mono_ali_5k exp/tri1
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_10k data/lang_train exp/tri1 exp/tri1_ali_10k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_10k data/lang_train exp/tri1_ali_10k exp/tri2b
fi

if [ $stage -le 11 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_10k data/lang_train exp/tri2b exp/tri2b_ali_10k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_10k data/lang_train exp/tri2b_ali_10k exp/tri3b

fi

if [ $stage -le 12 ]; then
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_clean_100 data/lang_train \
    exp/tri3b exp/tri3b_ali_clean_100

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train_clean_100 data/lang_train \
                      exp/tri3b_ali_clean_100 exp/tri4b
fi

if [ $stage -le 16 ]; then
  # align the new, combined set, using the tri4b model
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_clean_460 data/lang_train exp/tri4b exp/tri4b_ali_clean_460

  # create a larger SAT model, trained on the 460 hours of data.
  steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
                      data/train_clean_460 data/lang_train exp/tri4b_ali_clean_460 exp/tri5b
fi

if [ $stage -le 18 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/train_960 data/lang_train exp/tri5b exp/tri5b_ali_960

  # train a SAT model on the 960 hour mixed data.  Use the train_quick.sh script
  # as it is faster.
  steps/train_quick.sh --cmd "$train_cmd" \
                       7000 150000 data/train_960 data/lang_train exp/tri5b_ali_960 exp/tri6b

fi

if [ $stage -le 19 ]; then
  $basic_cmd --mem 16G exp/tri6b/graph_bpe.5000.varikn/log/mkgraph.log utils/mkgraph.sh \
    data/lang_bpe.5000.varikn/ exp/tri6b/ exp/tri6b/graph_bpe.5000.varikn
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 \
    exp/tri6b/graph_bpe.5000.varikn data/dev_clean exp/tri6b/decode_dev_clean_bpe.5000.varikn
  steps/decode_fmllr.sh --cmd "$basic_cmd" --nj 8 \
    exp/tri6b/graph_bpe.5000.varikn data/dev_other exp/tri6b/decode_dev_other_bpe.5000.varikn
fi

if [ $stage -le 20 ]; then
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/train_960 data/lang_train exp/tri6b exp/tri6b_ali_960
  steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
    data/dev_all data/lang_train exp/tri6b exp/tri6b_ali_dev_all
fi

if [ $stage -le 21 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang_train/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang_train..."
      echo " ... not sure what to do. Exiting."
      exit 1;
    fi
  else
    cp -r data/lang_train data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 22 ]; then
  local/chain/build_new_tree.sh exp/chain/tree
fi

if [ $stage -le 23 ]; then
  srun --mem 24G --time 1-12:0:0 -c8 \
    local/chain/make_shards.py 100 shards/train_960 \
      --num-proc 8 \
      --wavscp data/train_960/split100/JOB/wav.scp \
      --text data/train_960/split100/JOB/text \
      --aliark "gunzip -c exp/chain/tree/ali.JOB.gz | ali-to-pdf exp/chain/tree/final.mdl ark:- ark:- |"

  srun --mem 6G --time 12:0:0 -c2 \
    local/chain/make_shards.py 8 shards/dev_all \
      --num-proc 2 \
      --wavscp data/dev_all/split8/JOB/wav.scp \
      --text data/dev_all/split8/JOB/text \
      --aliark "gunzip -c exp/chain/tree/ali.valid.JOB.gz | ali-to-pdf exp/chain/tree/final.mdl ark:- ark:- |"
fi

if [ $stage -le 24 ]; then
  local/chain/prepare_graph_clustered.sh
fi

if [ $stage -le 25 ]; then
  local/chain/run_training.sh
fi

if [ $stage -le 26 ]; then
  $basic_cmd --mem 16G exp/chain/graph/graph_bpe.5000.varikn/log/mkgraph.log utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    data/lang_bpe.5000.varikn/ exp/chain/tree exp/chain/graph/graph_bpe.5000.varikn
fi

if [ $stage -le 27 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 28 ]; then
  local/chain/run_training.sh \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml"
fi

if [ $stage -le 29 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J-contd/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other/ \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J-contd/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 30 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J-contd/2602-2256units/decode_test_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/test_other/ \
    --hparams "hyperparams/chain/New-CRDNN-J-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-J-contd/2602-2256units/decode_test_other_bpe.5000.varikn_acwt1.0"
fi

# TODO: Document FF-10 training.

if [ $stage -le 33 ]; then
  local/chain/decode.sh --datadir data/dev_clean \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_dev_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/dev_other \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_dev_other_bpe.5000.varikn_acwt1.0"
fi

if [ $stage -le 34 ]; then
  local/chain/decode.sh --datadir data/test_clean \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_test_clean_bpe.5000.varikn_acwt1.0"
  local/chain/decode.sh --datadir data/test_other \
    --hparams "hyperparams/chain/New-CRDNN-FF-10-contd.yaml" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --decodedir "exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_test_other_bpe.5000.varikn_acwt1.0"
fi
