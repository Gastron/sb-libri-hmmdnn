# LibriSpeech experiments for HMM/DNN vs. E2E Attention comparison

Both the experiments use model configurations that were 

## Librispeech HMM / DNN recipe with SpeechBrain for neural networks

GMMs and some data prep copied from https://github.com/kaldi-asr/kaldi/tree/master/egs/librispeech/s5 which is under Apache 2.0 licence

## Librispeech Attention models with SpeechBrain

Note that this uses the same shards as the HMM/DNN training so you have to run the HMM/DNN recipe up until the point where the shards get prepped to start this.



