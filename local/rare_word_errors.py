#!/usr/bin/env python3
import speechbrain as sb
from speechbrain.utils import edit_distance
import numpy as np
import pathlib
import re
import statistics
import bisect
import scipy.stats
import operator
from collections import Counter

def load_text(path):
    texts = {}
    with open(path) as fin:
        for line in fin:
            uttid, *text = line.strip().split()
            # HACK!
            uttid = uttid.replace("é", "e")
            texts[uttid] = text
    return texts

def format_number(number):
    """Make a 3 significant digit string from number"""
    nstr = "+" if number >=0 else ""
    nstr += ("{:.1f}" if abs(number) // 10 else "{:.2f}").format(number)
    return nstr

def format_relative(number):
    nstr = "+" if number >=0 else ""
    if abs(number) // 10:
        return nstr + str(round(number)) + "\\% "
    else :
        return nstr + "{:.1f}".format(number) + "\\%"

def format_credib(number):
    return str(round(number)) + "\\%"


def split_to_quantiles(textdict, n):
    quantile_cutoffs = statistics.quantiles((len(v) for v in textdict.values()), n=n)
    quantiles = [{} for _ in range(n)]
    for uttid, text in textdict.items():
        v = len(text)
        q_index = bisect.bisect_left(quantile_cutoffs, v)
        quantiles[q_index][uttid] = text
    quantile_strings = [f"l <= {quantile_cutoffs[0]}"] + \
            [f"{quantile_cutoffs[i-1]} < l <= {quantile_cutoffs[i]}" for i in range(1, n-1)] + \
            [f"{quantile_cutoffs[n-2]} < l"]
    return quantiles, quantile_strings

def wer_by_quantiles(ref_quantiles, cutoffs, hyp_dict):
    summaries_by_quantile = {}
    for ref_quantile, name in zip(ref_quantiles, cutoffs):
        quantile_details = edit_distance.wer_details_by_utterance(ref_quantile, hyp_dict)
        quantile_summary = edit_distance.wer_summary(quantile_details)
        summaries_by_quantile[name] = quantile_summary
    return summaries_by_quantile

def get_bootci(refpath, leftpath, rightpath):
    out, err, code = sb.utils.superpowers.run_shell(f"compute-wer-bootci ark:{refpath} ark:{leftpath} ark:{rightpath}")
    credibility = lines = float(out.decode().split("\n")[2].strip().split()[-1])
    if credibility < 50.0:
        out, err, code = sb.utils.superpowers.run_shell(f"compute-wer-bootci ark:{refpath} ark:{rightpath} ark:{leftpath}")
        credibility = lines = float(out.decode().split("\n")[2].strip().split()[-1])
    return credibility

def load_utt2spk(utt2spkpath):
    utt2spk = {}
    with open(utt2spkpath) as fi:
        for line in fi:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    return utt2spk


def run_analysis(refpath, hyppath, utt2spkpath, trainvocab):
    # params
    top_k_utts = 250
    top_k_spks = 5
    n_quantiles = 2


    ref = load_text(refpath)
    hyp = load_text(hyppath)

    S = 0
    D = 0
    NUM = 0
    for key, text in ref.items():
        if any(word not in trainvocab for word in text):
            ops = edit_distance.op_table(text, hyp[key])
            ali = edit_distance.alignment(ops)
            for step in ali:
                refindex = step[1]
                if refindex is not None and text[refindex] not in trainvocab:
                    if step[0] == "S":
                        S += 1
                    elif step[0] == "D":
                        D += 1
                    NUM+=1
    return S, D, NUM




def deal_with_hmm(path):
    if path.endswith(".txt"):
        return path
    if "dev" in path:
        hp_path = path
    elif "test-all" in path:
        hp_path = path.replace("test-all", "dev-all-fixed")
    elif "test2020" in path:
        hp_path = path.replace("test-all", "dev-all-fixed")
    else:
        hp_path = path.replace("test", "dev")
    wipf = hp_path +"/scoring_kaldi/wer_details/wip" 
    lmwtf = hp_path +"/scoring_kaldi/wer_details/lmwt" 
    with open(wipf) as fi:
        wip = fi.read().strip()
    with open(lmwtf) as fi:
        lmwt = fi.read().strip()
    return path + f"/scoring_kaldi/penalty_{wip}/{lmwt}.txt"

def find_refs(path):
    if "test_clean" in path:
        return "data/test_clean/text", "data/test_clean/utt2spk", "data/train_960/text"
    if "test_other" in path:
        return "data/test_other/text", "data/test_other/utt2spk", "data/train_960/text"
    if "dev_clean" in path:
        return "data/dev_clean/text", "data/dev_clean/utt2spk", "data/train_960/text"
    if "dev_other" in path:
        return "data/dev_other/text", "data/dev_other/utt2spk", "data/train_960/text"
    if "test-all" in path:
        return "fin-train20/speechbrain_2015-2020-kevat/data/parl-test-all/text", "fin-train20/speechbrain_2015-2020-kevat/data/parl-test-all/utt2spk", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2015-2020-train_cleaned/text"
    if "test2020" in path:
        return "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/text", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/utt2spk", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2015-2020-train_cleaned/text"
    if "test2021" in path:
        return "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/text", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/utt2spk", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2015-2020-train_cleaned/text"

def get_vocab(path):
    vocab = Counter()
    with open(path) as fin:
        for line in fin:
            vocab.update(line.strip().split())
    return vocab


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = deal_with_hmm(args.path)
    refpath, utt2spkpath, trainpath = find_refs(args.path)
    trainvocab = get_vocab(trainpath)
    S, D, NUM = run_analysis(refpath, path, utt2spkpath, trainvocab)
    print(f"{S}, {D}, {NUM}, {(S+D)/NUM*100.}")

