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

def load_text(path):
    if "rescored_text" in path:
        return load_text_spm(path)
    texts = {}
    with open(path) as fin:
        for line in fin:
            uttid, *text = line.strip().split()
            # HACK!
            uttid = uttid.replace("é", "e")
            texts[uttid] = text
    return texts

def load_text_spm(path):
    texts = {}
    with open(path) as fin:
        for line in fin:
            uttid, *text = line.strip().split()
            # HACK!
            uttid = uttid.replace("é", "e")
            text = "".join(text)
            if text[0] =="▁":
                text = text[1:]
            text = text.replace("▁", " ")
            texts[uttid] = text.split()
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
    if "rescored_text" in leftpath:
        leftpath = f"'local/wer_hyp_filter_spm <{leftpath}|'"
    if "rescored_text" in rightpath:
        rightpath = f"'local/wer_hyp_filter_spm <{rightpath}|'"
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


def run_analysis(refpath, leftpath, rightpath, utt2spkpath):
    # params
    top_k_utts = 250
    top_k_spks = 5
    n_quantiles = 2


    ref = load_text(refpath)
    left = load_text(leftpath)
    right = load_text(rightpath)
    utt2spk = load_utt2spk(utt2spkpath)


    left_details = edit_distance.wer_details_by_utterance(
        ref, 
        left, 
        compute_alignments=True,
        scoring_mode="strict"
    )
    left_by_spk = edit_distance.wer_details_by_speaker(
        left_details,
        utt2spk,
    )
    left_summary = edit_distance.wer_summary(left_details)
    left_top_wer, left_top_empty = edit_distance.top_wer_utts(left_details,top_k=top_k_utts)
    left_top_wer_spk = edit_distance.top_wer_spks(left_by_spk, top_k=top_k_spks)

    right_details = edit_distance.wer_details_by_utterance(
        ref, 
        right, 
        compute_alignments=True,
        scoring_mode="strict"
    )
    right_by_spk = edit_distance.wer_details_by_speaker(
        right_details,
        utt2spk,
    )
    right_summary = edit_distance.wer_summary(right_details)
    right_top_wer, right_top_empty  = edit_distance.top_wer_utts(right_details,top_k=top_k_utts)
    right_top_wer_spk = edit_distance.top_wer_spks(right_by_spk, top_k=top_k_spks)

    ref_quantiles, cutoffs = split_to_quantiles(ref, n=n_quantiles)
    left_quantile_summaries = wer_by_quantiles(ref_quantiles, cutoffs, left)
    right_quantile_summaries = wer_by_quantiles(ref_quantiles, cutoffs, right)
    #print("Left quantile WERs relative to overall left WER:")
    #print([f"{name}: {s['WER']/left_summary['WER']*100.0}" for name, s in left_quantile_summaries.items()])
    #print("Right quantile WERs, again relative:")
    #print([f"{name}: {s['WER']/right_summary['WER']*100.0}" for name, s in right_quantile_summaries.items()])

    #print("Right total WER:", right_summary['WER'])
    #print("Left total WER:",left_summary['WER'])
    diff = right_summary['WER']-left_summary['WER']
    rel_diff = (diff) *100 / right_summary['WER']

    ## START THE LINE:
    line = f"& {format_number(diff)} & {format_relative(rel_diff)}    "
    credibility = format_credib(get_bootci(refpath, leftpath, rightpath))
    #print(credibility)
    if len(credibility) == 4:
        line += f"& {credibility}          "
    elif len(credibility) == 5:
        line += f"& {credibility}         "

    cross_details = edit_distance.wer_details_by_utterance( 
    left,
    right,
    compute_alignments=True,
    scoring_mode="strict"
    ) 
    cross_summary = edit_distance.wer_summary(cross_details)
    line += f"& {cross_summary['SER']:.2f}   "
    wergetter = operator.itemgetter("WER")
    kt_utt = scipy.stats.kendalltau(list(map(wergetter, left_details)), list(map(wergetter, right_details)))
    wergetter = operator.itemgetter("WER")
    kt_spk = scipy.stats.kendalltau(list(map(wergetter, left_by_spk)), list(map(wergetter, right_by_spk)))
    line+= f"& {kt_utt[0]:.2f}  & {kt_spk[0]:.2f} \\\\"
    return line


def deal_with_hmm(path):
    if path.endswith(".txt"):
        return path
    if "rescored_text" in path:
        return path
    if "dev" in path:
        hp_path = path
    elif "test-all" in path:
        hp_path = path.replace("test-all", "dev-all-fixed")
    elif "test2020" in path:
        hp_path = path.replace("test2020", "dev-all-fixed")
    elif "test-2020" in path:
        hp_path = path.replace("test-2020", "dev-all-fixed")
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
        return "data/test_clean/text", "data/test_clean/utt2spk"
    if "test_other" in path:
        return "data/test_other/text", "data/test_other/utt2spk"
    if "dev_clean" in path:
        return "data/dev_clean/text", "data/dev_clean/utt2spk"
    if "dev_other" in path:
        return "data/dev_other/text", "data/dev_other/utt2spk"
    if "test-all" in path:
        return "fin-train20/speechbrain_2015-2020-kevat/data/parl-test-all/text", "fin-train20/speechbrain_2015-2020-kevat/data/parl-test-all/utt2spk"
    if "test2020" in path:
        return "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/text", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/utt2spk"
    if "test-2020" in path:
        return "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/text", "fin-train20/kaldi_2015-2020-kevat/s5/data/parl2020-test/utt2spk"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("leftpath")
    parser.add_argument("rightpath")
    args = parser.parse_args()
    leftpath = deal_with_hmm(args.leftpath)
    rightpath = deal_with_hmm(args.rightpath)
    refpath, utt2spkpath = find_refs(args.leftpath)
    line = run_analysis(refpath, leftpath, rightpath, utt2spkpath)
    print(line)

