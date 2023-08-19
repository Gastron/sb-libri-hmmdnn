#!/usr/bin/env python3
# This script just creates the figure, there's no CLI

import speechbrain as sb
from speechbrain.utils import edit_distance
import speechbrain.dataio.wer as wer_io
import numpy as np
import pathlib
import re
import matplotlib.pyplot as plt
import cycler
import statistics
import bisect
import scipy.stats
import operator
from collections import Counter

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 10}
plt.rc('font', **font)


sty_cycle = cycler.cycler(c=['tab:blue','tab:orange','tab:green']) + \
            cycler.cycler(ls=["-",":","--"])
sty_cycle = iter(sty_cycle)

max_streak=4

def load_text(path):
    texts = {}
    with open(path) as fin:
        for line in fin:
            uttid, *text = line.strip().split()
            # HACK!
            uttid = uttid.replace("é", "e")
            texts[uttid] = text
    return texts

def deal_with_hmm(path):
    if path.endswith(".txt"):
        return path
    elif not "chain" in path:
        raise ValueError("Don't know how to deal with" + path)
    if "dev" in path:
        hp_path = path
    else:
        hp_path = path.replace("test", "dev")
    wipf = hp_path +"/scoring_kaldi/wer_details/wip" 
    lmwtf = hp_path +"/scoring_kaldi/wer_details/lmwt" 
    with open(wipf) as fi:
        wip = fi.read().strip()
    with open(lmwtf) as fi:
        lmwt = fi.read().strip()
    return path + f"/scoring_kaldi/penalty_{wip}/{lmwt}.txt"

def analyse_edit_streaks(ref, hmm, aed):
    hmm_counter = Counter()
    aed_counter = Counter()
    for uttid, reftext in ref.items():
        hmmtext = hmm[uttid]
        aedtext = aed[uttid]
        hmm_ali = edit_distance.alignment(edit_distance.op_table(reftext, hmmtext))
        aed_ali = edit_distance.alignment(edit_distance.op_table(reftext, aedtext))       
        hmm_streak = 0
        for edit, ri, hi in hmm_ali:
            if edit == "=":
                if hmm_streak > 0:
                    hmm_counter[hmm_streak]+=1
                hmm_streak = 0
            else:
                hmm_streak+=1
        if hmm_streak > 0:
            hmm_counter[hmm_streak]+=1
        aed_streak = 0
        for edit, ri, hi in aed_ali:
            if edit == "=":
                if aed_streak > 0:
                    aed_counter[aed_streak]+=1
                aed_streak = 0
            else:
                aed_streak+=1
        if aed_streak > 0:
            aed_counter[aed_streak]+=1
    return hmm_counter, aed_counter


def calculate_ratios(ratios_to_edit, test_ref, hmm_hyp, aed_hyp, max_streak=4):
    hmm_edit_streaks, aed_edit_streaks = analyse_edit_streaks(test_ref, hmm_hyp, aed_hyp)
    for streak in range(1,max_streak+1):
        hmm_e = hmm_edit_streaks[streak]
        aed_e = aed_edit_streaks[streak]
        ratios_to_edit[streak].append(aed_e/hmm_e)

ratios_crdnn_trans = {streak_len: [] for streak_len in range(1,max_streak+1)}
ratios_w2v2_trans = {streak_len: [] for streak_len in range(1,max_streak+1)}
ratios_conformer_s_trans = {streak_len: [] for streak_len in range(1,max_streak+1)}
ratios_conformer_l_trans = {streak_len: [] for streak_len in range(1,max_streak+1)}

# 1. Libri Test Clean:
test_ref_path = "librispeech/exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_test_clean_bpe.5000.varikn_acwt1.0/scoring_kaldi/test_filt.txt"
test_ref = load_text(test_ref_path)
hmm_crdnn_trans_hyp_path = "librispeech/exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_test_clean_bpe.5000.varikn_acwt1.0/scoring_kaldi/penalty_0.5/10.txt"
aed_crdnn_trans_hyp_path = "librispeech/exp/attention/CRDNN-FF-10-contd/2602-5000units/text_test_clean_beam8_cov3.0_eos1.3_temp1.5_noattnshift.txt"
hmm_w2v2_trans_hyp_path = "librispeech/exp/chain/New-W2V2-F/2602-2240units/decode_test_clean_bpe.5000.varikn_acwt1.0/scoring_kaldi/penalty_0.5/10.txt"
aed_w2v2_trans_hyp_path = "librispeech/exp/attention/New-W2V2-F/2602-5000units/text_test_clean_beam12_cov3.0_eos1.3_temp1.0_noattnshift.txt"
hmm_crdnn_trans_hyp = load_text(hmm_crdnn_trans_hyp_path)
hmm_w2v2_trans_hyp = load_text(hmm_w2v2_trans_hyp_path)
aed_crdnn_trans_hyp = load_text(aed_crdnn_trans_hyp_path)
aed_w2v2_trans_hyp = load_text(aed_w2v2_trans_hyp_path)

hmm_conformer_s_path = deal_with_hmm("exp/chain/Conformer-I-small/3407-2256units/decode_test_clean_bpe.5000.varikn_acwt1.0/")
aed_conformer_s_path = "exp/attention/Conformer-I-small/3407-5000units/text_test_clean_beam66_temp1.15_ctc0.4_ckpts10.txt"
hmm_conformer_l_path = deal_with_hmm("exp/chain/Conformer-I/3407-2256units/decode_test_clean_bpe.5000.varikn_acwt1.0/")
aed_conformer_l_path = "exp/attention/Conformer-I/3407-5000units/text_test_clean_beam66_temp1.15_ctc0.4_ckpts10.txt"
hmm_conformer_s_hyp = load_text(hmm_conformer_s_path)
hmm_conformer_l_hyp = load_text(hmm_conformer_l_path)
aed_conformer_s_hyp = load_text(aed_conformer_s_path)
aed_conformer_l_hyp = load_text(aed_conformer_l_path)

calculate_ratios(ratios_crdnn_trans, test_ref, hmm_crdnn_trans_hyp, aed_crdnn_trans_hyp)
calculate_ratios(ratios_w2v2_trans, test_ref, hmm_w2v2_trans_hyp, aed_w2v2_trans_hyp)
calculate_ratios(ratios_conformer_s_trans, test_ref, hmm_conformer_s_hyp, aed_conformer_s_hyp)
calculate_ratios(ratios_conformer_l_trans, test_ref, hmm_conformer_l_hyp, aed_conformer_l_hyp)


# 2. Libri Test Other:
test_ref_path = "librispeech/exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_test_other_bpe.5000.varikn_acwt1.0/scoring_kaldi/test_filt.txt"
test_ref = load_text(test_ref_path)
hmm_crdnn_trans_hyp_path = "librispeech/exp/chain/New-CRDNN-FF-10-fixed-contd/2602-2256units/decode_test_other_bpe.5000.varikn_acwt1.0/scoring_kaldi/penalty_0.0/13.txt"
aed_crdnn_trans_hyp_path = "librispeech/exp/attention/CRDNN-FF-10-contd/2602-5000units/text_test_other_beam24_cov3.0_eos1.3_temp1.5_noattnshift.txt"
hmm_w2v2_trans_hyp_path = "librispeech/exp/chain/New-W2V2-F/2602-2240units/decode_test_other_bpe.5000.varikn_acwt1.0/scoring_kaldi/penalty_0.5/11.txt"
aed_w2v2_trans_hyp_path = "librispeech/exp/attention/New-W2V2-F/2602-5000units/text_test_other_beam12_cov3.0_eos1.3_temp1.0_noattnshift.txt"
hmm_crdnn_trans_hyp = load_text(hmm_crdnn_trans_hyp_path)
hmm_w2v2_trans_hyp = load_text(hmm_w2v2_trans_hyp_path)
aed_crdnn_trans_hyp = load_text(aed_crdnn_trans_hyp_path)
aed_w2v2_trans_hyp = load_text(aed_w2v2_trans_hyp_path)

hmm_conformer_s_path = deal_with_hmm("exp/chain/Conformer-I-small/3407-2256units/decode_test_other_bpe.5000.varikn_acwt1.0/")
aed_conformer_s_path = "exp/attention/Conformer-I-small/3407-5000units/text_test_other_beam66_temp1.15_ctc0.4_ckpts10.txt"
hmm_conformer_l_path = deal_with_hmm("exp/chain/Conformer-I/3407-2256units/decode_test_other_bpe.5000.varikn_acwt1.0/")
aed_conformer_l_path = "exp/attention/Conformer-I/3407-5000units/text_test_other_beam66_temp1.15_ctc0.4_ckpts10.txt"
hmm_conformer_s_hyp = load_text(hmm_conformer_s_path)
hmm_conformer_l_hyp = load_text(hmm_conformer_l_path)
aed_conformer_s_hyp = load_text(aed_conformer_s_path)
aed_conformer_l_hyp = load_text(aed_conformer_l_path)

calculate_ratios(ratios_crdnn_trans, test_ref, hmm_crdnn_trans_hyp, aed_crdnn_trans_hyp)
calculate_ratios(ratios_w2v2_trans, test_ref, hmm_w2v2_trans_hyp, aed_w2v2_trans_hyp)
calculate_ratios(ratios_conformer_s_trans, test_ref, hmm_conformer_s_hyp, aed_conformer_s_hyp)
calculate_ratios(ratios_conformer_l_trans, test_ref, hmm_conformer_l_hyp, aed_conformer_l_hyp)

# 3. FP 16
test_ref_path = "fin-train20/speechbrain_2015-2020-kevat/exp/mtl-am/New-CRDNN-J-contd/2602-2328units/decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/test_filt.txt"
test_ref = load_text(test_ref_path)

hmm_crdnn_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat/exp/mtl-am/New-CRDNN-J-contd/2602-2328units/decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/penalty_0.5/10.txt"
aed_crdnn_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat_e2e/exp/MWER/CRDNN-E-contd/2602-1750units/text_test-all_beam4_cov3.0_eos1.2_temp2.0_noattnshift.txt"
hmm_w2v2_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat/exp/mtl-am/New-W2V2-F/2602-2480units/decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/penalty_0.5/10.txt"
aed_w2v2_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat_e2e/exp/MWER/W2V2-F/2602-1750units/text_test-all_beam4_cov3.0_eos1.3_temp2.0_noattnshift.txt"
hmm_crdnn_trans_hyp = load_text(hmm_crdnn_trans_hyp_path)
hmm_w2v2_trans_hyp = load_text(hmm_w2v2_trans_hyp_path)
aed_crdnn_trans_hyp = load_text(aed_crdnn_trans_hyp_path)
aed_w2v2_trans_hyp = load_text(aed_w2v2_trans_hyp_path)

calculate_ratios(ratios_crdnn_trans, test_ref, hmm_crdnn_trans_hyp, aed_crdnn_trans_hyp)
calculate_ratios(ratios_w2v2_trans, test_ref, hmm_w2v2_trans_hyp, aed_w2v2_trans_hyp)

# 4. FP 20
test_ref_path = "fin-train20/speechbrain_2015-2020-kevat/exp/mtl-am/New-CRDNN-J-contd/2602-2328units/decode_parl-test2020_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/test_filt.txt"
test_ref = load_text(test_ref_path)

hmm_crdnn_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat/exp/mtl-am/New-CRDNN-J-contd/2602-2328units/decode_parl-test2020_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/penalty_0.5/10.txt"
aed_crdnn_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat_e2e/exp/MWER/CRDNN-E-contd/2602-1750units/text_test2021_beam4_cov3.0_eos1.3_temp2.0_noattnshift.txt"
hmm_w2v2_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat/exp/mtl-am/New-W2V2-F/2602-2480units/decode_parl-test2020_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/penalty_0.5/10.txt"
aed_w2v2_trans_hyp_path = "fin-train20/speechbrain_2015-2020-kevat_e2e/exp/MWER/W2V2-F/2602-1750units/text_test2021_beam4_cov3.0_eos1.3_temp2.0_noattnshift.txt"
hmm_crdnn_trans_hyp = load_text(hmm_crdnn_trans_hyp_path)
hmm_w2v2_trans_hyp = load_text(hmm_w2v2_trans_hyp_path)
aed_crdnn_trans_hyp = load_text(aed_crdnn_trans_hyp_path)
aed_w2v2_trans_hyp = load_text(aed_w2v2_trans_hyp_path)

calculate_ratios(ratios_crdnn_trans, test_ref, hmm_crdnn_trans_hyp, aed_crdnn_trans_hyp)
calculate_ratios(ratios_w2v2_trans, test_ref, hmm_w2v2_trans_hyp, aed_w2v2_trans_hyp)


fig, ((ax_crdnn, ax_w2v2),(ax_conf_s, ax_conf_l)) = plt.subplots(2,2,layout="constrained",figsize=[6.4,9.2])
ratio_values_crdnn = list(zip(*list(ratios_crdnn_trans.values())))
ratio_values_w2v2 = list(zip(*list(ratios_w2v2_trans.values())))
ratio_values_conf_s = list(zip(*list(ratios_conformer_s_trans.values())))
ratio_values_conf_l = list(zip(*list(ratios_conformer_l_trans.values())))
datasets = [ "Libri Test Clean", "Libri Test Other","FP Test16", "FP Test20"]
datasets_conf = ["Libri Test Clean", "Libri Test Other"]

crdnn_plots = []
for values in ratio_values_crdnn:
    crdnn_plots.append(ax_crdnn.scatter(range(1,5), values))
w2v2_plots = []
for values in ratio_values_w2v2:
    w2v2_plots.append(ax_w2v2.scatter(range(1,5), values))
conf_s_plots = []
for values in ratio_values_conf_s:
    conf_s_plots.append(ax_conf_s.scatter(range(1,5), values))
conf_l_plots = []
for values in ratio_values_conf_l:
    conf_l_plots.append(ax_conf_l.scatter(range(1,5), values))

ax_crdnn.set_xticks([1,2,3,4])
ax_w2v2.set_xticks([1,2,3,4])
ax_crdnn.set_ylim([0.3,1.9])
ax_w2v2.set_ylim([0.3,1.9])
ax_conf_s.set_xticks([1,2,3,4])
ax_conf_l.set_xticks([1,2,3,4])
ax_conf_s.set_ylim([0.3,1.9])
ax_conf_l.set_ylim([0.3,1.9])

ax_crdnn.legend(crdnn_plots, datasets)
ax_w2v2.legend(w2v2_plots, datasets, loc="upper left")
ax_conf_s.legend(crdnn_plots, datasets_conf, loc="lower left")
ax_conf_l.legend(w2v2_plots, datasets_conf, loc="lower left")
ax_conf_s.set_xlabel("Streak Length")
ax_conf_l.set_xlabel("Streak Length")
ax_crdnn.set_ylabel("AED:HMM Number of Streaks Ratio")
ax_conf_s.set_ylabel("AED:HMM Number of Streaks Ratio")
ax_crdnn.set_title("CRDNN")
ax_w2v2.set_title("wav2vec 2.0")
ax_conf_s.set_title("Conformer Small")
ax_conf_l.set_title("Conformer Large")

fig.savefig("edit_streaks.png")



