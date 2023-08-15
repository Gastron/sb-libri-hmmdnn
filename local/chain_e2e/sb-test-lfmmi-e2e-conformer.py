#!/usr/bin/env/python3
"""Finnish Parliament ASR"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import kaldi_io
import tqdm
from types import SimpleNamespace
sys.path.append("./local")
sys.path.append("./local/chain")
from make_shards import wavscp_to_output, segments_to_output
import pathlib

def setup(hparams, run_opts):
    """ Kind of mimics what Brain does """
    if "device" in run_opts:
        device = run_opts["device"]
    elif "device" in hparams:
        device = hparams["device"]
    else:
        device = "cpu"
    print("Device is:", device)
    if "cuda" in device:
        torch.cuda.set_device(int(device[-1]))
    modules = torch.nn.ModuleDict(hparams["modules"]).to(device)
    hparams = SimpleNamespace(**hparams)
    if hasattr(hparams, "checkpointer"):
        kwargs = {}
        if hasattr(hparams, "test_max_key"):
            kwargs["max_key"] = hparams.test_max_key
        elif hasattr(hparams, "test_min_key"):
            kwargs["min_key"] = hparams.test_min_key
        ckpt = hparams.checkpointer.find_checkpoint(**kwargs)
        hparams.checkpointer.load_checkpoint(ckpt)
        if getattr(hparams, "average_n_ckpts", 1) > 1:
            ckpts = hparams.checkpointer.find_checkpoints(
                max_num_checkpoints=hparams.average_n_ckpts,
                **kwargs
            )
            ckpt = sb.utils.checkpoints.average_checkpoints(
                ckpts, recoverable_name="model", device=device
            )
            hparams.model.load_state_dict(ckpt, strict=True)
            hparams.model.eval()
            print(f"Loaded the average of {len(ckpts)} best checkpoints")
        else:
            epoch = hparams.epoch_counter.current
            print("Loaded checkpoint from epoch", epoch, "at path", ckpt.path)
    modules.eval()
    return modules, hparams, device

def count_scp_lines(scpfile):
    lines = 0
    with open(scpfile) as fin:
        for _ in fin:
            lines += 1
    return lines

def run_test(modules, hparams, device):
    testdir = pathlib.Path(hparams.testdir)
    if (testdir / "segments").exists():
        num_utts = count_scp_lines(testdir / "segments")
        data_iter = segments_to_output(testdir / "segments", testdir / "wav.scp")
    else:
        num_utts = count_scp_lines(testdir / "wav.scp")
        data_iter = wavscp_to_output(testdir / "wav.scp")
    with open(hparams.test_probs_out, 'wb') as fo:
        with torch.no_grad():
            for uttid, data in tqdm.tqdm(data_iter, total=num_utts):
                audio = data["audio.pth"].to(device).unsqueeze(0)
                lengths=torch.tensor([1.]).to(device)
                feats = hparams.compute_features(audio)
                normalized = modules.normalize(feats, lengths=lengths,  epoch=1000)
                src = modules.CNN(normalized)
                encoded = modules.Transformer(
                    src, lengths,
                )
                lfmmi_out = modules.lfmmi_lin_out(encoded)
                # HACK if wrong number of outputs was given at first
                lfmmi_out = lfmmi_out[:,:,:hparams.real_num_units]
                if hparams.normalize_out:
                    lfmmi_out = hparams.log_softmax(lfmmi_out)
                kaldi_io.write_mat(fo, lfmmi_out.squeeze(0).cpu().numpy(), key=uttid)
    

if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    modules, hparams, device = setup(hparams, run_opts)
    run_test(modules, hparams, device)

