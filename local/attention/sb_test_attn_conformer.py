#!/usr/bin/env/python3
"""
Mostly taken from https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/transformer/train.py
"""

import os
import sys
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import webdataset as wds
import glob
import io
import torchaudio
sys.path.append("local/")
sys.path.append("local/chain")
import pathlib

from sb_train_attn_conformer import ASR, dataio_prepare
from sb_test_only_attn import KaldiData

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prepare(hparams)

    # TODO: No pretraining atm

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    test_dataloader_opts = hparams.get("test_dataloader_opts", hparams["valid_dataloader_opts"])

    # These might be needed if webdataset is not detected
    #train_dataloader_opts.setdefault("batch_size", None)
    #valid_dataloader_opts.setdefault("batch_size", None)

    print("Decoding", hparams["test_data_id"])
    test_data = KaldiData(hparams["test_data_dir"], 
            tokenizer = hparams["tokenizer"],
            bos_index = hparams["bos_index"],
            eos_index = hparams["eos_index"])
    test_loader_kwargs = hparams.get("test_loader_kwargs", {})
    if "collate_fn" not in test_loader_kwargs:
        test_loader_kwargs["collate_fn"] = sb.dataio.batch.PaddedBatch

    test_stats = asr_brain.evaluate(
        test_set=test_data,
        max_key="ACC",
        test_loader_kwargs = test_loader_kwargs
    )
