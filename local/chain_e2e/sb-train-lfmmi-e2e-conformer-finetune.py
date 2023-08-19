#!/usr/bin/env/python3
""" HMM/DNN ASR
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import webdataset as wds
from glob import glob
import io
import torchaudio
import tqdm
from pychain import ChainGraph, ChainGraphBatch 
import simplefst
import pathlib

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Brain class for speech recognition training
class LFMMIAM(sb.Brain):

    def __init__(self, train_fsts={}, threadpool_workers=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_fsts = train_fsts
        self.executor = ThreadPoolExecutor(max_workers = threadpool_workers)

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.wav
        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)
        # forward modules
        src = self.modules.CNN(feats)
        encoded = self.modules.Transformer(
            src, wav_lens,
        )

        lfmmi_out = self.modules.lfmmi_lin_out(encoded)
        return lfmmi_out

    def load_graph(self, uttid):
        try:
            fstpath, offset = self.train_fsts[uttid]
            return ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)
        except:
            return None

    def compute_objectives(self, predictions, batch, stage):
        lfmmi_out = predictions
        # Get the grahps:
        if stage == sb.Stage.TRAIN:
            futures = []
            for uttid in batch.__key__:
                futures.append(self.executor.submit(self.load_graph, uttid))
            graphs = []
            for future in futures:
                result = future.result()
                graphs.append(result)
                if result is None:
                    raise ValueError("Empty Graph I GUESS")
        else:
            graphs = batch.graph
        num_transitions = list(map(self.hparams.transgetter, graphs))
        output_lengths = (lfmmi_out.shape[1] * batch.wav.lengths).int().cpu()
        max_num_states = max(map(self.hparams.stategetter, graphs))
        numerator_graphs = ChainGraphBatch(
                graphs,
                max_num_transitions=max(num_transitions),
                max_num_states=max_num_states
        )
        lfmmi_loss = self.hparams.chain_loss(lfmmi_out, output_lengths, numerator_graphs)
        output_norm_loss = torch.linalg.norm(lfmmi_out,dim=2).mean()
        loss = lfmmi_loss + output_norm_loss*self.hparams.outnorm_scale
        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"] }, 
                min_keys=["loss"],
                num_to_keep=10,
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start(max_key=max_key, min_key=min_key)

        max_num_checkpoints=getattr(self.hparams, "average_n_ckpts", 1)
        if max_num_checkpoints > 1:
            ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
                max_num_checkpoints=max_num_checkpoints,
            )
            ckpt = sb.utils.checkpoints.average_checkpoints(
                ckpts, recoverable_name="model", device=self.device
            )

            self.hparams.model.load_state_dict(ckpt, strict=True)
            self.hparams.model.eval()
            print(f"Loaded the average of {len(ckpts)} best checkpoints")
        else:
            print(f"Loaded the checkpoint from {self.hparams.epoch_counter.current}")

    def fit_batch(self, batch):

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()


def move_numfsts_to_local_tmp(fstdir, tmpdir):
    """Copies the chain numerator FSTs onto a local disk"""
    fstdir = pathlib.Path(fstdir)
    tmpdir = pathlib.Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    sb.utils.superpowers.run_shell(f"rsync --update {fstdir}/num.*.ark {tmpdir}/")

def find_numfsts_in_local_tmp(fstdir, tmpdir):
    fstdir = pathlib.Path(fstdir)
    tmpdir = pathlib.Path(tmpdir)
    numfsts = {}
    for scpfile in fstdir.glob("num.*.scp"):
        with open(scpfile) as fin:
            for line in fin:
                uttid, data = line.strip().split()
                # HACK: WebDataset cannot handle periods in uttids:
                uttid = uttid.replace(".", "")
                arkpath, offset = data.split(":")
                arkpath = pathlib.Path(arkpath)
                newpath = tmpdir / arkpath.name
                numfsts[uttid] = (str(newpath), int(offset))
    return numfsts

def dataio_prepare(hparams, numfsts):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.


    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Dictionary containing "train", "valid", and "test" keys mapping to 
        WebDataset datasets dataloaders for them.
    """
    def load_valid_fst(sample, numfsts=numfsts):
        uttid = sample["__key__"]
        fstpath, offset = numfsts["valid"][uttid]
        sample["graph"] = ChainGraph(simplefst.StdVectorFst.read_ark(fstpath, offset), log_domain=True)
        return sample

    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(wav="audio.pth")
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    validdata = (
            wds.WebDataset(hparams["validshards"])
            .decode()
            .rename(wav="audio.pth")
            .map(load_valid_fst, handler=wds.warn_and_continue)
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                drop_end=False,
                **hparams["valid_dynamic_batch_kwargs"],
            )
    )
    return {"train": traindata, "valid": validdata}




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

    # Copy numerator FSTs to local drive:
    run_on_main(move_numfsts_to_local_tmp,
            args=[hparams["numfstdir"], hparams["numfsttmpdir"]],
    )
    run_on_main(move_numfsts_to_local_tmp,
            args=[hparams["valid_numfstdir"], hparams["valid_numfsttmpdir"]],
    )
    # fetch new paths:
    numfsts = {}
    numfsts["train"] = find_numfsts_in_local_tmp(hparams["numfstdir"], hparams["numfsttmpdir"])
    numfsts["valid"] = find_numfsts_in_local_tmp(hparams["valid_numfstdir"], hparams["valid_numfsttmpdir"])

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams, numfsts)
    # read valid data into memory:
    #datasets["valid"] = torch.utils.data.DataLoader(
    #        list(iter(datasets["valid"])),
    #        batch_size=None
    #)

    pt_kwargs = {}
    if hasattr(hparams, "test_max_key"):
        pt_kwargs["max_key"] = hparams.test_max_key
    elif hasattr(hparams, "test_min_key"):
        pt_kwargs["min_key"] = hparams.test_min_key

    if "device" in run_opts:
        device = run_opts["device"]
    elif "device" in hparams:
        device = hparams["device"]
    else:
        device = "cpu"
    pt_checkpointer = hparams["pretrain_checkpointer"]
    ckpt = pt_checkpointer.find_checkpoint(**pt_kwargs)
    print(f"Pretrain first from {ckpt.path}")
    pt_checkpointer.load_checkpoint(ckpt, device=device)
    if hparams.get("pretrain_n_ckpts", 1) > 1:
        ckpts = pt_checkpointer.find_checkpoints(
            max_num_checkpoints=hparams["pretrain_n_ckpts"],
            **pt_kwargs
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=device
        )
        hparams["model"].load_state_dict(ckpt, strict=True)
        print(f"Loaded the average of {len(ckpts)} best checkpoints")
    hparams["modules"]["CNN"].requires_grad_(False)

    # Trainer initialization
    asr_brain = LFMMIAM(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        train_fsts = numfsts["train"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    train_loader_kwargs = hparams["train_dataloader_opts"]
    train_loader_kwargs.setdefault("batch_size", None)
    #valid_loader_kwargs = hparams["valid_dataloader_opts"]
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs = train_loader_kwargs,
        valid_loader_kwargs = hparams.get("valid_loader_kwargs", {"batch_size": None})
    )
    
