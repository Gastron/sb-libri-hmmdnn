#!/usr/bin/env/python3
"""Finnish Parliament ASR
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
sys.path.append("local/")
import pathlib

logger = logging.getLogger(__name__)

# Brain class for speech recognition training
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        wavs, wav_lens = batch.wav
        tokens_bos, _ = batch.tokens_bos
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs = self.modules.env_corrupt(wavs, wav_lens)

        feats = self.modules.wav2vec2(wavs)
        if self.hparams.subsampling == 2:
            pass
        elif self.hparams.subsampling == 3:
            feats = torch.repeat_interleave(feats,2,dim=1)[:,::self.hparams.subsampling,:]
        elif self.hparams.subsampling == 4:
            feats = feats[:,::2,:]

        encoded_signal = self.modules.encoder(feats)

        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, batch.wav.lengths
        )

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}
        #p_seq = predictions["seq_logprobs"]
        #_, max_indices = torch.sort(p_seq, dim=2, descending=True)
        #for timestep, indices in enumerate(max_indices[0]):
        #    print("Time:", timestep)
        #    for i, ind in enumerate(indices[:2]):
        #        print("\tTop", i, self.hparams.tokenizer.id_to_piece(ind.item()), p_seq[0,timestep,ind].exp())
        #import sys; sys.exit()

        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_signal)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        elif stage == sb.Stage.VALID:
            predictions["tokens"], _ = self.hparams.valid_search(
                encoded_signal, batch.wav.lengths
            )
        elif stage == sb.Stage.TEST:
            predictions["tokens"], _ = self.hparams.test_search(
                encoded_signal, batch.wav.lengths
            )

        return predictions

    def is_ctc_active(self, stage):
        """Check if CTC is currently active.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Compute sequence loss against targets with EOS
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Add ctc loss if necessary. The total cost is a weighted sum of
        # ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, tokens_lens = batch.tokens
            loss_ctc = self.hparams.ctc_cost(
                predictions["ctc_logprobs"], tokens, batch.wav.lengths, tokens_lens
            )
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            specials = [self.hparams.bos_index, self.hparams.eos_index, self.hparams.unk_index]
            predictions["tokens"] = [
                    [token for token in pred if token not in specials]
                    for pred in predictions["tokens"]
            ]
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.trn]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.__key__, predicted_words, target_words)
            self.cer_metric.append(batch.__key__, predicted_words, target_words)

        return loss

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "wav2vec_opt", self.wav2vec_optimizer
                )

        for modulename in getattr(self.hparams, "frozen_modules", []):
            getattr(self.modules, modulename).requires_grad_(False)
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        should_step = self.step % self.hparams.grad_accumulation_factor == 0
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        (loss / self.hparams.grad_accumulation_factor).backward()

        if should_step:
            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()
        
            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr_model)
            old_lr_w2v, new_lr_w2v = self.hparams.lr_annealing_wav2vec(stage_stats["WER"])
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_w2v)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_model": old_lr_model, "lr_w2v": old_lr_w2v},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                num_to_keep=getattr(self.hparams, "ckpts_to_keep", 1)
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            if hasattr(self.hparams, "decode_text_file"):
                with open(self.hparams.decode_text_file, "w") as fo:
                    for utt_details in self.wer_metric.scores:
                        print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if getattr(self.hparams, "avg_ckpts", 1) > 1:
            ckpts = self.checkpointer.find_checkpoints(
                    max_key=max_key,
                    min_key=min_key,
                    max_num_checkpoints=self.hparams.avg_ckpts
            )
            model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    ckpts, "model" 
            )
            self.hparams.model.load_state_dict(model_state_dict)
            self.checkpointer.save_checkpoint(name=f"AVERAGED-{self.hparams.avg_ckpts}")


def dataio_prepare(hparams):
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

    def tokenize(sample):
        text = sample["trn"]
        # quick hack for one sample in text of test2021:
        text = text.replace(" <NOISE>", "")
        fulltokens = torch.LongTensor(
                [hparams["bos_index"]] + hparams["tokenizer"].encode(text) + [hparams["eos_index"]]
        )
        sample["tokens"] = fulltokens[1:-1]
        sample["tokens_bos"] = fulltokens[:-1]
        sample["tokens_eos"] = fulltokens[1:]
        return sample
    
    traindata = (
            wds.WebDataset(hparams["trainshards"])
            .decode()
            .rename(trn="transcript.txt", wav="audio.pth")
            .map(tokenize)
            .repeat()
            .then(
                sb.dataio.iterators.dynamic_bucketed_batch,
                **hparams["dynamic_batch_kwargs"]
            )
    )
    if "valid_dynamic_batch_kwargs" in hparams:
        validdata = (
                wds.WebDataset(hparams["validshards"])
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .then(
                    sb.dataio.iterators.dynamic_bucketed_batch,
                    drop_end=False,
                    **hparams["valid_dynamic_batch_kwargs"]
                )
        )
    else:
        validdata = (
                wds.WebDataset(hparams["validshards"])
                .decode()
                .rename(trn="transcript.txt", wav="audio.pth")
                .map(tokenize)
                .batched(
                    batchsize=hparams["validbatchsize"], 
                    collation_fn=sb.dataio.batch.PaddedBatch,
                    partial=True
                )
        )

    normalizer = sb.dataio.preprocess.AudioNormalizer()
    def normalize_audio(sample):
        signal = sample["wav"]
        samplerate = sample["meta"]["samplerate"]
        sample["wav"] = normalizer(signal, samplerate)
        sample["meta"]["samplerate"] = normalizer.sample_rate 
        return sample

    datas = {"train": traindata, "valid": validdata, }
    
    return datas

    




if __name__ == "__main__":
    import os
    print("SLURM_STEP_GPUS", os.environ.get("SLURM_STEP_GPUS"))
    print("SLURM_JOB_GPUS", os.environ.get("SLURM_JOB_GPUS"))

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Pretrain if defined:
    if "pretrainer" in hparams:
        ckpt_finder_kwargs = hparams.get("ckpt_finder_kwargs", {"min_key": "WER"})
        ckpt = hparams["ckpt_finder"].find_checkpoint(**ckpt_finder_kwargs)
        hparams["pretrainer"].collect_files(ckpt.path)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    train_loader_kwargs = hparams.get("train_loader_kwargs", {})
    train_loader_kwargs.setdefault("batch_size", None)
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs = train_loader_kwargs,
        valid_loader_kwargs = hparams.get("valid_loader_kwargs", {"batch_size": None})
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = asr_brain.evaluate(
        test_set=datasets[hparams["test_data_id"]],
        min_key="WER",
        test_loader_kwargs = hparams.get("test_loader_kwargs", {"batch_size": None})
    )
