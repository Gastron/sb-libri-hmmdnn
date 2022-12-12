#!/bin/bash

cmd="srun --mem 24G --time 2-0:0:0 -c5 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink"
hparams="hyperparams/attention/CRDNN-E.yaml"
py_script="local/attention/sb_train_attn.py"

. path.sh
. parse_options.sh

timesfailed=0
while ! $cmd python $py_script $hparams; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 5 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

