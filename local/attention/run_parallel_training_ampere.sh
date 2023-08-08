#!/bin/bash
#SBATCH --mem=96G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --output="exp/attention/Conformer-logging/slurm-%j.log"

hparams="hyperparams/attention/Conformer-E.yaml"
py_script="local/attention/sb_train_attn_conformer.py"
num_proc=4
master_port=47770  # NOTE! on the same machine, you have to use your own port
# Should upgrade to torch.distributed.run

. path.sh
. parse_options.sh

timesfailed=0
while ! python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port $master_port $py_script --bfloat16_mix_prec --distributed_launch $hparams; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 10 ]; then
    echo "Training crashed, restarting!"
    sleep 10
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

