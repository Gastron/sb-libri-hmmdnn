#!/bin/bash
#SBATCH --mem=72G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu,gpu-nvlink,dgx-spa
#SBATCH --gres=gpu:2
#SBATCH --constraint 'volta'
#SBATCH --cpus-per-task=10
#SBATCH --output="exp/attention/Conformer-logging/slurm-%j.log"

hparams="hyperparams/attention/Conformer-I2.yaml"
py_script="local/attention/sb_train_attn_conformer.py"
num_proc=2
master_port=47720  # NOTE! on the same machine, you have to use your own port
# Should upgrade to torch.distributed.run

. path.sh
. parse_options.sh

timesfailed=0
while ! python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port $master_port $py_script --distributed_launch $hparams; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 10 ]; then
    echo "Training crashed, restarting!"
    sleep 10
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

