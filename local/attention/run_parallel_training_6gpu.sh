#!/bin/bash
#SBATCH --mem=216G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu,gpu-nvlink,dgx-spa
#SBATCH --gres=gpu:6
#SBATCH --constraint 'volta'
#SBATCH --cpus-per-task=30
#SBATCH --output="exp/attention/Conformer-logging/slurm-%j.log"

hparams="hyperparams/attention/Conformer-I.yaml"
py_script="local/attention/sb_train_attn_conformer.py"
num_proc=6
master_port=47770  # NOTE! on the same machine, you have to use your own port
# Should upgrade to torch.distributed.run

. path.sh
. parse_options.sh

timesfailed=0
while ! python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port $master_port $py_script --distributed_launch $hparams; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 10 ]; then
    echo "Training crashed, restarting!"
    sleep 3
    # Kill all dangling processes:
    ps aux | grep "$hparams" | grep -v grep | cut -d" " -f 3 | xargs kill
    sleep 10
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

