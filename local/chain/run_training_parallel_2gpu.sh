#!/bin/bash
#SBATCH --mem=72G
#SBATCH --time=3-0:0:0
#SBATCH --partition=gpu,gpu-nvlink,dgx-spa
#SBATCH --gres=gpu:2
#SBATCH --constraint 'volta'
#SBATCH --cpus-per-task=10
#SBATCH --output="exp/chain/Conformer-logging/slurm-%j.log"

hparams="hyperparams/chain/Conformer-I.yaml"
treedir="exp/chain/tree/"
py_script="local/chain/sb-train-mtl-conformer.py"
num_proc=2
master_port=47776  # NOTE! on the same machine, you have to use your own port
# Should upgrade to torch.distributed.run

. path.sh
. parse_options.sh

num_units=$(tree-info $treedir/tree | grep "num-pdfs" | cut -d" " -f2)

timesfailed=0
while ! python -m torch.distributed.launch --nproc_per_node=$num_proc --master_port $master_port $py_script --distributed_launch $hparams --num_units $num_units --find_unused_parameters; do
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

