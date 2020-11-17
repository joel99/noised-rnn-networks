#!/bin/bash
#SBATCH --job-name=noisy-rnns
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 3
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --account=overcap
#SBATCH --output=slurm_logs/run-%j.out
#SBATCH --error=slurm_logs/run-%j.err

if [[ $# -eq 1 ]]
then
    echo "Here we go"
    python -u run.py --run-type train --exp-config configs/$1.yaml
elif [[ $# -ge 2 ]]
# Simply add the "-s" flag in order to sweep a directory (will run sequentially)
then
    echo ${@:2}
    python -u run.py --run-type train --exp-config configs/$1.yaml ${@:2}
# elif [[ $# -eq 3 ]]
# then
#     python -u run.py --run-type train --exp-config configs/$1.yaml --ckpt-path $1.$3.pth MODEL.GRAPH_FILE data/configs/graphs/$2
else
    echo "Expected args <variant> (ckpt)"
fi