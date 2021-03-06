#!/usr/bin/env bash
#SBATCH -J ktd
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem-per-cpu 4096
#SBATCH -C intel

#SBATCH --workdir=/homedtic/stotaro/ktd
#SBATCH -o /homedtic/stotaro/ktd/jobs/%N.%J.ktd.out # STDOUT
#SBATCH -e /homedtic/stotaro/ktd/jobs/%N.%J.ktd.err # STDOUT

set -x
module load PyTorch
source /homedtic/stotaro/ktd_env/bin/activate

python -O main.py --bench_id=$1 --agent_id=$2 --seed=$3
