#!/usr/bin/env bash

#SBATCH -J launcher
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --workdir=/homedtic/stotaro/ktd

set -x
mkdir -p jobs
sh ./cmd/clean.sh
module load PyTorch

python -O cmd/launcher.py
