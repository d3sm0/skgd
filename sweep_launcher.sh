#!/usr/bin/env bash
set -x
PROJECT_NAME='test'
wandb sweep $1 -p $PROJECT_NAME &>sweep_id.log
SWEEP_ID=$(grep -o 'ID: .*' sweep_id.log | cut -f2 --d ' ')
SWEEP_ADDR=d3sm0/$PROJECT_NAME/$SWEEP_ID
echo $SWEEP_ADDR
NUM_PROC=1
for i in $(seq 0 $NUM_PROC)
  do
    echo "starting machine $i for $SWEEP_ADDR"
    wandb agent $SWEEP_ADDR
    sleep 5
  done
