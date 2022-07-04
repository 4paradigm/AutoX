#!/usr/bin/env bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS=$1
PORT=${PORT:-29500}

PYTHONPATH=".":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT AutoTest.py --launcher pytorch ${@:4}
