#!/usr/bin/env bash

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=$1
GPUS=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.run \
--nnodes=$NNODES \
--node_rank=$NODE_RANK \
--nproc_per_node=$GPUS \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
$(dirname "$0")/../evaluate.py --cfg-path $CONFIG ${@:3}