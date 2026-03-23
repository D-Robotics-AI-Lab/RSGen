#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29555}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}


#python3 tools/train.py configs/fsod/yolox_s_100e_coco1mix_512p.py --auto-scale-lr
# bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg.py 8 --auto-scale-lr  --work-dir work_dirs/hbbox_ori_cnet_xs_finetune_wo_mask_wo_gated_train_w_edge2edge_controlnet