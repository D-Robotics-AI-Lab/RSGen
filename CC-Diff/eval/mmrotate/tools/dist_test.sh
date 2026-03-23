#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

# bash tools/dist_test.sh configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py /data/vepfs/public/xianbao01.hou/dataset/CC-Diff/eval_mmrotate/ficgen_gen_ori_obbox_dota_wo_fgcnet_wo_edge2edge_train/latest.pth 1 --eval mAP --show-dir /data/vepfs/public/xianbao01.hou/new/ccdiff/vis/ficgen_gen_ori_obbox_dota_wo_fgcnet_wo_edge2edge_train

# bash tools/dist_test.sh \
#     configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py \
#     /data/vepfs/public/xianbao01.hou/dataset/CC-Diff/eval_mmrotate/ficgen_gen_ori_obbox_dota_wo_fgcnet_wo_edge2edge_train/latest.pth \
#     8 \
#     --out results_wo.pkl \
#     --eval mAP

