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

# bash tools/dist_test.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg.py /data/vepfs/users/xianbao01.hou/CC-Diff/eval/mmdetection/work_dirs/ficgen_gen_ori_obbox_rsvg_wo_fgcnet_wo_edge2edge_train/latest.pth 8 --eval bbox --out results_wo.pkl
# bash tools/dist_test.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg.py /data/vepfs/users/xianbao01.hou/CC-Diff/eval/mmdetection/work_dirs/ficgen_gen_ori_obbox_rsvg_w_fgcnet_wo_edge2edge_train/latest.pth 8 --eval bbox --out results_w.pkl
# bash tools/dist_test.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_rsvg_test_dota.py /data/vepfs/users/xianbao01.hou/CC-Diff/eval/mmdetection/work_dirs/migc_w_ours_w_edge2edge_fix_loss_1800_train/epoch_12.pth 8 --eval bbox
# bash tools/dist_test.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_dota.py /data/vepfs/users/xianbao01.hou/CC-Diff/eval/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_dota/epoch_12.pth 8 --eval bbox
