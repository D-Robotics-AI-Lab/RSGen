#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
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
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 42 \
    --launcher pytorch ${@:3}


# cd /data/vepfs/users/xianbao01.hou/CC-Diff/eval/mmrotate && bash tools/dist_train.sh configs/s2anet/s2anet_r50_fpn_3x_hrsc_le135.py 8 --work-dir /data/vepfs/public/xianbao01.hou/new/ccdiff/mmrotate/hrsc_ori_s2anet_2
# export MMCV_WITH_OPS=1  export TORCH_CUDA_ARCH_LIST="8.9" L20

# bash tools/dist_train.sh configs/s2anet/s2anet_r50_fpn_3x_hrsc_le135.py 8 --work-dir /data/vepfs/public/xianbao01.hou/new/ccdiff/mmrotate/FICGen_HRSC_train_ref_w_ours_wo_edge2edge_fft_hf_out_train_seed42_1real_1syn \ 
# --cfg-options data.train.ann_file='/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/HRSC2016_Combined_w_fgcnet_wo_edge2edge_epoch100_200/ImageSets/trainval.txt' \
#             data.train.ann_subdir='/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/HRSC2016_Combined_w_fgcnet_wo_edge2edge_epoch100_200/Annotations' \
#             data.train.img_subdir='/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/HRSC2016_Combined_w_fgcnet_wo_edge2edge_epoch100_200/AllImages' \
#             data.train.img_prefix='/data/vepfs/public/xianbao01.hou/new/ccdiff/ficgen_data/HRSC2016_Combined_w_fgcnet_wo_edge2edge_epoch100_200/AllImages'   