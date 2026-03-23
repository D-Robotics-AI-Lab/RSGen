export HOME=.chche
accelerate launch --main_process_port=13854 --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 train_ficgen_fgcontrol_dota.py \
    --pretrained_model_name_or_path=/data/vepfs/public/xianbao01.hou/model/sd1-5 \
    --train_data_dir=/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/images \
    --img_patch_path=/data/vepfs/users/xianbao01.hou/CC-Diff/dataset/filter_train/results_obb/img_patch \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --allow_tf32 \
    --checkpointing_steps=15000 \
    --max_train_steps=75000 \
    --learning_rate=8e-5 \
    --max_grad_norm=1 \
    --lr_scheduler=constant --lr_warmup_steps=500 \
    --output_dir=/data/vepfs/public/xianbao01.hou/model/CC-Diff/FICGen/FICGen_dota_train_8e5_75000
