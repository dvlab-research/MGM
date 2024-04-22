#!/bin/bash
PRETRAIN_NAME=MGM-7B-Pretrain
FINETUNE_NAME=MGM-7B
AUX_SIZE=768
DROP_PATH=False
# delete --hostfile hostfile and change --per_device_train_batch_size if trained on single machine

deepspeed --hostfile hostfile \
    mgm/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version plain \
    --data_path ./data/MGM-Pretrain/mgm_pretrain.json \
    --image_folder ./data/MGM-Pretrain \
    --vision_tower model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_size_aux $AUX_SIZE \
    --drop_path $DROP_PATH \
    --bf16 True \
    --output_dir ./work_dirs/$PRETRAIN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


deepspeed --hostfile hostfile \
    mgm/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path model_zoo/LLM/vicuna/7B-V1.5 \
    --version v1 \
    --data_path ./data/MGM-Finetune/mgm_instruction.json \
    --image_folder ./data/MGM-Finetune \
    --vision_tower model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --pretrain_mm_mlp_adapter ./work_dirs/$PRETRAIN_NAME/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_size_aux $AUX_SIZE \
    --drop_path $DROP_PATH \
    --bf16 True \
    --output_dir ./work_dirs/$FINETUNE_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
