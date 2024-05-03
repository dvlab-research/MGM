#!/bin/bash
PRETRAIN_NAME=MGM-8B-LLaMA-3-Pretrain
FINETUNE_NAME=MGM-8B-LLaMA-3-HD
AUX_SIZE=1536
IMAGE_GRID=2
IMAGE_GLOBAL=True

# delete --hostfile hostfile and change --per_device_train_batch_size if trained on single machine

deepspeed --hostfile ../hostfile \
    mgm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path model_zoo/LLM/llama-3/Meta-Llama-3-8B-Instruct \
    --version llama_3 \
    --data_path ./data/MGM-Finetune/mgm_instruction.json \
    --image_folder ./data/MGM-Finetune \
    --vision_tower model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --image_grid $IMAGE_GRID \
    --image_global $IMAGE_GLOBAL \
    --pretrain_mm_mlp_adapter ./work_dirs/$PRETRAIN_NAME/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_size_aux $AUX_SIZE \
    --bf16 True \
    --output_dir ./work_dirs/$FINETUNE_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
