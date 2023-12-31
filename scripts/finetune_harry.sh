#!/bin/bash
export CUDA_VISIBLE_DEVICES=0  # Set the GPU to use (0 for the first GPU)
# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
#PROMPT_VERSION="llava_llama_2"
#MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path ../LLaVA_backup/checkpoints/vicuna-7b-v1.3  \
    --version $PROMPT_VERSION \
    --data_path Data/ucsf_data/LLaVA_heme_train.json \
    --image_folder Data/ucsf_data/image_folder \
    --vision_tower ../LLaVA_backup/checkpoints/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ../LLaVA_backup/checkpoints/llava-pretrain-vicuna-7b-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ../LLaVA_backup/checkpoints/llava-vicuna-v1-3-7b-just-testingshit \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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


