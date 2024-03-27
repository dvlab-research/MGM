#!/bin/bash

CKPT="Mini-Gemini/Mini-Gemini-7B-HD"

CUDA_VISIBLE_DEVICES=0 python -m minigemini.eval.model_vqa_loader \
    --model-path work_dirs/$CKPT \
    --question-file data/MiniGemini-Eval/MME/llava_mme.jsonl \
    --image-folder data/MiniGemini-Eval/MME/MME_Benchmark_release_version \
    --answers-file data/MiniGemini-Eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd data/MiniGemini-Eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
