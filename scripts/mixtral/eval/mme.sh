#!/bin/bash
CKPT="Mini-Gemini/Mini-Gemini-8x7B"

python -m minigemini.eval.model_vqa_loader \
    --model-path work_dirs/$CKPT \
    --question-file data/MiniGemini-Eval/MME/llava_mme.jsonl \
    --image-folder data/MiniGemini-Eval/MME/MME_Benchmark_release_version \
    --answers-file data/MiniGemini-Eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode mistral_instruct


cd data/MiniGemini-Eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
