Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models
=============================================================================

Mini-Gemini supports a series of dense and MoE Large Language Models (LLMs) from 2B to 34B with image understanding, reasoning, and generation simultaneously. We build this repo based on LLaVA.

Release
-------
[04/15] 🔥 The Hugging Face demo is available. It's a 13B-HD version, welcome to watch and try.
[03/28] 🔥 Mini-Gemini is coming! We release the paper, demo, code, models, and data for Mini-Gemini!

Contents
--------
- Demo
- Install
- Model
- Preparation
- Train
- Evaluation
- Examples
- Citation
- Acknowledgement
- License
- Introduction to Mini-Gemini
- Motivation
- Key Features
- Usage Examples
- Performance Metrics
- Comparison to Other Models
- Contributing Guidelines
- Future Roadmap
- Community and Support
- License and Legal Information

Demo
----
We provide some selected examples in this section. More examples can be found in our project page. Feel free to try our online demo!

Install
-------
Please follow the instructions below to install the required packages.

NOTE: If you want to use Mini-Gemini-2B, please ensure to install the latest version Transformers (>=4.38.0).

Clone this repository
git clone https://github.com/dvlab-research/MiniGemini.git

Install Package
conda create -n minigemini python=3.10 -y
conda activate minigemini
cd MiniGemini
pip install --upgrade pip # enable PEP 660 support
pip install -e .

Install additional packages for training cases
pip install ninja
pip install flash-attn --no-build-isolation

Model
-----
The framework of Mini-Gemini is conceptually simple: dual vision encoders are utilized to provide low-resolution visual embedding and high-resolution candidates; patch info mining is proposed to conduct patch-level mining between high-resolution regions and low-resolution visual queries; LLM is utilized to marry text with images for both comprehension and generation at the same time.

We provide all our fully finetuned models on Stage 1 and 2 data for Mini-Gemini:

| Model             | LR  | HR  | Base LLM               | Vision Encoder            | Finetuning Data       | Finetuning schedule | Download |
|-------------------|-----|-----|------------------------|---------------------------|-----------------------|----------------------|----------|
| Mini-Gemini-2B   | 336 | 768 | Gemma-2B               | CLIP-L                    | MiniGemini-Instruct   | full_ft-1e           | ckpt     |
| Mini-Gemini-7B   | 336 | 768 | Vicuna-7B-v1.5         | CLIP-L                    | MiniGemini-Instruct   | full_ft-1e           | ckpt     |
| Mini-Gemini-13B  | 336 | 768 | Vicuna-13B-v1.5        | CLIP-L                    | MiniGemini-Instruct   | full_ft-1e           | ckpt     |
| Mini-Gemini-8x7B | 336 | 768 | Mixtral-8x7B-Instruct-v0.1 | CLIP-L                 | MiniGemini-Instruct   | full_ft-1e           | ckpt     |
| Mini-Gemini-34B  | 336 | 768 | Nous-Hermes-2-Yi-34B   | CLIP-L                    | MiniGemini-Instruct   | full_ft-1e           | ckpt     |

Here are the pretrained weights on Stage 1 data only:

| Model             | LR  | HR  | Base LLM               | Vision Encoder            | Pretrain Data          | Finetuning schedule | Download |
|-------------------|-----|-----|------------------------|---------------------------|------------------------|----------------------|----------|
| Mini-Gemini-2B   | 336 | 768 | Gemma-2B               | CLIP-L                    | MiniGemini-Pretrain    | 1e                   | ckpt     |
| Mini-Gemini-7B   | 336 | 768 | Vicuna-7B-v1.5         | CLIP-L                    | MiniGemini-Pretrain    | 1e                   | ckpt     |
| Mini-Gemini-13B  | 336 | 768 | Vicuna-13B-v1.5        | CLIP-L                    | MiniGemini-Pretrain    | 1e                   | ckpt     |
| Mini-Gemini-8x7B | 336 | 768 | Mixtral-8x7B-Instruct-v0.1 | CLIP-L                 | MiniGemini-Pretrain    | 1e                   | ckpt     |
| Mini-Gemini-34B  | 336 | 768 | Nous-Hermes-2-Yi-34B   | CLIP-L                    | MiniGemini-Pretrain    | 1e                   | ckpt     |

Preparation
-----------
Dataset
We provide the processed data for Mini-Gemini training. For model pretraining, please download the following the training image-based data and organize them as:

-> means put the data in the local folder.

LLaVA Images -> data/MiniGemini-Pretrain/images, data/MiniGemini-Finetune/llava/LLaVA-Pretrain/images
ALLaVA Caption -> data/MiniGemini-Pretrain/ALLaVA-4V

For model finetuning, please download the following the instruction data and organize them as:

-> means put the data in the local folder.

COCO train2017 -> data/MiniGemini-Finetune/coco
GQA -> data/MiniGemini-Finetune/gqa
OCR-VQA (we save all files as .jpg) -> data/MiniGemini-Finetune/ocr_vqa
TextVQA (not included for training) -> data/MiniGemini-Finetune/textvqa
VisualGenome part1, VisualGenome part2 -> data/MiniGemini-Finetune/vg
ShareGPT4V-100K -> data/MiniGemini-Finetune/sam, share_textvqa, wikiart, web-celebrity, web-landmark
LAION GPT4V -> data/MiniGemini-Finetune/gpt4v-dataset
ALLaVA Instruction -> data/MiniGemini-Pretrain/ALLaVA-4V
DocVQA -> data/MiniGemini-Finetune/docvqa
ChartQA -> data/MiniGemini-Finetune/chartqa
DVQA -> data/MiniGemini-Finetune/dvqa
AI2D -> data/MiniGemini-Finetune/ai2d

For model evaluation, please follow this link for preparation. We use some extra benchmarks for evaluation. please download the following the training image-based data and organize them as:

-> means put the data in the local folder.

MMMU -> data/MiniGemini-Eval/MMMU
MMB -> data/MiniGemini-Eval/MMB
MathVista -> data/MiniGemini-Eval/MathVista

Please put the pretrained data, finetuned data, and eval data in MiniGemini-Pretrain, MiniGemini-Finetune, and MiniGemini-Eval subset following Structure.

For meta info, please download the following files and organize them as in Structure.

Data file name    | Size
-----------------|------
minigemini_pretrain.json | 1.68 G
minigemini_instruction.json | 1.79 G
minigemini_generation_pure_text.json | 0.04 G

IMPORTANT: minigemini_generation_pure_text.json is a generation-related subset. DO NOT merge it with minigemini_instruction.json as it is already included in it. You may merge this file with your customized LLM/VLM SFT dataset to enable the reasoning generation ability.

Pretrained Weights
We recommend users to download the pretrained weights from the following link CLIP-Vit-L-336, OpenCLIP-ConvNeXt-L, Gemma-2b-it, Vicuna-7b-v1.5, Vicuna-13b-v1.5, Mixtral-8x7B-Instruct-v0.1, and Nous-Hermes-2-Yi-34B , and put them in model_zoo following Structure.

Structure
---------
The folder structure should be organized as follows before training.

MiniGemini
├── minigemini
├── scripts
├── work_dirs
│ ├── Mini-Gemini
│ │ ├── Mini-Gemini-2B
│ │ ├── ...
├── model_zoo
│ ├── LLM
│ │ ├── gemma
│ │ │ ├── gemma-2b-it
│ │ ├── vicuna
│ │ │ ├── 7B-V1.5
│ │ │ ├── 13B-V1.5
│ │ ├── mixtral
│ │ │ ├── Mixtral-8x7B-Instruct-v0.1
│ │ ├── Nous-Hermes-2-Yi-34B
│ ├── OpenAI
│ │ ├── clip-vit-large-patch14-336
│ │ ├── openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup
├── data
│ ├── MiniGemini-Pretrain
│ │ ├── minigemini_pretrain.json
│ │ ├── images
│ │ ├── ALLaVA-4V
│ ├── MiniGemini-Finetune
│ │ ├── minigemini_instruction.json
│ │ ├── llava
│ │ ├── coco
│ │ ├── gqa
│ │ ├── ocr_vqa
│ │ ├── textvqa
│ │ ├── vg
│ │ ├── gpt4v-dataset
│ │ ├── sam
│ │ ├── share_textvqa
│ │ ├── wikiart
│ │ ├── web-celebrity
│ │ ├── web-landmark
│ │ ├── ALLaVA-4V
│ │ ├── docvqa
│ │ ├── chartqa
│ │ ├── dvqa
│ │ ├── ai2d
│ ├── MiniGemini-Eval
│ │ ├── MMMU
│ │ ├── MMB
│ │ ├── MathVista
│ │ ├── ...


Train
-----
Mini-Gemini training consists of two stages: (1) feature alignment stage: bridge the vision and language tokens; (2) instruction tuning stage: teach the model to follow multimodal instructions.

Mini-Gemini is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the per_device_train_batch_size and increase the gradient_accumulation_steps accordingly. Always keep the global batch size the same: per_device_train_batch_size x gradient_accumulation_steps x num_gpus.

Please make sure you download and organize the data following Preparation before training.

NOTE: Please set hostfile for 2 machine training and hostfile_4 for 4 machine training.

If you want to train and finetune Mini-Gemini, please run the following command for Mini-Gemini-7B with image size 336:

bash scripts/llama/train/stage_1_2_full_v7b_336_hr_768.sh


or for Mini-Gemini-13B with image size 336:

bash scripts/llama/train/stage_1_2_full_v13b_336_hr_768.sh

Because we reuse the pre-trained projecter weights from the Mini-Gemini-7B, you can directly use the Mini-Gemini-7B-HD with image size 672 for stage-2 instruction tuning:

bash scripts/llama/train/stage_2_full_v7b_672_hr_1536.sh


Please find more training scripts of gemma, llama, mixtral, and yi in scripts/.

Evaluation
----------
We perform evaluation on several image-based benchmarks. Please download the evaluation data following Preparation and organize them as in Structure.

| Model             | LLM                 | Res. | Link  | TextVQA | MMB  | MME            | MM-Vet | MMMU_val | MMMU_test | MathVista |
|-------------------|---------------------|------|-------|---------|------|----------------|--------|----------|-----------|-----------|
| Mini-Gemini-2B   | Gemma-2B            | 336  | ckpt  | 56.2    | 59.8 | 1341/312       | 31.1   | 31.7     | 29.1      | 29.4      |
| Mini-Gemini-7B   | Vicuna-7B-v1.5      | 336  | ckpt  | 65.2    | 69.3 | 1523/316       | 40.8   | 36.1     | 32.8      | 31.4      |
| Mini-Gemini-13B  | Vicuna-13B-v1.5     | 336  | ckpt  | 65.9    | 68.5 | 1565/322       | 46.0   | 38.1     | 33.5      | 37.0      |
| Mini-Gemini-8x7B | Mixtral-8x7B        | 336  | ckpt  | 68.7    | 69.2 | 1822/329       | 51.0   | 40.5     | 36.6      | 37.6      |
| Mini-Gemini-34B  | Nous-Hermes-2-Yi-34B| 336 | ckpt  | 71.4    | 72.8 | 2083/370       | 55.3   | 42.7     | 39.1      | 42.3      |

Here are the performance metrics of Mini-Gemini on various benchmarks. The model's performance is assessed based on accuracy and other relevant metrics across different datasets and tasks.

Examples
--------
Here are some examples demonstrating the capabilities of Mini-Gemini:

1. Image Captioning:
   Input: An image of a cat sitting on a table.
   Output: "A cat sitting on a table next to a window."

2. Visual Question Answering (VQA):
   Image: ![Sample Image](link_to_sample_image.jpg)
   Question: What color is the cat?
   Answer: The cat is black and white.

3. Text-to-Image Generation:
   Input: "A description of a beach with palm trees and a sunset."
   Output: ![Generated Image](link_to_generated_image.jpg)

Citation
--------
If you find Mini-Gemini useful in your research, please consider citing:
@article{minigemini2024,
title={Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models},
author={Authors},
journal={Journal/Conference},
year={2024}
}


Acknowledgement
---------------
We thank the open-source community for their contributions, especially in the development of the underlying libraries and frameworks.

License
-------
Mini-Gemini is licensed under the MIT License. See the LICENSE file for details.

Introduction to Mini-Gemini
---------------------------
Mini-Gemini is a multimodal model that combines vision and language understanding for various tasks such as image captioning, visual question answering, and text-to-image generation.

Motivation
----------
The motivation behind Mini-Gemini is to explore the potential of multi-modality in language models and harness the synergy between vision and language for enhanced performance in various tasks.

Key Features
------------
- Integration of vision and language understanding.
- Simultaneous comprehension and generation of text and images.
- Support for various benchmarks and evaluation metrics.

Usage Examples
--------------
Mini-Gemini can be used for tasks such as image captioning, visual question answering, and text-to-image generation. It can generate captions for images, answer questions about visual content, and generate images from textual descriptions.

Performance Metrics
-------------------
Mini-Gemini's performance is evaluated based on accuracy, BLEU scores, and other relevant metrics across different datasets and tasks.

Comparison to Other Models
---------------------------
Mini-Gemini outperforms previous models in tasks such as image captioning, visual question answering, and text-to-image generation due to its multi-modal architecture and comprehensive training.

Contributing Guidelines
------------------------
Contributions to Mini-Gemini are welcome! Please follow the guidelines in the CONTRIBUTING.md file.

Future Roadmap
--------------
Future developments of Mini-Gemini may include enhancements to its architecture, support for additional tasks and benchmarks, and optimization for performance and efficiency.

Community and Support
----------------------
Join our community to get support, share ideas, and collaborate on Mini-Gemini-related projects.

License and Legal Information
-----------------------------
Mini-Gemini is licensed under the MIT License. See the LICENSE file for details.

For any legal inquiries or concerns, please contact us at legal@minigemini.org.

For more information, visit our website: https://www.minigemini.org
