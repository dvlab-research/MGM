import argparse
import torch
import os
import json
from tqdm import tqdm

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def create_one_query(problem, shot_num, shot_type, use_caption):


    ### [1] Demo prompt
    demo_prompt = ""

    ### [2] Test query
    # problem info
    question = problem['question']
    unit = problem['unit']
    choices = problem['choices']
    # caption = problem['caption']
    precision = problem['precision']
    question_type = problem['question_type']
    answer_type = problem['answer_type']

    # hint
    if shot_type == 'solution':
        if question_type == "multi_choice":
            assert answer_type == "text"
            hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
        else:
            assert answer_type in ["integer", "float", "list"]
            if answer_type == "integer":
                hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."
            
            elif answer_type == "float" and precision == 1:
                hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."
            
            elif answer_type == "float" and precision == 2:
                hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."
            
            elif answer_type == "list":
                hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
    else:
        assert shot_type == 'code'
        hint_text = "Hint: Please generate a python code to solve the problem"

    # question
    question_text = f"Question: {question}"
    if unit:
        question_text += f" (Unit: {unit})"

    # choices
    if choices:
        # choices: (A) 1.2 (B) 1.3 (C) 1.4 (D) 1.5
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A')+i)}) {choice}")
        choices_text = "\n".join(texts)
    else:
        choices_text = ""

    # prompt
    if shot_type == 'solution':
        prompt = "Solution: "
    else:
        assert shot_type == 'code'
        prompt = "Python code: "
    
    elements = [hint_text, question_text, choices_text]
    test_query = "\n".join([e for e in elements if e != ""])

    ### [3] Final query
    query = demo_prompt + "\n\n" + test_query
    query = query.strip()
    return query


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           load_8bit=args.load_8bit)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = [dict(pid=pid, info=qs) for pid, qs in questions.items()]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if os.path.exists(answers_file):
        file = open(answers_file, "r")
        pred_contents = [json.loads(line) for line in file]
        done_pid = [sample['pid'] for sample in pred_contents]
    else:
        done_pid = []
    ans_file = open(answers_file, "a")

    for i, line in enumerate(tqdm(questions)):
        idx = line['pid']
        info = line['info']
        if idx in done_pid:
            continue

        qs = create_one_query(
            problem = info, 
            shot_num = 0,
            shot_type = 'solution',
            use_caption = False,
        )
        query = qs

        if 'image' in info:
            image_file = info["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            
            if hasattr(model.config, 'image_size_aux'):
                if not hasattr(image_processor, 'image_size_raw'):
                    image_processor.image_size_raw = image_processor.crop_size.copy()
                image_processor.crop_size['height'] = model.config.image_size_aux
                image_processor.crop_size['width'] = model.config.image_size_aux
                image_processor.size['shortest_edge'] = model.config.image_size_aux
            
            image_tensor = process_images([image], image_processor, model.config)[0]
            
            image_grid = getattr(model.config, 'image_grid', 1)
            if hasattr(model.config, 'image_size_aux'):
                raw_shape = [image_processor.image_size_raw['height'] * image_grid, 
                            image_processor.image_size_raw['width'] * image_grid]
                image_tensor_aux = image_tensor
                image_tensor = torch.nn.functional.interpolate(image_tensor[None], 
                                                            size=raw_shape, 
                                                            mode='bilinear', 
                                                            align_corners=False)[0]
            else:
                image_tensor_aux = []

            if image_grid >= 2:            
                raw_image = image_tensor.reshape(3, 
                                                image_grid,
                                                image_processor.image_size_raw['height'],
                                                image_grid,
                                                image_processor.image_size_raw['width'])
                raw_image = raw_image.permute(1, 3, 0, 2, 4)
                raw_image = raw_image.reshape(-1, 3,
                                            image_processor.image_size_raw['height'],
                                            image_processor.image_size_raw['width'])
                
                if getattr(model.config, 'image_global', False):
                    global_image = image_tensor
                    if len(global_image.shape) == 3:
                        global_image = global_image[None]
                    global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[image_processor.image_size_raw['height'],
                                                                image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                    # [image_crops, image_global]
                    raw_image = torch.cat([raw_image, global_image], dim=0)
                image_tensor = raw_image.contiguous()
            
            images = image_tensor[None].to(dtype=model.dtype, device='cuda', non_blocking=True)
            images_aux = image_tensor_aux[None].to(dtype=model.dtype, device='cuda', non_blocking=True) if len(image_tensor_aux)>0 else None
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            images = None
            images_aux = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        terminators = tokenizer.eos_token_id
        if "llama_3" in args.conv_mode:
            terminators = [terminators, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                images_aux=images_aux,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=terminators,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        info['query'] = query
        info['response'] = outputs
        ans_file.write(json.dumps(info) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)