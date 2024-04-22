"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial

from mgm.constants import WORKER_HEART_BEAT_INTERVAL
from mgm.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from mgm.model.builder import load_pretrained_model
from mgm.mm_utils import process_images, load_image_from_base64, tokenizer_image_token
from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread
try:
    from diffusers import StableDiffusionXLPipeline
except:
    print('please install diffusers==0.26.3')

try:
    from paddleocr import PaddleOCR
except:
    print('please install paddleocr following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md')

import io
import base64

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device, use_flash_attn=False):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device, use_flash_attn=use_flash_attn)
        # self.is_multimodal = 'llava' in self.model_name.lower()
        self.is_multimodal = True

        if hasattr(self.model.config, 'image_size_aux'):
            if not hasattr(self.image_processor, 'image_size_raw'):
                self.image_processor.image_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = self.model.config.image_size_aux
            self.image_processor.crop_size['width'] = self.model.config.image_size_aux
            self.image_processor.size['shortest_edge'] = self.model.config.image_size_aux

        # ocr model
        self.ocr_model = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")

        # diffusion model
        max_gpu_index = torch.cuda.device_count() - 1
        device_last = torch.device(f'cuda:{max_gpu_index}')
        print(torch.cuda.device_count(), '++++++', device_last)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        ).to(device=device_last)

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=30)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def add_content(self, prompt, new_content):
        if '[INST]' in prompt:
            split_index = prompt.rfind(' [/INST]')
        elif '<|im_end|>' in prompt:
            split_index = prompt.rfind('<|im_end|>')
        else:
            split_index = prompt.rfind('###Assistant:')
        left_prompt = prompt[:split_index]
        right_prompt = prompt[split_index:]
        prompt = left_prompt + new_content + right_prompt
        return prompt

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        gen_image = params.get("gen_image", False)
        use_ocr = params.get("use_ocr", False)
        num_image_tokens = 0

        if gen_image:
            prompt = self.add_content(prompt, ' <GEN>')
        print(prompt)

        if images is not None and len(images) > 0 and self.is_multimodal:  # len(images) = 1
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]

                # add OCR tokens
                if use_ocr:
                    str_in_image = ''
                    for image in images:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format)
                        img_byte_arr = img_byte_arr.getvalue()
                        result = self.ocr_model.ocr(img_byte_arr, cls=True) 
                        
                        if result[0] is not None:
                            result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
                            if len(result) > 0:
                                str_in_image += ', '.join(result)
                    # print('OCR Token: ' + str_in_image)
                    if len(str_in_image) > 0:
                        prompt = self.add_content(prompt, '\nReference OCR Token: ' + str_in_image + '\n')

                image_tensor = process_images(images, image_processor, model.config)

                image_grid = getattr(model.config, 'image_grid', 1)
                if hasattr(model.config, 'image_size_aux'):
                    raw_shape = [image_processor.image_size_raw['height'] * image_grid,
                                 image_processor.image_size_raw['width'] * image_grid]
                    image_tensor_aux = image_tensor 
                    image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                                   size=raw_shape,
                                                                   mode='bilinear',
                                                                   align_corners=False) # # torch.Size([1, 3, 336, 336])
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
                    image_tensor = image_tensor.unsqueeze(0)

                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                image_tensor_aux = image_tensor_aux.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                image_tensor = None
            image_args = {"images": image_tensor, "images_aux": image_tensor_aux}
        else:
            image_tensor = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return

        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"
        torch.cuda.empty_cache()

        if gen_image and "<h>" in generated_text and "</h>" in generated_text:
            # common_neg_prompt = "blur, lowres, bad anatomy, bad hands, cropped, worst quality"
            common_neg_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
            prompt = generated_text.split("<h>")[1].split("</h>")[0]
            # yield json.dumps({"text": prompt, "error_code": 0}).encode() + b"\0"

            output_img = self.pipe(prompt, negative_prompt=common_neg_prompt).images[0]
            buffered = io.BytesIO()
            output_img.save(buffered, format='JPEG')
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            torch.cuda.empty_cache()

            generated_text = generated_text.split("<h>")[0] + '\n' + 'Prompt: ' + prompt + '\n'
            yield json.dumps({"text": generated_text, "image": img_b64_str, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device,
                         use_flash_attn=args.use_flash_attn)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")