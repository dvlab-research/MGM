import argparse
import torch

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
try:
    from diffusers import StableDiffusionXLPipeline
except:
    print('please install diffusers==0.26.3')

try:
    from paddleocr import PaddleOCR
except:
    print('please install paddleocr following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md')


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()
    
    if args.ocr and args.image_file is not None:
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")
        result = ocr.ocr(args.image_file)   
        str_in_image = ''
        if result[0] is not None:
            result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
            if len(result) > 0:
                str_in_image = ', '.join(result)
                print('OCR Token: ' + str_in_image)
    
    if args.gen:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        ).to("cuda")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    if '8x7b' in model_name.lower():
        conv_mode = "mistral_instruct"
    elif '34b' in model_name.lower():
        conv_mode = "chatml_direct"
    elif '2b' in model_name.lower():
        conv_mode = "gemma"
    else:
        conv_mode = "vicuna_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if args.image_file is not None:
        images = []
        if ',' in args.image_file:
            images = args.image_file.split(',')
        else:
            images = [args.image_file]
        
        image_convert = []
        for _image in images:
            image_convert.append(load_image(_image))
    
        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy()
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux
        
        # Similar operation in model_worker.py
        image_tensor = process_images(image_convert, image_processor, model.config)
    
        image_grid = getattr(model.config, 'image_grid', 1)
        if hasattr(model.config, 'image_size_aux'):
            raw_shape = [image_processor.image_size_raw['height'] * image_grid,
                        image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
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
    
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(model.device, dtype=torch.float16)
    else:
        images = None
        image_tensor = None
        image_tensor_aux = []


    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if args.ocr and len(str_in_image) > 0:
            inp = inp + '\nReference OCR Token: ' + str_in_image + '\n'
        if args.gen:
            inp = inp + ' <GEN>'
        # print(inp, '====')

        if images is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = (DEFAULT_IMAGE_TOKEN + '\n')*len(images) + inp
            conv.append_message(conv.roles[0], inp)
            images = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # add image split string
        if prompt.count(DEFAULT_IMAGE_TOKEN) >= 2:
            final_str = ''
            sent_split = prompt.split(DEFAULT_IMAGE_TOKEN)
            for _idx, _sub_sent in enumerate(sent_split):
                if _idx == len(sent_split) - 1:
                    final_str = final_str + _sub_sent
                else:
                    final_str = final_str + _sub_sent + f'Image {_idx+1}:' + DEFAULT_IMAGE_TOKEN
            prompt = final_str
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        conv.messages[-1][-1] = outputs
        
        if args.gen and '<h>' in outputs and '</h>' in outputs:
            common_neg_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
            prompt = outputs.split("</h>")[-2].split("<h>")[-1]
            output_img = pipe(prompt, negative_prompt=common_neg_prompt).images[0]
            output_img.save(args.output_file)
            print(f'Generate an image, save at {args.output_file}')

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None) # file_0.jpg,file_1.jpg for multi image
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--output-file", type=str, default='generate.png')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)