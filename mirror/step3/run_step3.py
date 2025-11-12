import re
import os
import json
import torch
import argparse

from tqdm import tqdm
from os.path import join as pjoin

from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from diffusers import DDIMScheduler

from mirror.src.photomaker import PhotoMakerStableDiffusionXLPipeline

def generate(generator, image_path_list, prompt, negative_prompt, num_steps, start_merge_step):
    input_id_images = [load_image(img_path) for img_path in image_path_list]
    
    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=len(image_path_list),
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
    ).images
    return images

def process_entry(args, generator, entry):
    dialog_idx = '-'.join(entry['idx'].split("-")[:-1])
    turn_idx = entry['idx'].split("-")[-1]
    save_dir = pjoin(args.save_dir, dialog_idx)

    image_path_list = entry['image_path']
    prompt = entry['prompt']
    negative_prompt = entry['negative_prompt'] + ", missing limbs, mutilated"

    images = generate(
        generator=generator,
        image_path_list=image_path_list,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=args.num_steps,
        start_merge_step=args.start_merge_step
    )
    os.makedirs(save_dir, exist_ok=True)
    for src, image in zip(image_path_list, images):
        src_image_name = src.split('/')[-1].split('.')[0]
        resize_image = image.resize((336, 336))
        resize_image.save(pjoin(save_dir, f"{turn_idx}_{src_image_name}.png"))
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="photomaker_prompts/prompt_0.jsonl")
    parser.add_argument("--save_dir", type=str, default="utterfeat/")
    parser.add_argument("--base_model_path", type=str, default="SG161222/RealVisXL_V3.0")
    parser.add_argument("--photomaker_path", type=str, default="TencentARC/PhotoMaker")
    parser.add_argument("--start_merge_step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--style_strength_ratio", type=int, default=20)
    args = parser.parse_args()
    device = "cuda"

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading Model...")
    photomaker_path = hf_hub_download(
        repo_id=args.photomaker_path, 
        filename="photomaker-v1.bin", 
        repo_type="model"
    )
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
        trigger_word="img",
        pm_version= 'v1',
    )
    pipe.id_encoder.to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    args.start_merge_step = int(float(args.style_strength_ratio) / 100 * args.num_steps)
    if args.start_merge_step > 30:
        args.start_merge_step = 30

    print(args)
    print(f"Load Prompts ...")
    utterfeat_prompts = [json.loads(q) for q in open(args.prompt_path, 'r')]

    for i, entry in tqdm(enumerate(utterfeat_prompts), total=len(utterfeat_prompts), desc=f"Generate Facial Expression: {args.prompt_path}"):
        process_entry(args, generator=generator,  entry=entry)


