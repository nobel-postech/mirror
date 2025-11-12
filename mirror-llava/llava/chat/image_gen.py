import re
import os
import torch
import random
import requests

from tqdm import tqdm
from os.path import join as pjoin

from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from diffusers import DDIMScheduler

from llava.chat.photomaker import PhotoMakerStableDiffusionXLPipeline

IMG_VERIFICATION_URL = os.getenv("IMG_VERIFICATION_URL", "")
IMG_DESCRIPTION_URL = os.getenv("IMG_DESCRIPTION_URL", "")

def extract_text_in_parentheses(text):
    result = re.findall(r'\[(.*?)\]', text)
    return result

class ImageGenerator:
    def __init__(self, 
                photomaker_path="TencentARC/PhotoMaker",
                base_model_path="SG161222/RealVisXL_V3.0",
                save_dir="./images",
                num_steps=50,
                seed=19,
                style_strength_ratio=20
            ):
        self.num_steps = num_steps
        self.start_merge_step = int(float(style_strength_ratio) / 100 * self.num_steps)
        if self.start_merge_step > 30:
            self.start_merge_step = 30       
            
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server_url = f"{IMG_VERIFICATION_URL}"
        
        self.prompt_gen = PromptGenerator()

        self.load_model(photomaker_path=photomaker_path,
                        base_model_path=base_model_path)
        
    def set_seed(self, seed):
        self.seed = seed

    def verify_image(self, image_path, base_image_path, client_statement):
        # Prepare the files to send to the Flask server
        with open(image_path, 'rb') as img_file:
            with open(base_image_path, 'rb') as base_file:
                files = {
                    'image': img_file,
                    'base_image': base_file
                }
                # Form data for the request
                data = {'statement': client_statement}
                # Send the POST request to the Flask server
                response = requests.post(self.server_url, files=files, data=data)
                return response.json()

    def load_model(self, 
                   photomaker_path,
                   base_model_path):
        photomaker_path = hf_hub_download(
            repo_id=photomaker_path, 
            filename="photomaker-v1.bin", 
            repo_type="model"
        )
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)
        
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",
            pm_version= 'v1',
        )
        pipe.id_encoder.to(self.device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()
        
        self.pipe = pipe
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
    
    def generate(self, image_path_list, prompt, negative_prompt):
        input_id_images = [load_image(img_path) for img_path in image_path_list]
        images = self.pipe(
            prompt=prompt,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=len(image_path_list),
            num_inference_steps=self.num_steps,
            start_merge_step=self.start_merge_step,
            generator=self.generator,
        ).images
        return images
            
    
    def run(self, origin, gender, dialogue, statement, dialog_idx, turn_idx):
        save_dir = pjoin(self.save_dir, dialog_idx)

        image_path_list = [origin]
        prompts = self.prompt_gen.get_prompt(gender, dialogue, statement)
        prompt, negative_prompt = prompts['prompt'], prompts['negative_prompt']
        
        try_num = 10
        for _ in range(try_num):
            images = self.generate(
                image_path_list=image_path_list,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
            os.makedirs(save_dir, exist_ok=True)
            assert len(image_path_list) == 1

            src = image_path_list[0]
            image = images[0]
            src_image_name = src.split('/')[-1].split('.')[0]
            save_path = pjoin(save_dir, f"{turn_idx}_{src_image_name}.png")

            resize_image = image.resize((336, 336))
            resize_image.save(save_path)
            
            is_ok = self.verify_image(image_path=save_path, base_image_path=src, client_statement=extract_text_in_parentheses(statement)[0])
            if is_ok: break
            
            self.set_seed(random.choice([42, 19, 24, 18]))
        return prompt, negative_prompt, save_path


class PromptGenerator:
    def __init__(self):
        self.server_url = f"{IMG_VERIFICATION_URL}"

    def get_prompt(self, gender, dialogue, utter, **kwargs):
        data = {'history': dialogue, 'client_utt': utter}
        response = requests.post(self.server_url, data=data)
        gen_prompts = self.parse_description(response)

        output = {
            'prompt': f"portrait photo of a {gender.lower().strip()} img, perfect face, natural skin, high detail, {gen_prompts['prompt']}",
            'negative_prompt': f"nsfw, lowres, bad anatomy, bad hands, grayscale photograph, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, {gen_prompts['negative_prompt']}"
            }
        return output

    def parse_description(self, text):
        try:
            text = text.strip().replace("\n\n", "\n")
            pattern_facial_exp = r"Facial Expression Description[\s]*:[\s\-\t\n]*(.+?)(?=\nContrasting Facial Expression Description|$)"
            pattern_contrasting_exp = r"Contrasting Facial Expression Description[\s]*:[\s\-\t\n]*(.+)$"

            facial_match = re.search(pattern_facial_exp, text, re.DOTALL)
            contrasting_match = re.search(pattern_contrasting_exp, text, re.DOTALL)

            prompt = facial_match.group(1).strip() if facial_match else None
            negative_prompt = contrasting_match.group(1).strip() if contrasting_match else None
                    
            return {
                'prompt': prompt,
                'negative_prompt': negative_prompt
            }
        except Exception as e:
            print(f"[TEXT] {text}")
            raise e