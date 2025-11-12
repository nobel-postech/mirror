
import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from ast import literal_eval
from os.path import join as pjoin

from mirror.src.prompts import *

def parse_description(text):
    try:
        prompt, negative_prompt = text.strip().split("\n")
        prompt = prompt.split(":", 1)[1].strip()
        negative_prompt = negative_prompt.split(":", 1)[1].strip()
    
        return {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }
    except Exception as e:
        print(f"[TEXT] {text}")
        raise e
    
def write_photomaker_prompts(llama_results, proc_celeba_df, save_path):
    save_file = open(save_path, 'w')
    for entry in tqdm(llama_results, total=len(llama_results)):
        dialog_idx = '-'.join(entry['custom_id'].split("-")[:4])
        row = entire_df[entire_df['idx'] == dialog_idx]
        if len(row) == 0: continue
        
        identity = row['identity'].iloc[0]
        identity_row = proc_celeba_df[proc_celeba_df['identity'] == identity]
        if 'dominant_gender' in identity_row.iloc[0]:
            gender = identity_row.iloc[0]['dominant_gender']
        else:
            gender = 'female' if identity_row.iloc[0]['celeb_male'] < 0 else 'male'
        image_paths = list(identity_row['img_path']) 
        try:
            prompts = parse_description(entry['response'])
            new_entry = {
                'idx': entry['custom_id'],
                'dialog_idx': dialog_idx,
                'image_path': image_paths,
                'prompt': f"portrait photo of a {gender.lower().strip()} img, perfect face, natural skin, high detail, {prompts['prompt']}",
                'negative_prompt': f"nsfw, lowres, bad anatomy, bad hands, grayscale photograph, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, {prompts['negative_prompt']}"
            }
            json.dump(new_entry, save_file, ensure_ascii=False)
            save_file.write('\n')

        except Exception as e:
            raise e
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_result_path", type=str, default="llama3_8b_result.jsonl")
    parser.add_argument("--celeba_path", type=str, default="proc_celeba.csv")
    parser.add_argument("--save_path", type=str, default="photomaker_prompts/prompt_0.jsonl")
    args = parser.parse_args()

    print(f"Loading Data...")
    entire_df = pd.read_csv(args.data_path, converters={
        'proc_dialogue': literal_eval
    })
    llama_results = [json.loads(q) for q in open(args.llm_result_path, 'r')]

    proc_celeba_df = pd.read_csv(args.celeba_path)
    proc_celeba_df['age_group'] = proc_celeba_df['age'].apply(lambda x: int(round(int(x) / 10) * 10))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    write_photomaker_prompts(llama_results, proc_celeba_df, save_path=args.save_path)