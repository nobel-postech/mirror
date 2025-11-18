
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


def get_row_by_index(df: pd.DataFrame, idx: str):
    try:
        return df[df["idx"] == idx].iloc[0]
    except Exception:
        raise KeyError(f"Index '{idx}' not found in DataFrame.")

def get_identity_row(identity_df: pd.DataFrame, identity: str):
    try:
        return identity_df[identity_df["identity"] == identity].iloc[0]
    except Exception:
        raise KeyError(f"Identity '{identity}' not found in CelebA DataFrame.")


def determine_gender(identity_row: pd.Series) -> str:
    if "dominant_gender" in identity_row:
        return identity_row["dominant_gender"].lower().strip()
    
    return "female" if identity_row.get("celeb_male", 1) < 0 else "male"


def build_prompt(gender: str, base_prompt: str) -> str:
    return (
        f"portrait photo of a {gender} img, perfect face, natural skin, "
        f"high detail, {base_prompt}"
    )

def build_negative_prompt(base_negative: str) -> str:
    return (
        "nsfw, lowres, bad anatomy, bad hands, grayscale photograph, text, error, "
        "missing fingers, extra digit, fewer digits, cropped, worst quality, "
        "low quality, normal quality, jpeg artifacts, signature, watermark, "
        "username, blurry, " + base_negative
    )

def write_photomaker_prompts(entire_df, llm_results, celeba_df, save_path):
    save_file = open(save_path, 'w')
    for entry in tqdm(llm_results, total=len(llm_results)):
        dialog_idx = '-'.join(entry['custom_id'].split("-")[:4])

        try:
            row = get_row_by_index(entire_df, dialog_idx)
            identity_row = get_identity_row(celeba_df, row['identity'])
        except Exception as e:
            print(e)
            continue
        
        gender = determine_gender(identity_row)
        image_path = identity_row['img_path'] # celeba/img_align_celeba/####.jpg
        try:
            prompts = parse_description(entry['response'])
            prompt_str = build_prompt(gender=gender.lower().strip(), base_prompt=prompts['prompt'])
            negative_prompt_str = build_negative_prompt(base_negative=prompts['negative_prompt'])

            new_entry = {
                'idx': entry['custom_id'],
                'dialog_idx': dialog_idx,
                'image_path': [image_path],
                'prompt': prompt_str,
                'negative_prompt': negative_prompt_str
            }
            json.dump(new_entry, save_file, ensure_ascii=False)
            save_file.write('\n')
        except Exception as e:
            raise e
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="mirror_data.csv")
    parser.add_argument("--llm_result_path", type=str, default="llama3_8b_result.jsonl")
    parser.add_argument("--celeba_path", type=str, default="proc_celeba.csv")
    parser.add_argument("--save_path", type=str, default="photomaker_prompts/prompt_0.jsonl")
    args = parser.parse_args()

    print(f"Loading Data...")
    entire_df = pd.read_csv(args.data_path, converters={
        'proc_dialogue': literal_eval
    })
    llm_results = [json.loads(q) for q in open(args.llm_result_path, 'r')]
    celeba_df = pd.read_csv(args.celeba_path)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    write_photomaker_prompts(
        entire_df=entire_df, 
        llm_results=llm_results, 
        celeba_df=celeba_df, 
        save_path=args.save_path
    )
