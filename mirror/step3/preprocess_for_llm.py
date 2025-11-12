
import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from ast import literal_eval
from os.path import join as pjoin

from mirror.src.prompts import *

def build_dialogue_history(proc_dialogue):
    history = []
    for utt in proc_dialogue:
        if utt['speaker'].strip().lower() == 'therapist':
            speaker = 'Therapist'
        elif utt['speaker'].strip().lower() == 'client':
            speaker = 'Client'
        else:
            speaker = 'Client'
        if utt['stage_direction']:
            history += [f"{speaker}: [{utt['stage_direction'].strip()}] {utt['statement']}"]
        else:
            history += [f"{speaker}: {utt['statement']}"]

    return '\n'.join(history).strip()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--save_path", type=str, default="annot_data/llama3_8b_prompt.jsonl")
    args = parser.parse_args()

    print(f"Loading Data...")
    entire_df = pd.read_csv(args.data_path, converters={
        'proc_dialogue': literal_eval
    })

    build_message = None 
    if 'llama' in args.model_name.lower():
        build_message = build_message_for_llama
    elif 'qwen' in args.model_name.lower():
        build_message = build_message_for_qwen
    else:
        raise Exception(args.model_name)
    
    for i, row in tqdm(entire_df.iterrows(), total=len(entire_df), desc=f"Preprocess prompt for {args.model_name}"):
        dialog = row['proc_dialogue']
        for t, utt in enumerate(dialog):
            if utt['speaker'].strip().lower() == 'therapist': continue
            
            history = build_dialogue_history(dialog[:t])
            client_utt = f"Client: [{utt['stage_direction'].strip()}] {utt['statement']}"

            messages = build_message(dialogue=history, utter=client_utt)
            entry = {
                "custom_id": f"{row['idx']}-turn:{t}",
                "messages": messages,
            }
            with open(args.save_path, "a+") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')    
