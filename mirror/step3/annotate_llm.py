
import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from glob import iglob
from ast import literal_eval
from os.path import join as pjoin

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def split_list_into_chunks(lst, chunk_size=2):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="annot_data/llama3_prompt.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="/model/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--save_path", type=str, default="annot_data/llama3_8b_result.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading Prompt...")
    prompts = [json.loads(q) for q in open(args.prompt_path, 'r')]
    # prompts = split_list_into_chunks(prompts, args.batch_size)
    
    print(f"Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(
        model=args.model_name_or_path, 
        tensor_parallel_size=1,
        device="cuda"
    ) 

    sampling_params = SamplingParams(
        temperature=0.7,     
        top_p=0.9,           
        max_tokens=128       
    )
    annot_cache = [] if not os.path.exists(args.save_path) else [json.loads(q) for q in open(args.save_path, 'r')]
    annot_cache = list(map(lambda x: x['custom_id'], annot_cache))
    
    for batch in tqdm(prompts, total=len(prompts)):
        prompts_chat_applied = [tokenizer.apply_chat_template(
            prompt['messages'], 
            tokenize=False, 
            add_generation_prompt=True
        ) for prompt in batch]

        outputs = llm.generate(prompts_chat_applied, sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]

        for prompt, response in zip(batch, generated_texts):
            entry = {
                "custom_id": prompt['custom_id'],
                "model": args.model_name_or_path, 
                "messages": prompt['messages'],
                "response": response
            }
            with open(args.save_path, "a+") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')