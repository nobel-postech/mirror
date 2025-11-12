import os
import re
import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import iglob
from ast import literal_eval
from os.path import join as pjoin

from mirror.utils.cactus_utils import parse_intake_forms
from mirror.agent.counseling_gen import CounselingGenerator
from mirror.agent.personality import (
    generate_personality_description,
    PERSONALITY_TRAITS
)

def fetch_gpt_response(gen, custom_id, client_persona, cognitive_distortion, 
                       distorted_thoughts, reason_counseling, personality_trait,
                       cbt_plan):
    history = gen.get_response(
        client_persona=client_persona,
        cognitive_distortion=cognitive_distortion,
        distorted_thoughts=distorted_thoughts,
        reason_counseling=reason_counseling,
        personality_trait=personality_trait,
        cbt_plan=cbt_plan)
        
    return {
        'id': custom_id,
        'client_info': {
            'client_persona': client_persona,
            'cognitive_distortion': cognitive_distortion,
            'distorted_thoughts': distorted_thoughts,
            'personality_trait': personality_trait,
        },
        'conversation': history
    }

def create_batch_request_file(gen, custom_id, client_persona, cognitive_distortion, 
                       distorted_thoughts, reason_counseling, personality_trait,
                       cbt_plan):
    message = gen.get_message(
        custom_id=custom_id,
        client_persona=client_persona,
        cognitive_distortion=cognitive_distortion,
        distorted_thoughts=distorted_thoughts,
        reason_counseling=reason_counseling,
        personality_trait=personality_trait,
        cbt_plan=cbt_plan)
    return message

def generate_introductions(personal_info):
    intro = f"{personal_info.get('name', 'Unknown')} is a {personal_info.get('age', 'Unknown')}-year-old {personal_info.get('gender', 'Unknown')} working as a {personal_info.get('occupation', 'Unknown')}."
    if 'education' in personal_info:
        intro += f" With a background in {personal_info['education']}."
    if 'marital status' in personal_info:
        intro += f" {personal_info['name']} is currently {personal_info['marital status']}."
    if 'family details' in personal_info:
        intro += f" {personal_info['family details'].capitalize()}."
    return intro


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--prompt_ver", type=str, default="default")
    parser.add_argument("--data_path", type=str, default="../data/cactus_source.csv")
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument('--fetch_api', action='store_true', default=False)
    args = parser.parse_args()

    print("Data path:", args.data_path)
    subset_patternreframe = pd.read_csv(args.data_path,
                   converters={
                       'patterns': literal_eval,
                   })
    
    print("Preprocessing Cactus...")
    subset_patternreframe = parse_intake_forms(subset_patternreframe)
    
    subset = 0
    save_path = pjoin(args.save_dir, f"from_cactus_patternreframe_wo_roleplay.jsonl") 
    if not args.fetch_api:
        save_path = pjoin(args.save_dir, f"prompts_{subset}.jsonl")
    
    print("Generate Screenplay...")
    gen = CounselingGenerator(model=args.model, version=args.prompt_ver)
    call_function = fetch_gpt_response if args.fetch_api else create_batch_request_file
    
    for i, sample in tqdm(subset_patternreframe.iterrows(), total=len(subset_patternreframe)):
        custom_id = f"cactus-{i}"

        client_persona = generate_introductions(sample['personal_info'])
        cognitive_distortion = ', '.join(sample['patterns'])
        distorted_thoughts = sample['thought']
        if not "\n1. " in sample['cbt_plan']: continue
        cbt_plan =  "1. " + sample['cbt_plan'].split("\n1. ")[-1]

        for j, trait in enumerate(PERSONALITY_TRAITS):
            idx = f"{custom_id}-{trait['trait_name'].lower().replace(" ", "-")}"

            personality_trait = generate_personality_description(trait)
            entry = call_function(
                gen=gen,
                custom_id=idx,
                client_persona=client_persona,
                cognitive_distortion=cognitive_distortion,
                distorted_thoughts=distorted_thoughts,
                reason_counseling=sample['reason_counseling'],
                personality_trait=personality_trait,
                cbt_plan=cbt_plan
            )
            with open(save_path, "a+") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        if not args.fetch_api and (i) % 1000 == 0 and (i) > 0:
            subset += 1
            save_path = pjoin(args.save_dir, f"prompts_{subset}.jsonl")