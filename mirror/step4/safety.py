
import os
import argparse
import pandas as pd

from tqdm import tqdm
from ast import literal_eval
from os.path import join as pjoin

from mirror.src.model.canary import Canary
from mirror.utils.data_utils import write_line

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="mirror_data.csv")
    parser.add_argument("--canary_dir", type=str, default="./step4/data/models/canary")
    parser.add_argument("--save_path", type=str, default="./results/canary/result.jsonl")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"Loading Model...")
    canary = Canary(canary_dir=args.canary_dir)
    
    print(f"Loading Data...")
    mirror = pd.read_csv(args.data_path, converters={
        'proc_dialogue': literal_eval
    })
    for i, row in tqdm(mirror.iterrows(), total=len(mirror), desc="Canary Filtering"):
        dialogue = []
        for t, utt in enumerate(row['proc_dialogue']):
            utt['t'] = t
            dialogue += [utt]

        client_statements = list(filter(lambda x: x['speaker'].lower().strip() != 'therapist', dialogue))
        client_batch = list(map(lambda x: x['statement'].strip(), client_statements))
        client_result = canary.chirp(client_batch)
        
        therapist_statements = list(filter(lambda x: x['speaker'].lower().strip() == 'therapist', dialogue))
        therapist_batch = list(map(lambda x: x['statement'].strip(), therapist_statements))
        therapist_result = canary.chirp(client_batch)

        client_output, therapist_output = [], []
        for stat, res in zip(client_statements, client_result):
            stat['safety'] = res
            client_output += [stat]
        for stat, res in zip(therapist_statements, therapist_result):
            stat['safety'] = res
            therapist_output += [stat]
        
        entry = {
            'idx': row['idx'],
            'client_safety': client_output,
            'therapist_safety': therapist_output,
        }
        write_line(path=args.save_path, entry=entry)