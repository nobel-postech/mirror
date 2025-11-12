import re
import json
import argparse
import pandas as pd

from glob import iglob
from tqdm import tqdm
from os.path import join as pjoin

def parse_counseling_notes(text):
    sections = {
        "personal_info": r"### Personal Information ###:\s*(.*?)\s*###",
        "personality": r"### Personality Traits ###:\s*(.*?)\s*###",
        "distorted_thought": r"### Distorted Thoughts ###:\s*(.*?)\s*###",
        "thinking_trap": r"### Thinking Trap ###:\s*(.*?)\s*###",
        "reason_for_seeking_counseling": r"### Reason for Seeking Counseling ###:\s*(.*?)\s*##",
        "cbt_plan": r"## CBT Plan ##\s*(.*?)\s*\*\*"
    }
    parsed_data = {}
    for section, pattern in sections.items():
        match = re.search(pattern, text, re.DOTALL)
        parsed_data[section] = match.group(1).strip() if match else "Not found"
    return parsed_data

def get_personality_idx(personality):
    if 'Behavioral Resistance' in personality:
        return 'behavioral-resistance'
    elif 'Emotional Resistance' in personality:
        return 'emotional-resistance'
    elif 'Cognitive Resistance' in personality:
        return 'cognitive-resistance'
    elif 'No Resistance' in personality:
        return 'no-resistance'

def get_prompt_df(args):
    data_rows = []
    for prompt_path in iglob(pjoin(args.batch_input_dir, "*.jsonl")):
        prompts = [json.loads(q) for q in open(prompt_path, 'r')]
        for item in prompts:
            source = parse_counseling_notes(item['body']['messages'][-1]['content'])
            
            entry = {
                'idx': item['custom_id'],
                'model': item['body']['model'],
            }        
            entry.update(source)
            data_rows += [entry]
    return pd.DataFrame(data_rows)

def get_result_df(args):
    data_rows = []
    for output_path in iglob(pjoin(args.batch_output_dir, "*.jsonl")):    
        results = [json.loads(q) for q in open(output_path, 'r')]
        
        for item in results:
            idx = item['custom_id']
            dialogue = item['response']['body']['choices'][0]['message']['content']
            data_rows += [{
                'idx': idx,
                'dialogue': dialogue
            }]
    return pd.DataFrame(data_rows)

def parse_conversation(conversation, name=None):
    parsed_conversation = []
    
    pattern = r"^(Therapist|Client):\s*\[(.*?)\]\s*(.*)"
    
    for line in list(map(lambda x: x.strip(), conversation.splitlines())):
        if len(line) == 0: continue
        
        match = re.match(pattern, line.strip())
        if match:
            speaker = match.group(1)   
            description = match.group(2)
            statement = match.group(3)
            parsed_conversation.append({
                "speaker": speaker,
                "stage_direction": description,
                "statement": statement
            })
        else:
            if not ('Client' in line or 'Therapist' in line):
                if not name: continue
                retry_pattern = rf"^({name}|{r'|'.join(name.split())}):\s*\[(.*?)\]\s*(.*)"
                match = re.match(retry_pattern, line)
                if match:
                    speaker = match.group(1)   
                    description = match.group(2)
                    statement = match.group(3)
                    
                    parsed_conversation.append({
                        "speaker": speaker,
                        "stage_direction": description,
                        "statement": statement
                    })
            elif not '[' in line:
                retry_pattern = r"^(Therapist|Client):\s*(.*)"
                match = re.match(retry_pattern, line)
                if match:
                    speaker = match.group(1)
                    statement = match.group(2)
                    parsed_conversation.append({
                        "speaker": speaker,
                        "stage_direction": None,
                        "statement": statement
                    })
            else:
                print(line)
    return parsed_conversation


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="mirror_data.csv")
    parser.add_argument("--batch_input_dir", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--batch_output_dir", type=str, default="annot_data/llama3_8b_prompt.jsonl")
    args = parser.parse_args()

    prompt_df = get_prompt_df(args)
    results_df = get_result_df(args)

    entire_df = prompt_df.merge(results_df, on='idx')

    proc_dialogue = []
    for i, row in tqdm(entire_df.iterrows(), total=len(entire_df), desc="Postprocess dialogue"):
        client_name = row['personal_info'].split(" is a", 1)[0].strip()
            
        dialogue = parse_conversation(row['dialogue'], name=client_name)
        proc_dialogue += [dialogue]
        
    entire_df['proc_dialogue'] = proc_dialogue
    entire_df.dropna(inplace=True)

    entire_df['turn'] = entire_df['proc_dialogue'].apply(lambda x: len(x) / 2)
    entire_df.drop(entire_df[entire_df['turn'] == 0].index, inplace=True)
    entire_df['resistance'] = entire_df['idx'].apply(lambda x: ' '.join(x.split("-")[1:]))

    entire_df.to_csv(args.save_path, index=False)
    