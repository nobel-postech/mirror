
import os
import re
import json
import argparse
from glob import iglob
from tqdm import tqdm
from os.path import join as pjoin
from mirror.utils.data_utils import write_line

# pip install tensorrt tf-keras
from mirror.step4.attr import DeepFace

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="images/")
    parser.add_argument("--save_path", type=str, default="./results/attr/result.jsonl")
    parser.add_argument("--drop_path", type=str, default="./results/attr/drop.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"Loading Data...")
    image_paths = list(iglob(pjoin(args.image_dir, "*/*.*")))
    
    drop_image_path_list = []
    for i, path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Analyze Facial Image"):
        dialog_id, filename = path.split("/")[-2:]
        entry = {
            'img_path': path,
            'dialog_id': dialog_id,
            'turn_id': filename.split("_")[0],
            'identity': filename.split("_")[-1].split(".")[0],
        }
        try:
            objs = DeepFace.analyze(
                img_path = path, 
                actions = ['age', 'gender', 'emotion'],
            )
            assert len(objs) == 1
            entry.update(objs[0])
            del entry['gender']
            for e in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                if not e in entry['emotion']: continue
                entry['emotion'][e] = float(entry['emotion'][e])
            
            write_line(args.save_path, entry=entry)
        
        except Exception as e:
            drop = {'img_path': path, 'error': str(e)}
            write_line(args.drop_path, entry=drop)
            continue