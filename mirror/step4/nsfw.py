import os
import argparse

from PIL import Image
from glob import iglob
from tqdm import tqdm
from os.path import join as pjoin
from transformers import pipeline

from mirror.utils.data_utils import write_line

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="images/")
    parser.add_argument("--save_path", type=str, default="results/nsfw/result.jsonl")
    parser.add_argument("--drop_path", type=str, default="results/nsfw/drop.jsonl")
    parser.add_argument("--model", type=str, default="Falconsai/nsfw_image_detection")

    args = parser.parse_args()
    print(f"Loading Data...")
    image_path_list = list(iglob(pjoin(args.image_dir, "*/*.*")))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print(f"Loading Model...")
    classifier = pipeline("image-classification", model=args.model)
    
    for i, path in tqdm(enumerate(image_path_list), total=len(image_path_list), desc="Analyze Facial Image"):
        dialog_id, filename = path.split("/")[-2:]
        entry = {
            'img_path': path,
            'dialog_id': dialog_id,
            'turn_id': filename.split("_")[0],
            'identity': filename.split("_")[-1].split(".")[0],
            'error': ''
        }
        img = Image.open(path)
        try:
            outputs = classifier(img)
            entry['score'] = { x['label'] : x['score'] for x in outputs }
            
            if entry['score']['nsfw'] >= entry['score']['normal']:
                write_line(path=args.drop_path, entry=entry)
            else:
                write_line(path=args.save_path, entry=entry)
        
        except Exception as e:
            entry['error'] = str(e)
            write_line(path=args.drop_path, entry=entry)
            continue