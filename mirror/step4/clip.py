
import os
import re
import json
import clip
import torch
import argparse
import pandas as pd

from PIL import Image
from glob import iglob
from tqdm import tqdm
from ast import literal_eval
from os.path import join as pjoin
from mirror.utils.data_utils import write_line

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="mirror_data.csv")
    parser.add_argument("--image_dir", type=str, default="images/")
    parser.add_argument("--save_path", type=str, default="./results/clip/result.jsonl")
    parser.add_argument("--drop_path", type=str, default="./results/clip/drop.jsonl")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print(f"Loading Data...")
    mirror = pd.read_csv(args.data_path, converters={
        'proc_dialogue': literal_eval
    })
    cos = torch.nn.CosineSimilarity(dim=0)
    for i, row in tqdm(mirror.iterrows(), total=len(mirror), desc="Calculate Cosine Similiarity"):
        assert os.path.exists(pjoin(args.image_dir, row['idx']))
        image_path_list = list(iglob(pjoin(args.image_dir, row['idx'], "*.*")))
        image_path_list = sorted(image_path_list, key=lambda x: int(x.split("/")[-1].split("_")[0].split(":")[-1]))
        
        client_statements = []
        for t, utt in enumerate(row['proc_dialogue']):
            if utt['speaker'].lower() == 'therapist': continue
            utt['t'] = t
            client_statements += [utt]

        for img, statement in zip(image_path_list, client_statements):
            image = preprocess(Image.open(img)).unsqueeze(0).to(device)
            text = clip.tokenize([f"A facial photo with {statement['stage_direction'].lower()}"]).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            similarity = cos(image_features[0], text_features[0]).item()
            if similarity < args.threshold:
                drop = {'img_path': img, 'similarity': str(similarity)}
                write_line(args.drop_path, entry=drop)
                continue
            entry = {
                'idx': row['idx'],
                'img_path': img,
                'statement': statement,
            }
            write_line(args.save_path, entry=entry)