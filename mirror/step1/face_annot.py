
import os
import re
import json
import argparse
import pandas as pd

from tqdm import tqdm
from os.path import join as pjoin

# pip install tensorrt tf-keras
from deepface import DeepFace

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="annot_data/celeba.csv")
    parser.add_argument("--img_data_dir", type=str, default="/data/celeba/img_align_celeba")
    parser.add_argument("--save_path", type=str, default="annot_data/proc_celeba_all.csv")
    parser.add_argument('--drop_duplicated', action='store_true')

    args = parser.parse_args()
    print(f"Loading Data...")
    celeba_df = pd.read_csv(args.data_path)
    if args.drop_duplicated:
        celeba_df = celeba_df.drop_duplicates(subset=['identity'])

    data_rows = []
    for i, row in tqdm(celeba_df.iterrows(), total=len(celeba_df), desc="Analyze Facial Image"):
        img_path = pjoin(args.img_data_dir, row['image'])
        entry = {
            'img_path': img_path,
            'identity': row['identity'],
            'celeb_male': row['Male'],
            'celeb_young': row['Young'],
        }
        try:
            objs = DeepFace.analyze(
                img_path = img_path, 
                actions = ['age', 'gender', 'emotion'],
            )
            entry.update(objs[0])
            data_rows += [entry]
        except ValueError as ve:
            continue

    proc_celeba_df = pd.DataFrame(data_rows)
    proc_celeba_df.to_csv(args.save_path, index=False)