import os
import re
import json
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from glob import iglob
from tqdm import tqdm
from ast import literal_eval
from os.path import join as pjoin
from mirror.utils.data_utils import write_line

# pip install tensorrt tf-keras
from mirror.step4.attr import DeepFace
from scipy.spatial.distance import cosine

def cosine_similarity(emb_a, emb_b):
    """Calculate cosine similarity between two embeddings."""
    return 1 - cosine(emb_a, emb_b) 

def get_embedding(image_path):
    """Extract facial embedding from an image."""
    try:
        embedding_objs = DeepFace.represent(img_path=image_path)
        return embedding_objs[0]["embedding"]
    except Exception as e:
        raise RuntimeError(f"Failed to get embedding for {image_path}: {str(e)}")

def crop_face(image_path, face_region, save_path):
    """Crop a face from an image based on given face region."""
    try:
        img = Image.open(image_path)
        left, top = face_region['x'], face_region['y']
        right, bottom = left + face_region['w'], top + face_region['h']
        
        cropped_image = img.crop((left, top, right, bottom))
        cropped_image.save(save_path)
        return save_path
    except Exception as e:
        raise RuntimeError(f"Failed to crop face from {image_path}: {str(e)}")

def process_identity_preservation(args):
    """Main function to process identity preservation."""
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    print(f"Loading Data: {args.data_path}")
    attr_df = pd.read_csv(args.data_path, converters={'region': literal_eval, 'proc_dialogue': literal_eval})

    for _, row in tqdm(attr_df.iterrows(), total=len(attr_df), desc="Processing Identity Preservation"):
        dialog_id = row['idx']
        original_image_path = row['img_path']

        if not os.path.exists(pjoin(args.image_dir, dialog_id)):
            continue

        for t, utt in enumerate(row['proc_dialogue']):
            if utt['speaker'].strip().lower() == 'therapist':
                continue

            identity = os.path.splitext(os.path.basename(original_image_path))[0]
            generated_image_path = pjoin(args.image_dir, dialog_id, f"turn:{t}_{identity}.png")

            entry = {
                'idx': dialog_id,
                't': t,
                'image_a_path': original_image_path,
                'image_b_path': generated_image_path,
            }

            try:
                if not os.path.exists(generated_image_path):
                    raise FileNotFoundError(f"Generated image not found: {generated_image_path}")

                deepface_analysis = DeepFace.analyze(img_path=generated_image_path, actions=['age'])
                if not deepface_analysis:
                    raise ValueError(f"DeepFace analysis failed for {generated_image_path}")

                cropped_a_path = crop_face(original_image_path, row['region'], pjoin(args.temp_dir, 'cropped_A.jpg'))
                cropped_b_path = crop_face(generated_image_path, deepface_analysis[0]['region'], pjoin(args.temp_dir, 'cropped_B.jpg'))

                emb_a = np.array(get_embedding(cropped_a_path))
                emb_b = np.array(get_embedding(cropped_b_path))

                entry['cosine_similarity'] = str(cosine_similarity(emb_a, emb_b))
                write_line(path=args.save_path, entry=entry)

            except Exception as e:
                entry['error'] = str(e)
                write_line(path=args.drop_path, entry=entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="mirror_w_annot.csv")
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--image_dir", type=str, default="images/")
    parser.add_argument("--save_path", type=str, default="./results/identity/results.jsonl")
    parser.add_argument("--drop_path", type=str, default="./results/identity/drop.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    process_identity_preservation(args)
