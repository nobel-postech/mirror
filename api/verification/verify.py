import torch
import clip
import os
import re

import numpy as np

from PIL import Image
from deepface import DeepFace
from transformers import pipeline
from scipy.spatial.distance import cosine

from os.path import join as pjoin

def cosine_similarity(emb_a, emb_b):
    cosine_sim = 1 - cosine(emb_a, emb_b) 
    return cosine_sim

class ImageVerifier:
    def __init__(self, temp_dir="./temp", device=None):
        self.temp_dir = temp_dir
        self.device = device
        os.makedirs(self.temp_dir, exist_ok=True)
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.cos = torch.nn.CosineSimilarity(dim=0)

        self.nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    def _image_crop(self, image_path, face_region, save_path='cropped_image.jpg'):
        img = Image.open(image_path)

        left = face_region['x']
        top = face_region['y']
        right = left + face_region['w']
        bottom = top + face_region['h']
        
        cropped_image = img.crop((left, top, right, bottom))
        cropped_image.save(save_path)
        return save_path

    def get_embedding(self, image_path):
        embedding_objs = DeepFace.represent(img_path=image_path)
        assert len(embedding_objs) == 1, len(embedding_objs)
        return embedding_objs[0]["embedding"]

    def is_same_identity(self, image_a_path, image_b_path):
        image_a_save_path = pjoin(self.temp_dir, f'cropped_A.jpg')
        image_b_save_path = pjoin(self.temp_dir, f'cropped_B.jpg')
        try:
            a_objs = DeepFace.analyze(
                    img_path = image_a_path, 
                    actions = ['gender'],
                )
            b_objs = DeepFace.analyze(
                    img_path = image_b_path, 
                    actions = ['gender'],
                )
            assert len(a_objs) == 1 and len(b_objs) == 1

            self._image_crop(image_a_path, face_region=a_objs[0]['region'], save_path=image_a_save_path)
            self._image_crop(image_b_path, face_region=b_objs[0]['region'], save_path=image_b_save_path)
            emb_a = np.array(self.get_embedding(image_a_save_path))
            emb_b = np.array(self.get_embedding(image_b_save_path))

            cos_sim = cosine_similarity(emb_a, emb_b)
        except Exception as e:
            print(e)
            return False
        
        return cos_sim >= 0.3

    def is_related_image(self, image_path, client_statement):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize([f"A facial photo with {client_statement.lower()}"])
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
             
        similarity = self.cos(image_features[0], text_features[0]).item()
        return similarity >= 0.2

    def is_safe_image(self, image_path):
        img = Image.open(image_path)
        outputs = self.nsfw_classifier(img)
        score = { x['label'] : x['score'] for x in outputs }
        return score['nsfw'] < score['normal']
    
    def run(self, image_path, base_image_path, client_statement):
        if not self.is_safe_image(image_path): 
            print("[Error] NSFW Image")
            return False
        if not self.is_related_image(image_path, client_statement): 
            print("[Error] Not Related Image with client statement")
            return False

        if not self.is_same_identity(image_path, base_image_path): 
            print("[Error] Not Same Identity")
            return False
        return True
