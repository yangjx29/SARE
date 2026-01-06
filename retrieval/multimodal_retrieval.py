import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers import Blip2Processor, Blip2Model
from transformers import AutoProcessor, BlipForImageTextRetrieval
from sklearn.metrics.pairwise import cosine_similarity

  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import is_similar
from cvd.description_generator import CDVCaptioner
  
  
import json
import base64
from torch.nn import functional as F

class MultimodalRetrieval:
    def __init__(self, image_encoder_name="./models/Clip/clip-vit-base-patch32", text_encoder_name="./models/Clip/clip-vit-base-patch32", fusion_method="concat", device="cuda" if torch.cuda.is_available() else "cpu"): 
        self.device = device
        self.fusion_method = fusion_method
        self.clip_model = CLIPModel.from_pretrained(image_encoder_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(image_encoder_name)

        blip_load_path='./models/Blip/blip2-opt-6.7b-coco'

        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
          
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()

    def init_blip(self):
          
          
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
          
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()

    def extract_multimodal_feat_blip(self, image_path: str, text: str):
          
        img = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.blip_model(**inputs, output_attentions=False, output_hidden_states=True)
     
            if hasattr(outputs, 'text_last_hidden_state') and outputs.text_last_hidden_state is not None:
                fused = outputs.text_last_hidden_state[:, 0, :]    
            else:
                fused = outputs.question_embeds 
     
        fused = fused.squeeze(0)
        fused = fused.mean(dim=0) 
        fused = fused / (fused.norm(p=2) + 1e-12)
        return fused.detach().cpu().numpy()

    def extract_image_feat(self, image_path):    
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.clip_model.get_image_features(**inputs).cpu().numpy()
        feat = feat.flatten()
        norm = np.linalg.norm(feat) + 1e-12
        feat = feat / norm
        return feat    

    def extract_text_feat(self, text):  
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.get_text_features(**inputs).cpu().numpy()
        feat = feat.flatten()   
        norm = np.linalg.norm(feat) + 1e-12
        feat = feat / norm
        return feat    

      
    def fuse_features(self, img_feat, text_feat):
        if self.fusion_method == "concat":
            return np.concatenate([img_feat, text_feat])
        elif self.fusion_method == "average":
            return (img_feat + text_feat) / 2
        elif self.fusion_method == "weighted":
            alpha = 0.7
            return alpha * text_feat + (1 - alpha) * img_feat
        elif self.fusion_method == "cross_atten":
            raise RuntimeError("blip_cross requires raw image path and text; call extract_multimodal_feat_blip().")
        else:
            raise ValueError("Invalid fusion method. Use 'concat' or 'average'.")

    def l2_normalize(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray: 
        norm = np.linalg.norm(x) + eps
        return x / norm

    def load_gallery_from_json(self,load_path):  
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[0]
        gallery = {}
        for cat, value in data.items():
            arr = np.array(value, dtype=np.float32)
            gallery[cat] = arr
        return gallery

    def topk_search(self, query_feat: np.ndarray, gallery_feats: np.ndarray, gallery_cats, k: int = 1):
        if gallery_feats.ndim != 2:
            raise ValueError("gallery_feats must be 2D [N, D]")
        q = self.l2_normalize(query_feat.astype(np.float32))
        G = gallery_feats    
        sims = G @ q    
        k = min(k, sims.shape[0])
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx], [gallery_cats[i] for i in idx]

    def build_template_gallery(self, mllm_bot, train_samples, description_generator, superclass, kshot=5,region_num=3):       
        gallery = {}

        for cat, paths in train_samples.items():
            cat_feats = []
            for i, path in enumerate(paths):
                description = description_generator.generate_description(mllm_bot, path, train_samples, superclass, kshot, region_num, label=cat, label_id=i)

                if self.fusion_method == "cross_atten":
                    fused_feat = self.extract_multimodal_feat_blip(path, description)
     
                else:
                    img_feat = self.extract_image_feat(path)
                    text_feat = self.extract_text_feat(description)
                    fused_feat = self.fuse_features(img_feat, text_feat)
                  
                cat_feats.append(fused_feat)
            if cat_feats: 
                gallery[cat] = np.mean(cat_feats, axis=0)   
            else:
                raise ValueError(f"No features extracted for category {cat}")
                
        return gallery   

    def fgvc_via_multimodal_retrieval(self, mllm_bot, query_image_path, gallery, description_generator, superclass, use_rag=True,topk=1):
        description = description_generator.generate_description_inference(mllm_bot,query_image_path, superclass)
        if self.fusion_method == "cross_atten":
            query_feat = self.extract_multimodal_feat_blip(query_image_path, description)
        else:
            img_feat = self.extract_image_feat(query_image_path)
            text_feat = self.extract_text_feat(description)
            query_feat = self.fuse_features(img_feat, text_feat) 
        gallery_cats = list(gallery.keys())
        gallery_feats = np.array([gallery[cat] for cat in gallery_cats])    
          
        if query_feat.shape[-1] != gallery_feats.shape[-1]:
            raise ValueError(f"Dim mismatch: query {query_feat.shape[-1]} vs gallery {gallery_feats.shape[-1]}. Ensure same fusion_method for gallery and query.") 
        cos_sims = np.dot(query_feat, gallery_feats.T)    
        beta = 0.1
        affinities = np.exp(-beta * (1 - cos_sims))     
        affinity_scores = {gallery_cats[i]: affinities[i] for i in range(len(gallery_cats))}
          
        if use_rag:
            topk_categories = sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
            topk_cat_names = [cat for cat, score in topk_categories]
            topk_scores = [score for cat, score in topk_categories]
            rag_prompt = self._construct_rag_prompt(topk_cat_names, topk_scores, superclass)
            from PIL import Image
            query_image = Image.open(query_image_path).convert("RGB")
            reply, final_prediction = mllm_bot.describe_attribute(query_image, rag_prompt)    
     
              
            predicted_category = self._extract_final_category(final_prediction, topk_cat_names)
        else:
            predicted_category = max(affinity_scores, key=affinity_scores.get)
     
        
     
        return predicted_category, affinity_scores
    def evaluate_fgvc(self, mllm_bot, test_samples, gallery, description_generator, superclass, use_rag=True, topk = 1):  
        correct = 0
        total = 0
        for true_cat, paths in test_samples.items():
            for path in paths:
                predicted_cat, _ = self.fgvc_via_multimodal_retrieval(mllm_bot, path, gallery, description_generator, superclass, use_rag,topk)
                if is_similar(predicted_cat,true_cat):
                    correct += 1
     
                total += 1
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def _construct_rag_prompt(self, top5_categories, top5_scores, superclass):  
        candidates_text = ""
        for i, (cat, score) in enumerate(zip(top5_categories, top5_scores)):
            candidates_text += f"{i+1}. {cat} (scores: {score:.4f})\n"
        
        rag_prompt = f    
        
        return rag_prompt

    def _extract_final_category(self, mllm_output, top5_categories):
        cleaned_output = mllm_output[0].strip().replace('.', '').replace(',', '').replace('-', '').replace('.', '')

        for category in top5_categories:
            if category.lower() in cleaned_output.lower():
                return category
        for category in top5_categories:
              
            category_words = category.lower().split()
            if any(word in cleaned_output.lower() for word in category_words if len(word) > 2):
                return category
        return top5_categories[0]

