import os
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import warnings

  
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

if TYPE_CHECKING:
    from agents.mllm_bot_qwen_2_5_vl import MLLMBot
from retrieval.multimodal_retrieval import MultimodalRetrieval
from utils.fileios import dump_json, load_json, dump_json_override


class KnowledgeBaseBuilder:
    """
    Knowledge base builder for creating and managing image and text knowledge bases
    """
    
    def __init__(self, 
                 image_encoder_name="./models/Clip/clip-vit-base-patch32",
                 text_encoder_name="./models/Clip/clip-vit-base-patch32", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 cfg=None,
                 dataset_info=None):
        
        from datetime import datetime
        init_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.device = device
        self.cfg = cfg
        self.dataset_info = dataset_info or {}

        if self.dataset_info:
            pass
        else:
            pass
        
        self.retrieval = MultimodalRetrieval(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            fusion_method='weighted',
            device=device
        )
        
        self.image_knowledge_base = {}
        self.text_knowledge_base = {}
        self.category_descriptions = {}
        
    def augment_image(self, image_path: str, augmentation_type: str = "all") -> List[str]:
            
        image = Image.open(image_path).convert("RGB")
        augmented_paths = []
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        base_dir = os.path.dirname(image_path)
        
        if augmentation_type in ["all", "rotation"]:
              
            for angle in [90, 180, 270]:
                rotated = image.rotate(angle, expand=True)
                rotated_path = os.path.join(base_dir, f"{base_name}_rot_{angle}.jpg")
                rotated.save(rotated_path)
                augmented_paths.append(rotated_path)
        
        if augmentation_type in ["all", "brightness"]:
              
            for factor in [0.7, 1.3]:
                enhancer = ImageEnhance.Brightness(image)
                bright = enhancer.enhance(factor)
                bright_path = os.path.join(base_dir, f"{base_name}_bright_{factor}.jpg")
                bright.save(bright_path)
                augmented_paths.append(bright_path)
        
        if augmentation_type in ["all", "contrast"]:
              
            for factor in [0.8, 1.2]:
                enhancer = ImageEnhance.Contrast(image)
                contrast = enhancer.enhance(factor)
                contrast_path = os.path.join(base_dir, f"{base_name}_contrast_{factor}.jpg")
                contrast.save(contrast_path)
                augmented_paths.append(contrast_path)
        
        if augmentation_type in ["all", "blur"]:
              
            for radius in [1, 2]:
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                blur_path = os.path.join(base_dir, f"{base_name}_blur_{radius}.jpg")
                blurred.save(blur_path)
                augmented_paths.append(blur_path)
        
        if augmentation_type in ["all", "flip"]:  
            flipped_h = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip_h_path = os.path.join(base_dir, f"{base_name}_flip_h.jpg")
            flipped_h.save(flip_h_path)
            augmented_paths.append(flip_h_path)
            
            flipped_v = image.transpose(Image.FLIP_TOP_BOTTOM)
            flip_v_path = os.path.join(base_dir, f"{base_name}_flip_v.jpg")
            flipped_v.save(flip_v_path)
            augmented_paths.append(flip_v_path)
        
        return augmented_paths
    
    def get_wiki_description(self, category_name: str) -> str:
        try:
            url = f"https://en.wikipedia.org/wiki/{category_name.replace('_', ' ').replace('-', ' ')}"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 100 and category_name.lower() in text.lower():
                    return text
            
            return f"Information about {category_name} from Wikipedia."
        except:
            return f"Description for {category_name} category."
    
    def generate_category_description(self, mllm_bot: "MLLMBot", category_name: str, 
                                   sample_images: List[str] = None) -> str:
        
        wiki_desc = self.get_wiki_description(category_name)
     
        if sample_images and len(sample_images) > 0:
              
            prompt = f"Please provide a detailed description of {category_name} based on these images."
            
            images = [Image.open(img_path).convert("RGB") for img_path in sample_images[:3]]    
            
            try:
                reply, description = mllm_bot.describe_attribute(images, prompt)
                if isinstance(description, list):
                    description = " ".join(description)
                
                del images
                import gc
                gc.collect()
                
                return description
            except Exception as e:
                
                del images
                import gc
                gc.collect()
                return wiki_desc
        else:
            return wiki_desc
    
    def build_image_knowledge_base(self, train_samples: Dict[str, List[str]], 
                                 augmentation: bool = True,
                                 completed_categories: set = None,
                                 on_category_complete: callable = None) -> Dict[str, np.ndarray]:
            
        from datetime import datetime
     
        image_kb = {}
        
        if completed_categories and self.image_knowledge_base:
            for cat in completed_categories:
                if cat in self.image_knowledge_base:
                    image_kb[cat] = self.image_knowledge_base[cat]
        
        completed_categories = completed_categories or set()
        
        total_categories = len(train_samples)
        total_images = sum(len(paths) for paths in train_samples.values())
          
        skipped_images = sum(len(train_samples[cat]) for cat in completed_categories if cat in train_samples)
        processed_images = skipped_images
        
        for cat_idx, (category, image_paths) in enumerate(train_samples.items()):
              
            if category in completed_categories:
                continue
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cat_progress_pct = (cat_idx / total_categories) * 100
            img_progress_pct = (processed_images / total_images) * 100 if total_images > 0 else 0
            remaining_categories = total_categories - cat_idx
            remaining_images = total_images - processed_images
            
            category_features = []
            
            for img_idx, img_path in enumerate(image_paths):
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                img_progress_pct = (processed_images / total_images) * 100 if total_images > 0 else 0
                remaining_images = total_images - processed_images
                
                try:
                    feat = self.retrieval.extract_image_feat(img_path)
                    category_features.append(feat)
                    processed_images += 1
                except Exception as e:
                    processed_images += 1
                    continue
                
                processed_images += 1
            
            if category_features:
                  
                category_features = np.array(category_features)
                avg_feature = np.mean(category_features, axis=0)
                image_kb[category] = avg_feature
                
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                completed_count = cat_idx + 1
                cat_completed_pct = (completed_count / total_categories) * 100
                img_completed_pct = (processed_images / total_images) * 100 if total_images > 0 else 0
                
                if on_category_complete:
                    on_category_complete(category)
            else:
                pass
        
        self.image_knowledge_base = image_kb
        return image_kb
    
    def build_text_knowledge_base(self, mllm_bot: "MLLMBot", train_samples: Dict[str, List[str]],
                                  completed_categories: set = None,
                                  on_category_complete: callable = None) -> Dict[str, np.ndarray]:
            
        from datetime import datetime
     
        text_kb = {}
        
        if completed_categories and self.text_knowledge_base:
            for cat in completed_categories:
                if cat in self.text_knowledge_base:
                    text_kb[cat] = self.text_knowledge_base[cat]
        
        completed_categories = completed_categories or set()
        
        total_categories = len(train_samples)
        processed_categories = len(completed_categories)
        
        for cat_idx, (category, image_paths) in enumerate(train_samples.items()):
              
            if category in completed_categories:
                continue
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cat_progress_pct = (cat_idx / total_categories) * 100
            remaining_categories = total_categories - cat_idx
            
            description = self.generate_category_description(mllm_bot, category, image_paths)
            self.category_descriptions[category] = description
            
            try:
                text_feat = self.retrieval.extract_text_feat(description)
                text_kb[category] = text_feat
                processed_categories += 1
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                completed_pct = (processed_categories / total_categories) * 100
                
                if on_category_complete:
                    on_category_complete(category)
            except Exception as e:
     
                continue
        
        self.text_knowledge_base = text_kb
        return text_kb
    
    def build_knowledge_base(self, mllm_bot: "MLLMBot", train_samples: Dict[str, List[str]], 
                           augmentation: bool = True,
                           completed_image_categories: set = None,
                           completed_text_categories: set = None,
                           on_image_category_complete: callable = None,
                           on_text_category_complete: callable = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        
        image_kb = self.build_image_knowledge_base(train_samples, augmentation, completed_image_categories, on_image_category_complete)
        
        text_kb = self.build_text_knowledge_base(mllm_bot, train_samples, completed_text_categories, on_text_category_complete)
        
        return image_kb, text_kb
    
    def initialize_lcb_stats_with_labels(self, train_samples: Dict[str, List[str]], 
                                       fast_thinking_module, top_k: int = 5) -> Dict:
            
        stats_summary = {
            "total_samples": 0,
            "correct_predictions": 0,
            "category_stats": {},
            "initialization_accuracy": 0.0
        }
        
        fast_thinking_module.category_stats = defaultdict(lambda: {"n": 0, "m": 0})
        fast_thinking_module.total_predictions = 0
        
        correct_count = 0
        total_count = 0
        
        for category, image_paths in train_samples.items():
              
            category_correct = 0
            category_total = 0
              
            for img_path in image_paths:
                  
                  
                result = fast_thinking_module.fast_thinking_pipeline(img_path, top_k)
                
                predicted_category = result.get("fused_top1", "unknown")
                
                  
                from utils.util import is_similar
                is_correct = is_similar(predicted_category, category, threshold=0.5)
                
                  
                fast_thinking_module.update_stats(category, is_correct)
                
                if is_correct:
                    correct_count += 1
                    category_correct += 1
                
                total_count += 1
                category_total += 1
            
              
            stats_summary["category_stats"][category] = {
                "total_samples": category_total,
                "correct_predictions": category_correct,
                "accuracy": category_correct / category_total if category_total > 0 else 0.0
            }
            
     
        
          
        stats_summary["total_samples"] = total_count
        stats_summary["correct_predictions"] = correct_count
        stats_summary["initialization_accuracy"] = correct_count / total_count if total_count > 0 else 0.0
        fast_thinking_module.save_stats()
        return stats_summary
    
    def save_knowledge_base(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
          
        image_kb_path = os.path.join(save_dir, "image_knowledge_base.json")
        image_kb_to_save = {cat: feat.tolist() for cat, feat in self.image_knowledge_base.items()}
        dump_json(image_kb_path, image_kb_to_save)
        
          
        text_kb_path = os.path.join(save_dir, "text_knowledge_base.json")
        text_kb_to_save = {cat: feat.tolist() for cat, feat in self.text_knowledge_base.items()}
        dump_json(text_kb_path, text_kb_to_save)
        
          
        desc_path = os.path.join(save_dir, "category_descriptions.json")
        dump_json(desc_path, self.category_descriptions)
    
    def update_knowledge_base(self, save_dir: str):
            
        os.makedirs(save_dir, exist_ok=True)
        
          
        image_kb_path = os.path.join(save_dir, "image_knowledge_base.json")
        image_kb_to_save = {cat: feat.tolist() for cat, feat in self.image_knowledge_base.items()}
        dump_json_override(image_kb_path, image_kb_to_save)
        
          
        text_kb_path = os.path.join(save_dir, "text_knowledge_base.json")
        text_kb_to_save = {cat: feat.tolist() for cat, feat in self.text_knowledge_base.items()}
        dump_json_override(text_kb_path, text_kb_to_save)
        
          
        desc_path = os.path.join(save_dir, "category_descriptions.json")
        dump_json_override(desc_path, self.category_descriptions)
    
    def load_knowledge_base(self, load_dir: str):
            
          
        image_kb_path = os.path.join(load_dir, "image_knowledge_base.json")
        if os.path.exists(image_kb_path):
            image_kb_data = load_json(image_kb_path)
              
            if isinstance(image_kb_data, dict):
                self.image_knowledge_base = {cat: np.array(feat) for cat, feat in image_kb_data.items()}
            elif isinstance(image_kb_data, list) and len(image_kb_data) > 0:
                  
                if isinstance(image_kb_data[0], dict):
                    self.image_knowledge_base = {cat: np.array(feat) for cat, feat in image_kb_data[0].items()}
                else:
     
                    self.image_knowledge_base = {}
            else:
     
                self.image_knowledge_base = {}
        
          
        text_kb_path = os.path.join(load_dir, "text_knowledge_base.json")
        if os.path.exists(text_kb_path):
            text_kb_data = load_json(text_kb_path)
              
            if isinstance(text_kb_data, dict):
                self.text_knowledge_base = {}
                for cat, feat in text_kb_data.items():
                    feat_array = np.array(feat)
                      
                    if feat_array.ndim == 1 and len(feat_array) > 0:
                        self.text_knowledge_base[cat] = feat_array
                    else:
     
            elif isinstance(text_kb_data, list) and len(text_kb_data) > 0:
                  
                if isinstance(text_kb_data[0], dict):
                    self.text_knowledge_base = {}
                    for cat, feat in text_kb_data[0].items():
                        feat_array = np.array(feat)
                          
                        if feat_array.ndim == 1 and len(feat_array) > 0:
                            self.text_knowledge_base[cat] = feat_array
                        else:
     
                else:
     
                    self.text_knowledge_base = {}
            else:
     
                self.text_knowledge_base = {}
        
          
        desc_path = os.path.join(load_dir, "category_descriptions.json")
        if os.path.exists(desc_path):
            desc_data = load_json(desc_path)
              
            if isinstance(desc_data, dict):
                self.category_descriptions = desc_data
            elif isinstance(desc_data, list) and len(desc_data) > 0:
                  
                if isinstance(desc_data[0], dict):
                    self.category_descriptions = desc_data[0]
                else:
     
                    self.category_descriptions = {}
            else:
     
                self.category_descriptions = {}
        
          
        belief_path = os.path.join(load_dir, "self_belief.txt")
        if os.path.exists(belief_path):
            with open(belief_path, 'r', encoding='utf-8') as f:
                self.self_belief = f.read()
     
     
     
     
     
        
          
        if self.text_knowledge_base:
            expected_dim = None
            problem_categories = []
            for cat, feat in self.text_knowledge_base.items():
                if not isinstance(feat, np.ndarray):
                    feat = np.array(feat)
                if expected_dim is None:
                    expected_dim = len(feat)
                elif len(feat) != expected_dim:
                    problem_categories.append((cat, feat.shape))
            
            if problem_categories:
     
                for cat, shape in problem_categories[:5]:
     
                if len(problem_categories) > 5:
     
            else:
     
    
    def image_retrieval(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
            
        if not self.image_knowledge_base:
            raise ValueError("Image knowledge base is empty, please build knowledge base first")
        
          
        query_feat = self.retrieval.extract_image_feat(query_image_path)
        
          
        similarities = []
        for category, feat in self.image_knowledge_base.items():
            sim = np.dot(query_feat, feat)    
            similarities.append((category, sim))
        
          
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def text_retrieval(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
            
        if not self.text_knowledge_base:
            raise ValueError("Text knowledge base is empty, please build knowledge base first")
        
          
        query_feat = self.retrieval.extract_text_feat(query_text)
        
          
        similarities = []
        for category, feat in self.text_knowledge_base.items():
            sim = np.dot(query_feat, feat)    
            similarities.append((category, sim))
        
          
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def incremental_update(self, category: str, image_path: str, structured_description: str,key_regions: List[Dict], confidence: float, mllm_bot, save_dir: str,
                         signals: Dict = None) -> bool:
            
        signals = signals or {}
        
          
        quality_score = self._evaluate_update_quality(category, confidence, signals)
        
        if quality_score < 0.7:    
     
            return False
        
     
          
        self._update_image_knowledge_base(category, image_path, quality_score)
        
          
          
        
          
        self.update_knowledge_base(save_dir)
        
     
        return True

    
    def _evaluate_update_quality(self, category: str, confidence: float, signals: Dict) -> float:
            
        quality_factors = []
        
          
        conf_factor = min(1.0, confidence / 0.8)    
        quality_factors.append(("confidence", conf_factor, 0.3))
        
          
        lcb_value = signals.get('lcb', 0.5)
        lcb_factor = min(1.0, lcb_value / 0.7)    
        quality_factors.append(("lcb", lcb_factor, 0.25))
        
          
        fast_slow_consistent = signals.get('fast_slow_consistent', False)
        consistency_factor = 1.0 if fast_slow_consistent else 0.3
        quality_factors.append(("consistency", consistency_factor, 0.2))
        
          
        fused_prob = signals.get('fused_top1_prob', 0.5)
        fused_factor = min(1.0, fused_prob / 0.7)
        quality_factors.append(("fused_prob", fused_factor, 0.15))
        
          
        rank_fast = signals.get('rank_in_fast_topk', 10)    
        rank_enh = signals.get('rank_in_enhanced_topk', 10)
        rank_factor = max(0.0, 1.0 - (min(rank_fast, rank_enh) / 5.0))    
        quality_factors.append(("ranking", rank_factor, 0.1))
        
          
        total_score = sum(factor * weight for _, factor, weight in quality_factors)
        
     
        
        return total_score
    
    def _update_image_knowledge_base(self, category: str, image_path: str, quality_score: float):
            
          
        new_feat = self.retrieval.extract_image_feat(image_path)
        
        if category in self.image_knowledge_base:
              
            old_feat = self.image_knowledge_base[category]
            
              
            new_weight = min(0.1, quality_score * 0.04)
            old_weight = 1.0 - new_weight
            
              
            updated_feat = old_weight * old_feat + new_weight * new_feat
            self.image_knowledge_base[category] = updated_feat
            
    def _update_text_knowledge_base(self, category: str, structured_description: str, 
                                  mllm_bot, quality_score: float): 
        try: 
            if structured_description and len(structured_description.strip()) > 10:
                prompt = f"""Based on the detailed structured description of this {category} image, generate a concise but discriminative text description that captures the key visual characteristics for fine-grained recognition.

                Structured description: {structured_description}

                Focus on distinctive features that help distinguish this category from similar ones."""


                try:
                      
                    new_description = structured_description    
                except:
                    new_description = f"Updated description for {category}: {structured_description[:100]}..."
            else:
                new_description = f"Updated description for {category} category."
            
              
            new_text_feat = self.retrieval.extract_text_feat(new_description)
            
            if category in self.text_knowledge_base:
                  
                old_text_feat = self.text_knowledge_base[category]
                
                  
                new_weight = min(0.2, quality_score * 0.3)
                old_weight = 1.0 - new_weight
                
                updated_text_feat = old_weight * old_text_feat + new_weight * new_text_feat
                self.text_knowledge_base[category] = updated_text_feat
                
                  
                if category in self.category_descriptions:
                    old_desc = self.category_descriptions[category]
                    self.category_descriptions[category] = f"{old_desc}\n\nUpdated: {new_description}"
                else:
                    self.category_descriptions[category] = new_description
                
     
            else:
                  
                self.text_knowledge_base[category] = new_text_feat
                self.category_descriptions[category] = new_description
     
                
        except Exception as e:
     
            raise

