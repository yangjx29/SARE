
import os
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import json
import re
from collections import Counter
import hashlib
from functools import lru_cache

if TYPE_CHECKING:
    from agents.mllm_bot_qwen_2_5_vl import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from experience_base_builder import ExperienceBaseBuilder
from fast_thinking import FastThinking
from utils.util import is_similar
from data import DATA_STATS
from difflib import get_close_matches


class SlowThinkingOptimized:
    """
    Slow thinking system optimized for reasoning with experience base
    """
    
    def __init__(self, mllm_bot: "MLLMBot", knowledge_base_builder: KnowledgeBaseBuilder,
                 fast_thinking: FastThinking,
                 experience_base_builder: ExperienceBaseBuilder=None,
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 simplified_reasoning: bool = True,
                 use_experience_base: bool = True,
                 top_k_experience: int = 1,
                 dataset_info: dict = None):
            
        from datetime import datetime
        init_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.mllm_bot = mllm_bot
        self.kb_builder = knowledge_base_builder
        self.fast_thinking = fast_thinking
        self.exp_builder = experience_base_builder
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.simplified_reasoning = simplified_reasoning
        self.use_experience_base = use_experience_base
        self.top_k_experience = top_k_experience
        self.dataset_info = dataset_info or {}
        
        if not self.dataset_info or 'experiment_dir_full' not in self.dataset_info:
            raise ValueError(
                "SlowThinkingOptimized initialization failed: must provide complete dataset_info. "
                "dataset_info should contain 'experiment_dir_full' field to avoid contaminating knowledge base."
            )
        self.default_save_dir = os.path.join(
            self.dataset_info['experiment_dir_full'],
            'knowledge_base'
        )
        
        self._mllm_cache = {}
        self._description_cache = {}
        
        dataset_name = self._get_dataset_name_from_info()
        if dataset_name not in DATA_STATS:
            raise ValueError(
                f"SlowThinkingOptimized initialization failed: unknown dataset '{dataset_name}'. "
                f"Available datasets: {list(DATA_STATS.keys())}"
            )
        self.current_dataset_stats = DATA_STATS[dataset_name]
        
        self.normalized_to_original = {
            self.normalize_name(cls): cls for cls in self.current_dataset_stats['class_names']
        }
        self.normalized_class_names = list(self.normalized_to_original.keys())
        
    def _get_dataset_name_from_info(self) -> str:
        
        if 'full_name' in self.dataset_info:
            full_name = self.dataset_info['full_name'].lower()
            for dataset_name in DATA_STATS.keys():
                if dataset_name in full_name:
                    return dataset_name
        
        if 'experiment_dir' in self.dataset_info:
            exp_dir = self.dataset_info['experiment_dir'].lower()
            for dataset_name in DATA_STATS.keys():
                if dataset_name in exp_dir:
                    return dataset_name
        
        raise ValueError(
            "Cannot infer dataset name from dataset_info. "
            "Please ensure dataset_info contains 'full_name' or 'experiment_dir' field."
        )
    
    def set_experience_base(self, experience_base_builder: ExperienceBaseBuilder):
            
        self.exp_builder = experience_base_builder
    
    def _get_self_belief_context(self) -> str:
            
        if not self.use_experience_base or not self.exp_builder:
            return ""
        try:
            policy_text = self.exp_builder.get_self_belief()
        except AttributeError:
            return self.exp_builder.INITIAL_SELF_BELIEF
        if not policy_text or not isinstance(policy_text, str):
            return ""
        return f"Self-Belief Policy Context:\n{policy_text.strip()}"
    
    def normalize_name(self, name):
            
        name = name.lower()
        name = re.sub(r'[-_]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()
    
    def _get_cache_key(self, image_path: str, prompt: str) -> str:
            
        if not self.enable_cache:
            return None
        cache_key = hashlib.md5(f"{image_path}_{prompt}".encode()).hexdigest()
        return cache_key
    
    def _cached_mllm_call(self, image_path: str, prompt: str, image: Image.Image = None) -> str:
            
        cache_key = self._get_cache_key(image_path, prompt)
        if cache_key and cache_key in self._mllm_cache:
            return self._mllm_cache[cache_key]
        
        if image is None:
            image = Image.open(image_path).convert("RGB")
        
        reply, response = self.mllm_bot.describe_attribute(image, prompt)
        if isinstance(response, list):
            response = " ".join(response)
        
        if cache_key:
            if len(self._mllm_cache) >= self.cache_size:
                  
                oldest_key = next(iter(self._mllm_cache))
                del self._mllm_cache[oldest_key]
            self._mllm_cache[cache_key] = response
        
        return response
    
    def reasoning_with_experience_base(self, query_image_path: str, fast_result: Dict,
                                       top_k_candidates: List[str],
                                       experience_context: str = "",
                                       top_k: int = 5) -> Dict:
            
        image = Image.open(query_image_path).convert("RGB")
        
          
        candidates = []
        fused_results = fast_result.get("fused_results", [])
        for cat, score in fused_results:
            candidates.append((str(cat), float(score)))
        
        candidate_text = ""
        for i, (cat, sc) in enumerate(candidates, start=1):
            candidate_text += f"{i}. {cat} (score: {sc:.4f})\n"
 
        prompt += """\nPlease analyze the image step by step and provide:
            1. Your reasoning chain (CoT) following the steps above
            2. Your final prediction (only the category name)
            Format your response as:
            Reasoning: [your step-by-step reasoning]
            Prediction: [category name]"""

        reply, response = self.mllm_bot.describe_attribute(image, prompt)
        if isinstance(response, list):
            response = " ".join(response)
        
        cot = ""
        prediction = "unknown"
        
        reasoning_match = re.search(r'Reasoning[:\s]+(.*?)(?=Prediction|$)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            cot = reasoning_match.group(1).strip()
        else:
              
            cot = response
        
        prediction_match = re.search(r'Prediction[:\s]+([^\n]+)', response, re.IGNORECASE)
        if prediction_match:
              
            prediction = prediction_match.group(1).strip()
        else:
              
            lines = response.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                  
                for cat in top_k_candidates:
                    if cat.lower() in last_line.lower() or last_line.lower() in cat.lower():
                        prediction = cat
                        break
                if prediction == "unknown":
                      
                    prediction = top_k_candidates[0] if top_k_candidates else "unknown"
          
        predicted_category = self._correct_category_name(prediction)
        
          
        confidence = 0.0
        if top_k_candidates:
              
            for cat, score in fast_result.get("fused_results", []):
                if self.normalize_name(cat) == self.normalize_name(predicted_category):
                    confidence = float(score)
                    break
        
        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "reasoning": cot,
            "fast_result": fast_result,
            "top_k_candidates": top_k_candidates,
            "simplified": True
        }
            
    
    def simplified_reasoning_with_enhanced(self, query_image_path: str, fast_result: Dict,
                                          enhanced_results: List[Tuple[str, float]], 
                                          top_k: int = 5) -> Dict:
            
        image = Image.open(query_image_path).convert("RGB")
        
          
        candidates = []
        
          
        fused_results = fast_result.get("fused_results", [])
        for cat, score in fused_results[:top_k]:
            candidates.append((str(cat), float(score)))
        
          
        for cat, score in (enhanced_results or [])[:top_k]:
            candidates.append((str(cat), float(score)))
        
          
        dedup = {}
        for cat, sc in candidates:
            if cat not in dedup or sc > dedup[cat]:
                dedup[cat] = sc
        merged = sorted([(c, s) for c, s in dedup.items()], key=lambda x: x[1], reverse=True)
        
          
        candidate_text = ""
        for i, (cat, sc) in enumerate(merged[:top_k * 2], start=1):    
            candidate_text += f"{i}. {cat} (score: {sc:.4f})\n"
        
          
    prompt = f"""You are an expert in fine-grained visual recognition. Look at the image and determine the most likely subclass from the following candidates:

        {candidate_text}

        Fast thinking analysis:
        - Image-based prediction: {fast_result.get('img_category', 'unknown')} (confidence: {fast_result.get('img_confidence', 0):.3f})
        - Text-based prediction: {fast_result.get('text_category', 'unknown')} (confidence: {fast_result.get('text_confidence', 0):.3f})
        - Fused prediction: {fast_result.get('fused_top1', 'unknown')} (confidence: {fast_result.get('fused_top1_prob', 0):.3f})
        - Fused margin: {fast_result.get('fused_margin', 0):.3f}

        Enhanced retrieval results:
        - Top candidate: {enhanced_results[0][0] if enhanced_results else 'unknown'} (score: {(enhanced_results[0][1] if enhanced_results else 0.0):.4f})

        Please analyze the image carefully and consider:
        1. The visual characteristics of the object (size, shape, color, texture, distinctive features)
        2. The similarity scores from both fast thinking and enhanced retrieval
        3. The consistency between different predictions
        4. Any distinctive features that help distinguish between similar categories
        5. The confidence margins between top candidates

        Pay special attention to:
        - Breed-specific features (ear shape, tail, muzzle, body proportions)
        - Color patterns and markings
        - Size and overall appearance
        - Any unique identifying traits

        Return ONLY a JSON object with the following fields:
        {{
            "predicted_category": "exact category name",
            "confidence": 0.0-1.0,
            "reasoning": "brief but detailed rationale for your decision",
            "key_features": "main visual features supporting your decision",
            "alternative_candidates": ["category1", "category2"] (if applicable)
        }}"""
        
        try:
            response = self._cached_mllm_call(query_image_path, prompt, image)
            
              
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "no rationale")
                    key_features = result.get("key_features", "no features")
                    alternative_candidates = result.get("alternative_candidates", [])
                    info = f"{reasoning} | Key Features: {key_features}"
                    if alternative_candidates:
                        info += f" | Alternatives: {', '.join(alternative_candidates)}"
                else:
                      
                    fallback = enhanced_results[0][0] if enhanced_results else (merged[0][0] if merged else "unknown")
                    predicted_category = fallback
                    confidence = float(enhanced_results[0][1]) if enhanced_results else (float(merged[0][1]) if merged else 0.5)
                    info = "JSON parsing failed; fallback to enhanced retrieval top candidate"
            except (json.JSONDecodeError, ValueError):
                  
                fallback = enhanced_results[0][0] if enhanced_results else (merged[0][0] if merged else "unknown")
                predicted_category = fallback
                confidence = float(enhanced_results[0][1]) if enhanced_results else (float(merged[0][1]) if merged else 0.5)
                info = "JSON parsing failed; fallback to enhanced retrieval top candidate"
            
              
            predicted_category = self._correct_category_name(predicted_category)
            
            return {
                "predicted_category": predicted_category,
                "confidence": confidence,
                "reasoning": info,
                "enhanced_results": enhanced_results,
                "fast_result": fast_result,
                "simplified": True
            }
            
        except Exception as e:
     
              
            fallback = enhanced_results[0][0] if enhanced_results else (merged[0][0] if merged else "unknown")
            conf = float(enhanced_results[0][1]) if enhanced_results else (float(merged[0][1]) if merged else 0.0)
            return {
                "predicted_category": self._correct_category_name(fallback),
                "confidence": conf,
                "reasoning": f"Simplified reasoning failed: {str(e)}",
                "enhanced_results": enhanced_results,
                "fast_result": fast_result,
                "simplified": True
            }
    
    def simplified_reasoning_pipeline(self, query_image_path: str, fast_result: Dict, 
                                     top_k: int = 5) -> Dict:
            
        image = Image.open(query_image_path).convert("RGB")
        
          
        candidates = []
        fast_top1 = fast_result.get("fused_top1") or fast_result.get("predicted_category")
        if isinstance(fast_top1, str) and len(fast_top1) > 0:
            candidates.append((fast_top1, float(fast_result.get("fused_top1_prob", 1.0))))
        
          
        fused_results = fast_result.get("fused_results", [])
        for cat, score in fused_results[:top_k]:
            if cat not in [c[0] for c in candidates]:
                candidates.append((str(cat), float(score)))
        
          
        dedup = {}
        for cat, sc in candidates:
            if cat not in dedup or sc > dedup[cat]:
                dedup[cat] = sc
        merged = sorted([(c, s) for c, s in dedup.items()], key=lambda x: x[1], reverse=True)
        
          
        candidate_text = ""
        for i, (cat, sc) in enumerate(merged, start=1):
            candidate_text += f"{i}. {cat} (score: {sc:.4f})\n"
        
          
        prompt = f"""You are an expert in fine-grained visual recognition. Look at the image and determine the most likely subclass from the following candidates:

{candidate_text}

Fast thinking analysis:
- Image-based prediction: {fast_result.get('img_category', 'unknown')} (confidence: {fast_result.get('img_confidence', 0):.3f})
- Text-based prediction: {fast_result.get('text_category', 'unknown')} (confidence: {fast_result.get('text_confidence', 0):.3f})
- Fused prediction: {fast_result.get('fused_top1', 'unknown')} (confidence: {fast_result.get('fused_top1_prob', 0):.3f})
- Fused margin: {fast_result.get('fused_margin', 0):.3f}

Please analyze the image carefully and consider:
1. The visual characteristics of the object (size, shape, color, texture, distinctive features)
2. The similarity scores from fast thinking
3. The consistency between different predictions
4. Any distinctive features that help distinguish between similar categories
5. The confidence margins between top candidates

Pay special attention to:
- Breed-specific features (ear shape, tail, muzzle, body proportions)
- Color patterns and markings
- Size and overall appearance
- Any unique identifying traits

Return ONLY a JSON object with the following fields:
{{
    "predicted_category": "exact category name",
    "confidence": 0.0-1.0,
    "reasoning": "brief but detailed rationale for your decision",
    "key_features": "main visual features supporting your decision",
    "alternative_candidates": ["category1", "category2"] (if applicable)
}}"""  
        
        try:
            response = self._cached_mllm_call(query_image_path, prompt, image)
            
              
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "no rationale")
                    key_features = result.get("key_features", "no features")
                    alternative_candidates = result.get("alternative_candidates", [])
                    info = f"{reasoning} | Key Features: {key_features}"
                    if alternative_candidates:
                        info += f" | Alternatives: {', '.join(alternative_candidates)}"
                else:
                      
                    fallback = merged[0][0] if merged else (fast_top1 or "unknown")
                    predicted_category = fallback
                    confidence = float(merged[0][1]) if merged else 0.5
                    info = "JSON parsing failed; fallback to top candidate"
            except (json.JSONDecodeError, ValueError):
                fallback = merged[0][0] if merged else (fast_top1 or "unknown")
                predicted_category = fallback
                confidence = float(merged[0][1]) if merged else 0.5
                info = "JSON parsing failed; fallback to top candidate"
            
              
            predicted_category = self._correct_category_name(predicted_category)
            
            return {
                "predicted_category": predicted_category,
                "confidence": confidence,
                "reasoning": info,
                "simplified": True,
                "fast_result": fast_result
            }
            
        except Exception as e:
     
              
            fallback = merged[0][0] if merged else (fast_top1 or "unknown")
            conf = float(merged[0][1]) if merged else 0.0
            return {
                "predicted_category": self._correct_category_name(fallback),
                "confidence": conf,
                "reasoning": f"Simplified reasoning failed: {str(e)}",
                "simplified": True,
                "fast_result": fast_result
            }
    
    def _correct_category_name(self, predicted_category: str) -> str:
            
        if predicted_category not in self.current_dataset_stats['class_names']:
              
            norm_pred = self.normalize_name(predicted_category)
            
              
            if norm_pred in self.normalized_class_names:
                corrected_category = self.normalized_to_original[norm_pred]
     
                return corrected_category
            else:
                  
                close_matches = get_close_matches(norm_pred, self.normalized_class_names, n=1, cutoff=0.3)
                if close_matches:
                    best_match_norm = close_matches[0]
                    corrected_category = self.normalized_to_original[best_match_norm]
     
                    return corrected_category
                else:
     
                    return predicted_category
        return predicted_category
    
    def enhanced_retrieval_only(self, query_image_path: str, fast_result: Dict, 
                               top_k: int = 5) -> List[Tuple[str, float]]:
            
        try:
              
            img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
            
              
            img_img_similarities = {}
            for category, kb_feat in self.kb_builder.image_knowledge_base.items():
                sim = np.dot(img_feat, kb_feat)
                img_img_similarities[category] = float(sim)
            
              
            img_text_similarities = {}
            for category, text_feat in self.kb_builder.text_knowledge_base.items():
                sim = np.dot(img_feat, text_feat)
                img_text_similarities[category] = float(sim)
            
              
            fused_results = fast_result.get("fused_results", [])
            fast_scores = {}
            for category, score in fused_results[:top_k * 3]:    
                fast_scores[category] = float(score)
            
              
            weighted_similarities = {}
            for category in set(list(img_img_similarities.keys()) + 
                               list(img_text_similarities.keys()) + 
                               list(fast_scores.keys())):
                img_img_score = img_img_similarities.get(category, 0.0)
                img_text_score = img_text_similarities.get(category, 0.0)
                fast_score = fast_scores.get(category, 0.0)
                
                if category in fast_scores:
                      
                    weighted_sim = (fast_score * 0.4 + 
                                   img_img_score * 0.35 + 
                                   img_text_score * 0.25)
                else:
                      
                    weighted_sim = (img_img_score * 0.55 + 
                                   img_text_score * 0.45)
                
                weighted_similarities[category] = weighted_sim
            
              
            sorted_results = sorted(weighted_similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]
            
        except Exception as e:
     
            return []
    
    def slow_thinking_pipeline_optimized(self, query_image_path: str, fast_result: Dict, 
                                        top_k: int = 5, save_dir: str = None) -> Dict:
        
        fused_results = fast_result.get("fused_results", [])
        top_k_candidates = [cat for cat, score in fused_results]
        
        experience_context = ""
        fused_top1_prob = float(fast_result.get("fused_top1_prob", 0.0))
        fused_margin = float(fast_result.get("fused_margin", 0.0))
        
        policy_context = self._get_self_belief_context()
        
        experience_context = policy_context
        
        result = self.reasoning_with_experience_base(
            query_image_path=query_image_path,
            fast_result=fast_result,
            top_k_candidates=top_k_candidates,
            experience_context=experience_context,
            top_k=top_k
        )
        
          
        predicted_category = result.get("predicted_category", "unknown")
        confidence = result.get("confidence", 0.0)
          
        
        return result
    
    def _update_stats(self, predicted_category: str, confidence: float, 
                     fast_result: Dict, slow_result: Dict, save_dir: str = None):
            
          
        fused_top1_prob = float(fast_result.get("fused_top1_prob", 0.0))
        fused_margin = float(fast_result.get("fused_margin", 0.0))
        fused_top1 = str(fast_result.get("fused_top1", "unknown"))
        fast_slow_consistent = is_similar(fused_top1, predicted_category, threshold=0.5)
        
          
        lcb_map = fast_result.get('lcb_map', {}) or {}
        lcb_value = float(lcb_map.get(predicted_category, 0.5)) if isinstance(lcb_map, dict) else 0.5
        
          
        confidence_checks = [
            float(confidence) >= 0.85,    
            (fused_top1_prob >= 0.88 and fused_margin >= 0.18),    
            (fast_slow_consistent and lcb_value >= 0.75 and float(confidence) >= 0.75),    
            (float(confidence) >= 0.75 and lcb_value >= 0.70)    
        ]
        
        is_confident_for_stats = any(confidence_checks)
        
          
        self.fast_thinking.update_stats(predicted_category, is_confident_for_stats, used_slow_thinking=True)
        
        used_experience = slow_result.get("used_experience_base", False)
        print(f"Stats update: category={predicted_category}, confidence={confidence:.3f}, LCB={lcb_value:.3f}, "
              f"consistency={fast_slow_consistent}, used_experience={used_experience}, update_m={is_confident_for_stats}")
