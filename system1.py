    

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import json
import os
from functools import lru_cache

from knowledge_base_builder import KnowledgeBaseBuilder
from utils.util import is_similar


class FastThinkingOptimized:
    def __init__(self, knowledge_base_builder: KnowledgeBaseBuilder, 
                 confidence_threshold: float = 0.8,
                 similarity_threshold: float = 0.7,
                   
                 fusion_weight: float = 0.05,
                 softmax_temp: float = 0.07,
                   
                 fused_conf_threshold: float = 0.75,    
                 fused_margin_threshold: float = 0.15,    
                 per_modality_conf_threshold: float = 0.65,    
                 consider_topk_overlap: bool = True,
                 topk_for_overlap: int = 3,
                   
                 stats_file: str = None,    
                 lcb_threshold: float = 0.65,    
                 lcb_threshold_adaptive: bool = True,    
                 lcb_threshold_min: float = 0.55,    
                 lcb_threshold_max: float = 0.75,    
                 prior_strength: float = 2.0,
                 prior_p: float = 0.6,
                 lcb_eta: float = 1.0,
                 lcb_alpha: float = 0.5,
                 lcb_epsilon: float = 1e-6,
                   
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                   
                 dataset_info: dict = None):

        self.kb_builder = knowledge_base_builder
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.fusion_weight = fusion_weight
        self.softmax_temp = max(1e-6, softmax_temp)
        
          
        self.fused_conf_threshold = fused_conf_threshold
        self.fused_margin_threshold = fused_margin_threshold
        self.per_modality_conf_threshold = per_modality_conf_threshold
        self.consider_topk_overlap = consider_topk_overlap
        self.topk_for_overlap = max(1, topk_for_overlap)
        
          
        self.lcb_threshold = lcb_threshold
        self.lcb_threshold_adaptive = lcb_threshold_adaptive
        self.lcb_threshold_min = lcb_threshold_min
        self.lcb_threshold_max = lcb_threshold_max
        self.prior_strength = prior_strength
        self.prior_p = prior_p
        self.lcb_eta = lcb_eta
        self.lcb_alpha = lcb_alpha
        self.lcb_epsilon = lcb_epsilon
        
          
        self.dataset_info = dataset_info or {}
        if stats_file is None:
            if not self.dataset_info or 'stats_file_full' not in self.dataset_info:
                raise ValueError(
                    "FastThinkingOptimized initialization failed: must provide dataset_info or explicitly specify stats_file. "
                    "dataset_info should contain 'stats_file_full' field to avoid contaminating knowledge base."
                )
              
            self.stats_file = self.dataset_info['stats_file_full']
     
        else:
            self.stats_file = stats_file

        self.total_predictions = 1
        self.category_stats = defaultdict(lambda: {"n": 0, "m": 0})
        self.load_stats()
        
          
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._similarity_cache = {}    
        self._lcb_cache = {}    
        
          
        self.performance_stats = {
            "fast_path_count": 0,
            "slow_path_count": 0,
            "fast_path_correct": 0,
            "slow_path_correct": 0
        }
    
    def load_stats(self): 
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.category_stats = defaultdict(lambda: {"n": 0, "m": 0}, data.get("category_stats", {}))
                self.total_predictions = data.get("total_predictions", 0)
                  
                if "performance_stats" in data:
                    self.performance_stats = data["performance_stats"]
     
    def save_stats(self):
            
        data = {
            "category_stats": dict(self.category_stats),
            "total_predictions": self.total_predictions,
            "performance_stats": self.performance_stats
        }
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_cache_key(self, query_image_path: str, top_k: int) -> str:
            
        if not self.enable_cache:
            return None
          
        cache_key = hashlib.md5(f"{query_image_path}_{top_k}".encode()).hexdigest()
        return cache_key
    
    def _get_similarity_cache_key(self, img_path: str, category: str) -> str:
            
        if not self.enable_cache:
            return None
        return hashlib.md5(f"{img_path}_{category}".encode()).hexdigest()
    
    def image_to_image_retrieval(self, query_image_path: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
            
        try:
            results = self.kb_builder.image_retrieval(query_image_path, top_k)
            if not results:
                return "unknown", 0.0, []
            best_category, best_score = results[0]
            return best_category, best_score, results
        except Exception as e:
     
            return "unknown", 0.0, []
    
    def image_to_text_retrieval(self, query_image_path: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
            
        query_img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
        similarities = []
        for category, text_feat in self.kb_builder.text_knowledge_base.items():
              
            if not isinstance(text_feat, np.ndarray):
                text_feat = np.array(text_feat)
            
              
            if text_feat.shape != query_img_feat.shape:
     
                continue
            
            sim = np.dot(query_img_feat, text_feat)
            similarities.append((category, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]
        if not results:
            return "unknown", 0.0, []
        best_category, best_score = results[0]
        return best_category, best_score, results
    
    def _to_probs(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
            
        if not results:
            return {}
        class_to_score = {}
        for cname, score in results:
            if cname not in class_to_score:
                class_to_score[cname] = score
            else:
                class_to_score[cname] = max(class_to_score[cname], score)
        scores = np.array(list(class_to_score.values()), dtype=np.float32)
        scores = scores / self.softmax_temp
        scores = scores - scores.max()
        exps = np.exp(scores)
        probs = exps / (exps.sum() + 1e-12)
        return {c: float(p) for c, p in zip(class_to_score.keys(), probs)}
    
    def _rrf(self, results: List[Tuple[str, float]], k: int = 60) -> Dict[str, float]:
            
        rrf = {}
        for rank, (cname, _) in enumerate(results, start=1):
            rrf[cname] = rrf.get(cname, 0.0) + 1.0 / (k + rank)
        return rrf
    
    def fuse_results(self, img_results: List[Tuple[str, float]], 
                    text_results: List[Tuple[str, float]], 
                    fusion_weight: Optional[float] = None) -> List[Tuple[str, float]]:
            
        alpha = self.fusion_weight if fusion_weight is None else fusion_weight
        img_probs = self._to_probs(img_results)
        text_probs = self._to_probs(text_results)
        img_rrf = self._rrf(img_results)
        text_rrf = self._rrf(text_results)
        
        categories = set(img_probs.keys()) | set(text_probs.keys()) | set(img_rrf.keys()) | set(text_rrf.keys())
        fused = []
        for c in categories:
            p_img = img_probs.get(c, 0.0)
            p_txt = text_probs.get(c, 0.0)
            rrf_img = img_rrf.get(c, 0.0)
            rrf_txt = text_rrf.get(c, 0.0)
            score = alpha * p_img + (1 - alpha) * p_txt + 0.1 * (rrf_img + rrf_txt)
            fused.append((c, float(score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused
    
    def calculate_lcb(self, category: str, confidence_scores: List[float]) -> float:
            
        import math
        stats = self.category_stats[category]
        n_raw = stats["n"]
        m_raw = stats["m"]
        
          
        n = n_raw + self.prior_strength
        m = m_raw + self.prior_p * self.prior_strength
        p_hat = m / (n + self.lcb_epsilon)
        
          
        if len(confidence_scores) > 1:
            probs = np.array(confidence_scores)
            probs = probs / (probs.sum() + 1e-12)
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12)
        else:
            entropy = 0.0
        
          
        if n_raw > 0:
            confidence_term = self.lcb_eta * math.sqrt(math.log(max(1, self.total_predictions)) / (2 * n + 1))
        else:
            confidence_term = self.lcb_eta * math.sqrt(math.log(max(1, self.total_predictions)))
        
          
        lcb = p_hat - confidence_term - self.lcb_alpha * entropy
        return max(0.0, min(1.0, lcb))
    
    def _get_adaptive_lcb_threshold(self) -> float:
            
        if not self.lcb_threshold_adaptive:
            return self.lcb_threshold
        
          
        fast_path_acc = 0.0
        if self.performance_stats["fast_path_count"] > 0:
            fast_path_acc = self.performance_stats["fast_path_correct"] / self.performance_stats["fast_path_count"]
        
          
        slow_path_acc = 0.0
        if self.performance_stats["slow_path_count"] > 0:
            slow_path_acc = self.performance_stats["slow_path_correct"] / self.performance_stats["slow_path_count"]
        
          
        if fast_path_acc > 0.85:
            adaptive_threshold = min(self.lcb_threshold_max, self.lcb_threshold + 0.05)
          
        elif fast_path_acc < 0.70:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.05)
        else:
            adaptive_threshold = self.lcb_threshold
        
        return adaptive_threshold
    
    def trigger_lcb_optimized(self, img_category: str, text_category: str, 
                             img_confidence: float, text_confidence: float,
                             fused_top1: str, fused_top1_prob: float, fused_margin: float,
                             topk_overlap: bool, name_soft_agree: bool) -> Tuple[bool, str, float, Dict]:
            
        trigger_reason = {}
        
        if fused_top1_prob >= self.fused_conf_threshold and fused_margin >= self.fused_margin_threshold:
            trigger_reason["type"] = "high_confidence_margin"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        categories_match_soft = is_similar(img_category, text_category, threshold=self.similarity_threshold) or name_soft_agree
        if categories_match_soft:
              
            if img_confidence >= self.per_modality_conf_threshold or text_confidence >= self.per_modality_conf_threshold:
                trigger_reason["type"] = "modality_consistency"
                trigger_reason["img_conf"] = img_confidence
                trigger_reason["text_conf"] = text_confidence
                return False, fused_top1, float(max(img_confidence, text_confidence)), trigger_reason
        
        if img_category == text_category or is_similar(img_category, text_category, threshold=0.4):
            if fused_top1_prob >= 0.2:    
                trigger_reason["type"] = "exact_modality_match"
                trigger_reason["fused_prob"] = fused_top1_prob
                return False, fused_top1, fused_top1_prob, trigger_reason
        
          
        if categories_match_soft and fused_top1_prob >= 0.3:
            trigger_reason["type"] = "soft_modality_match"
            trigger_reason["fused_prob"] = fused_top1_prob
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        if fused_top1_prob >= 0.40 and fused_margin >= 0.06:
            trigger_reason["type"] = "medium_confidence_margin"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
          
        if fused_top1_prob >= 0.4:
            trigger_reason["type"] = "high_prob_low_margin"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
          
        if fused_top1_prob >= 0.4 and fused_margin >= 0.04:
            if categories_match_soft:
                trigger_reason["type"] = "relaxed_confidence_modality"
                trigger_reason["fused_prob"] = fused_top1_prob
                trigger_reason["margin"] = fused_margin
                return False, fused_top1, fused_top1_prob, trigger_reason
        
          
        if fused_top1_prob >= 0.4 and topk_overlap:
            trigger_reason["type"] = "medium_prob_overlap_early"
            trigger_reason["fused_prob"] = fused_top1_prob
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        if fused_top1_prob >= 0.4 and fused_margin >= 0.05:
            trigger_reason["type"] = "pre_lcb_confidence"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        confidence_scores = [
            max(0.0, min(1.0, float(img_confidence))),
            max(0.0, min(1.0, float(text_confidence))),
            max(0.0, min(1.0, float(fused_top1_prob)))
        ]
        category_for_lcb = fused_top1
        
          
        if category_for_lcb not in self.category_stats:
            self.category_stats[category_for_lcb] = {"n": 0, "m": 0}
        
          
        lcb_value = self.calculate_lcb(category_for_lcb, confidence_scores)
        
          
        adaptive_threshold = self._get_adaptive_lcb_threshold()
        
          
        if lcb_value >= adaptive_threshold:
            trigger_reason["type"] = "lcb_pass"
            trigger_reason["lcb_value"] = lcb_value
            trigger_reason["threshold"] = adaptive_threshold
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        if lcb_value >= (adaptive_threshold * 0.7) and fused_top1_prob >= 0.5:
            trigger_reason["type"] = "lcb_near_threshold"
            trigger_reason["lcb_value"] = lcb_value
            trigger_reason["threshold"] = adaptive_threshold
            trigger_reason["fused_prob"] = fused_top1_prob
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        if fused_top1_prob >= 0.5 and fused_margin >= 0.05:
              
            if is_similar(img_category, text_category, threshold=0.4):
                trigger_reason["type"] = "high_prob_modality_match"
                trigger_reason["fused_prob"] = fused_top1_prob
                trigger_reason["margin"] = fused_margin
                return False, fused_top1, fused_top1_prob, trigger_reason        
          
        trigger_reason["type"] = "need_slow_thinking"
        trigger_reason["lcb_value"] = lcb_value
        trigger_reason["threshold"] = adaptive_threshold
        trigger_reason["fused_prob"] = fused_top1_prob
        trigger_reason["margin"] = fused_margin
        avg_confidence = (img_confidence + text_confidence) / 2
        return True, "conflict", avg_confidence, trigger_reason
    
    def update_stats(self, category: str, is_correct: bool, used_slow_thinking: bool = False):
            
        self.category_stats[category]["n"] += 1
        if is_correct:
            self.category_stats[category]["m"] += 1
        
        self.total_predictions += 1
        
          
        if used_slow_thinking:
            self.performance_stats["slow_path_count"] += 1
            if is_correct:
                self.performance_stats["slow_path_correct"] += 1
        else:
            self.performance_stats["fast_path_count"] += 1
            if is_correct:
                self.performance_stats["fast_path_correct"] += 1
        
        self.save_stats()
    
    def fast_thinking_pipeline(self, query_image_path: str, top_k: int = 5) -> Dict:
            
          
        img_category, img_confidence, img_results = self.image_to_image_retrieval(query_image_path, top_k)
        
          
        text_category, text_confidence, text_results = self.image_to_text_retrieval(query_image_path, top_k)
        
          
        fused_results = self.fuse_results(img_results, text_results)
        fused_top1 = fused_results[0][0] if fused_results else img_category
        
          
        fused_scores = np.array([s for _, s in fused_results], dtype=np.float32) if fused_results else np.array([1.0], dtype=np.float32)
        fused_scaled = fused_scores / self.softmax_temp
        fused_scaled = fused_scaled - fused_scaled.max()
        fused_exps = np.exp(fused_scaled)
        fused_probs = fused_exps / (fused_exps.sum() + 1e-12)
        fused_top1_prob = float(fused_probs[0]) if fused_probs.size > 0 else 1.0
        fused_margin = float(fused_probs[0] - fused_probs[1]) if fused_probs.size > 1 else fused_top1_prob
        
          
        img_topk = [c for c, _ in img_results[:self.topk_for_overlap]]
        text_topk = [c for c, _ in text_results[:self.topk_for_overlap]]
        topk_overlap = any(c in text_topk for c in img_topk)
        
          
        name_soft_agree = False
        for ci in img_topk:
            for ct in text_topk:
                if is_similar(ci, ct, threshold=self.similarity_threshold):
                    name_soft_agree = True
                    break
            if name_soft_agree:
                break
        
          
        img_probs = self._to_probs(img_results)
        text_probs = self._to_probs(text_results)
        img_confidence = float(max(img_probs.values())) if img_probs else 0.0
        text_confidence = float(max(text_probs.values())) if text_probs else 0.0
        
          
        need_slow_thinking, predicted_category, confidence, trigger_reason = self.trigger_lcb_optimized(
            img_category, text_category, img_confidence, text_confidence,
            fused_top1, fused_top1_prob, fused_margin, topk_overlap, name_soft_agree
        )
          
        lcb_map = {}
        confidence_scores = [img_confidence, text_confidence, fused_top1_prob]
        lcb_value = self.calculate_lcb(fused_top1, confidence_scores)
        lcb_map[fused_top1] = lcb_value
        
          
        predicted_fast = fused_top1
        
        result = {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "need_slow_thinking": need_slow_thinking,
            "fused_top1": fused_top1,
            "predicted_fast": predicted_fast,
            "img_category": img_category,
            "text_category": text_category,
            "img_confidence": img_confidence,
            "text_confidence": text_confidence,
            "fused_results": fused_results,
            "fused_top1_prob": fused_top1_prob,
            "fused_margin": fused_margin,
            "topk_overlap": topk_overlap,
            "name_soft_agree": name_soft_agree,
            "img_results": img_results,
            "text_results": text_results,
            "lcb_map": lcb_map,
            "trigger_reason": trigger_reason    
        }
        
        return result
