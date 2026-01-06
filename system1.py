    

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
    
    def trigger_dynamic_sare(self, category: str, p_hat_c: float, 
                         P_fuse_distribution: np.ndarray, top1_prob: float) -> Tuple[bool, Dict]:

        eta = 1.0   
        alpha = 0.5 
        threshold_T = 0.6 


        stats = self.category_stats.get(category, {"n_c": 1}) 
        n_c = stats.get("n_c", 1)
        N_total = self.total_support_samples 


        if n_c > 0:
            hoeffding_term = eta * np.sqrt(np.log(N_total) / (2 * n_c))
        else:
            hoeffding_term = 1.0 # Max penalty if unseen


        entropy_term = -np.sum(P_fuse_distribution * np.log(P_fuse_distribution + 1e-12))
        weighted_entropy = alpha * entropy_term


        G_score = p_hat_c - hoeffding_term - weighted_entropy

        trigger_details = {
            "G_score": float(G_score),
            "p_hat_c": float(p_hat_c),
            "hoeffding_penalty": float(hoeffding_term),
            "entropy_penalty": float(weighted_entropy),
            "n_c": n_c,
            "threshold": threshold_T
        }


        if G_score > threshold_T:
            return False, trigger_details
        else:
            return True, trigger_details  
        
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

            img_dict = dict(img_results)
            text_dict = dict(text_results)
            all_candidates = list(set(img_dict.keys()) | set(text_dict.keys()))


            v_scores = np.array([img_dict.get(c, -1e9) for c in all_candidates])
            t_scores = np.array([text_dict.get(c, -1e9) for c in all_candidates])

            # Apply temperature scaling
            P_img = softmax(v_scores / self.softmax_temp)
            P_text = softmax(t_scores / self.softmax_temp)

            # 4. Linear Fusion (Eq. 3) 
            # "P_fuse = lambda * P_img + (1 - lambda) * P_text" where lambda = 0.3
            lambda_val = 0.3
            P_fuse = lambda_val * P_img + (1 - lambda_val) * P_text

            fuse_dict = {c: p for c, p in zip(all_candidates, P_fuse)}
            
            # Sort to find Top-1 candidate
            sorted_candidates = sorted(fuse_dict.items(), key=lambda x: x[1], reverse=True)
            top1_category, top1_prob = sorted_candidates[0]

            # 5. Reciprocal Rank Fusion (Eq. 4) [cite: 559, 561]
            # "R(c) = Sum(1 / (k + r_m(c)))" where k=60
            # Note: Ranks must be 1-based.
            k_rrf = 60.0
            
            # Get ranks for the top1_category in both lists
            # If not present in top_k, assign a large rank penalty (e.g., top_k + 1)
            rank_v = next((i + 1 for i, (c, _) in enumerate(img_results) if c == top1_category), top_k + 100)
            rank_t = next((i + 1 for i, (c, _) in enumerate(text_results) if c == top1_category), top_k + 100)
            
            RRF_score = (1 / (k_rrf + rank_v)) + (1 / (k_rrf + rank_t))

            # 6. Final Retrieval Confidence (Eq. 5) [cite: 566]
            # "p_hat_c = P_fuse(c) + beta * R(c)" where beta = 0.1
            beta_val = 0.1
            p_hat_c = top1_prob + (beta_val * RRF_score)

            # 7. Dynamic Trigger Decision (Eq. 6) 
            need_slow_thinking, trigger_details = self.trigger_dynamic_sare(
                category=top1_category,
                p_hat_c=p_hat_c,
                P_fuse_distribution=P_fuse, # Needed for Entropy
                top1_prob=top1_prob
            )

            result = {
                "predicted_category": top1_category,
                "need_slow_thinking": need_slow_thinking,
                "confidence": p_hat_c,
                "fused_top1_prob": top1_prob,
                "rrf_score": RRF_score,
                "trigger_details": trigger_details,
                "img_results": img_results,
                "text_results": text_results
            }
            
            return result
