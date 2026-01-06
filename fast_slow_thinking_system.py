import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import torch
from collections import defaultdict
import threading
from datetime import datetime

from knowledge_base_builder import KnowledgeBaseBuilder
from experience_base_builder import ExperienceBaseBuilder
from fast_thinking import FastThinking
from fast_thinking_optimized import FastThinkingOptimized
from slow_thinking_optimized import SlowThinkingOptimized
from slow_thinking import SlowThinking
from utils.fileios import dump_json, load_json
from utils.util import is_similar


class FastSlowThinkingSystem:
    """
    Fast-Slow Thinking System that combines fast thinking (similarity-based) 
    and slow thinking (LLM-based reasoning) for image classification.
    """
    
    def __init__(self, 
                 model_tag: str = "Qwen2.5-VL-7B",
                 model_name: str = "Qwen2.5-VL-7B",
                 image_encoder_name: str = "./models/Clip/clip-vit-base-patch32",
                 text_encoder_name: str = "./models/Clip/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cfg: Optional[Dict] = None,
                 dataset_info: Optional[Dict] = None,
                 mllm_bot_class = None,
                 checkpoint_mgr = None):
            
        self.device = device
        self.cfg = cfg or {}
        self.dataset_info = dataset_info or {}

        if mllm_bot_class is None:
            from agents.mllm_bot_qwen_2_5_vl import MLLMBot as DefaultMLLMBot
            mllm_bot_class = DefaultMLLMBot
        
        self.mllm_bot = mllm_bot_class(
            model_tag=model_tag,
            model_name=model_name,
            device=device,
            checkpoint_mgr=checkpoint_mgr
        )
        
        self.kb_builder = KnowledgeBaseBuilder(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            device=device,
            cfg=cfg,
            dataset_info=self.dataset_info
        )
        
        self.fast_thinking = FastThinkingOptimized(
            knowledge_base_builder=self.kb_builder,
            confidence_threshold=self.cfg.get('confidence_threshold', 0.8),
            similarity_threshold=self.cfg.get('similarity_threshold', 0.7),
            dataset_info=self.dataset_info
        )
        
        self.slow_thinking = SlowThinkingOptimized(
            mllm_bot=self.mllm_bot,
            knowledge_base_builder=self.kb_builder,
            fast_thinking=self.fast_thinking,
            dataset_info=self.dataset_info
        )
        
        self.exp_builder = None
        
        self.memory_monitor_stop = threading.Event()
        self.memory_monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.memory_monitor_thread.start()

    def __del__(self):
            
        self.cleanup()
        
    def _monitor_memory(self):
            
        while not self.memory_monitor_stop.is_set():
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_free = memory_total - memory_allocated
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
     
            self.memory_monitor_stop.wait(3)
    
    def cleanup(self):
            
        try:
              
            if hasattr(self, 'memory_monitor_stop'):
                self.memory_monitor_stop.set()
            if hasattr(self, 'memory_monitor_thread') and self.memory_monitor_thread.is_alive():
                self.memory_monitor_thread.join(timeout=2)
            
            if hasattr(self, 'mllm_bot') and self.mllm_bot:
                self.mllm_bot.cleanup()
                del self.mllm_bot
            if hasattr(self, 'kb_builder'):
                del self.kb_builder
            if hasattr(self, 'fast_thinking'):
                del self.fast_thinking
            if hasattr(self, 'slow_thinking'):
                del self.slow_thinking
            if hasattr(self, 'exp_builder'):
                del self.exp_builder
            torch.cuda.empty_cache()
     
        except Exception as e:
     
    def build_knowledge_base(self, train_samples: Dict[str, List[str]], 
                           save_dir: str = "./knowledge_base",
                           augmentation: bool = True,
                           completed_image_categories: set = None,
                           completed_text_categories: set = None,
                           on_image_category_complete: callable = None,
                           on_text_category_complete: callable = None) -> Tuple[Dict, Dict]:
        
        if completed_image_categories:
            pass
        if completed_text_categories:
            pass
        
        image_kb, text_kb = self.kb_builder.build_knowledge_base(
            self.mllm_bot, train_samples, augmentation,
            completed_image_categories, completed_text_categories,
            on_image_category_complete, on_text_category_complete
        )
        self.kb_builder.save_knowledge_base(save_dir)
        self.initialize_experience_base()
        
          
        exp_checkpoint = self.exp_builder._load_experience_checkpoint(save_dir)
        
        self.exp_builder.build_experience_base(
            train_samples,
            max_iterations=1, 
            max_reflections_per_iter=3, 
            top_k=5,
            save_dir=save_dir,
            checkpoint_state=exp_checkpoint
        )
        self.exp_builder.save_experience_base(save_dir)
        
        stats_summary = self.kb_builder.initialize_lcb_stats_with_labels(
            train_samples, self.fast_thinking, top_k=5
        )

        return image_kb, text_kb
    
    def load_knowledge_base(self, load_dir: str = "./knowledge_base"):
        self.kb_builder.load_knowledge_base(load_dir)
     
    
    def initialize_experience_base(self):
            
        if self.exp_builder is None:
     
            self.exp_builder = ExperienceBaseBuilder(
                mllm_bot=self.mllm_bot,
                knowledge_base_builder=self.kb_builder,
                fast_thinking_module=self.fast_thinking,
                slow_thinking_module=self.slow_thinking,
                device=self.device,
                dataset_info=self.dataset_info
            )
     
        return self.exp_builder
    
    def build_experience_base(self,
                              validation_samples: Dict[str, List[str]],
                              max_iterations: int = 3,
                              top_k: int = 5,
                              max_reflections_per_iter: int = 5,
                              min_improvement: float = 0.01,
                              max_samples_per_category: Optional[int] = None,
                              save_dir: str = "./experience_base") -> Dict:
            
          
        exp_builder = self.initialize_experience_base()
        
          
        result = exp_builder.build_experience_base(
            validation_samples=validation_samples,
            max_iterations=max_iterations,
            top_k=top_k,
            max_reflections_per_iter=max_reflections_per_iter,
            min_improvement=min_improvement,
            max_samples_per_category=max_samples_per_category
        )
        
          
        exp_builder.save_experience_base(save_dir)
        
        return result
    
    def load_experience_base(self, load_dir: str = "./experience_base"):
            
          
        exp_builder = self.initialize_experience_base()
        
          
        exp_builder.load_experience_base(load_dir)
        
          
        if hasattr(self.slow_thinking, 'set_experience_base'):
            self.slow_thinking.set_experience_base(exp_builder)

    def classify_single_image(self, query_image_path: str, 
                            use_slow_thinking: bool = None,
                            top_k: int = 5) -> Dict:
            
     
        start_time = time.time()
        
        fast_result = self.fast_thinking.fast_thinking_pipeline(query_image_path, top_k)
        fast_time = time.time() - start_time
        
        result = {
            "query_image": query_image_path,
            "fast_result": fast_result,
            "fast_time": fast_time,
            "total_time": fast_time
        }
        
          
        need_slow_thinking = use_slow_thinking if use_slow_thinking is not None else fast_result["need_slow_thinking"]
        
        if need_slow_thinking:
    
            slow_start_time = time.time()

            slow_result = self.slow_thinking.slow_thinking_pipeline_optimized(
                query_image_path, fast_result, top_k
            )         
            slow_time = time.time() - slow_start_time
        
            fast_pred = fast_result.get("fused_top1", fast_result.get("predicted_category", "unknown"))
            slow_pred = slow_result["predicted_category"]
            fast_conf = fast_result.get("fused_top1_prob", fast_result.get("confidence", 0.0))
            slow_conf = slow_result.get("confidence", 0.0)
            fast_slow_consistent = is_similar(fast_pred, slow_pred, threshold=0.5)
            
            result.update({
                "slow_result": slow_result,
                "slow_time": slow_time,
                "total_time": fast_time + slow_time,
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "final_reasoning": final_reasoning,
                "used_slow_thinking": True,
                "fast_slow_consistent": is_similar(fast_pred, slow_pred, threshold=0.5)
            })
        else:
     
            final_prediction = fast_result["predicted_category"]
            final_confidence = fast_result["confidence"]
            
            result.update({
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "final_reasoning": "Fast thinking result",
                "used_slow_thinking": False
            })
        
        total_time = time.time() - start_time
        result["total_time"] = total_time
        
     
     
        
        return result
    
    def classify_batch_images(self, query_image_paths: List[str],
                            use_slow_thinking: bool = None,
                            top_k: int = 5) -> List[Dict]:
            
     
        results = []
        
        for i, img_path in enumerate(tqdm(query_image_paths, desc="Classification progress")):
            try:
                result = self.classify_single_image(img_path, use_slow_thinking, top_k)
                results.append(result)
            except Exception as e:
     
                results.append({
                    "query_image": img_path,
                    "final_prediction": "error",
                    "final_confidence": 0.0,
                    "error": str(e)
                })
        
     
        return results
    
    def evaluate_on_dataset(self, test_samples: Dict[str, List[str]],
                          use_slow_thinking: bool = None,
                          top_k: int = 5) -> Dict:
            
     
        
        all_results = []
        correct_count = 0
        total_count = 0
        fast_thinking_count = 0
        slow_thinking_count = 0
        
        for true_category, image_paths in test_samples.items():
     
            
            for img_path in image_paths:
                try:
                    result = self.classify_single_image(img_path, use_slow_thinking, top_k)
                    
                      
                    predicted = result["final_prediction"]
                    is_correct = is_similar(predicted, true_category, threshold=0.5)
                    
                    if is_correct:
                        correct_count += 1
                    
                    total_count += 1
                    
                    if result.get("used_slow_thinking", False):
                        slow_thinking_count += 1
                    else:
                        fast_thinking_count += 1
                    
                    result["true_category"] = true_category
                    result["is_correct"] = is_correct
                    all_results.append(result)
                    
                except Exception as e:
     
                    total_count += 1
                    all_results.append({
                        "query_image": img_path,
                        "true_category": true_category,
                        "final_prediction": "error",
                        "is_correct": False,
                        "error": str(e)
                    })
        
          
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        fast_thinking_ratio = fast_thinking_count / total_count if total_count > 0 else 0.0
        slow_thinking_ratio = slow_thinking_count / total_count if total_count > 0 else 0.0
        
          
        total_times = [r.get("total_time", 0) for r in all_results if "total_time" in r]
        avg_time = np.mean(total_times) if total_times else 0.0
        
          
        confidences = [r.get("final_confidence", 0) for r in all_results if "final_confidence" in r]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        evaluation_result = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "fast_thinking_count": fast_thinking_count,
            "slow_thinking_count": slow_thinking_count,
            "fast_thinking_ratio": fast_thinking_ratio,
            "slow_thinking_ratio": slow_thinking_ratio,
            "avg_time": avg_time,
            "avg_confidence": avg_confidence,
            "detailed_results": all_results
        }
        
     
     
     
     
     
     
        
        return evaluation_result
    
    def save_results(self, results: List[Dict], save_path: str):
            
          
        cleaned_results = []
        for result in results:
            cleaned_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    cleaned_result[key] = value
                else:
                    cleaned_result[key] = str(value)
            cleaned_results.append(cleaned_result)
        
        dump_json(save_path, cleaned_results)
     
    
    def get_system_stats(self) -> Dict:
            
        stats = {
            "image_kb_size": len(self.kb_builder.image_knowledge_base),
            "text_kb_size": len(self.kb_builder.text_knowledge_base),
            "device": self.device,
            "model_tag": self.mllm_bot.model_tag if hasattr(self.mllm_bot, 'model_tag') else "unknown",
            "confidence_threshold": self.fast_thinking.confidence_threshold,
            "similarity_threshold": self.fast_thinking.similarity_threshold
        }
        return stats
    
    def _final_decision(self, query_image_path: str, fast_result: Dict, slow_result: Dict, top_k: int = 5) -> Tuple[str, float, str]:
            
        from PIL import Image
        import json
        import re
        
        image = Image.open(query_image_path).convert("RGB")
        
          
        fast_candidates = fast_result.get("fused_results", [])[:top_k]
        fast_candidates_text = ""
        for i, (category, score) in enumerate(fast_candidates):
            fast_candidates_text += f"{i+1}. {category} (fast similarity: {score:.4f})\n"
        
          
        slow_candidates = slow_result.get("enhanced_results", [])[:top_k]
        slow_candidates_text = ""
        for i, (category, score) in enumerate(slow_candidates):
            slow_candidates_text += f"{i+1}. {category} (slow similarity: {score:.4f})\n"
        
          
        structured_description = slow_result.get("structured_description", "")
        
        prompt = f"""You are an expert in fine-grained visual recognition. I need you to make a final decision between two different analysis approaches.

Fast Thinking Analysis (CLIP-based retrieval):
{fast_candidates_text}

Slow Thinking Analysis (MLLM + detailed analysis):
{slow_candidates_text}

Detailed visual analysis from slow thinking:
{structured_description}

Please carefully analyze the image and consider:
1. The visual characteristics described in the detailed analysis
2. The similarity scores from both approaches
3. The consistency between fast and slow thinking results
4. Which approach provides more reliable evidence for classification

Make your final prediction in JSON format:
{{
    "predicted_category": "exact category name",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of your decision process",
    "chosen_approach": "fast" or "slow" or "hybrid",
    "key_evidence": "main visual evidence supporting your decision"
}}"""
         
        
        try:
            reply, response = self.mllm_bot.describe_attribute(image, prompt)
            if isinstance(response, list):
                response = " ".join(response)
            
              
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "No reasoning provided")
                    chosen_approach = result.get("chosen_approach", "hybrid")
                    key_evidence = result.get("key_evidence", "No evidence provided")
                    
                      
                    enhanced_reasoning = f"Final Decision - {chosen_approach.upper()}: {reasoning} | Key Evidence: {key_evidence}"
                    
                else:
                      
                    predicted_category = slow_result["predicted_category"]
                    confidence = slow_result["confidence"]
                    reasoning = "JSON parsing failed, using slow thinking result as fallback"
                    enhanced_reasoning = reasoning
                    
            except (json.JSONDecodeError, ValueError):
                predicted_category = slow_result["predicted_category"]
                confidence = slow_result["confidence"]
                reasoning = "JSON parsing failed, using slow thinking result as fallback"
                enhanced_reasoning = reasoning
            
            return predicted_category, confidence, enhanced_reasoning
            
        except Exception as e:
     
            import traceback
            traceback.print_exc()
            
              
            predicted_category = slow_result.get("predicted_category", slow_result.get("category", "unknown"))
            confidence = slow_result.get("confidence", slow_result.get("similarity", 0.0))
            
              
            if not predicted_category or predicted_category == "unknown":
                if fast_result and "predicted_category" in fast_result:
                    predicted_category = fast_result["predicted_category"]
                    confidence = fast_result.get("confidence", 0.0)
                else:
                      
                    fused_results = fast_result.get("fused_results", [])
                    if fused_results:
                        predicted_category = fused_results[0][0]
                        confidence = fused_results[0][1]
                    else:
                        predicted_category = "unknown"
                        confidence = 0.0
            
            return predicted_category, confidence, f"Final decision failed: {str(e)}"   
    