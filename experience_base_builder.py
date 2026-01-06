import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from tqdm import tqdm
from PIL import Image
import re
from collections import defaultdict
import yaml

if TYPE_CHECKING:
    from agents.mllm_bot_qwen_2_5_vl import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from utils.fileios import dump_json, load_json
from utils.util import is_similar

  
def get_experience_number():
        
    try:
        from discovering import hyperparams
        return hyperparams.experience_number
    except ImportError:
     
        return 8


class ExperienceBaseBuilder:
    """
    Experience base builder for managing and building experience-based knowledge
    """
    
    INITIAL_SELF_BELIEF = "I am a visual classification system that learns from experience to make better predictions."
    
    def __init__(self,
                 mllm_bot: "MLLMBot",
                 knowledge_base_builder: KnowledgeBaseBuilder,
                 fast_thinking_module,
                 slow_thinking_module,
                 device: str = "cuda",
                 dataset_info: dict = None):
            
        self.mllm_bot = mllm_bot
        self.kb_builder = knowledge_base_builder
        self.fast_thinking = fast_thinking_module
        self.slow_thinking = slow_thinking_module
        self.device = device
        self.dataset_info = dataset_info or {}

        self.max_strategy_rules = get_experience_number()      
        self.strategy_rules = []
        self.next_rule_id = 1
        self.self_belief_core = self.INITIAL_SELF_BELIEF
        self.self_belief = self._compose_self_belief_prompt()

        self.pseudo_trajectories = []    

        self.reflection_history = []    
 
        self.save_dir = None

    def initialize_self_belief(self, custom_belief: Optional[str] = None):
            
        self.strategy_rules = []
        self.next_rule_id = 1
        if custom_belief:
            self.self_belief_core = custom_belief
        else:
            self.self_belief_core = self.INITIAL_SELF_BELIEF
        self.self_belief = self._compose_self_belief_prompt()

    def _compose_self_belief_prompt(self) -> str:
            
        if not self.strategy_rules:
            return self.self_belief_core
        
        rules_text = "\n\nPolicy Memory (learned strategy rules):\n"
        for idx, rule in enumerate(self.strategy_rules, start=1):
            condition = rule.get("applicability_signals", "applicable to challenging cases")
            rules_text += f"{idx}. {rule['rule']} (Trigger: {condition})\n"
        return f"{self.self_belief_core}{rules_text}"
    
    def _refresh_self_belief_prompt(self):
            
        self.self_belief = self._compose_self_belief_prompt()
    
    def generate_cot_with_self_belief(self, 
                                      image_path: str,
                                      top_k_candidates: List[Tuple[str, float]],
                                      world_belief_context: Optional[str] = None) -> Tuple[str, str]:

        image = Image.open(image_path).convert("RGB")
        
        candidate_text = ""
        for i, (cat, score) in enumerate(top_k_candidates, start=1):
            candidate_text += f"{i}. {cat} (similarity: {score:.4f})\n"

        prompt = f"""{self.self_belief}
        Candidate classes (highly likely to contain the correct option):
        {candidate_text}"""     

        if world_belief_context:
            prompt += f"\n\nCategory descriptions:\n{world_belief_context}\n"
        
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
                for cat, _ in top_k_candidates:
                    if cat.lower() in last_line.lower() or last_line.lower() in cat.lower():
                        prediction = cat
                        break
                if prediction == "unknown":
                    prediction = top_k_candidates[0][0] if top_k_candidates else "unknown"
            
        print(f"cot: {cot}")
        print(f"prediction: {prediction}")
        return cot, prediction
    
    def build_pseudo_trajectories(self,
                                  validation_samples: Dict[str, List[str]],
                                  top_k: int = 5,
                                  max_samples_per_category: Optional[int] = None,
                                  completed_categories: set = None,
                                  on_category_complete: callable = None) -> List[Dict]:
            
        from datetime import datetime
     
        self.pseudo_trajectories = []
        
        completed_categories = completed_categories or set()
        if completed_categories:
            pass
        
        total_samples = sum(min(len(paths), max_samples_per_category) if max_samples_per_category else len(paths) 
                           for paths in validation_samples.values())
        skipped_samples = sum(min(len(validation_samples[cat]), max_samples_per_category) if max_samples_per_category else len(validation_samples[cat]) 
                             for cat in completed_categories if cat in validation_samples)
        processed = skipped_samples
        total_categories_count = len(validation_samples)
        completed_categories_count = len(completed_categories)
        
        for cat_idx, (true_category, image_paths) in enumerate(validation_samples.items()):
              
            if true_category in completed_categories:
                continue
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cat_progress = (cat_idx / total_categories_count) * 100
            img_progress = (processed / total_samples) * 100 if total_samples > 0 else 0
            remaining_categories = total_categories_count - cat_idx
            remaining_images = total_samples - processed
            
            if max_samples_per_category:
                image_paths = image_paths[:max_samples_per_category]
            
              
            world_belief_context = self._get_world_belief_context(true_category)
            
            for img_idx, img_path in enumerate(image_paths):
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                overall_progress = (processed / total_samples) * 100 if total_samples > 0 else 0
                remaining_images = total_samples - processed
     
                
                  
                fast_result = self.fast_thinking.fast_thinking_pipeline(img_path, top_k)
                top_k_candidates = fast_result.get("fused_results", [])[:top_k]
                
                if not top_k_candidates:
                    processed += 1
                    continue
                
                cot, prediction = self.generate_cot_with_self_belief(
                    img_path, top_k_candidates, world_belief_context
                )
                  
                trajectory = {
                    "image_path": img_path,
                    "cot": cot,
                    "prediction": prediction,
                    "label": true_category,
                    "top_k_candidates": top_k_candidates,
                    "is_correct": is_similar(prediction, true_category, threshold=0.3)
                }
                
                self.pseudo_trajectories.append(trajectory)
                processed += 1
            
              
            completed_categories_count += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cat_completed_pct = (completed_categories_count / total_categories_count) * 100
            img_completed_pct = (processed / total_samples) * 100 if total_samples > 0 else 0
     
            
              
            if on_category_complete:
                on_category_complete(true_category)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return self.pseudo_trajectories
    
    def _get_world_belief_context(self, category: str) -> str:
            
        if hasattr(self.kb_builder, 'category_descriptions'):
              
            description = self.kb_builder.category_descriptions.get(category, "")
            if description:
                return f"{category}: {description}"
        return f"{category}: No description available"
    
    def _normalize_rule_text(self, text: str) -> str:
            
        cleaned = text.lower().strip()
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def _aggregate_rules(self, candidates: List[Dict]) -> List[Dict]:
            
        aggregated = {}
        for candidate in candidates:
            rule_text = candidate.get("rule", "").strip()
            if not rule_text:
                continue
            normalized = self._normalize_rule_text(rule_text)
            if not normalized:
                continue
            if normalized in aggregated:
                aggregated_rule = aggregated[normalized]
                aggregated_rule["support"] += candidate.get("support", 1)
                aggregated_rule["source_images"].extend(candidate.get("source_images", []))
                aggregated_rule["labels"].extend(candidate.get("labels", []))
                aggregated_rule["failure_patterns"].append(candidate.get("failure_pattern", ""))
            else:
                candidate["normalized"] = normalized
                candidate.setdefault("support", 1)
                candidate.setdefault("source_images", [])
                candidate.setdefault("labels", [])
                candidate.setdefault("failure_patterns", [])
                aggregated[normalized] = candidate
        return list(aggregated.values())
    
    def _add_strategy_rule(self, rule_entry: Dict) -> bool:
            
        rule_text = rule_entry.get("rule", "").strip()
        if not rule_text:
            return False
        
        normalized = self._normalize_rule_text(rule_text)
        if not normalized:
            return False
        
        added = False
        for existing in self.strategy_rules:
            if existing.get("normalized") == normalized:
                existing["support"] += rule_entry.get("support", 1)
                existing["source_images"].extend(rule_entry.get("source_images", []))
                existing["labels"].extend(rule_entry.get("labels", []))
                existing["failure_patterns"].extend(rule_entry.get("failure_patterns", []))
                existing["priority"] = rule_entry.get("priority", existing.get("priority", "medium"))
                added = True
                break
        
        if not added:
            rule_entry["normalized"] = normalized
            rule_entry["id"] = f"R{self.next_rule_id}"
            self.next_rule_id += 1
            rule_entry.setdefault("support", 1)
            rule_entry.setdefault("source_images", [])
            rule_entry.setdefault("labels", [])
            rule_entry.setdefault("failure_patterns", [])
            rule_entry.setdefault("priority", "medium")
            rule_entry.setdefault("applicability_signals", "challenging scenarios")
            rule_entry.setdefault("effectiveness", 0.0)
            rule_entry.setdefault("applicability", 0.0)
            self.strategy_rules.append(rule_entry)
            self.reflection_history.append({
                "rule_id": rule_entry["id"],
                "rule": rule_entry["rule"],
                "failure_pattern": rule_entry.get("failure_pattern", ""),
                "source_images": rule_entry.get("source_images", []),
                "added_at": len(self.reflection_history) + 1
            })
            added = True
        
        if len(self.strategy_rules) > self.max_strategy_rules:
            self._manage_rule_capacity()
        
        return added
    
    def _manage_rule_capacity(self):
            
        if len(self.strategy_rules) <= self.max_strategy_rules:
            return
        
        ranked = self._rank_rules_for_retention()
        if ranked:
            keep_ids = set(ranked.get("keep_ids", []))
            if keep_ids:
                self.strategy_rules = [rule for rule in self.strategy_rules if rule["id"] in keep_ids]
        if len(self.strategy_rules) > self.max_strategy_rules:
            self.strategy_rules = self.strategy_rules[:self.max_strategy_rules]
    
    def _rank_rules_for_retention(self) -> Dict:
            
        summary_lines = []
        for rule in self.strategy_rules:
            summary_lines.append(
                f"{rule['id']}: {rule['rule']} | support={rule.get('support', 1)} | "
                f"priority={rule.get('priority', 'medium')} | effectiveness={rule.get('effectiveness', 0.0)} | "
                f"applicability={rule.get('applicability', 0.0)}"
            )
        
        prompt = f"""You are maintaining a policy rule cache for fine-grained recognition. Capacity is {self.max_strategy_rules}.
Rules:
{chr(10).join(summary_lines)}

Please decide which rules to keep to maximize generalization coverage. Return ONLY a JSON object:
{{
    "keep_ids": ["R1", "R2", ...],  // exactly {self.max_strategy_rules} IDs if possible
    "drop_ids": ["R3", ...]
}}"""  
        
        dummy_image = Image.new('RGB', (224, 224), color='white')
        reply, response = self.mllm_bot.describe_attribute(dummy_image, prompt)
        if isinstance(response, list):
            response = " ".join(response)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            decision = json.loads(json_str)
            return decision
        return {}
    
    def reflect_on_failed_samples(self,
                                  failed_trajectories: List[Dict],
                                  max_reflections: int = 5) -> List[Dict]:
            
        if not failed_trajectories:
            return []
        
        selected_samples = failed_trajectories[:max_reflections]
        
        rule_candidates = []
        
        for i, traj in enumerate(selected_samples, start=1):
            
            world_belief_context = self._get_world_belief_context(traj['label'])
            
              
            candidate_text = ""
            for j, (cat, score) in enumerate(traj['top_k_candidates'], start=1):
                candidate_text += f"{j}. {cat} (similarity: {score:.4f})\n"
            
              
            reflection_prompt = f"""You are reviewing a failed fine-grained recognition case.
Candidate classes (highly likely to contain the correct option):
{candidate_text}
Your reasoning: "{traj['cot']}"
Prediction: {traj['prediction']}
Ground truth: {traj['label']}

Please reflect at the strategy level:
1. Correctness: Did your reasoning correctly interpret visual evidence?
2. Consistency: Was your step-by-step logic consistent with known class characteristics?
3. Rationality: Did you focus on the right discriminative parts? Did you ignore key details?
4. Generalization: What general lesson can be learned to avoid similar mistakes on other images?
Then, propose a domain-general rule of thumb that applies to future FGVR cases.

Return ONLY a JSON object:
{{
    "failure_pattern": "concise description of the challenging scenario",
    "root_cause": "brief diagnosis of the reasoning flaw",
    "improved_rule": "Actionable instruction starting with a verb",
    "applicability_signals": "When should this rule be recalled?",
    "priority": "high|medium|low"
}}"""
            
              
            image = Image.open(traj['image_path']).convert("RGB")
            reply, response = self.mllm_bot.describe_attribute(image, reflection_prompt)
            if isinstance(response, list):
                response = " ".join(response)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
            else:
                parsed = {}
            
            improved_rule = parsed.get("improved_rule", "").strip()
            failure_pattern = parsed.get("failure_pattern", "").strip()
            applicability_signals = parsed.get("applicability_signals", "").strip()
            priority = parsed.get("priority", "medium").strip()
            
            candidate = {
                "rule": improved_rule,
                "failure_pattern": failure_pattern,
                "applicability_signals": applicability_signals or "similar ambiguous cases",
                "priority": priority or "medium",
                "support": 1,
                "source_images": [traj['image_path']],
                "labels": [traj['label']],
                "root_cause": parsed.get("root_cause", "").strip(),
                "source_reasoning": traj['cot']
            }
            rule_candidates.append(candidate)
        
        refined_rules = self._aggregate_rules(rule_candidates)
        return refined_rules
    
    def update_self_belief(self, refined_rules: List[Dict]) -> bool:
            
        if not refined_rules:
     
            return False
        
        updated = False
        for rule_entry in refined_rules:
            added = self._add_strategy_rule(rule_entry)
            updated = updated or added

        if updated:
            self._refresh_self_belief_prompt()

        return updated
    
    def evaluate_with_current_belief(self,
                                     validation_samples: Dict[str, List[str]],
                                     top_k: int = 5,
                                     completed_categories: set = None,
                                     on_category_complete: callable = None) -> Dict:
            
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
     
        
        completed_categories = completed_categories or set()
        if completed_categories:
     
        
        correct_count = 0
        total_count = 0
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
          
        total_samples = sum(len(paths) for paths in validation_samples.values())
        skipped_samples = sum(len(validation_samples[cat]) for cat in completed_categories if cat in validation_samples)
        processed = skipped_samples
        total_categories_count = len(validation_samples)
        completed_categories_count = len(completed_categories)
        
        for cat_idx, (true_category, image_paths) in enumerate(validation_samples.items()):
              
            if true_category in completed_categories:
                continue
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cat_progress = (cat_idx / total_categories_count) * 100
            img_progress = (processed / total_samples) * 100 if total_samples > 0 else 0
            remaining_categories = total_categories_count - cat_idx
            remaining_images = total_samples - processed
     
     
     
            
            world_belief_context = self._get_world_belief_context(true_category)
            
            for img_idx, img_path in enumerate(image_paths):
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                overall_progress = (processed / total_samples) * 100 if total_samples > 0 else 0
                remaining_images = total_samples - processed
     
                
                  
                fast_result = self.fast_thinking.fast_thinking_pipeline(img_path, top_k)
                top_k_candidates = fast_result.get("fused_results", [])[:top_k]

                
                  
                cot, prediction = self.generate_cot_with_self_belief(
                    img_path, top_k_candidates, world_belief_context
                )
                
                  
                is_correct = is_similar(prediction, true_category, threshold=0.4)
     
                if is_correct:
                    correct_count += 1
                    category_stats[true_category]["correct"] += 1
                
                total_count += 1
                category_stats[true_category]["total"] += 1
                processed += 1
            
              
            completed_categories_count += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cat_completed_pct = (completed_categories_count / total_categories_count) * 100
            img_completed_pct = (processed / total_samples) * 100 if total_samples > 0 else 0
     
            
              
            if on_category_complete:
                on_category_complete(true_category)
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        result = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "category_stats": dict(category_stats)
        }
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
     
        
        return result
    
    def build_experience_base(self,
                              validation_samples: Dict[str, List[str]],
                              max_iterations: int = 3,
                              top_k: int = 5,
                              max_reflections_per_iter: int = 5,
                              min_improvement: float = 0.01,
                              max_samples_per_category: Optional[int] = None,
                              save_dir: str = None,
                              checkpoint_state: Dict = None) -> Dict:
            
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        checkpoint_state = checkpoint_state or {}
        completed_initial_eval = checkpoint_state.get('completed_initial_eval', False)
        completed_iterations = checkpoint_state.get('completed_iterations', 0)
        completed_final_eval = checkpoint_state.get('completed_final_eval', False)
        
        if checkpoint_state:
            pass
        
        if not completed_initial_eval:
            self.initialize_self_belief()
        else:
            if checkpoint_state.get('best_self_belief'):
                self.self_belief = checkpoint_state['best_self_belief']
        
        if not completed_initial_eval:
            initial_performance = self.evaluate_with_current_belief(validation_samples, top_k)
            best_accuracy = initial_performance["accuracy"]
            best_self_belief = self.self_belief
            
            if save_dir:
                self._save_experience_checkpoint(save_dir, {
                    'completed_initial_eval': True,
                    'completed_iterations': 0,
                    'completed_final_eval': False,
                    'initial_accuracy': best_accuracy,
                    'best_accuracy': best_accuracy,
                    'best_self_belief': best_self_belief,
                    'iteration_results': []
                })
        else:
            best_accuracy = checkpoint_state.get('best_accuracy', 0.0)
            best_self_belief = checkpoint_state.get('best_self_belief', self.self_belief)
            initial_performance = {"accuracy": checkpoint_state.get('initial_accuracy', best_accuracy)}
     
        
        iteration_results = checkpoint_state.get('iteration_results', [])
        
          
        start_iteration = completed_iterations + 1
        
        for iteration in range(start_iteration, max_iterations + 1):
     
     
     
            
              
            trajectories = self.build_pseudo_trajectories(
                validation_samples, top_k, max_samples_per_category
            )
            
              
            failed_trajectories = [t for t in trajectories if not t['is_correct']]
            
            if not failed_trajectories:
     
                break
            
     
            
              
            refined_rules = self.reflect_on_failed_samples(
                failed_trajectories, max_reflections_per_iter
            )
            
            if not refined_rules:
     
                  
                if save_dir:
                    self._save_experience_checkpoint(save_dir, {
                        'completed_initial_eval': True,
                        'completed_iterations': iteration,
                        'completed_final_eval': False,
                        'initial_accuracy': initial_performance["accuracy"],
                        'best_accuracy': best_accuracy,
                        'best_self_belief': best_self_belief,
                        'iteration_results': iteration_results
                    })
                continue
            else:
     
              
            updated = self.update_self_belief(refined_rules)
            if not updated:
     
                  
                if save_dir:
                    self._save_experience_checkpoint(save_dir, {
                        'completed_initial_eval': True,
                        'completed_iterations': iteration,
                        'completed_final_eval': False,
                        'initial_accuracy': initial_performance["accuracy"],
                        'best_accuracy': best_accuracy,
                        'best_self_belief': best_self_belief,
                        'iteration_results': iteration_results
                    })
                continue
            
              
            new_performance = self.evaluate_with_current_belief(validation_samples, top_k)
            new_accuracy = new_performance["accuracy"]
            
            performance_change = new_accuracy - best_accuracy
            
     
            
              
            if performance_change >= min_improvement:
     
                best_accuracy = new_accuracy
                best_self_belief = self.self_belief
            else:
     
                  
                self.self_belief = best_self_belief
            
              
            iteration_result = {
                "iteration": iteration,
                "failed_samples_count": len(failed_trajectories),
                "refined_rules": [rule.get("rule", "") for rule in refined_rules],
                "accuracy_before": best_accuracy - performance_change if performance_change >= min_improvement else best_accuracy,
                "accuracy_after": new_accuracy,
                "performance_change": performance_change,
                "strategy_accepted": performance_change >= min_improvement
            }
            iteration_results.append(iteration_result)
            
              
            if save_dir:
                self._save_experience_checkpoint(save_dir, {
                    'completed_initial_eval': True,
                    'completed_iterations': iteration,
                    'completed_final_eval': False,
                    'initial_accuracy': initial_performance["accuracy"],
                    'best_accuracy': best_accuracy,
                    'best_self_belief': best_self_belief,
                    'iteration_results': iteration_results
                })
            
              
            if performance_change < min_improvement and iteration > 1:
     
                break
        
          
        self.self_belief = best_self_belief
        
          
        if not completed_final_eval:
            final_performance = self.evaluate_with_current_belief(validation_samples, top_k)
        else:
            final_performance = {"accuracy": checkpoint_state.get('final_accuracy', best_accuracy)}
     
        
        result = {
            "initial_accuracy": initial_performance["accuracy"],
            "final_accuracy": final_performance["accuracy"],
            "improvement": final_performance["accuracy"] - initial_performance["accuracy"],
            "best_self_belief": best_self_belief,
            "iteration_results": iteration_results,
            "total_iterations": len(iteration_results)
        }
        
          
        if save_dir:
            self._save_experience_checkpoint(save_dir, {
                'completed_initial_eval': True,
                'completed_iterations': max_iterations,
                'completed_final_eval': True,
                'initial_accuracy': initial_performance["accuracy"],
                'final_accuracy': final_performance["accuracy"],
                'best_accuracy': best_accuracy,
                'best_self_belief': best_self_belief,
                'iteration_results': iteration_results
            })
        
     
     
     
     
     
     
        
        return result
    
    def _save_experience_checkpoint(self, save_dir: str, state: Dict):
            
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, "experience_build_checkpoint.json")
        dump_json(checkpoint_path, state)
     
    
    def _load_experience_checkpoint(self, save_dir: str) -> Optional[Dict]:
            
        checkpoint_path = os.path.join(save_dir, "experience_build_checkpoint.json")
        if os.path.exists(checkpoint_path):
            try:
                state = load_json(checkpoint_path)
                
                  
                if isinstance(state, list):
                    if len(state) > 0:
                        state = state[-1]    
     
     
                    else:
     
                        return None
                elif isinstance(state, dict):
     
                else:
     
                    return None
                
                  
                required_fields = ['completed_initial_eval', 'completed_iterations', 'completed_final_eval']
                missing_fields = [f for f in required_fields if f not in state]
                if missing_fields:
     
                      
                    state.setdefault('completed_initial_eval', False)
                    state.setdefault('completed_iterations', 0)
                    state.setdefault('completed_final_eval', False)
                    state.setdefault('initial_accuracy', 0.0)
                    state.setdefault('best_accuracy', 0.0)
                    state.setdefault('best_self_belief', '')
                    state.setdefault('iteration_results', [])
                
                return state
                
            except Exception as e:
     
                import traceback
                traceback.print_exc()
                return None
        return None
    
    def save_experience_base(self, save_dir: str):
            
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
          
        belief_path = os.path.join(save_dir, "self_belief.txt")
        with open(belief_path, 'w', encoding='utf-8') as f:
            f.write(self.self_belief)
        
          
        trajectories_path = os.path.join(save_dir, "pseudo_trajectories.json")
        dump_json(trajectories_path, self.pseudo_trajectories)
        
          
        history_path = os.path.join(save_dir, "reflection_history.json")
        dump_json(history_path, self.reflection_history)
        
          
        rules_path = os.path.join(save_dir, "strategy_rules.json")
        dump_json(rules_path, self.strategy_rules)
        
     
    
    def load_experience_base(self, load_dir: str):
            
        self.save_dir = load_dir
        
          
        belief_path = os.path.join(load_dir, "self_belief.txt")
        if os.path.exists(belief_path):
            with open(belief_path, 'r', encoding='utf-8') as f:
                self.self_belief = f.read()
              
     
        
          
          
          
          
          
        
          
          
          
          
          
        
          
          
          
          
          
          
          
          
          
          
          
          
          
        
          
          
        
     
    
    def get_self_belief(self) -> str:
            
        return self.self_belief
    
    def get_pseudo_trajectories(self) -> List[Dict]:
            
        return self.pseudo_trajectories  
if __name__ == "__main__":
      
    from agents.mllm_bot_qwen_2_5_vl import MLLMBot
    from knowledge_base_builder import KnowledgeBaseBuilder
    from fast_thinking_optimized import FastThinkingOptimized
    from slow_thinking_optimized import SlowThinkingOptimized
    
      
    mllm_bot = MLLMBot(model_tag="Qwen2.5-VL-7B", model_name="Qwen2.5-VL-7B", device="cuda")
    kb_builder = KnowledgeBaseBuilder(device="cuda")
    fast_thinking = FastThinkingOptimized(kb_builder, device="cuda")
    slow_thinking = SlowThinkingOptimized(mllm_bot, kb_builder, fast_thinking, device="cuda")
    
      
    exp_builder = ExperienceBaseBuilder(
        mllm_bot=mllm_bot,
        knowledge_base_builder=kb_builder,
        fast_thinking_module=fast_thinking,
        slow_thinking_module=slow_thinking,
        device="cuda"
    )
    
      
    validation_samples = {
        "Chihuahua": ["path/to/chihuahua1.jpg", "path/to/chihuahua2.jpg"],
        "Shiba Inu": ["path/to/shiba1.jpg", "path/to/shiba2.jpg"]
    }
    
      
    result = exp_builder.build_experience_base(validation_samples, max_iterations=1)
    
      
    exp_builder.save_experience_base("./experience_base")
    
     

