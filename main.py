import warnings
import torch
import os
import argparse
import json
from tqdm import tqdm
from termcolor import colored
from collections import Counter, defaultdict
from utils.configuration import setup_config, seed_everything
from utils.fileios import dump_json, load_json, dump_txt

from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY
from data.extract_from_testsets import get_test_images_by_percentage, validate_test_set, get_dataset_key_for_test
import importlib
from description_generator import Description_Generator
from retrieval.multimodal_retrieval import MultimodalRetrieval
from fast_slow_thinking_system import FastSlowThinkingSystem
from utils.util import is_similar
import re
import hashlib
import numpy as np
import yaml
import sys
import subprocess

import pprint
import time
import signal

def get_mllm_bot_class(model_name: str):
    if model_name not in MODEL_MODULE_MAPPING:
        raise ValueError(f"Model {model_name} not supported")
    module_path = MODEL_MODULE_MAPPING[model_name]
    try:
        module = importlib.import_module(module_path)
        MLLMBot = getattr(module, 'MLLMBot')
        return MLLMBot
    except Exception as e:
        raise ValueError(f"Error: {e}")

def extract_images_from_knowledge_base(cfg, folder_suffix='', args=None): 
    global randomly_extract_discoverying_set
    dataset_name = cfg['dataset_name']
    if randomly_extract_discoverying_set:
        if folder_suffix == '_random':
            num_per_category = 'random'
            output_suffix = 'random'
        else:
            try:   
                num_per_category = int(folder_suffix[1:])    
                output_suffix = folder_suffix[1:]    
            except (ValueError, IndexError):
                num_per_category = 1
                output_suffix = '1'
        try:
            from data.extract_from_trainsets import extract_and_save_discovery_set
            extract_type = 'knowledge_base' if (args and args.mode == 'build_knowledge_base') else 'fast_slow'
            json_discovery = extract_and_save_discovery_set(
                dataset_name, 
                num_per_category, 
                cfg.get('seed', None if discovery_data_true_random else 42),
                output_suffix,
                extract_type
            )
            return json_discovery
        except Exception as e:
            randomly_extract_discoverying_set = False
            return DATA_DISCOVERY[dataset_name](cfg, folder_suffix=folder_suffix) 
    else:
        return DATA_DISCOVERY[dataset_name](cfg, folder_suffix=folder_suffix)

def load_dataset_config(): 
    global DATASET_CONFIG
    config_path = os.path.join(os.path.dirname(__file__), "configs", "datasets_list.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        DATASET_CONFIG = yaml.safe_load(f)
    return DATASET_CONFIG

def get_dataset_info(dataset_name: str) -> dict:
    global DATASET_CONFIG
    if DATASET_CONFIG is None:
        load_dataset_config()
    if dataset_name not in DATASET_CONFIG['dataset_mapping']:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG['dataset_mapping'].keys())}")
    dataset_info = DATASET_CONFIG['dataset_mapping'][dataset_name].copy()
    experiments_root = DATASET_CONFIG.get('experiments_root', './experiments')
    return dataset_info

def set_current_dataset(dataset_name: str):
    global CURRENT_DATASET
    CURRENT_DATASET = get_dataset_info(dataset_name)
    return CURRENT_DATASET

def prepare_test_samples(cfg, args):      
    test_samples = defaultdict(list)
    if args.use_test_data:
        dataset_key = get_dataset_key_for_test(cfg)
        data_root = cfg.get('data_root', './datasets') 
        try:  
            dataset_info = get_dataset_info(cfg.get('dataset_name'))
            experiments_root = dataset_info.get('experiments_root', './experiments')
            experiment_dir = dataset_info.get('experiment_dir', cfg.get('dataset_name'))
            json_test_file = os.path.join(experiments_root, experiment_dir, 'images_split', 'images_test.json')
            if os.path.exists(json_test_file):
                with open(json_test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                total_images = sum(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                                  for entry in test_data)
                num_classes = len(set(entry[0] for entry in test_data))
                test_stats = {
                    'test_dir': json_test_file,
                    'num_classes': num_classes,
                    'total_images': total_images,
                    'avg_images_per_class': total_images / num_classes if num_classes > 0 else 0,
                    'min_images_per_class': min(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                                              for entry in test_data),
                    'max_images_per_class': max(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                                              for entry in test_data)
                }
            else:  
                test_stats = validate_test_set(dataset_key, data_root)
        except Exception as e:
            raise ValueError(f"Error: {e}")
        if os.path.exists(json_test_file):
            with open(json_test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f) 
            all_images = []
            for entry in test_data:
                if len(entry) >= 3:
                    class_name, class_id, image_paths = entry[0], entry[1], entry[2] 
                    if isinstance(image_paths, list):
                        for img_path in image_paths:
                            all_images.append((img_path, class_name))
                    else:
                        all_images.append((image_paths, class_name))
            if args.test_percentage >= 100.0:
                sampled_images = all_images
            else:
                import random
                random.seed(cfg.get('seed', 42) if not test_data_true_random else None)
                sample_count = max(1, int(len(all_images) * args.test_percentage / 100))
                sampled_images = random.sample(all_images, sample_count)
        else:
            sampled_images = get_test_images_by_percentage(
                dataset_key, 
                args.test_percentage,
                seed=cfg.get('seed', 42),
                use_true_random=test_data_true_random
            )

        for img_path, class_name in sampled_images:
            test_samples[class_name].append(img_path)
        
        avg_per_class = len(sampled_images) / len(test_samples) if test_samples else 0
    else:
        if args.test_data_dir is None:
            raise ValueError("") 
        if args.test_data_dir.endswith('.json'):
              
     
            test_samples = load_test_data_from_json(args.test_data_dir, cfg.get('dataset_name'))
        else:
            dataset_key = get_dataset_key_for_test(cfg)
            from data.class_name_mapper import (
                get_dataset_name_from_key,
                standardize_test_class_name
            )
            dataset_name = get_dataset_name_from_key(dataset_key)
            
            for raw_class_name in os.listdir(args.test_data_dir):
                class_dir = os.path.join(args.test_data_dir, raw_class_name)
                if os.path.isdir(class_dir):
                      
                    if dataset_name:
                        class_name = standardize_test_class_name(raw_class_name, dataset_name)
                    else:
                        class_name = raw_class_name
                    
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(class_dir, img_name)
                            test_samples[class_name].append(img_path)
            
            total_images = sum(len(paths) for paths in test_samples.values())

    return dict(test_samples)


class Hyperparameters: 
    def __init__(self, experience_number: int = 8, classify_top_k: int = 10, use_experience_base: bool = True, vocabulary_free: bool = False):
        self.experience_number = experience_number    
        self.classify_top_k = classify_top_k          
        self.use_experience_base = use_experience_base    
        self.vocabulary_free = vocabulary_free        
    def __repr__(self):
        return f"Hyperparameters(experience_number={self.experience_number}, classify_top_k={self.classify_top_k}, use_experience_base={self.use_experience_base}, vocabulary_free={self.vocabulary_free})"

hyperparams = Hyperparameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discovery', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--mode',  
                        type=str, 
                        default='build_knowledge_base', 
                        choices=['build_knowledge_base', 'evaluate'],    
                        help='operating mode for each stage')  
    parser.add_argument('--config_file_env',  
                        type=str,  
                        default='./configs/env_machine.yml',    
                        help='location of host environment related config file')  
    parser.add_argument('--config_file_expt',    
                        type=str,  
                        default='./configs/expts/bird200_all.yml', 
                        help='location of host experiment related config file') 
      
    parser.add_argument('--num_per_category',    
                        type=str, 
                        default='3',  
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'random'], 
                        )
      
    parser.add_argument('--model',
                        type=str,
                        default=DEFAULT_MODEL,
                        choices=SUPPORTED_MODELS,
                        help=f'MLLM model selection, supported: {", ".join(SUPPORTED_MODELS)}')
    
    parser.add_argument('--knowledge_base_dir', type=str, default='./knowledge_base', help='knowledge base directory')
    parser.add_argument('--query_image', type=str, default=None, help='query image path for classification')
    parser.add_argument('--test_data_dir', type=str, default=None, help='test data directory for evaluation')
    parser.add_argument('--use_test_data', action='store_true', help='use images_test directory for testing')
    parser.add_argument('--test_percentage', type=float, default=100.0, help='percentage of test images to use (0-100)')
    parser.add_argument('--results_out', type=str, default='./results.json', help='output path for results')
    parser.add_argument('--use_slow_thinking', type=bool, default=None, help='force use slow thinking (None for auto)')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='confidence threshold for fast thinking')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='similarity threshold for trigger mechanism')
    
    parser.add_argument('--experience_number', type=int, default=8, help='Maximum number of experiences in experience base')
    parser.add_argument('--classify_top_k', type=int, default=10, help='Number of top_k candidates for classification')
    parser.add_argument('--use_experience_base', type=bool, default=True, help='Whether to use experience base (for ablation experiments)')
    parser.add_argument('--vocabulary_free', type=bool, default=False, help='Whether to use open vocabulary (for ablation experiments)')

    args = parser.parse_args()    

    hyperparams.experience_number = args.experience_number
    hyperparams.classify_top_k = args.classify_top_k
    hyperparams.use_experience_base = args.use_experience_base
    hyperparams.vocabulary_free = args.vocabulary_free
    
    cfg = setup_config(args.config_file_env, args.config_file_expt)  
    
    dataset_name = cfg.get('dataset_name', 'dog')    
    set_current_dataset(dataset_name)
    
    check_and_generate_json_files(dataset_name)
    
    seed_everything(cfg['seed']) 

    expt_id_suffix = f"_{args.num_per_category}"    

    cuda_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    pprint.pprint(vars(args))

    pprint.pprint(cfg)

    _mllm_bot_cache = [None]    
    
    system = None
    
    def load_mllm_bot_if_needed():
        if _mllm_bot_cache[0] is None:
     
            _mllm_bot_cache[0] = get_mllm_bot_class(args.model)
        return _mllm_bot_cache[0]

    if args.mode == 'build_knowledge_base':
        checkpoint_mgr = CheckpointManager(
            dataset_name=cfg.get('dataset_name', 'dog'),
            mode='build_knowledge_base',
            experiment_dir='./experiments',
            project_root=os.path.dirname(os.path.abspath(__file__))
        )
          
        if checkpoint_mgr.check_conflict(is_knowledge_base_build=True):
            checkpoint_mgr.wait_with_warning(wait_seconds=20, interval=10)
          
        checkpoint_mgr.start_session(hyperparams={
            'kshot': int(args.num_per_category) if args.num_per_category != 'random' else -1,
            'experience_number': hyperparams.experience_number,
            'classify_top_k': hyperparams.classify_top_k
        })
        
        def signal_handler(signum, frame):
     
            checkpoint_mgr.record_error(
                stage=StageType.IMAGE_KB_BUILD,
                error=KeyboardInterrupt("User interruption"),
                return_code=128 + signum
            )
            sys.exit(128 + signum)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            MLLMBot = load_mllm_bot_if_needed()
    
            system = FastSlowThinkingSystem(
                model_tag=args.model,
                model_name=args.model,
                device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
                cfg=cfg,
                dataset_info=CURRENT_DATASET,
                mllm_bot_class=MLLMBot,
                checkpoint_mgr=checkpoint_mgr
            )
                
            data_discovery = extract_images_from_knowledge_base(cfg, folder_suffix=expt_id_suffix, args=args)
            train_samples = defaultdict(list)
            for name, path in data_discovery.subcat_to_sample.items():
                for p in path:
                    train_samples[name].append(p)

            completed_image_categories = set(checkpoint_mgr.get_completed_categories(StageType.IMAGE_KB_BUILD))
            completed_text_categories = set(checkpoint_mgr.get_completed_categories(StageType.TEXT_KB_BUILD))
            
            if len(completed_image_categories) >= len(train_samples):
     
            if len(completed_text_categories) >= len(train_samples):

            if not completed_image_categories and not completed_text_categories:
     
            checkpoint_mgr.update_stage_progress(
                StageType.IMAGE_KB_BUILD,
                status=CheckpointStatus.IN_PROGRESS,
                total_categories=len(train_samples)
            )
            
            def on_image_category_complete(category: str):  
                checkpoint_mgr.mark_category_completed(StageType.IMAGE_KB_BUILD, category, verbose=False)
                checkpoint_mgr.save_checkpoint(verbose=False)
            
            def on_text_category_complete(category: str): 
                checkpoint_mgr.mark_category_completed(StageType.TEXT_KB_BUILD, category, verbose=False)
                checkpoint_mgr.save_checkpoint(verbose=False)  
              
            system.load_knowledge_base(args.knowledge_base_dir)
            image_kb, text_kb = system.build_knowledge_base(
                train_samples, 
                save_dir=args.knowledge_base_dir,
                augmentation=True,
                completed_image_categories=completed_image_categories,
                completed_text_categories=completed_text_categories,
                on_image_category_complete=on_image_category_complete,
                on_text_category_complete=on_text_category_complete
            )
              
            checkpoint_mgr.update_stage_progress(
                StageType.IMAGE_KB_BUILD,
                status=CheckpointStatus.COMPLETED
            )
            checkpoint_mgr.update_stage_progress(
                StageType.TEXT_KB_BUILD,
                status=CheckpointStatus.COMPLETED
            )
            checkpoint_mgr.mark_completed()
            
        except Exception as e:
            checkpoint_mgr.record_error(
                stage=StageType.IMAGE_KB_BUILD,
                error=e,
                return_code=1
            )
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    elif args.mode == 'evaluate': 
          
        def signal_handler(signum, frame):
     
            checkpoint_mgr.record_error(
                stage=StageType.FAST_SLOW_EVAL,
                error=KeyboardInterrupt("User interruption"),
                return_code=128 + signum
            )
            sys.exit(128 + signum)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:  
            MLLMBot = load_mllm_bot_if_needed()
            system = FastSlowThinkingSystem(
                model_tag=args.model,
                model_name=args.model,
                device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
                cfg=cfg,
                dataset_info=CURRENT_DATASET,
                mllm_bot_class=MLLMBot,
                checkpoint_mgr=checkpoint_mgr
            )
              
            system.load_knowledge_base(args.knowledge_base_dir)
            system.load_experience_base(args.knowledge_base_dir)
            test_samples = prepare_test_samples(cfg, args)
            from data.result_saver import (
                save_classification_result,
                create_result_entry,
                get_experiment_dir_from_dataset_info
            )
            from datetime import datetime

            completed_categories = checkpoint_mgr.get_completed_categories(StageType.FAST_SLOW_EVAL)
            saved_results = checkpoint_mgr.get_classification_results()
            saved_stats = checkpoint_mgr.get_statistics()
            
              
            correct = saved_stats.get('correct', 0)
            total = saved_stats.get('total', 0)
            fast_only_correct = saved_stats.get('fast_only_correct', 0)
            slow_triggered = saved_stats.get('slow_triggered', 0)
            slow_triggered_correct = saved_stats.get('slow_triggered_correct', 0)
            
            classification_results = list(saved_results)
            
            project_root = os.path.dirname(os.path.abspath(__file__))
            
            pbar = tqdm(test_samples.items())
            for cat_idx, (true_cat, paths) in enumerate(pbar):
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if checkpoint_mgr.should_skip_category(StageType.FAST_SLOW_EVAL, true_cat):
                    continue
                
                checkpoint_mgr.update_stage_progress(
                    StageType.FAST_SLOW_EVAL,
                    current_category=true_cat,
                    current_image_index=0
                )
                
                for img_idx, path in enumerate(paths):
                    try:
                        result = system.classify_single_image(path, use_slow_thinking=None, top_k=hyperparams.classify_top_k)
                        
                        pred = result.get('final_prediction', 'unknown')
                        ok = is_similar(pred, true_cat, threshold=0.3)
                        used_slow = result.get('used_slow_thinking', False)
                        
                        fast_result = result.get('fast_result', {})
                        fast_result_data = {
                            'predicted_category': fast_result.get('predicted_category', 'unknown'),
                            'predicted_fast': fast_result.get('predicted_fast', 'unknown'),
                            'confidence': fast_result.get('confidence', 0.0),
                            'fused_top1': fast_result.get('fused_top1', 'unknown'),
                            'fused_top1_prob': fast_result.get('fused_top1_prob', 0.0),
                            'need_slow_thinking': fast_result.get('need_slow_thinking', False),
                            'img_category': fast_result.get('img_category', 'unknown'),
                            'text_category': fast_result.get('text_category', 'unknown')
                        }
                        
                        slow_result_data = {}
                        if used_slow:
                            slow_result = result.get('slow_result', {})
                            slow_result_data = {
                                'predicted_category': slow_result.get('predicted_category', 'unknown'),
                                'confidence': slow_result.get('confidence', 0.0),
                                'reasoning': slow_result.get('reasoning', '')
                            }
                        
                        result_entry = create_result_entry(
                            label=true_cat,
                            prediction=pred,
                            is_correct=ok,
                            fast_result=fast_result_data,
                            slow_result=slow_result_data if used_slow else None,
                            image_path=path,
                            confidence=result.get('final_confidence', 0.0),
                            project_root=project_root
                        )
                        classification_results.append(result_entry)
                        checkpoint_mgr.add_classification_result(result_entry)
                        
                        if ok:
                            correct += 1
                            if not used_slow:
                                fast_only_correct += 1
                            if used_slow:
                                slow_triggered_correct += 1
                        else:
                        if used_slow:
                            slow_triggered += 1
                        
                        total += 1

                        if total % 3 == 0:
                            checkpoint_mgr.update_stage_progress(
                                StageType.FAST_SLOW_EVAL,
                                completed_images=total,
                                current_image_index=img_idx + 1
                            )
                            checkpoint_mgr.save_checkpoint()
                            
                    except Exception as e:
                        checkpoint_mgr.update_stage_progress(
                            StageType.FAST_SLOW_EVAL,
                            current_image_index=img_idx
                        )
                        checkpoint_mgr.record_error(
                            stage=StageType.FAST_SLOW_EVAL,
                            error=e,
                            return_code=1
                        )
                        raise
                
                checkpoint_mgr.mark_category_completed(StageType.FAST_SLOW_EVAL, true_cat)
                checkpoint_mgr.save_checkpoint()

            acc = correct / total if total > 0 else 0.0
            fast_only_acc = fast_only_correct / (total-slow_triggered) if (total-slow_triggered) > 0 else 0.0
            slow_trigger_ratio = slow_triggered / total if total > 0 else 0.0
            slow_trigger_acc = slow_triggered_correct / slow_triggered if slow_triggered > 0 else 0.0

            try:
                dataset_key = get_dataset_key_for_test(cfg)
                experiment_dir = get_experiment_dir_from_dataset_info(CURRENT_DATASET)
                if experiment_dir:
                    metadata = {
                        'accuracy': acc,
                        'correct': correct,
                        'total': total,
                        'fast_only_correct': fast_only_correct,
                        'slow_triggered': slow_triggered,
                        'slow_triggered_correct': slow_triggered_correct,
                        'fast_only_acc': fast_only_acc,
                        'slow_trigger_ratio': slow_trigger_ratio,
                        'slow_trigger_acc': slow_trigger_acc,
                        'resume_count': len(checkpoint_mgr.checkpoint_data.get('resume_history', []))
                    }
                    
                    test_data_type = 'test' if args.use_test_data else 'discovery'
                    test_percentage = args.test_percentage if args.use_test_data else None
                    
                    save_path = save_classification_result(
                        dataset_name=dataset_key,
                        experiment_dir=experiment_dir,
                        results=classification_results,
                        metadata=metadata,
                        test_data_type=test_data_type,
                        test_percentage=test_percentage
                    )
     
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            checkpoint_mgr.update_stage_progress(
                StageType.FAST_SLOW_EVAL,
                status=CheckpointStatus.COMPLETED,
                completed_categories=len(test_samples),
                completed_images=total
            )
            checkpoint_mgr.mark_completed()
            
        except Exception as e:
            checkpoint_mgr.record_error(
                stage=StageType.FAST_SLOW_EVAL,
                error=e,
                return_code=1
            )
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        raise NotImplementedError 
