import sys
import os
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.util import encode_base64, prepare_qwen2_5_input, get_important_image_tokens, create_attention_mask
from utils.model_loader import (
    load_qwen_model_safe, 
    ModelLoadTimeoutError, 
    ModelLoadLockError,
    cleanup_model_locks
)

import torch
from os import path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from agents.CFG import CFGLogits 
from agents.attention import qwen_modify, qwen_modify_with_importance
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce

QWEN = {
    'Qwen2.5-VL-7B': 'Qwen/Qwen2.5-VL-7B-Instruct'
      
}

ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t ' \
                     'know honestly. Don\'t imagine any contents that are not in the image.'

SUB_ANSWER_INSTRUCTION = 'Answer: '    def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n + n_addition_q):]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []

    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + 'Question: {}'.format(questions[-1])
    else:
        chat_log = chat_log[:-2]
    return chat_logdef trim_answer(answer):
    if isinstance(answer, list):
        return answer
    answer = answer.split('Question:')[0].replace('\n', ' ').strip()
    return answerclass MLLMBot:
    def __init__(self, model_tag, model_name, pai_enable_attn=False, device='cpu', device_id=0, bit8=False, max_answer_tokens=-1, checkpoint_mgr=None):
        self.model_tag = model_tag
        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens
        self.checkpoint_mgr = checkpoint_mgr
        local_model_path_abs = "./models/Qwen"
        local_model_path = path.join(local_model_path_abs, QWEN[self.model_tag].split('/')[-1])
        if self.checkpoint_mgr and self.checkpoint_mgr.should_skip_model_loading():
        try:
              
            if self.checkpoint_mgr:
                self.checkpoint_mgr.start_model_loading(
                    model_name=f"{model_tag} (MLLMBot)",
                    model_path=local_model_path,
                    timeout_seconds=600,    
                    max_retries=2
                )
            cleanup_model_locks("./models")  
            self.qwen2_5, self.qwen2_5_processor = load_qwen_model_safe(
                model_tag=model_tag,
                model_path=local_model_path,
                device=device,
                device_id=device_id,
                bit8=bit8,
                timeout=600,    
                max_retries=2,
                use_lock=True
            )  
            self.device = 'cpu' if device == 'cpu' else f'cuda:{device_id}'
            self.bit8 = bit8
            if device != 'cpu' and hasattr(self.qwen2_5, 'gradient_checkpointing_enable'):
                self.qwen2_5.gradient_checkpointing_enable()
   
            if self.checkpoint_mgr:
                self.checkpoint_mgr.complete_model_loading(success=True)
            
            if device != 'cpu':
        except (ModelLoadTimeoutError, ModelLoadLockError) as e:
            error_msg = f"ERROR: {e}"

            if self.checkpoint_mgr:
                self.checkpoint_mgr.complete_model_loading(success=False, error_msg=error_msg)
              
            if self.checkpoint_mgr and self.checkpoint_mgr.can_retry_model_loading():
                retry_count = self.checkpoint_mgr.get_model_loading_retry_count() + 1
                self.checkpoint_mgr.record_model_loading_retry(retry_count, str(e))
     
            else:
            raise
            
        except Exception as e:
            error_msg = f"ERROR: {e}"

            if self.checkpoint_mgr:
                self.checkpoint_mgr.complete_model_loading(success=False, error_msg=error_msg)
            
            raise
        
        self.pai_enable_attn = pai_enable_attn     
        self.pai_alpha = 0.5             
        self.pai_layers = (10, 28)       
        self.pai_enable_cfg = False      
        self.pai_gamma = 1.1             
        self.num_map = 0
        
    def __del__(self):
        try:
            if hasattr(self, 'qwen2_5'):
                del self.qwen2_5
            if hasattr(self, 'qwen2_5_processor'):
                del self.qwen2_5_processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
              
            cleanup_model_locks("./models")
        except Exception as e:
     
    
    def cleanup(self):
        try:
            if hasattr(self, 'qwen2_5'):
                del self.qwen2_5
            if hasattr(self, 'qwen2_5_processor'):
                del self.qwen2_5_processor
            torch.cuda.empty_cache()
        except Exception as e:
     
        
    def _get_model_device(self):
        try:
            return self.qwen2_5.model.embed_tokens.weight.device
        except Exception:
              
            try:
                return next(self.qwen2_5.parameters()).device
            except Exception:
                return torch.device(self.device)
          def _resolve_img_token_span(self, messages, inputs):

        try:
            input_ids = inputs.input_ids
            if input_ids is None:
     
                return None, None
            seq_len = input_ids.shape[1]
              
            tokenizer = self.qwen2_5_processor.tokenizer
              
            vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
            image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
            vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
     
            input_ids_list = input_ids[0].tolist()

            if vision_start_id in input_ids_list and vision_end_id in input_ids_list:
                img_start = input_ids_list.index(vision_start_id)
                img_end   = input_ids_list.index(vision_end_id) + 1    
                return img_start, img_end
            else:
                return None, None
        except Exception as e:
            return None, None

    def _inject_qwen_pai_attention(self, img_start_idx, img_end_idx):
        if img_start_idx is None or img_end_idx is None:
            return
        qwen_modify(self.qwen2_5, self.pai_layers[0], self.pai_layers[1], True, self.pai_alpha, False, img_start_idx, img_end_idx)

    def _inject_qwen_pai_attention_with_importance(self, img_start_idx, img_end_idx, important_tokens_info):
        if img_start_idx is None or img_end_idx is None:
            return
        importance_weights = important_tokens_info['weights']    
        important_indices = important_tokens_info['important_indices']    
        
        qwen_modify_with_importance(self.qwen2_5, self.pai_layers[0], self.pai_layers[1], True, self.pai_alpha, False, img_start_idx, img_end_idx, importance_weights, important_indices)

    def get_name(self):
        return self.model_name
    
    def _resize_image_if_needed(self, image: Image.Image, max_size: int = pre_define_max_size) -> Image.Image:  
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > max_size:
            scale = max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
        return image


    def __call_qwen2_5(self, raw_image, prompt, max_new_tokens=256):
        if isinstance(raw_image, Image.Image):
            raw_image = [raw_image]

        content = []
          
        for img in raw_image:
              
            img = self._resize_image_if_needed(img, max_size=pre_define_max_size)
            image_str = encode_base64(img)
            content.append({"type": "image", "image": f'data:image;base64,{image_str}'})
        content.append({"type": "text", "text": prompt})
          
        messages = [
            {
                "role": "user",
                "content": content
            }
        ] 
        model_device = self._get_model_device()
        inputs = prepare_qwen2_5_input(messages, self.qwen2_5_processor)
        inputs = inputs.to(model_device)   
        
          
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        if "image_grid_thw" in inputs: 
            pass

        if self.pai_enable_attn:
              
            from agents.qwen2_5_methods import rel_attention_qwen2_5
            general_question = 'Write a general description of the image.'
              
            formatted_prompt = f"<image>\nUSER: {prompt} Answer the question with a concise phrase.\nASSISTANT:"
            general_prompt = f"<image>\nUSER: {general_question} Answer the question with a concise phrase.\nASSISTANT:"
            att_map = rel_attention_qwen2_5(raw_image[0], formatted_prompt, general_prompt, self.qwen2_5, self.qwen2_5_processor)
            import matplotlib.pyplot as plt
            path='./maps/'
            os.makedirs(path, exist_ok=True)
            plt.imshow(att_map, interpolation='none')
            plt.axis('off')
            plt.savefig(
                fname=f"{path}/attention_map_{self.num_map}.png",    
                dpi=300,                    
                bbox_inches='tight',          
                pad_inches=0                  
            )
            self.num_map +=1

            important_tokens_info = get_important_image_tokens(att_map, inputs, self.qwen2_5_processor, threshold=1)
     
            if len(important_tokens_info['important_indices']) > 0:
            img_start_idx, img_end_idx = self._resolve_img_token_span(messages, inputs)

            self._inject_qwen_pai_attention_with_importance(img_start_idx, img_end_idx, important_tokens_info)
            
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3    
            memory_reserved = torch.cuda.memory_reserved() / 1024**3     
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3    
            memory_free = memory_total - memory_allocated
      
            if memory_free < 12:  
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                  
                memory_after_clear = torch.cuda.memory_allocated() / 1024**3
                memory_free_after = memory_total - memory_after_clear
     
                if memory_free_after < 10:
                    torch.cuda.reset_peak_memory_stats()
     
        with torch.no_grad():
            generated_ids = self.qwen2_5.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                       
                use_cache=True,                        
                num_beams=1,                           
                pad_token_id=self.qwen2_5_processor.tokenizer.eos_token_id,  
            )
            
        input_ids = inputs.input_ids
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        reply = self.qwen2_5_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        del inputs, input_ids, generated_ids, generated_ids_trimmed

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_free = memory_total - memory_after
     
            if memory_free < 10:
                torch.cuda.empty_cache()
 
                if memory_reserved > memory_free and memory_free < 10:
                    torch.cuda.reset_peak_memory_stats()
      
        return reply

    def answer_chat_log(self, raw_image, chat_log, n_qwen2_5_context=-1):
        qwen2_5_prompt = '\n'.join([ANSWER_INSTRUCTION,
                                  get_chat_log(chat_log['questions'],chat_log['answers'],
                                               last_n=n_qwen2_5_context), SUB_ANSWER_INSTRUCTION]
                                 )
        reply = self.__call_qwen2_5(raw_image, qwen2_5_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def tell_me_the_obj(self, raw_image, super_class, super_unit):
        std_prompt = f"Questions: What is the {super_unit} of the {super_class} in this photo? Answer:"
          
        reply = self.__call_qwen2_5(raw_image, std_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def describe_attribute(self, raw_image, attr_prompt, max_new_tokens=256):
        reply = self.__call_qwen2_5(raw_image, attr_prompt, max_new_tokens)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply
    
    def compare_attention_enhancement(self, raw_image, attr_prompt, save_dir="./experiments/attention_comparison"):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        original_attn = self.pai_enable_attn
        self.pai_enable_attn = False
        
        reply_no_enhance, _ = self.describe_attribute(raw_image, attr_prompt)
     
        self.pai_enable_attn = True
        
        reply_with_enhance, _ = self.describe_attribute(raw_image, attr_prompt)
     
        self.pai_enable_attn = original_attn
    
        with open(os.path.join(save_dir, "comparison_results.txt"), "w", encoding="utf-8") as f:
            f.write("ATTENTION ENHANCEMENT COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Prompt: {attr_prompt}\n\n")
            f.write(f"Without enhancement: {reply_no_enhance}\n\n")
            f.write(f"With enhancement: {reply_with_enhance}\n\n")
            f.write(f"Enhancement layers: {self.pai_layers}\n")
            f.write(f"Alpha value: {self.pai_alpha}\n")
        return reply_no_enhance, reply_with_enhance

    def caption(self, raw_image):
          
        std_prompt = 'a photo of'
        reply = self.__call_qwen2_5(raw_image, std_prompt)
        reply = reply.replace('\n', ' ').strip()    
        return reply

    def call_llm(self, prompts):
        prompts_temp = self.qwen2_5_processor(None, prompts, return_tensors="pt")
        model_device = self._get_model_device()
        input_ids = prompts_temp['input_ids'].to(model_device)
        attention_mask = prompts_temp['attention_mask'].to(model_device)

        prompts_embeds = self.qwen2_5.language_model.get_input_embeddings()(input_ids)

        with torch.no_grad():
            outputs = self.qwen2_5.language_model.generate(
                inputs_embeds=prompts_embeds,
                attention_mask=attention_mask)

        outputs = self.qwen2_5_processor.decode(outputs[0], skip_special_tokens=True)
        return outputs
