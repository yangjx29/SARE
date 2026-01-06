import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from agents.mllm_bot_qwen_2_5_vl import MLLMBot
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
import json
from utils.fileios import dump_json, load_json, dump_txt 
from utils.util import *

class Description_Generator:
    def __init__(self, image_encoder_name="./models/Clip/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu", cfg = None):
        self.device = device
        self.cfg = cfg
        self.clip_model = CLIPModel.from_pretrained(image_encoder_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(image_encoder_name)

    def extract_image_feat(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            feat = self.clip_model.get_image_features(**inputs).cpu().numpy()

        return feat

    def select_references(self, target_image_path, train_samples, t=5):
        cluster_centers = {}    

        for category, paths in train_samples.items():
            feats = [self.extract_image_feat(p) for p in paths]
            cluster_centers[category] = np.mean(feats, axis=0)

        f_t = self.extract_image_feat(target_image_path)
        similarities = {category: cosine_similarity(f_t, center)[0][0] for category, center in cluster_centers.items()}   
        top_t_cats = sorted(similarities, key=similarities.get, reverse=True)[:t] 
        ref_paths = [random.choice(train_samples[category]) for category in top_t_cats]
        save_path_references = self.cfg['path_references'] + str(t)    
        ref_paths = "\n".join(ref_paths)
        dump_txt(save_path_references, ref_paths)

        return ref_paths

    def discover_regions(self, mllm_bot, target_image_path, ref_paths, superclass, s=3):
        ref_paths = ref_paths.split("\n")
        t = len(ref_paths)

        if t == 1:
            images = [Image.open(target_image_path).convert("RGB")]
        else:
            images = [Image.open(target_image_path).convert("RGB")] + [Image.open(p).convert("RGB") for p in ref_paths]

        prompt = (
            f"We provide {t} images from different categories within the {superclass} that share similar visual features, "
            f"and use them as references to generate {s} discriminative visual regions(eg,. descriptive phrases like white fur, thin straight legs) for distinguishing the target image's category. "
            "The first image is the target, followed by the references(if have). Output the regions as a comma-separated list."
        )
        outputs = []
        reply, trimmed_reply = mllm_bot.describe_attribute(images, prompt)

        if isinstance(outputs, str):
            trimmed_reply = trimmed_reply.lower().strip()
            outputs = trimmed_reply
            regions = ", ".join(outputs)
        else:
            for target_reference in trimmed_reply:
                output_string = target_reference.lower().strip() 
                feats = output_string.split(",")
                for f in feats:
                    outputs.append(f.strip())

        regions = outputs[:s]    
        return regions   

    def discover_regions_inference(self, mllm_bot, target_image_path, ref_paths, superclass, s=3): 
        ref_paths = ref_paths.split("\n")
        t = len(ref_paths)

        if t == 1:
            images = [Image.open(target_image_path).convert("RGB")]
        else:
            images = [Image.open(target_image_path).convert("RGB")] + [Image.open(p).convert("RGB") for p in ref_paths]

        prompt = (
            f"We provide an image from {superclass}, "
            f"please generate {s} visual regions that are discriminative compared to other subclasses (e.g., descriptive phrases like white fur, thin straight legs) to distinguish the target image's category. Output the regions as a comma-delimited list."
        )
        outputs = []
        reply, trimmed_reply = mllm_bot.describe_attribute(images, prompt)

        if isinstance(outputs, str):
            trimmed_reply = trimmed_reply.lower().strip()
            outputs = trimmed_reply
            regions = ", ".join(outputs)
        else:
            for target_reference in trimmed_reply:
                output_string = target_reference.lower().strip() 
                feats = output_string.split(",")
                for f in feats:
                    outputs.append(f.strip())

        regions = outputs[:s]    
        return regions   

    def describe_attributes(self, mllm_bot, target_image_path, regions, superclass="dog", label=None, label_id=None):     
        image = Image.open(target_image_path).convert("RGB")
        descriptions = []

        for region in regions:    
            if label is not None:
                prompt = f"Given an image, describe the visual attributes of {region} in the given {superclass} category, whose specific category is {label}."
            else:
                prompt = f"Given an image, describe the visual attributes of {region} in the {superclass} category."

            reply, output = mllm_bot.describe_attribute(image, prompt)
            description = ", ".join(output)
            descriptions.append(description)

        return descriptions

    def summarize_attributes(self, mllm_bot, target_image_path, descriptions, superclass, regions, label=None, label_id=None):    
        image = Image.open(target_image_path).convert("RGB")
        attr_info = "\n".join([f"**Region**: {r}\nDescription: {d}" for r, d in zip(regions, descriptions)])

        if label is not None: 
            prompt = (
                f"Summarize the information you get about the {label} from the attribute description:\n{attr_info}\n"
                f"Output a structured description."
            )
        else: 
            prompt = (
                f"Summarize the information you get about the {superclass} from the attribute description:\n{attr_info}\n"
                "Output a concise structured description."
            )

        reply, output = mllm_bot.describe_attribute(image, prompt)
        output = ", ".join(output)
        output = output.lower()

        if label is not None:
            result_data = {
                "img_path": target_image_path,
                "label": label if label is not None else "unknown",
                "label_id": label_id if label_id is not None else -1,
                "regions": regions,
                "descriptions": descriptions,
                "superclass": superclass
            }

            save_path_descriptions = self.cfg['path_descriptions']
            dump_json(save_path_descriptions, result_data)

        return output.strip()

    def generate_description_inference(self, mllm_bot, target_image_path, superclass, t=5, s=3):
        ref_path = ""
        regions = self.discover_regions_inference(mllm_bot, target_image_path, ref_path, superclass, s)

        descriptions = self.describe_attributes(mllm_bot, target_image_path, regions, superclass)
        final_description = self.summarize_attributes(mllm_bot, target_image_path, descriptions, superclass, regions)

        return final_description

    def generate_description(self, mllm_bot, target_image_path, train_samples, superclass, kshot=5, s=3, label=None, label_id=None):
        ref_paths = self.select_references(target_image_path, train_samples, kshot)
        regions = self.discover_regions(mllm_bot, target_image_path, ref_paths, superclass, s)

        descriptions = self.describe_attributes(mllm_bot, target_image_path, regions, superclass, label, label_id)
        final_description = self.summarize_attributes(mllm_bot, target_image_path, descriptions, superclass, regions, label, label_id)

        return final_description
        