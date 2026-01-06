# SARE: Sample-wise Adaptive Reasoning for Training-free Fine-grained Visual Recognition

This is the anonymous repository for the paper submission. This repository contains the implementation of SARE, a self-adaptive reasoning enhancement framework for fine-grained visual recognition.

## 📋 Overview

SARE proposes a dual-system framework for Fine-Grained Visual Recognition (FGVR) inspired by human cognitive processes:
- **System 1 (Fast Retrieval-based Perception)**: Rapid, intuitive judgments using a **Multimodal Prototype Library** to handle straightforward samples efficiently.
- **System 2 (Experience-guided Nuanced Reasoning)**: Deliberate, step-by-step analysis invoked only for ambiguous cases, utilizing a **Self-Reflective Experience Library** to focus on discriminative details.

![Overview](fig/overview.png)

## 🎯 Key Features

- **Statistics-based Dynamic Trigger**: A mechanism that dynamically routes samples to System 2 based on fused confidence scores, historical category difficulty, and candidate ambiguity.
- **Self-Reflective Experience Mechanism**: A closed-loop process that distills "reusable discriminative guidance" from past failures (without parameter updates) to prevent repeated errors.
- **Dual-System Architecture**: Synergizes fast retrieval with nuanced reasoning to optimize the trade-off between accuracy and computational efficiency.

## 📚 Knowledge Base Construction
SARE constructs three lightweight offline libraries to support adaptive inference:
1. **Multimodal Prototype Library**: Contains visual (CLIP embeddings) and textual (LVLM descriptions) prototypes for fast System 1 retrieval.
2. **Statistical Retrieval Library**: Records class-conditional retrieval history to calibrate uncertainty and support the dynamic trigger.
3. **Self-Reflective Experience Library**: Stores structured decision rules abstracted from model self-reflection on hard samples.

## 📁 Project Structure

```
SARE/
├── agents/
│   └── mllm_bot.py              # MLLM interface
├── retrieval/
│   └── multimodal_retrieval.py  # Multimodal retrieval module
├── main.py                       # Main
├── system1.py                    # Fast thinking system
├── system2.py                    # Slow thinking system
├── fast_slow_thinking_system.py # Integrated dual-system
├── knowledge_base_builder.py    # Knowledge base construction
├── experience_base_builder.py   # Experience base construction
├── description_generator.py     # Visual description generation
├── config_template.yaml         # Configuration template
├── requirements.txt             
└── fig/                         # Figures and visualizations
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA 11.8+ 
- 24GB+ GPU memory (for Qwen2.5-VL-7B)


### Quick Start

#### 1. Build Knowledge Base

First, build the category-level knowledge base :

```bash
python main.py \
    --mode build_knowledge_base \
    --config_file_env ./configs/env_machine.yml \
    --config_file_expt ./configs/expts/dog120_all.yml \
    --num_per_category 10 \
    --knowledge_base_dir ./experiments/dog120/knowledge_base
```

This creates:
- Category prototypes
- Visual descriptions
- Self-belief statistics

#### 2. Build Experience Base (Optional)

Build the instance-level experience base:

```bash
python main.py \
    --mode build_experience_base \
    --config_file_env ./configs/env_machine.yml \
    --config_file_expt ./configs/expts/dog120_all.yml \
    --experience_base_dir ./experiments/dog120/experience_base
```

#### 3. Run Inference

Run the SARE inference:

```bash
python main.py \
    --mode fast_slow \
    --config_file_env ./configs/env_machine.yml \
    --config_file_expt ./configs/expts/dog120_all.yml \
    --knowledge_base_dir ./experiments/dog120/knowledge_base \
    --test_data_dir ./datasets/dogs_120/test \
    --results_out ./results/dog120_results.json \
```
![main_results](fig/main_results.png)
