# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

LLaMA-Factory is a unified platform for fine-tuning 100+ large language models with various training approaches including LoRA, QLoRA, full-parameter fine-tuning, DPO, PPO, reward modeling, and more. It supports both command-line and web-based interfaces for training, evaluation, and inference.

## Architecture

The codebase follows a layered architecture as documented in `src/llamafactory/__init__.py`:
```
api, webui > chat, eval, train > data, model > hparams > extras
```

### Key Components

- **`src/llamafactory/api/`** - OpenAI-style API server implementation
- **`src/llamafactory/chat/`** - Chat interfaces with multiple backends (HuggingFace, vLLM, SGLang)
- **`src/llamafactory/train/`** - Training workflows organized by method:
  - `pt/` - Pre-training
  - `sft/` - Supervised Fine-Tuning 
  - `dpo/` - Direct Preference Optimization
  - `ppo/` - Proximal Policy Optimization
  - `rm/` - Reward Modeling
  - `kto/` - KTO Training
- **`src/llamafactory/data/`** - Data loading, processing, and formatting
- **`src/llamafactory/model/`** - Model loading, adapters, and utilities
- **`src/llamafactory/hparams/`** - Hyperparameter definitions and parsing
- **`src/llamafactory/extras/`** - Utilities, constants, logging, and environment handling

### Configuration System

All training configurations are YAML-based with a hierarchical structure:
- **Model**: `model_name_or_path`, `trust_remote_code`, `template`
- **Method**: `stage` (pt/sft/dpo/ppo/rm/kto), `finetuning_type` (lora/qlora/full)
- **Dataset**: `dataset`, `cutoff_len`, `max_samples`
- **Output**: `output_dir`, `logging_steps`, `save_steps`
- **Training**: Batch size, learning rate, scheduler, optimization settings

## Common Commands

### Training Commands

#### LoRA Fine-tuning
```bash
# Supervised fine-tuning with LoRA
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# Pre-training with LoRA
llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml

# DPO training with LoRA
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml

# PPO training with LoRA
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml

# Reward modeling
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml
```

#### QLoRA (Quantized LoRA)
```bash
# 4/8-bit quantized training
llamafactory-cli train examples/train_qlora/llama3_lora_sft_otfq.yaml
```

#### Full Parameter Fine-tuning
```bash
# Full fine-tuning (requires FORCE_TORCHRUN=1 for multi-GPU)
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml
```

#### Multi-GPU/Multi-Node Training
```bash
# Single node, multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# Multi-node training
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### Custom Parameters
```bash
# Override config parameters
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml learning_rate=1e-5 logging_steps=1
```

### Inference Commands

#### Chat Interface
```bash
# CLI chat
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml

# Web chat interface
llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

#### API Server
```bash
# Launch OpenAI-style API with vLLM backend
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true

# Launch API with HuggingFace backend
llamafactory-cli api examples/inference/llama3.yaml
```

### Model Export/Merging

```bash
# Merge LoRA adapters
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

# Export with quantization
llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```

### Evaluation

```bash
# Evaluate on benchmarks (MMLU, C-Eval, CMMLU)
llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml
```

### Web UI

```bash
# Launch LlamaBoard (complete web interface)
llamafactory-cli webui
```

## Development Commands

### Code Quality
```bash
# Run linting and formatting
make style

# Check code quality without fixing
make quality

# Run all quality checks
make commit

# Run tests
make test
```

### Build and Package
```bash
# Build package
make build
```

## Dataset Management

### Dataset Configuration
- All datasets are configured in `data/dataset_info.json`
- Supports HuggingFace Hub, ModelScope Hub, and local datasets
- Two main formats: **Alpaca** (instruction-input-output) and **ShareGPT** (conversation format)

### Environment Variables for Data Sources
```bash
# Use ModelScope instead of HuggingFace
export USE_MODELSCOPE_HUB=1

# Use OpenMind Hub
export USE_OPENMIND_HUB=1
```

## Environment Variables

### Training Control
- `FORCE_TORCHRUN=1` - Force distributed training
- `CUDA_VISIBLE_DEVICES=0,1` - Specify GPUs to use
- `ASCEND_RT_VISIBLE_DEVICES` - For Ascend NPU devices

### Multi-node Setup
- `NNODES` - Number of nodes
- `NODE_RANK` - Current node rank  
- `MASTER_ADDR` - Master node address
- `MASTER_PORT` - Master node port

### Logging and Monitoring
- `WANDB_API_KEY` - Weights & Biases API key
- `SWANLAB_API_KEY` - SwanLab API key
- `LLAMAFACTORY_VERBOSITY=WARN` - Set logging level
- `WANDB_DISABLED=true` - Disable W&B logging

### Performance Optimization
- `RECORD_VRAM=1` - Enable VRAM recording
- `DISABLE_VERSION_CHECK=1` - Skip version checks
- `OPTIM_TORCH=1` - Enable DDP optimizations

## Hardware Considerations

### Memory Requirements (estimated)
- **QLoRA (4-bit)**: ~6GB for 7B, ~12GB for 14B, ~24GB for 30B
- **LoRA (16-bit)**: ~16GB for 7B, ~32GB for 14B, ~64GB for 30B  
- **Full (16-bit)**: ~60GB for 7B, ~120GB for 14B, ~300GB for 30B

### Supported Quantization
- **Bits & Bytes**: 4/8-bit quantization
- **GPTQ**: 4/8-bit quantization
- **AWQ**: 4-bit quantization
- **AQLM**: 2-bit quantization
- **HQQ/EETQ**: Advanced quantization methods

## Inference Backends

### Available Backends
- **HuggingFace**: Default, most compatible
- **vLLM**: ~270% faster inference, better for serving
- **SGLang**: Alternative fast inference backend

### Backend Selection
```yaml
infer_backend: vllm  # choices: [huggingface, vllm, sglang]
```

## Key Configuration Files

### Training Examples
- `examples/train_lora/` - LoRA training configurations
- `examples/train_qlora/` - Quantized LoRA configurations  
- `examples/train_full/` - Full parameter fine-tuning
- `examples/inference/` - Inference configurations
- `examples/merge_lora/` - Model merging configurations

### Data Configuration
- `data/dataset_info.json` - Dataset definitions and metadata
- `data/README.md` - Dataset format documentation

## Monitoring and Logging

### Supported Loggers
- **TensorBoard**: Built-in support
- **Weights & Biases**: Set `report_to: wandb`
- **SwanLab**: Set `use_swanlab: true`
- **MLflow**: Set `report_to: mlflow`

### LlamaBoard
Built-in web interface provides:
- Training progress monitoring
- Loss curves and metrics
- Model comparison tools
- Experiment management

## Common Issues and Solutions

### Multi-GPU Training
- Always use `FORCE_TORCHRUN=1` for multi-GPU full fine-tuning
- LoRA training automatically uses all visible GPUs

### Memory Issues  
- Use QLoRA for large models on limited memory
- Adjust `per_device_train_batch_size` and `gradient_accumulation_steps`
- Enable gradient checkpointing in config

### Dataset Issues
- Ensure dataset is properly defined in `dataset_info.json`
- Verify column mappings match your data format
- Use `overwrite_cache: true` when changing data preprocessing

### NPU Support
- Use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`  
- Set `do_sample: false` if inference fails on NPU
- Install proper CANN toolkit and kernels

This codebase is optimized for both research and production use cases, with extensive support for various hardware configurations and training methodologies.
