# Tool_call-agent: Fine-tuned Qwen3-1.7B for Advanced Tool Calling

[![Model](https://img.shields.io/badge/Model-Qwen3--1.7B-blueviolet)](https://huggingface.co/Qwen/Qwen3-1.7B)
[![Hardware](https://img.shields.io/badge/Hardware-NPU%20%2F%20GPU%20Support-green)](#)

---

## 📖 项目简介 (Introduction)

**Tool_call-agent** 是一个基于 **Qwen3-1.7B** 的全栈智能体开发项目。  
项目通过 **LLaMA-Factory** 框架对模型进行 LoRA 微调，使其在极小的参数量下展现出卓越的工具调用（Tool Calling）能力。

本项目不仅包含完整的数据集处理与训练流程，还实现了一个高度优化的推理引擎。该引擎能够自动适配国产化 **NPU (Ascend)** 与 **NVIDIA GPU** 硬件，并针对内存占用进行了深度优化（如半精度推理、高度简洁prompt等）。模型能够精准解析复杂的系统指令，将用户意图转化为标准的 Python 函数调用格式，适用于各种端侧智能助手与自动化场景。

---

## 1. Project Overview

The **Tool_call-agent** repository demonstrates an end-to-end pipeline for enabling sophisticated tool-use capabilities on the **Qwen3-1.7B** model.  
To mitigate the limitations of small-scale models in producing structured outputs (such as JSON and function calls), this project adopts a high-density system instruction set combined with a regex-based parsing mechanism, achieving an **≈80% tool invocation success rate** within a specialized “决赛数据集 (Competition Instruction Set)” context.

---

## 2. Key Features

- **Hardware Agnostic**: Built-in support for both **NVIDIA GPU (CUDA)** and **Huawei Ascend (NPU)** via `torch_npu` detection.  
- **Inference Optimization**:  
  - Employs `torch.float16` and `low_cpu_mem_usage=True` for minimal memory footprint.  
  - Uses `torch.inference_mode()` and greedy decoding to accelerate generation.  
- **Complex Logic Mapping**:  
  - **Format Enforcement**: Strict adherence to `<tool>ToolName(param=value)</tool>` syntax.  
- **Robust Parsing**: Regex-based post-processor that handles missing parentheses and extracts the final execution command from long-form reasoning.

---

## 3. Repository Structure

```text
.
├── LLaMA-Factory           # Training Framework & LoRA configurations
│   ├── data                # Fine-tuning datasets (ShareGPT format)
│   ├── examples            # Training YAML configurations
│   └── saves               # Exported LoRA checkpoints/weights
├── demo                    # Core Inference & Agent Logic
│   ├── agent.py            # CustomAgent with NPU/GPU support and Tool logic
│   ├── bash_run.py         # Batch evaluation script with tqdm progress bar
│   ├── data                # Competition test sets and smoke tests (.jsonl)
│   ├── prompts             # Tool definitions (intent_name, slots, etc.)
│   └── results             # Evaluation logs and JSON output results
└── README.md

```

## 4. Implementation Details

### Agent Logic
- The `BaseLLM` class encapsulates the HuggingFace Transformers pipeline, specifically disabling `enable_thinking` for Qwen3 to focus on direct tool generation.  
- The `CustomAgent` dynamically builds a tool registry by reading a JSONL instruction set and injecting it into a highly detailed System Prompt.  

### Benchmarking
The project includes a benchmarking script that:  
- Loads `.json` or `.jsonl` test data.  
- Pre-processes multi-turn dialogue history.  
- Executes batch inference with a progress bar (`tqdm`).  
- Exports structured results for accuracy analysis.  

---

## 5. Installation & Quick Start

### Prerequisites
- Python 3.10+  
- PyTorch (with CUDA or NPU support)  
- Transformers, PEFT, Accelerate, tqdm  

### Quick Start
```bash
# Run demo
python demo/bash_run.py
