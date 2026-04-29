+++
date = '2026-04-29T17:00:00+08:00'
draft = false
title = 'TensorRT-LLM 入门实战：Llama 模型量化与推理'
description = 'TensorRT-LLM 入门教程，将 Llama-3.2-1B-Instruct 模型转换为 TensorRT-LLM 格式并进行 INT8 量化推理'
tags = ["LLM", "TensorRT", "推理优化"]
categories = ["practice"]
+++

# TensorRT-LLM 入门实战：Llama 模型量化与推理

本文将介绍如何使用 TensorRT-LLM 将 HuggingFace 格式的 Llama 模型转换为优化后的 TensorRT Engine，并进行 INT8 weight-only 量化推理。通过这个实践项目，你可以了解 LLM 推理优化的基本流程和核心概念。

---

## 目录

1. [项目概述](#1-项目概述)
2. [TensorRT-LLM 简介](#2-tensorrt-llm-简介)
3. [环境准备](#3-环境准备)
4. [模型准备](#4-模型准备)
5. [模型转换流程](#5-模型转换流程)
6. [推理测试](#6-推理测试)
7. [踩坑记录](#7-踩坑记录)
8. [总结](#8-总结)

---

## 1. 项目概述

### 1.1 项目目标

将 `Llama-3.2-1B-Instruct` 模型从 HuggingFace 格式转换为 TensorRT-LLM 格式，并进行 INT8 weight-only 量化，最终实现高效的推理部署。

### 1.2 为什么需要推理优化？

大语言模型（LLM）的推理面临几个核心挑战：

| 挑战 | 说明 |
|:-----|:-----|
| **内存占用大** | Llama-3.2-1B 原始模型约 2.4GB，更大模型如 Llama-70B 需要数十 GB 内存 |
| **推理延迟高** | 自回归生成需要逐 token 推理，延迟累积明显 |
| **吞吐量受限** | 单次推理占用资源多，批量推理效率低 |

TensorRT-LLM 通过以下方式解决这些问题：

- **量化**：将 FP16/FP32 权重压缩为 INT8/INT4，减少内存和带宽需求
- **Kernel Fusion**：合并多个 GPU 操作为单一 kernel，减少 kernel launch 开销
- **优化 Attention**：使用 Flash Attention、Paged Attention 等高效实现
- **KV Cache 管理**：优化缓存策略，支持更长的序列和更高的 batch size

### 1.3 项目结构

```
tensorrt-llm-study/
├── llm-model/
│   └── Llama-3.2-1B-Instruct/     # 原始 HuggingFace 模型 (~2.4GB)
│       ├── model.safetensors      # 模型权重
│       ├── config.json            # 模型配置
│       ├── tokenizer.json         # tokenizer 文件
│       └── original/              # 原始 PyTorch 格式
│
├── trtllm-output/
│   ├── llama32_1b_int8wo_ckpt/    # TensorRT-LLM Checkpoint (~2GB)
│   │   ├── config.json            # 转换后的配置
│   │   └── rank0.safetensors      # 量化后的权重
│   │
│   └── llama32_1b_int8wo_engine/  # TensorRT Engine (~2GB)
│       ├── config.json            # Engine 配置
│       └── rank0.engine           # 优化后的 Engine 文件
│
└── sh.txt                         # 操作命令记录
```

---

## 2. TensorRT-LLM 简介

### 2.1 什么是 TensorRT-LLM？

TensorRT-LLM 是 NVIDIA 推出的高性能 LLM 推理优化库，基于 TensorRT 构建，专门针对 Transformer 架构的模型进行优化。它提供了：

- **模型转换工具**：将 HuggingFace、PyTorch 等格式的模型转换为 TensorRT Engine
- **量化支持**：INT8、INT4、FP8 等多种量化方案
- **推理 API**：Python 和 C++ 推理接口
- **分布式推理**：支持多卡 Tensor Parallelism 和 Pipeline Parallelism

### 2.2 核心概念

#### Checkpoint vs Engine

| 概念 | 说明 |
|:-----|:-----|
| **Checkpoint** | 中间格式，包含量化后的权重和模型配置信息，尚未生成最终的执行计划 |
| **Engine** | 最终可执行文件，包含完整的计算图、优化后的 kernel 和权重，可直接用于推理 |

转换流程：`原始模型 → Checkpoint → Engine`

#### Weight-Only Quantization

**Weight-Only Quantization（仅权重量化）** 是一种轻量级的量化策略：

- 只量化模型权重（Weights），激活值（Activations）保持 FP16
- 权重从 FP16 压缩为 INT8 或 INT4，推理时动态反量化
- 精度损失较小，因为权重是静态的，量化误差可控
- 实现简单，不需要校准数据（Calibration Data）

与之相对的是 **Weight-Activation Quantization（权重量化+激活量化）**：

- 权重和激活值都量化，内存和带宽收益更大
- 需要校准数据来确定激活值的量化范围
- 精度可能受影响，需要精细调优

#### Tensor Parallelism (TP)

**Tensor Parallelism（张量并行）** 是一种分布式推理策略：

- 将模型的权重切分到多个 GPU 上
- 每个 GPU 只存储部分权重，减少单卡内存压力
- 推理时各卡并行计算，通信同步中间结果
- 适合大模型（如 Llama-70B）单卡无法容纳的场景

`TP_size=1` 表示单卡推理，无切分。

---

## 3. 环境准备

### 3.1 实验环境

本项目使用的实际环境配置如下：

| 配置项 | 版本 |
|:-------|:-----|
| **操作系统** | Ubuntu 22.04 |
| **NVIDIA Driver** | 595.58.03 |
| **CUDA Version** | 13.2 |
| **GPU** | NVIDIA GPU（支持 CUDA 13.2） |
| **内存** | 32GB |
| **磁盘** | 200GB |
| **TensorRT-LLM** | v1.0.0 (Docker 镜像) |

通过 `nvidia-smi` 查看本机 GPU 信息：

```
NVIDIA-SMI 595.58.03
Driver Version: 595.58.03
CUDA Version: 13.2
```

### 3.2 Docker 环境

TensorRT-LLM 官方提供 Docker 镜像，包含所有依赖：

```bash
# 拉取官方镜像
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.0.0
```

使用 Docker 的优势：

- **环境隔离**：避免版本冲突，官方镜像已预装所有依赖
- **源码集成**：镜像内包含 TensorRT-LLM 源码和示例脚本
- **快速启动**：无需手动安装 CUDA、cuDNN、TensorRT 等

### 3.3 启动容器

```bash
docker run --rm \
  -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus=all \
  -v /home/minge/my_repo/tensorrt-llm-study/llm-model/Llama-3.2-1B-Instruct:/models/Llama-3.2-1B-Instruct:ro \
  -v /home/minge/my_repo/tensorrt-llm-study/trtllm-output:/trtllm-output \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0
```

参数解释：

| 参数 | 说明 |
|:-----|:-----|
| `--ipc=host` | 共享主机 IPC namespace，提升共享内存性能 |
| `--ulimit memlock=-1` | 不限制锁定内存大小，允许 GPU 使用更多内存 |
| `--ulimit stack=67108864` | 增大栈空间，避免大模型推理时栈溢出 |
| `--gpus=all` | 使用所有可用 GPU |
| `-v ...:ro` | 挂载模型目录为只读，保护原始模型 |
| `-v ...` | 挂载输出目录，持久化转换结果 |
| `-p 8000:8000` | 端口映射，用于后续 Triton Server |

---

## 4. 模型准备

### 4.1 下载 Llama-3.2-1B-Instruct

从 ModelScope（国内镜像）下载模型，避免 HuggingFace 连接问题：

```bash
# 安装 Git LFS
git lfs install

# 克隆模型（包含大文件）
git clone https://www.modelscope.cn/LLM-Research/Llama-3.2-1B-Instruct.git
```

### 4.2 模型文件说明

```
Llama-3.2-1B-Instruct/
├── config.json            # 模型架构配置
├── model.safetensors      # 模型权重（约 2.4GB）
├── tokenizer.json         # Tokenizer 配置
├── tokenizer_config.json  # Tokenizer 参数
├── special_tokens_map.json # 特殊 token 映射
└── generation_config.json # 生成参数配置
```

**config.json 关键字段**：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "vocab_size": 128256
}
```

| 字段 | 说明 |
|:-----|:-----|
| `hidden_size` | 隐藏层维度，决定模型容量 |
| `num_hidden_layers` | Transformer 层数 |
| `num_attention_heads` | 注意力头数 |
| `vocab_size` | 词表大小 |

---

## 5. 模型转换流程

### 5.1 转换流程概述

```
┌─────────────────────────────────────────────────────────────┐
│                      模型转换流程                            │
├─────────────────────────────────────────────────────────────┤
│  1. HuggingFace 模型 (.safetensors)                         │
│     ↓                                                       │
│  2. convert_checkpoint.py → TensorRT-LLM Checkpoint         │
│     ↓                                                       │
│  3. trtllm-build → TensorRT Engine (.engine)                │
│     ↓                                                       │
│  4. run.py → 推理测试                                        │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Step 1：转换为 Checkpoint

在容器内执行：

```bash
cd /app/tensorrt_llm/examples/models/core/llama

python convert_checkpoint.py \
  --model_dir /models/Llama-3.2-1B-Instruct \
  --output_dir /trtllm-output/llama32_1b_int8wo_ckpt \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int8 \
  --tp_size 1
```

参数详解：

| 参数 | 说明 |
|:-----|:-----|
| `--model_dir` | 原始 HuggingFace 模型路径 |
| `--output_dir` | Checkpoint 输出路径 |
| `--dtype float16` | 激活值数据类型，FP16 平衡精度与性能 |
| `--use_weight_only` | 启用仅权重量化 |
| `--weight_only_precision int8` | 权重量化为 INT8 |
| `--tp_size 1` | Tensor Parallelism size，1 表示单卡 |

#### 量化原理

INT8 weight-only 量化过程：

```
FP16 权重 (W) → INT8 权重 (W_int8) + Scale (S)

推理时：
W_dequantized = W_int8 * S  # 动态反量化
Output = Activation @ W_dequantized
```

**为什么 INT8 量化可行？**

- FP16 每个权重占 2 字节，INT8 占 1 字节，内存减半
- 权重在推理时是静态的，量化误差可预先计算并补偿
- NVIDIA GPU 有专门的 INT8 Tensor Core，反量化效率高

#### 转换输出结果

执行转换命令后，输出如下：

```
Total time of reading and converting: 8.440 s
Total time of saving checkpoint: 1.780 s
Total time of converting checkpoints: 00:00:10
```

转换耗时约 **10 秒**，主要时间用于读取模型权重和量化处理。

生成的 Checkpoint 文件：

```
llama32_1b_int8wo_ckpt/
├── config.json         # 量化后的模型配置
└── rank0.safetensors   # INT8 量化后的权重（约 2GB）
```

**config.json 关键配置**：

```json
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "vocab_size": 128256,
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "quantization": {
        "quant_algo": "W8A16",
        "group_size": 128
    },
    "mapping": {
        "tp_size": 1,
        "pp_size": 1
    }
}
```

关键配置说明：

| 字段 | 说明 |
|:-----|:-----|
| `quant_algo: "W8A16"` | Weight INT8 + Activation FP16，即 Weight-Only 量化 |
| `group_size: 128` | 量化分组大小，每组共享一个 scale |
| `max_position_embeddings` | 最大序列长度 131072（Llama-3.2 支持超长上下文） |

### 5.3 Step 2：构建 Engine

```bash
trtllm-build \
  --checkpoint_dir /trtllm-output/llama32_1b_int8wo_ckpt \
  --output_dir /trtllm-output/llama32_1b_int8wo_engine \
  --gemm_plugin auto
```

参数详解：

| 参数 | 说明 |
|:-----|:-----|
| `--checkpoint_dir` | 上一步生成的 Checkpoint 目录 |
| `--output_dir` | Engine 输出路径 |
| `--gemm_plugin auto` | 自动选择最优 GEMM（矩阵乘法）实现 |

#### GEMM Plugin

**GEMM（General Matrix Multiply）** 是神经网络的核心计算操作。TensorRT-LLM 提供多种 GEMM 实现：

| Plugin | 说明 |
|:--------|:------|
| `auto` | 自动选择最优实现（推荐） |
| `cublas` | NVIDIA cuBLAS 库实现 |
| `cublaslt` | cuBLASLt，支持 INT8 |
| `fpA_intB` | FP16 激活 + INT8 权重的混合精度实现 |

选择 `auto` 时，TensorRT 会根据 GPU 型号、量化类型自动选择最优 Plugin。

#### Engine 构建过程

Engine 构建会执行以下优化：

1. **计算图优化**：合并可融合的操作，消除冗余计算
2. **Kernel 选择**：为每个操作选择最优 GPU Kernel
3. **内存规划**：预分配推理所需内存，避免动态分配开销
4. **权重嵌入**：将量化后的权重嵌入 Engine 文件

构建时间通常较长（几分钟到几十分钟），但只需构建一次，后续推理可直接使用 Engine。

#### Engine 构建输出结果

执行构建命令后，输出如下：

```
[04/29/2026-11:04:07] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:31
[04/29/2026-11:04:07] [TRT] [I] Serialized 27 bytes of code generator cache.
[04/29/2026-11:04:07] [TRT] [I] Serialized 134685 bytes of compilation cache.
[04/29/2026-11:04:07] [TRT] [I] Serialized 12 timing cache entries
[04/29/2026-11:04:07] [TRT-LLM] [I] Timing cache serialized to model.cache
[04/29/2026-11:04:07] [TRT-LLM] [I] Build phase peak memory: 10421.96 MB, children: 6296.07 MB
[04/29/2026-11:04:08] [TRT-LLM] [I] Serializing engine to /trtllm-output/llama32_1b_int8wo_engine/rank0.engine...
[04/29/2026-11:04:09] [TRT-LLM] [I] Engine serialized. Total time: 00:00:01
[04/29/2026-11:04:09] [TRT-LLM] [I] Total time of building all engines: 00:00:34
```

构建耗时约 **34 秒**，峰值内存占用约 **10.4 GB**。

生成的 Engine 文件：

```
llama32_1b_int8wo_engine/
├── config.json     # Engine 配置（包含构建参数）
├── rank0.engine    # TensorRT Engine 文件（约 2GB）
└── model.cache     # Timing Cache（用于加速后续构建）
```

**Engine config.json 关键配置**：

```json
{
    "version": "1.0.0",
    "pretrained_config": {
        "quantization": {
            "quant_algo": "W8A16"
        }
    },
    "build_config": {
        "max_input_len": 1024,
        "max_seq_len": 131072,
        "max_batch_size": 2048,
        "max_num_tokens": 8192,
        "kv_cache_type": "PAGED",
        "plugin_config": {
            "gemm_plugin": "auto",
            "gpt_attention_plugin": "auto",
            "weight_only_quant_matmul_plugin": "float16"
        },
        "auto_parallel_config": {
            "cluster_key": "NVIDIA-GeForce-RTX-4060"
        }
    }
}
```

关键配置说明：

| 字段 | 说明 |
|:-----|:-----|
| `max_batch_size: 2048` | 最大支持 batch size |
| `kv_cache_type: "PAGED"` | 使用分页 KV Cache，提高内存利用率 |
| `weight_only_quant_matmul_plugin` | INT8 权重的矩阵乘法实现 |
| `cluster_key` | 自动检测的 GPU 型号（RTX 4060） |

---

## 6. 推理测试

### 6.1 运行推理脚本

```bash
cd /app/tensorrt_llm/examples/models/core/llama

# 英文测试
python /app/tensorrt_llm/examples/run.py \
  --engine_dir /trtllm-output/llama32_1b_int8wo_engine \
  --tokenizer_dir /models/Llama-3.2-1B-Instruct \
  --max_output_len 128 \
  --input_text "Explain what TensorRT-LLM is in one sentence."

# 中文测试（使用 Python session）
python /app/tensorrt_llm/examples/run.py \
  --engine_dir /trtllm-output/llama32_1b_int8wo_engine \
  --tokenizer_dir /models/Llama-3.2-1B-Instruct \
  --max_output_len 128 \
  --input_text "请用三句话解释 TensorRT-LLM 的作用。" \
  --use_py_session
```

参数说明：

| 参数 | 说明 |
|:-----|:-----|
| `--engine_dir` | Engine 目录路径 |
| `--tokenizer_dir` | Tokenizer 文件路径 |
| `--max_output_len` | 最大输出 token 数 |
| `--input_text` | 输入提示词 |
| `--use_py_session` | 使用 Python session（默认使用 C++ session） |

### 6.2 测试结果

#### 中文测试结果

输入：`"请用三句话解释 TensorRT-LLM 的作用。"`

输出：

```
Input [Text 0]: "<|begin_of_text|>请解释 TensorRT-LLM 的作用。"
Output [Text 0 Beam 0]: "TensorRT-LLM 是一个用于计算机语言转换的工具，能够将计算机语言的语句转换为计算机语言的语句。它可以将计算机语言的语句转换为计算机语言的语句，例如从Python到C++。

TensorRT-LL"
```

**输出分析**：

- 模型成功生成了回复，但输出内容质量有限
- 1B 参数的小模型在中文理解和生成上能力有限
- 输出在达到 `max_output_len=128` 时截断
- 使用 INT8 量化后，推理正常工作，量化未破坏模型基本功能

#### 性能数据总结

| 阶段 | 耗时 | 内存占用 |
|:-----|:-----|:---------|
| Checkpoint 转换 | 10 秒 | - |
| Engine 构建 | 34 秒 | 10.4 GB（峰值） |
| Engine 文件大小 | - | ~2 GB |

**GPU 信息**：系统自动检测到 `NVIDIA-GeForce-RTX-4060`，TensorRT-LLM 会针对该 GPU 优化 Engine。

## 7. 踩坑记录

### 7.1 nvidia-container-toolkit 安装问题

**问题描述**：

Docker 使用 GPU 时需要安装 `nvidia-container-toolkit`，但安装过程遇到依赖问题。

**解决方案**：

参考 [CSDN 博客](https://blog.csdn.net/devshilei/article/details/144233180) 的方法解决。

核心步骤：

```bash
# 设置 GPG key 和 repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 更新并安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 7.2 网络连接问题

**问题描述**：

安装过程中遇到网络问题：

- `Could not handshake`：TLS/SSL 握手失败
- 连接超时
- `无法定位软件包`：apt 源无法访问
- `gpg: 找不到有效的 OpenPGP 数据`：GPG key 下载失败

**根本原因**：

IPv6 连接不稳定，部分服务器对 IPv6 支持不佳。

**解决方案**：

禁用 IPv6 后重新安装：

```bash
# 临时禁用 IPv6
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1

# 永久禁用（编辑 /etc/sysctl.conf）
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# 重新执行安装命令
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 7.3 问题总结

| 问题 | 原因 | 解决方案 |
|:-----|:-----|:----------|
| nvidia-container-toolkit 安装失败 | 依赖配置问题 | 按官方文档配置 repository |
| 网络握手失败 | IPv6 不稳定 | 禁用 IPv6 |
| GPG key 下载失败 | 网络问题 | 禁用 IPv6 后重试 |
| 软件包无法定位 | apt 源未更新 | 正确配置 repository 后 apt-get update |

---

## 8. 总结

### 8.1 项目要点回顾

| 步骤 | 命令 | 输出 |
|:------|:------|:------|
| 模型下载 | `git clone` | HuggingFace 模型目录 |
| Checkpoint 转换 | `convert_checkpoint.py` | 量化后的权重文件 |
| Engine 构建 | `trtllm-build` | 可执行的 Engine 文件 |
| 推理测试 | `run.py` | 模型输出结果 |

### 8.2 关键知识点

1. **Weight-Only Quantization**：仅量化权重，激活保持 FP16，精度损失小，实现简单

2. **Checkpoint vs Engine**：Checkpoint 是中间格式，Engine 是最终可执行文件

3. **Tensor Parallelism**：大模型切分到多卡，减少单卡内存压力

4. **GEMM Plugin**：矩阵乘法的优化实现，INT8 量化需要特定 Plugin 支持

### 8.3 INT8 Weight-Only 的优缺点

| 优点 | 缺点 |
|:-----|:-----|
| 内存占用减半 | 精度有轻微损失 |
| 无需校准数据 | 激活值仍为 FP16，带宽优化有限 |
| 实现简单 | 相比 INT4，压缩率较低 |
| 精度损失可控 | 大模型收益更明显，小模型收益有限 |

### 8.4 后续学习方向

- **其他量化方案**：INT4 Weight-Only、FP8、Weight-Activation Quantization
- **KV Cache 优化**：Paged Attention、KV Cache 量化
- **分布式推理**：多卡 Tensor Parallelism
- **Triton Inference Server**：生产级服务部署
- **性能调优**：Batch size、Max sequence length 对性能的影响

---

## 参考资料

- [TensorRT-LLM 官方文档](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Llama 示例](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama)
- [NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [nvidia-container-toolkit 踩坑解决](https://blog.csdn.net/devshilei/article/details/144233180)