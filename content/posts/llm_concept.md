+++
date = '2026-04-22T09:00:24+08:00'
draft = false
title = 'LLM 核心概念速查表'
description = '大语言模型从基础到部署的完整概念索引'
tags = ["LLM"]
categories = ["Writting"]
+++


## 一、模型类型与训练相关

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 大语言模型 | Large Language Model (LLM) | 基于海量文本训练的语言生成与理解核心模型，通常参数量在十亿级别以上 |
| 基础模型 / 底座模型 | Base Model / Foundation Model | 仅完成预训练、未做指令微调的原始模型，具备通用语言能力但需二次优化才能直接使用 |
| 指令微调模型 | Instruct-tuned Model | 经过指令-回答数据微调，可直接响应用户指令，开箱即用 |
| 对话模型 | Chat Model | 专门面向多轮对话场景优化，适配日常聊天、问答交互 |
| 推理模型 | Reasoning Model | 经过强化学习训练，具备深度思考和逐步推理能力的模型，如 o1、DeepSeek-R1 |
| MoE 模型 | Mixture of Experts | 混合专家架构模型，通过路由机制激活部分参数，在保持大参数量的同时降低推理成本 |
| 预训练 | Pre-training | 模型初期大规模无监督学习阶段，用于学习语言规律、积累世界知识 |
| 后训练 | Post-training | 预训练之后的训练阶段，包括指令微调、对齐、强化学习等 |
| 微调 | Fine-tuning (FT) | 在底座模型基础上，用特定任务数据继续训练，提升专项能力 |
| 指令微调 | Instruction Tuning | 特指用"指令-响应"格式数据微调，提升模型遵循用户意图的能力 |
| 对齐 | Alignment | 让模型输出符合人类价值观、伦理规范和用户预期的过程 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习，通过奖励模型优化模型输出质量 |
| DPO | Direct Preference Optimization | 直接偏好优化，无需训练奖励模型的对齐方法，训练更简单高效 |
| 高效微调 | Parameter-Efficient Fine-Tuning (PEFT) | 仅微调模型少量参数，大幅节省显存和计算资源，适配本地部署 |

## 二、模型结构与参数

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 参数 | Parameter | 模型内部可学习的权重和偏置，决定模型的学习能力 |
| 参数量 | Number of Parameters | 模型参数的总数量，7B=7 Billion（70亿），8B=80亿，是模型规模的核心指标 |
| 活跃参数 | Active Parameters | MoE模型推理时实际参与计算的参数数量，决定实际推理开销 |
| Transformer | Transformer | 目前大模型的主流基础架构，由编码器（Encoder）和解码器（Decoder）组成 |
| 注意力机制 | Attention | 模型核心机制，用于关注文本中不同部分的关联关系，理解上下文 |
| 自注意力 | Self-Attention | Transformer的核心模块，让文本中每个token都能关注到自身及其他token |
| 多头注意力 | Multi-Head Attention (MHA) | 并行使用多组注意力机制，捕捉文本不同维度的关联，提升模型表达能力 |
| GQA | Grouped-Query Attention | 分组查询注意力，在性能和效率间取得平衡，LLaMA2等模型采用 |
| MQA | Multi-Query Attention | 多查询注意力，显著减少KV Cache显存占用，提升推理速度 |
| 上下文窗口 | Context Window | 模型一次能处理的最大文本长度范围，超出范围会丢失上下文信息 |
| 上下文长度 | Context Length | 同"上下文窗口"，单位为token，常见规格有8K、32K、128K、1M等 |
| RoPE | Rotary Position Embedding | 旋转位置编码，主流的位置编码方式，支持外推扩展上下文长度 |
| 滑动窗口注意力 | Sliding Window Attention | 限制注意力范围以降低计算复杂度，Mistral等模型采用 |
| Flash Attention | Flash Attention | 高效注意力计算实现，大幅减少显存占用并加速训练和推理 |

## 三、Tokenizer（分词器）相关

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 分词器 | Tokenizer | 大模型的"语言翻译官"，负责将人类文本转换为模型可识别的最小单位（token） |
| 词元 / 令牌 | Token | 模型处理文本的最小单位，可是汉字、子词、英文单词或字符 |
| 词汇表 | Vocabulary | 分词器内置的所有token集合，模型仅能识别词汇表内的token |
| 词汇表大小 | Vocabulary Size | 词汇表中token的总数，影响模型参数量和语言覆盖能力 |
| 子词分词 | Subword Tokenization | 主流分词方式，将文本拆分为词、子词或字符，兼顾效率和稀有词覆盖度 |
| 字节对编码 | Byte Pair Encoding (BPE) | 最常用的分词算法，LLaMA、GPT、Qwen等主流模型均采用 |
| SentencePiece | SentencePiece | 端到端的分词工具，支持BPE和Unigram算法，训练新分词器的首选 |
| 编码 | Encode | 分词器的核心功能之一，将人类文本 → token → 数字ID（模型可计算的格式） |
| 解码 | Decode | 分词器的核心功能之一，将模型输出的数字ID → token → 人类可读懂的文本 |
| 特殊标记 | Special Tokens | 分词器中预留的特定功能token，如 `<PAD>`、`<EOS>`、`<BOS>` 等 |
| 填充标记 | Padding Token | 用于将不同长度的序列对齐到相同长度，便于批量处理 |
| 序列截断 | Truncation | 当输入超过最大长度时，截断文本以适应模型限制 |

## 四、量化与精度相关

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 量化 | Quantization | 通过降低模型权重的精度，减少显存占用和计算量，适配本地低配置GPU |
| INT4 / INT8 量化 | INT4 / INT8 Quantization | 最常用的低精度量化格式，INT4占用显存最少，适配8G显存显卡 |
| FP16 | FP16 | 半精度浮点格式（16位），比FP32节省一半显存，常用于中端GPU推理 |
| BF16 | BF16 | Brain Float 16，与FP16相比有更大的动态范围，现代GPU首选精度格式 |
| FP32 | FP32 | 单精度浮点格式（32位），模型训练的默认精度，显存占用最大 |
| GPTQ 量化 | GPTQ | 常用的训练后量化方案，兼顾量化速度和效果，支持多数主流LLM |
| AWQ 量化 | AWQ | 激活感知权重量化，量化后模型性能下降少，显存占用更低 |
| GGUF 格式 | GGUF Format | llama.cpp项目推出的模型格式，支持CPU推理和多种量化级别，本地部署首选 |
| GGML | GGML | GGUF的前身格式，已逐渐被GGUF取代 |
| bitsandbytes | bitsandbytes | NVIDIA GPU上的8位优化器库，也支持4位/8位量化加载模型 |
| 量化感知训练 | Quantization-Aware Training (QAT) | 训练过程中模拟量化效果，使模型适应低精度，精度损失更小 |
| 混合精度 | Mixed Precision | 训练或推理时混合使用不同精度，在效率和精度间取得平衡 |

## 五、模型格式与转换

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 模型格式 | Model Format | 模型权重存储的文件格式，不同框架和推理引擎有不同偏好 |
| SafeTensors | SafeTensors | Hugging Face 推出的安全模型格式，防止恶意代码注入，目前主流格式 |
| PyTorch (.pt/.bin) | PyTorch Format | PyTorch 原生模型格式，支持 pickling，存在安全风险 |
| ONNX | ONNX | 开放神经网络交换格式，支持跨框架部署，兼容性好 |
| GGUF | GGUF | llama.cpp 专用格式，支持CPU/GPU混合推理，本地部署主流选择 |
| TensorRT 引擎 | TensorRT Engine | NVIDIA TensorRT 优化后的二进制格式，推理速度最快但不跨平台 |
| 模型转换 | Model Conversion | 将模型从一种格式转换为另一种，如 PyTorch → ONNX → TensorRT |
| 权重共享 | Weight Tying | 输入嵌入层和输出层共享权重，减少参数量 |
| 分片模型 | Sharded Model | 将大模型拆分为多个文件存储，便于下载和加载 |

## 六、推理引擎与框架

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 推理引擎 | Inference Engine | 优化模型推理速度的工具，如 vLLM、LMDeploy、TensorRT-LLM |
| vLLM | vLLM | 高吞吐量推理引擎，采用 PagedAttention 技术，生产环境首选 |
| LMDeploy | LMDeploy | 商汤科技推出的推理引擎，支持 Turbomind 和 PyTorch 后端 |
| TensorRT-LLM | TensorRT-LLM | NVIDIA 官方推理引擎，针对 NVIDIA GPU 深度优化，性能最佳 |
| llama.cpp | llama.cpp | C++ 实现的轻量级推理框架，支持CPU推理，本地部署首选 |
| Ollama | Ollama | 基于 llama.cpp 的封装工具，简化本地模型部署，一键运行 |
| Text Generation WebUI | Text Generation WebUI | 常用的本地大模型图形界面，支持多种后端和模型格式 |
| Hugging Face Transformers | Transformers | 最流行的模型库和框架，提供统一的模型加载和推理接口 |
| PyTorch | PyTorch | Meta 开发的深度学习框架，灵活易用，研究和生产皆适用 |
| ONNX Runtime | ONNX Runtime | 微软推出的跨平台推理引擎，支持ONNX格式模型 |
| OpenVINO | OpenVINO | Intel 推出的推理优化工具包，针对 Intel CPU/GPU 优化 |
| SGLang | SGLang | 高效的结构化生成语言，优化复杂提示词的执行效率 |
| Triton Inference Server | Triton | NVIDIA 推出的生产级推理服务框架，支持多模型多后端 |

## 七、推理优化技术

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 推理 | Inference | 模型加载权重后，接收用户输入（prompt）并生成输出的过程 |
| 预填充 | Prefill | 处理用户输入的初始阶段，并行计算所有输入token的KV Cache |
| 解码 | Decoding | 逐token生成输出的阶段，每步生成一个token |
| 显存 | GPU Memory / VRAM | 显卡专用内存，决定能加载的模型规模，如RTX 4090 24G即24G VRAM |
| 共享内存 | Shared Memory | 从系统内存中划分给显卡使用的部分，速度慢，无法用于正常模型推理 |
| KV Cache | KV Cache | 推理时存储上下文注意力信息，加速后续生成，减少重复计算 |
| PagedAttention | PagedAttention | vLLM 的核心创新，将 KV Cache 分页管理，显存利用率接近100% |
| 连续批处理 | Continuous Batching | 动态调整批次，有请求完成立即加入新请求，大幅提升吞吐量 |
| 投机采样 | Speculative Decoding | 用小模型快速生成候选token，大模型验证，加速推理 |
| 投机执行 | Speculative Execution | 推测采样的另一说法，通过并行计算加速生成 |
| Flash Attention | Flash Attention | IO感知的高效注意力算法，减少显存访问，加速训练和推理 |
| Flash Decoding | Flash Decoding | 优化解码阶段的注意力计算，提升长上下文生成速度 |
| 模型并行 | Model Parallelism | 将模型拆分到多个GPU上运行，支持超大模型推理 |
| 张量并行 | Tensor Parallelism (TP) | 将单个算子的计算拆分到多GPU，降低延迟 |
| 流水线并行 | Pipeline Parallelism (PP) | 将模型层拆分到多GPU，类似流水线处理，提升吞吐 |
| 序列并行 | Sequence Parallelism | 将长序列拆分到多GPU处理，支持超长上下文 |
| 批处理 | Batching | 同时处理多个请求，提升GPU利用率和吞吐量 |
| 动态批处理 | Dynamic Batching | 运行时动态组装批次，平衡延迟和吞吐 |
| 算子融合 | Operator Fusion | 将多个连续算子合并为一个，减少显存访问开销 |
| 图优化 | Graph Optimization | 优化计算图结构，如常量折叠、死代码消除等 |

## 八、服务部署与API

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| API 服务 | API Service | 通过HTTP接口提供模型推理能力，便于集成到各类应用 |
| OpenAI 兼容 API | OpenAI Compatible API | 遵循 OpenAI API 格式的接口，方便切换不同模型后端 |
| RESTful API | RESTful API | 基于 HTTP 的 API 设计风格，使用 JSON 格式交互 |
| gRPC | gRPC | 高性能 RPC 框架，支持流式传输，适合内部服务通信 |
| 流式响应 | Streaming Response | 通过 SSE 等技术逐token返回结果，提升用户体验 |
| SSE | Server-Sent Events | 服务端推送技术，常用于实现流式输出 |
| WebSocket | WebSocket | 全双工通信协议，支持实时双向数据传输 |
| 推理服务器 | Inference Server | 专门用于运行模型推理的服务端程序 |
| 模型服务化 | Model Serving | 将模型封装为可调用的服务，提供负载均衡、监控等能力 |
| TGI | Text Generation Inference | Hugging Face 推出的生产级推理服务器 |
| vLLM Server | vLLM Server | vLLM 内置的 OpenAI 兼容 API 服务器 |
| 负载均衡 | Load Balancing | 将请求分发到多个模型实例，提升整体处理能力 |
| 请求队列 | Request Queue | 排队等待处理的请求集合，合理管理避免服务过载 |
| 并发限制 | Concurrency Limit | 同时处理的请求数上限，防止资源耗尽 |
| 超时设置 | Timeout | 请求处理的最大等待时间，超时则返回错误 |
| 健康检查 | Health Check | 定期检测服务状态，确保服务可用性 |
| 模型热加载 | Hot Loading | 不停机加载新模型或更新模型，保证服务连续性 |
| 多模型部署 | Multi-model Deployment | 同一服务中部署多个模型，支持不同任务或A/B测试 |
| 容器化部署 | Container Deployment | 使用 Docker 等容器技术部署模型服务，便于管理和扩展 |
| Docker | Docker | 最流行的容器化平台，简化应用打包和部署 |
| Kubernetes | Kubernetes (K8s) | 容器编排平台，支持自动扩缩容和服务管理 |
| Helm Chart | Helm Chart | Kubernetes 应用的打包格式，简化部署配置 |
| Ray Serve | Ray Serve | 基于 Ray 的模型服务框架，支持自动扩缩容 |
| BentoML | BentoML | 机器学习模型服务化框架，简化部署流程 |

## 九、硬件与加速

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| GPU | Graphics Processing Unit | 图形处理器，大模型推理和训练的核心硬件 |
| CPU | Central Processing Unit | 中央处理器，可用于小模型推理，速度较慢 |
| TPU | Tensor Processing Unit | Google 专用的 AI 加速芯片，针对张量运算优化 |
| NPU | Neural Processing Unit | 神经网络处理器，如华为昇腾、苹果 Neural Engine |
| CUDA | CUDA | NVIDIA 的并行计算平台，GPU 加速的基础 |
| CUDA Core | CUDA Core | NVIDIA GPU 的基础计算单元 |
| Tensor Core | Tensor Core | NVIDIA GPU 的专用矩阵计算单元，大幅加速 AI 计算 |
| cuDNN | cuDNN | NVIDIA 的深度学习加速库，提供高效的神经网络算子 |
| NCCL | NCCL | NVIDIA 的多GPU通信库，支持分布式训练和推理 |
| 显存带宽 | Memory Bandwidth | 显存的数据传输速率，影响大模型加载和推理速度 |
| 内存带宽 | Memory Bandwidth | 系统内存的数据传输速率，影响CPU推理性能 |
| 互联带宽 | Interconnect Bandwidth | 多GPU之间数据传输速率，如 NVLink、PCIe |
| NVLink | NVLink | NVIDIA 的高速GPU互联技术，带宽远超 PCIe |
| PCIe | PCIe | 外设互联标准，GPU与CPU通信的主要通道 |
| 多卡推理 | Multi-GPU Inference | 使用多张GPU协同推理，支持更大模型或更高吞吐 |
| GPU 虚拟化 | GPU Virtualization | 将GPU资源切分给多个用户或任务使用 |
| MIG | Multi-Instance GPU | NVIDIA 技术，将单张GPU虚拟化为多个独立实例 |
| CPU 卸载 | CPU Offloading | 将部分模型权重存储在内存，按需加载到显存 |
| 磁盘卸载 | Disk Offloading | 将模型权重存储在磁盘，进一步降低内存需求 |
| 零拷贝 | Zero Copy | 减少数据在内存和显存间的复制，提升效率 |

## 十、生成与对话相关

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 提示词 | Prompt | 用户输入给模型的指令、问题或上下文，引导模型生成对应内容 |
| 系统提示 | System Prompt | 用于设定模型的角色、行为规则和回答风格（如"你是一名技术专家"） |
| 提示词模板 | Prompt Template | 预定义的提示词结构，便于快速构建标准化的输入 |
| 上下文学习 | In-Context Learning (ICL) | 通过在提示词中提供示例，让模型快速适应新任务 |
| 少样本学习 | Few-shot Learning | 在提示词中提供少量示例，引导模型理解任务格式 |
| 零样本学习 | Zero-shot Learning | 不提供任何示例，仅通过指令让模型完成任务 |
| 温度系数 | Temperature | 控制模型生成内容的随机性，值越高越随机，值越低越严谨 |
| Top-p 采样 | Top-p Sampling (Nucleus Sampling) | 从累积概率达到 p 的最小候选集中采样，控制生成多样性 |
| Top-k 采样 | Top-k Sampling | 只从概率最高的 k 个候选token中采样，过滤低概率选项 |
| 重复惩罚 | Repetition Penalty | 降低生成重复内容的概率，值越大惩罚越强 |
| 最大生成长度 | Max New Tokens | 模型在用户输入之外，最多能生成的token数量，防止生成过长内容 |
| 流式输出 | Streaming Output | 模型逐token实时输出结果，无需等待全部生成完成，提升交互体验 |
| 停止词 | Stop Strings / Stop Tokens | 遇到特定字符串或token时停止生成，控制输出边界 |
| 幻觉 | Hallucination | 模型编造不存在的事实、数据或逻辑，属于模型输出偏差的一种 |
| 提示词注入 | Prompt Injection | 恶意用户通过特殊输入覆盖系统提示，诱导模型执行非预期行为 |
| 越狱攻击 | Jailbreak Attack | 通过特定提示词绕过模型的安全限制，诱导生成有害内容 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，结合外部知识库提升模型回答的准确性和时效性 |
| 向量数据库 | Vector Database | 存储文本向量表示的数据库，支持相似度检索，RAG 系统核心组件 |
| 嵌入模型 | Embedding Model | 将文本转换为向量表示的模型，用于语义检索和相似度计算 |

## 十一、任务与能力相关

| 中文 | 英文 | 简要说明 |
|:---|:---|:---|
| 自然语言理解 | Natural Language Understanding (NLU) | 模型理解文本语义、情感、意图的能力（如读懂问题、分析情绪） |
| 自然语言生成 | Natural Language Generation (NLG) | 模型生成通顺、符合逻辑的人类语言的能力（如写文案、编故事） |
| 文本摘要 | Text Summarization | 模型提炼长文本核心内容，生成简洁摘要的能力 |
| 文本翻译 | Machine Translation | 模型在不同语言之间进行互译的能力（如中译英、英译日） |
| 代码生成 | Code Generation | 模型根据自然语言指令，生成符合语法、可运行代码的能力 |
| 代码补全 | Code Completion | 模型根据上下文自动补全代码的能力，提升开发效率 |
| 代码解释 | Code Explanation | 模型解释代码功能、逻辑和意图的能力 |
| 思维链 | Chain-of-Thought (CoT) | 让模型分步推理、逐步输出思考过程，提升复杂问题的解决能力 |
| 多模态 | Multimodal | 模型处理多种模态（文本、图像、音频、视频）的能力 |
| 视觉语言模型 | Vision-Language Model (VLM) | 能理解图像并生成文本描述或回答图像相关问题的模型 |
| 函数调用 | Function Calling | 模型根据用户需求调用外部工具或API的能力 |
| 工具使用 | Tool Use | 模型调用外部工具（如搜索、计算器）完成任务的能力 |
| Agent | AI Agent | 具有自主规划、工具使用、多步推理能力的智能体 |
| 评估基准 | Benchmark | 用于评估模型能力的标准化测试集，如 MMLU、GSM8K、HumanEval |
| 困惑度 | Perplexity (PPL) | 衡量模型预测文本能力的指标，值越低表示模型越好 |
| 输出吞吐量 | Output Throughput | 单位时间内模型生成的token数量，衡量推理效率 |
| 首 token 延迟 | Time to First Token (TTFT) | 从发送请求到生成第一个token的时间，影响用户等待体验 |
| 端到端延迟 | End-to-End Latency | 从发送请求到完成整个响应的总时间 |