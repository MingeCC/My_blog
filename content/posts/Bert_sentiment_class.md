+++
date = '2026-04-26T13:17:31+08:00'
draft = false
title = 'BERT 中文情感分析实战教程'
description = 'BERT 中文情感分析实战教程，基于预训练 BERT 模型进行中文文本情感分析任务'
tags = ["LLM"]
categories = ["practice"]
+++


# BERT 中文情感分析实战教程

本文将详细介绍如何使用预训练的 BERT 模型进行中文文本情感分析任务。项目基于 `bert-base-chinese` 模型，使用 ChnSentiCorp 数据集进行二分类训练（正面/负面情感判断）。

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境准备](#2-环境准备)
3. [Part 1：BERT 模型基础使用](#3-part-1bert-模型基础使用)
4. [Part 2：数据集准备](#4-part-2数据集准备)
5. [Part 3：自定义模型与训练](#5-part-3自定义模型与训练)
6. [Part 4：模型推理与应用](#6-part-4模型推理与应用)
7. [总结](#7-总结)

---

## 1. 项目概述

### 1.1 什么是 BERT？

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的预训练语言模型。其核心特点：

- **双向编码**：同时考虑上下文信息，理解更准确
- **预训练 + 微调**：在大规模语料上预训练，针对特定任务微调。**微调（Fine-tuning）** 指在预训练模型基础上，使用特定任务的少量数据进行训练，使模型适应具体应用场景（如情感分析、文本分类等）
- **迁移学习**：预训练模型可作为各种 NLP 任务的基础

### 1.2 项目目标

使用 `bert-base-chinese` 预训练模型，对中文文本进行情感二分类：

| 类别 | 说明 |
|:-----|:-----|
| 0（负面） | 如"这个产品质量很差，不值得购买" |
| 1（正面） | 如"服务态度很好，非常满意" |

### 1.3 项目结构

```
3_Bert_Model_train/
│
├── model/                          # 预训练模型缓存目录
│   └── models--google-bert--bert-base-chinese/
│       ├── refs/main                # 分支引用
│       └── snapshots/
│           └── 8f23c25b06e.../      # 模型版本ID（过长已省略）
│               ├── config.json      # 模型配置
│               ├── vocab.txt        # 词表（21128个中文token）
│               ├── tokenizer.json   # 分词器配置
│               └── model.safetensors # 模型权重
│       └── .no_exist/               # 缓存标记目录
│
├── Dataset/                        # 数据集目录
│   ├── dataset_dict.json           # 数据集索引
│   ├── lansinuote___chn_senti_corp/ # 原始缓存目录
│   │   └── default/0.0.0/b0c4c1.../
│   │       ├── chn_senti_corp-train.arrow
│   │       ├── chn_senti_corp-validation.arrow
│   │       ├── chn_senti_corp-test.arrow
│   │       └── dataset_info.json
│   ├── train/                      # 训练集
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── validation/                 # 验证集
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── test/                       # 测试集
│       ├── data-00000-of-00001.arrow
│       ├── dataset_info.json
│       └── state.json
│
│
├── ex1_1_Model_download.py         # 模型下载与加载
├── ex1_2_Bert_feature_extract.py   # Pipeline特征提取
├── ex1_3_AutoModelForMaskedLM_fill_mask.py  # 填空任务
├── ex1_4_AutoModelForNextSentencePrediction.py  # 下一句预测
│
├── ex2_1_Dataset_download.py       # 数据集下载
├── ex2_2_Make_dataset.py           # 自定义Dataset类
│
├── ex3_1_New_class_model.py        # 自定义分类模型
├── ex3_2_Train.py                  # 训练脚本
├── ex3_3_Eval.py                   # 评估脚本
├── ex3_4_Run_model.py              # 交互式推理
│
└── bert_fc_sentiment.pth           # 训练好的模型权重
```

---

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install transformers torch datasets
```

### 2.2 主要依赖说明

| 库 | 版本建议 | 作用 |
|:----|:----------|:------|
| `transformers` | ≥4.0 | HuggingFace 模型库，提供 BERT 及各类预训练模型。|
| `torch` | ≥1.8 | PyTorch 深度学习框架 |
| `datasets` | ≥2.0 | HuggingFace 数据集库 |

### 2.3 硬件要求

- **GPU**：推荐 CUDA 支持的显卡（训练速度提升 10 倍以上）
- **内存**：建议 8GB 以上
- **磁盘**：模型约 400MB，数据集约 50MB

---

## 3. Part 1：BERT 模型基础使用

在正式训练之前，先了解 BERT 模型的基本使用方式和三种典型任务。

### 3.1 模型下载与加载

**文件**：`ex1_1_Model_download.py`

#### 代码解析

```python
from transformers import (
    AutoTokenizer,                    # 自动加载分词器
    AutoModel,                        # 基础 BERT 模型（提取特征）
    AutoModelForMaskedLM,             # 填空任务模型
    AutoModelForNextSentencePrediction,  # 下一句预测模型
)

model_name = "google-bert/bert-base-chinese"  # 模型名称
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"  # 本地缓存路径

# 从本地加载模型
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=True  # 仅从本地加载，不联网下载
)
model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=True
)
```

#### 关键概念解释

**AutoTokenizer（分词器）**：
- 将中文文本转换为数字 ID 序列
- `bert-base-chinese` 使用字符级分词，词表大小 21128
- 例如："我喜欢学习" → `[101, 659, 1756, 2361, 1394, 102]`（101/102 是特殊token）

**词表（Vocabulary）**：
- 词表是分词器的核心组成部分，包含模型能识别的所有 token
- 每个 token 在词表中有一个唯一编号（ID）
- `bert-base-chinese` 词表包含 21128 个条目，覆盖常用汉字、标点符号和特殊标记
- 词表决定了模型能"理解"哪些字符，词表之外的字符会被拆分或替换为特殊标记

**local_files_only=True**：
- 首次运行需设置为 `False` 下载模型
- 下载后改为 `True`，避免重复下载

**三种模型类的区别**：

| 模型类 | 输出 | 适用场景 |
|:--------|:------|:----------|
| `AutoModel` | 隐藏层特征 (768维) | 特征提取、下游任务微调 |
| `AutoModelForMaskedLM` | 填空预测 | 文本补全 |
| `AutoModelForNextSentencePrediction` | 二分类输出 | 句子关系判断 |

---

### 3.2 Pipeline 快速推理

**文件**：`ex1_2_Bert_feature_extract.py`

#### 代码解析

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"

tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)

# 创建分类 pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")

# 推理
result = classifier("我喜欢学习人工智能。", top_k=2)
print(result)
```

#### Pipeline 说明

`pipeline` 是 transformers 提供的快速推理接口，封装了分词、推理、输出解析等步骤：

```
输入文本 → tokenizer编码 → 模型推理 → softmax → 输出结果
```

**常用 pipeline 类型**：

| pipeline 类型 | 任务 |
|:---------------|:------|
| `text-classification` | 文本分类 |
| `fill-mask` | 填空 |
| `sentiment-analysis` | 情感分析 |
| `question-answering` | 问答 |

**注意**：直接使用预训练的 `BertForSequenceClassification` 进行分类，输出是随机初始化的分类头，结果无意义。需要先在特定数据集上训练。

---

### 3.3 填空任务（Masked Language Modeling）

**文件**：`ex1_3_AutoModelForMaskedLM_fill_mask.py`

#### 代码解析

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch

model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)

# 创建填空 pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, 
                     device="cuda" if torch.cuda.is_available() else "cpu")

# 使用 [MASK] 遮盖一个词
text = f"我喜欢学{tokenizer.mask_token}语。"

# 预测被遮盖的词
results = fill_mask(text, top_k=5)

print("输入:", text)
for item in results:
    print(f"score={item['score']:.4f}, token={item['token_str']}, sentence={item['sequence']}")
```

#### 输出示例

```
输入: 我喜欢学[MASK]语。
score=0.8234, token=英, sentence=我喜欢学英语。
score=0.0521, token=法, sentence=我喜欢学法语。
score=0.0312, token=日, sentence=我喜欢学日语。
score=0.0189, token=韩, sentence=我喜欢学韩语。
score=0.0156, token=中, sentence=我喜欢学中文语。
```

#### 原理说明

BERT 预训练时采用 MLM（Masked Language Modeling）策略：

1. 随机遮盖 15% 的输入 token
2. 模型预测被遮盖位置的原始词
3. 通过大量训练，模型学会理解上下文

这是 BERT 能够理解语义的核心训练方式。

---

### 3.4 下一句预测（Next Sentence Prediction）

**文件**：`ex1_4_AutoModelForNextSentencePrediction.py`

#### 代码解析

```python
import torch
from transformers import AutoTokenizer, AutoModelForNextSentencePrediction

model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
model = AutoModelForNextSentencePrediction.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
model.eval()

# 输入两个句子
sentence_a = "我喜欢学习人工智能。"
sentence_b = "它可以帮助我们理解语言。"

# 编码（会将两个句子拼接）
inputs = tokenizer(sentence_a, sentence_b, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 输出是二分类 logits
probs = torch.softmax(outputs.logits, dim=-1)[0]
labels = ["是下一句", "不是下一句"]
prediction = labels[int(torch.argmax(probs))]

print("句子 A:", sentence_a)
print("句子 B:", sentence_b)
print(f"{labels[0]}: {probs[0].item():.4f}")
print(f"{labels[1]}: {probs[1].item():.4f}")
print("预测结果:", prediction)
```

#### 输出示例

```
句子 A: 我喜欢学习人工智能。
句子 B: 它可以帮助我们理解语言。
是下一句: 0.9823
不是下一句: 0.0177
预测结果: 是下一句
```

#### 原理说明

BERT 预训练的另一任务是 NSP：

1. 输入两个句子 A 和 B
2. 模型判断 B 是否是 A 的下一句
3. 使用特殊 token `[SEP]` 分隔句子，`[CLS]` 位置的输出用于分类

**[SEP] Token（Separator Token）**：
- `[SEP]` 是 BERT 的特殊 token，用于分隔不同的句子
- 在 NSP 任务中，句子 A 和句子 B 之间插入 `[SEP]`，让模型知道句子的边界
- 输入格式为：`[CLS] 句子A [SEP] 句子B [SEP]`
- 示例：`[CLS] 我喜欢学习人工智能 [SEP] 它可以帮助我们理解语言 [SEP]`

---

## 4. Part 2：数据集准备

### 4.1 数据集下载

**文件**：`ex2_1_Dataset_download.py`

#### 代码解析

```python
from datasets import load_dataset, load_from_disk

# 首次下载（取消注释执行）
# ds = load_dataset(
#     "lansinuote/ChnSentiCorp",
#     cache_dir=r"D:\Desktop\LLM_start\3_Bert_Model_train\Dataset"
# )

# 保存到本地磁盘
# ds.save_to_disk(r"D:\Desktop\LLM_start\3_Bert_Model_train\Dataset")

# 后续从本地加载
ds_local = load_from_disk(r"D:\Desktop\LLM_start\3_Bert_Model_train\Dataset")
print(ds_local)

# 查看数据
test_ds = ds_local["test"]
print(test_ds)

for data in test_ds:
    print(data)
```

#### 数据集说明

**ChnSentiCorp** 是中文情感分析数据集：

| 划分 | 样本数 |
|:------|:--------|
| train | 9600 |
| validation | 1200 |
| test | 1200 |

**数据格式**：

```python
{
    'text': '这个酒店服务态度很好，房间干净整洁',  # 文本内容
    'label': 1                                    # 标签（0=负面, 1=正面）
}
```

#### 数据示例

```
{'text': '房间太小了。和网上的图片反差太大。感觉被骗了。', 'label': 0}
{'text': '酒店设施陈旧，服务态度也很差，不推荐入住', 'label': 0}
{'text': '环境优雅，服务周到，下次还会选择入住', 'label': 1}
{'text': '性价比很高，房间宽敞明亮，非常满意', 'label': 1}
```

---

### 4.2 自定义 Dataset 类

**文件**：`ex2_2_Make_dataset.py`

#### 代码解析

```python
from torch.utils.data import Dataset
from datasets import load_from_disk

class mydataset(Dataset):
    """
    自定义 PyTorch Dataset 类，用于加载情感分析数据集
    
    Args:
        split: 数据划分，可选 'train', 'test', 'validation'
    """
    
    def __init__(self, split):
        # 从本地加载整个数据集
        self.dataset = load_from_disk(r"D:\Desktop\LLM_start\3_Bert_Model_train\Dataset")
        
        # 根据 split 选择对应划分
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        else:
            raise ValueError("Invalid split name. Use 'train', 'test', or 'validation'.")

    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)

    def __getitem__(self, item):
        """获取单个样本"""
        data = self.dataset[item]["text"]    # 获取文本
        label = self.dataset[item]["label"]  # 获取标签
        return data, label
    
if __name__ == "__main__":
    test_dataset = mydataset("test")
    
    for data in test_dataset:
        print(data)
```

#### Dataset 类说明

PyTorch 的 `Dataset` 类需要实现三个方法：

| 方法 | 作用 |
|:------|:------|
| `__init__` | 初始化，加载数据 |
| `__len__` | 返回数据集长度 |
| `__getitem__` | 根据索引返回单个样本 |

这样设计的目的是：
- 与 PyTorch 的 `DataLoader` 配合使用
- 支持批量加载、数据增强、shuffle 等功能
- 代码结构清晰，便于扩展

---

## 5. Part 3：自定义模型与训练

### 5.1 自定义分类模型

**文件**：`ex3_1_New_class_model.py`

#### 代码解析

```python
from transformers import BertModel
import torch

# 加载预训练 BERT 模型
pretrained_model = BertModel.from_pretrained(
    "google-bert/bert-base-chinese",
    cache_dir=r"D:\Desktop\LLM_start\3_Bert_Model_train\model",
    local_files_only=True
)

class Model(torch.nn.Module):
    """
    自定义 BERT 文本分类模型
    
    架构：BERT (冻结) + 全连接层 (可训练)
    """
    
    def __init__(self):
        super().__init__()
        self.bert = pretrained_model           # 预训练 BERT
        self.fc = torch.nn.Linear(768, 2)      # 分类头：768 → 2

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        前向传播
        
        Args:
            input_ids: token ID 序列
            attention_mask: 注意力掩码（区分真实token和padding）
            token_type_ids: token 类型 ID（区分句子A和句子B）
        
        Returns:
            logits: 分类概率 [batch_size, 2]
        """
        # BERT 前向传播（冻结参数，不参与训练）
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        # 取 [CLS] token 的输出作为句子表示
        # last_hidden_state: [batch_size, seq_len, 768]
        # [:, 0, :] 取第0个位置（[CLS]）的向量
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 全连接层分类
        outputs = self.fc(cls_output)  # [batch_size, 2]
        
        # softmax 转为概率
        logits = torch.softmax(outputs, dim=-1)
        
        return logits
```

#### 模型架构详解

```
输入文本
    ↓
Tokenizer 编码 → [input_ids, attention_mask, token_type_ids]
    ↓
BERT Encoder（冻结，不训练）
    ↓
[CLS] token 输出 (768维向量)
    ↓
全连接层 Linear(768, 2)
    ↓
Softmax
    ↓
输出概率 [负面概率, 正面概率]
```

#### 关键概念解释

**[CLS] Token**：
- BERT 在每个输入开头添加特殊 token `[CLS]`
- `[CLS]` 位置的输出向量被设计用于句子级别的任务
- 其 **Embedding（嵌入向量）** 融合了整个句子的语义信息

**Embedding（嵌入）**：
- Embedding 是将离散符号（如文字）转换为连续向量表示的技术
- 在 BERT 中，每个 token 被映射为一个 768 维的向量
- 这些向量在高维空间中表示 token 的语义含义，语义相近的词向量距离更近
- 例如："高兴"和"开心"的 embedding 向量会比较接近，而"高兴"和"悲伤"会距离较远

**冻结 BERT 参数**：
```python
with torch.no_grad():
    outputs = self.bert(...)
```
- `torch.no_grad()` 禁止梯度计算
- BERT 有约 1.1 亿参数，冻结后只训练全连接层的 1536 个参数
- 大大减少训练时间和资源消耗
- 预训练模型已具备强大的语义理解能力，无需重新学习

**attention_mask**：
- 告诉模型哪些位置是真实 token，哪些是 padding
- 真实 token 为 1，padding 为 0
- 避免 padding token 参与注意力计算

---

### 5.2 训练脚本

**文件**：`ex3_2_Train.py`

#### 完整代码解析

```python
from ex3_1_New_class_model import Model      # 导入自定义模型
from transformers import BertTokenizer        # 分词器
from ex2_2_Make_dataset import mydataset     # 导入数据集类
import torch
from torch.utils.data import DataLoader      # 数据加载器
from torch.optim import AdamW                # 优化器

# ==================== 配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"

# 加载分词器
token = BertTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=True
)

# ==================== 数据预处理函数 ====================
def collate_fn(data):
    """
    DataLoader 的批处理函数
    
    **collate_fn（整理函数）**：
    - collate_fn 是 DataLoader 的自定义批处理函数
    - 当 DataLoader 从 Dataset 中取出一批数据时，需要将这些数据整理成统一的 tensor 格式
    - 默认的 collate_fn 只能处理简单的数据格式，复杂场景需要自定义
    - 在本项目中，collate_fn 负责将文本列表批量编码为 token 序列，并转换为 tensor
    
    Args:
        data: list of (text, label) tuples
    
    Returns:
        input_ids, attention_mask, token_type_ids, labels (均为 tensor)
    """
    # 分离文本和标签
    sentes = [i[0] for i in data]    # 文本列表
    labels = [i[1] for i in data]    # 标签列表
    
    # 批量编码
    encoding = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,  # 待编码文本
        truncation=True,                  # 超长截断。**Truncation（截断）**：当文本长度超过 max_length 时，自动截断超出部分。BERT 最大支持 512 个 token，超出会导致错误，截断可确保输入长度符合模型要求
        padding="max_length",             # 填充到固定长度
        max_length=360,                   # 最大长度
        return_tensors="pt",              # 返回 PyTorch tensor
        return_length=True                # 返回实际长度
    )
    
    # 提取各部分
    input_ids = encoding["input_ids"]           # token ID
    attention_mask = encoding["attention_mask"] # 注意力掩码
    token_type_ids = encoding["token_type_ids"] # token 类型
    
    # 标签转为 tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_mask, token_type_ids, labels


# ==================== 数据加载 ====================
train_dataset = mydataset("train")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,        # 每批 64 个样本
    shuffle=True,         # 打乱顺序。**Shuffle（打乱）**：在每个 epoch 开始时随机打乱数据顺序，避免模型按固定顺序学习样本，有助于模型更泛化，防止因样本顺序产生的偏差
    drop_last=True,       # 丢弃不完整的最后一批
    collate_fn=collate_fn  # 批处理函数
)


# ==================== 训练主循环 ====================
if __name__ == "__main__":
    
    epochs = 10                          # 训练轮数
    print(device)
    
    model = Model().to(device)           # 初始化模型并移到 GPU
    
    # 只优化全连接层参数（BERT 已冻结）
    optimizer = AdamW(model.fc.parameters(), lr=5e-4)
    
    # 交叉熵损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        model.bert.eval()  # BERT 保持 eval 模式
        
        total_loss = 0.0
        total_correct = 0
        total_num = 0
        
        for step, batch in enumerate(train_dataloader, start=1):
            # 获取 batch 数据
            input_ids, attention_mask, token_type_ids, labels = batch
            
            # 移到 GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            
            # ===== 前向传播 =====
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # ===== 计算损失 =====
            loss = loss_fn(outputs, labels)
            
            # ===== 反向传播 =====
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数
            
            # ===== 统计 =====
            batch_size = labels.size(0)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_num += batch_size
            
            # 每 64 步打印进度
            if step % 64 == 0 or step == len(train_dataloader):
                avg_loss = total_loss / step
                train_acc = total_correct / total_num
                
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Step [{step}/{len(train_dataloader)}] "
                    f"Loss: {avg_loss:.4f} "
                    f"Train Acc: {train_acc:.4f}"
                )
        
        # 每轮结束统计
        avg_loss = total_loss / len(train_dataloader)
        train_acc = total_correct / total_num
        
        print(
            f"========== Epoch {epoch + 1}/{epochs} 完成 ==========\n"
            f"Avg Loss: {avg_loss:.4f}\n"
            f"Train Acc: {train_acc:.4f}\n"
        )
    
    # ===== 保存模型 =====
    torch.save(
        model.state_dict(),
        r"D:\Desktop\LLM_start\3_Bert_Model_train\bert_fc_sentiment.pth"
    )
    
    print("模型参数已保存")
```

#### 训练流程详解

```
┌─────────────────────────────────────────────────────────────┐
│                        训练流程                              │
├─────────────────────────────────────────────────────────────┤
│  1. DataLoader 加载一批数据 (64个样本)                        │
│  2. collate_fn 编码文本 → tensor                             │
│  3. 数据移到 GPU                                             │
│  4. 模型前向传播 → 输出概率                                   │
│  5. 计算损失 (CrossEntropyLoss)                              │
│  6. 反向传播计算梯度                                          │
│  7. 优化器更新全连接层参数                                     │
│  8. 统计准确率、损失                                          │
│  9. 循环直至所有 epoch 完成                                   │
│  10. 保存模型参数                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 关键参数解释

| 参数 | 值 | 说明 |
|:------|:-----|:------|
| `epochs` | 10 | 训练轮数，数据集被完整遍历的次数 |
| `batch_size` | 64 | 每批样本数，影响内存占用和训练稳定性 |
| `max_length` | 360 | 文本最大长度，超长截断，不足填充 |
| `lr` | 5e-4 | 学习率，控制参数更新步长 |

#### 为什么只训练 fc 层？

```python
optimizer = AdamW(model.fc.parameters(), lr=5e-4)
```

- BERT 参数冻结，不传入优化器
- 只更新全连接层的 1536 个参数
- 训练速度快，防止破坏预训练学到的语义知识

#### collate_fn 详解

`batch_encode_plus` 批量编码功能：

```python
encoding = token.batch_encode_plus(
    batch_text_or_text_pairs=sentes,  # 输入文本列表
    truncation=True,                  # 超过 max_length 截断
    padding="max_length",             # 填充到 max_length
    max_length=360,                   # 最大序列长度
    return_tensors="pt",              # 返回 PyTorch tensor
)
```

编码后返回的三个关键 tensor：

| tensor | 形状 | 说明 |
|--------|------|------|
| `input_ids` | [64, 360] | token ID 序列，是文本中每个字符在词表中的编号。例如 "我喜欢" 编码后可能得到 `[101, 659, 1756, 2361, 102]`，其中 101 是 `[CLS]`，102 是 `[SEP]`，其他数字对应汉字的词表编号 |
| `attention_mask` | [64, 360] | 1=真实token, 0=padding |
| `token_type_ids` | [64, 360] | 0=句子A, 1=句子B。用于区分输入中的不同句子。在单句子任务中全为 0，在双句子任务（如 NSP）中，第一句部分为 0，第二句部分为 1。这帮助模型理解哪些 token 属于哪个句子 |

---

### 5.3 模型评估

**文件**：`ex3_3_Eval.py`

#### 代码解析

```python
from ex3_1_New_class_model import Model
from ex2_2_Make_dataset import mydataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

# ==================== 配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
max_length = 360
model_path = r"D:\Desktop\LLM_start\3_Bert_Model_train\bert_fc_sentiment.pth"
model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"

token = BertTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=True
)

# ==================== collate_fn ====================
def collate_fn(data):
    sentes = [i[0] for i in data]
    labels = [i[1] for i in data]
    
    encoding = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_mask, token_type_ids, labels


# ==================== 评估函数 ====================
def evaluate(model, dataloader, loss_fn):
    """
    模型评估函数
    
    Args:
        model: 待评估模型
        dataloader: 测试数据加载器
        loss_fn: 损失函数
    
    Returns:
        avg_loss: 平均损失
        acc: 准确率
        stat: 统计信息字典
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    
    # 分类统计
    total_positive = 0    # 真实正面样本数
    total_negative = 0    # 真实负面样本数
    pred_positive = 0     # 预测正面样本数
    pred_negative = 0     # 预测负面样本数
    
    with torch.no_grad():  # 评估时不计算梯度
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels = batch
            
            # 移到 GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # 计算损失
            loss = loss_fn(outputs, labels)
            
            # 预测结果
            preds = torch.argmax(outputs, dim=1)
            
            batch_num = labels.size(0)
            
            # 累计统计
            total_loss += loss.item() * batch_num
            total_correct += (preds == labels).sum().item()
            total_num += batch_num
            
            # 分类统计
            total_positive += (labels == 1).sum().item()
            total_negative += (labels == 0).sum().item()
            pred_positive += (preds == 1).sum().item()
            pred_negative += (preds == 0).sum().item()
    
    avg_loss = total_loss / total_num
    acc = total_correct / total_num
    
    return avg_loss, acc, {
        "total_num": total_num,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "pred_positive": pred_positive,
        "pred_negative": pred_negative,
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("当前设备:", device)
    
    # 1. 加载测试集
    test_dataset = mydataset("test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print("测试集样本数:", len(test_dataset))
    print("测试集 batch 数:", len(test_dataloader))
    
    # 2. 加载训练好的模型
    model = Model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("模型参数加载成功:", model_path)
    
    # 3. 评估模型
    loss_fn = torch.nn.CrossEntropyLoss()
    
    test_loss, test_acc, stat = evaluate(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn
    )
    
    # 4. 打印结果
    print("========== 测试结果 ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")
    print(f"测试样本总数: {stat['total_num']}")
    print(f"真实正面样本数: {stat['total_positive']}")
    print(f"真实负面样本数: {stat['total_negative']}")
    print(f"预测正面样本数: {stat['pred_positive']}")
    print(f"预测负面样本数: {stat['pred_negative']}")
```

#### 评估输出示例

```
当前设备: cuda
测试集样本数: 1200
测试集 batch 数: 19
模型参数加载成功: D:\Desktop\LLM_start\3_Bert_Model_train\bert_fc_sentiment.pth
========== 测试结果 ==========
Test Loss: 0.1234
Test Acc : 0.9458
测试样本总数: 1200
真实正面样本数: 600
真实负面样本数: 600
预测正面样本数: 582
预测负面样本数: 618
```

---

## 6. Part 4：模型推理与应用

### 6.1 交互式推理脚本

**文件**：`ex3_4_Run_model.py`

#### 完整代码解析

```python
from ex3_1_New_class_model import Model
from transformers import BertTokenizer
import torch

# ==================== 配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\Desktop\LLM_start\3_Bert_Model_train\model"
model_path = r"D:\Desktop\LLM_start\3_Bert_Model_train\bert_fc_sentiment.pth"
max_length = 360
names = ["负面", "正面"]

# ==================== 加载 tokenizer ====================
tokenizer = BertTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=True
)

# ==================== 单句编码函数 ====================
def encode_sentence(sentence):
    """
    将单个句子编码为模型输入格式
    
    Args:
        sentence: 输入文本
    
    Returns:
        input_ids, attention_mask, token_type_ids (均为 tensor)
    """
    encoding = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[sentence],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    
    return input_ids, attention_mask, token_type_ids


# ==================== 加载训练好的模型 ====================
def load_trained_model():
    """加载训练好的模型"""
    model = Model().to(device)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 设置为评估模式
    model.eval()
    
    return model


# ==================== 预测函数 ====================
def predict(model, sentence):
    """
    对单个句子进行情感预测
    
    Args:
        model: 分类模型
        sentence: 输入文本
    
    Returns:
        pred: 预测类别 (0 或 1)
        label_name: 类别名称 ("负面" 或 "正面")
        negative_prob: 负面概率
        positive_prob: 正面概率
    """
    # 编码输入
    input_ids, attention_mask, token_type_ids = encode_sentence(sentence)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取概率
        probs = outputs  # 模型输出已经是 softmax 结果
        pred = torch.argmax(probs, dim=1).item()
    
    label_name = names[pred]
    negative_prob = probs[0][0].item()
    positive_prob = probs[0][1].item()
    
    return pred, label_name, negative_prob, positive_prob


# ==================== 主程序：交互式输入 ====================
if __name__ == "__main__":
    print("当前设备:", device)
    print("正在加载模型...")
    
    model = load_trained_model()
    
    print("模型加载完成")
    print("请输入测试文本，输入 q 退出")
    print("-" * 50)
    
    while True:
        sentence = input("请输入测试文本：").strip()
        
        if sentence.lower() == "q":
            print("测试结束")
            break
        
        if sentence == "":
            print("输入为空，请重新输入")
            continue
        
        # 进行预测
        pred, label_name, negative_prob, positive_prob = predict(model, sentence)
        
        # 输出结果
        print("模型判定:", label_name)
        print("预测类别:", pred)
        print(f"负面概率: {negative_prob:.4f}")
        print(f"正面概率: {positive_prob:.4f}")
        print("-" * 50)
```

#### 运行示例

```
当前设备: cuda
正在加载模型...
模型加载完成
请输入测试文本，输入 q 退出
--------------------------------------------------
请输入测试文本：这个酒店服务态度很好，房间干净整洁
模型判定: 正面
预测类别: 1
负面概率: 0.0234
正面概率: 0.9766
--------------------------------------------------
请输入测试文本：房间太小了，和图片反差太大，感觉被骗了
模型判定: 负面
预测类别: 0
负面概率: 0.9812
正面概率: 0.0188
--------------------------------------------------
请输入测试文本：q
测试结束
```

---

## 7. 总结

### 7.1 项目要点回顾

| 步骤 | 文件 | 核心内容 |
|:------|:------|:----------|
| 模型加载 | ex1_1 | 从 HuggingFace 下载并加载 `bert-base-chinese` |
| 基础推理 | ex1_2~1_4 | Pipeline 快速推理、填空任务、下一句预测 |
| 数据准备 | ex2_1~2_2 | 下载 ChnSentiCorp 数据集、自定义 Dataset 类 |
| 模型构建 | ex3_1 | 自定义分类模型：BERT + 全连接层 |
| 模型训练 | ex3_2 | 10 轮训练，冻结 BERT，只训练分类头 |
| 模型评估 | ex3_3 | 测试集准确率评估 |
| 模型应用 | ex3_4 | 交互式情感预测 |

### 7.2 关键技术点

1. **迁移学习思想**：利用预训练模型的语义理解能力，只需训练少量参数即可完成下游任务

2. **冻结策略**：保持 BERT 参数不变，只训练分类头，避免过拟合、节省资源

3. **[CLS] Token**：BERT 设计用于句子级任务的特殊 token，聚合全局语义信息

4. **DataLoader + collate_fn**：PyTorch 数据加载标准模式，支持批量编码

### 7.3 扩展方向

- **解冻微调**：解冻 BERT 后几层参数，进一步优化模型
- **其他任务**：序列标注、问答、文本匹配等
- **其他预训练模型**：RoBERTa、ERNIE、BERT-wwm 等中文优化模型

---

## 参考资料

- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)