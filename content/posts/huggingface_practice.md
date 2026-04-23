+++
date = '2026-04-22T10:00:00+08:00'
draft = true
title = 'Hugging Face 实战'
description = '从入门到精通的 Hugging Face 使用指南'
tags = ["Hugging Face", "Transformers", "NLP"]
categories = ["技术"]
+++


## 一、接入 Hugging Face 大模型 API

Hugging Face 提供了便捷的 Inference API，让开发者无需部署模型即可调用各种大语言模型。本节将详细介绍如何获取 API Token 并调用 Llama 3.2 模型。

### 1.1 获取 Access Token

首先需要在 Hugging Face 平台上获取访问令牌：

1. **登录账户**：访问 [Hugging Face](https://huggingface.co/) 并登录你的账户
2. **进入 Token 管理页面**：点击右上角头像 → `Access Tokens`
3. **创建新 Token**：点击 `Create new token` 按钮
4. **配置 Token 权限**：
   - 输入 Token 名称（如 `my-api-token`）
   - 勾选所需权限，推荐勾选 `Read` 权限用于 API 调用
5. **生成并保存**：点击 `Create token`，复制生成的 `HF_TOKEN` 并妥善保存

> ⚠️ 注意：Token 只会显示一次，请务必保存好。如果遗忘，需要重新生成。

![Access Tokens 页面](/images/huggingface_practice/Access%20Tokens.png)

### 1.2 选择并查看模型 API

Hugging Face 托管了海量预训练模型，许多模型提供了API的调用接口，当然也不是每个都提供，这里以 `meta-llama/Llama-3.2-1B-Instruct`为例：

1. **搜索模型**：在 Hugging Face 首页搜索框输入 `meta-llama/Llama-3.2-1B-Instruct`
2. **进入模型页面**：点击搜索结果进入模型详情页
3. **查看 API 调用方式**：找到 `Deploy` → `Inference Providers`
4. **选择服务商**：可以看到不同服务商的 API 端点和调用示例

![Inference Providers 页面](/images/huggingface_practice/Providers.png)

![API 调用代码示例](/images/huggingface_practice/Providers_code.png)

### 1.3 使用 Python 调用 API

以下是一个完整的 Python 示例，使用 `requests` 库调用 Llama 3.2 模型，此时需要传入 `HF_TOKEN` ：

```python
import os
import requests

os.environ["HF_TOKEN"] = "XXXX"

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    "Content-Type": "application/json",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "你是哪个模型，多大的?"
        }
    ],
    "model": "meta-llama/Llama-3.2-1B-Instruct:novita"
})

print(response["choices"][0]["message"]["content"])
```

### 1.4 API 输出结果
下图为 `meta-llama/Llama-3.2-1B-Instruct` 的输出结果
![API 输出结果示例](/images/huggingface_practice/Api_out.png)

### 1.5 常见问题

**Q: Token 验证失败怎么办？**

A: 检查 Token 是否正确复制，确认 Token 有足够的权限（需要 `Read` 权限）。

**Q: 模型调用超时？**

A: 部分模型需要较长加载时间，首次调用可能超时。可以增加 `timeout` 参数或稍后重试。

**Q: 如何查看可用模型？**

A: 访问 [Hugging Face Models](https://huggingface.co/models) 页面，筛选支持 Inference API 的模型


## 二、环境搭建与配置


## 三、Transformers 库基础


## 四、模型加载与使用


## 五、数据集处理


## 六、模型微调实战


## 七、模型上传与分享


## 八、推理部署


## 九、高级应用


## 十、常见问题与解决方案


## 参考资源