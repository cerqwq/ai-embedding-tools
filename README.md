# 🧮 AI Embedding Tools

AI嵌入工具，支持文本嵌入、图像嵌入、多模态嵌入。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenAI-API-green?logo=openai" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

## ✨ 特性

- 🏗️ 嵌入策略设计
- 💻 嵌入代码生成
- 🔍 相似度搜索设计
- 📈 嵌入质量评估
- ⚖️ 嵌入模型比较
- 📊 微调数据生成

## 🚀 快速开始

```bash
pip install openai

python tools.py
```

## 📖 使用

```python
from ai_embedding_tools import create_tools

tools = create_tools()

# 嵌入策略
strategy = tools.design_embedding_strategy("语义搜索", ["文本", "代码"])

# 嵌入代码
code = tools.generate_embedding_code("text-embedding-3-small", "文本")

# 相似度搜索
similarity = tools.design_similarity_search("电商")

# 嵌入评估
evaluation = tools.evaluate_embeddings(embeddings)

# 模型比较
comparison = tools.compare_embedding_models("语义搜索")

# 微调数据
fine_tuning = tools.generate_fine_tuning_data("分类", examples)
```

## 📁 项目结构

```
ai-embedding-tools/
├── tools.py       # 嵌入工具核心
└── README.md
```

## 📄 许可证

MIT License
