# AgentMem (LOCOMO Eval Pipeline)

本仓库保留了基于 mem0 的课程评测流程，便于快速复现 add / search / eval。

## 目录

- `evaluation/run_experiments.py`：运行 add/search
- `evaluation/evals.py`：评测，结束自动打印摘要
- `evaluation/print_eval.py`：单独打印 `eval.json` 摘要
- `results/`：示例输出（mem0 search 及评测结果）
- `.env.example`：环境变量模板

## 准备

1) 复制 `.env.example` 为 `.env`，填写：

```
OPENAI_API_KEY=
OPENAI_BASE_URL=
MEM0_API_KEY=
MEM0_PROJECT_ID=
MEM0_ORGANIZATION_ID=
MODEL=qwen3-8b
EMBEDDING_MODEL=text-embedding-v1
# 评测 LLM（可与运行模型分开）
EVAL_OPENAI_API_KEY=
EVAL_OPENAI_BASE_URL=
EVAL_MODEL=gpt-4o-mini
```

2) 安装依赖（使用已有 `.venv` 或自行创建）：

```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

3) 数据（如需重新下载）：

```
mkdir -p dataset
curl -L "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" -o dataset/locomo10.json
```

## 运行

- 写入记忆：

```
python evaluation/run_experiments.py --technique_type mem0 --method add
```

- 检索生成结果（示例 top_k=15）：

```
python evaluation/run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 15
```

- 评测并打印摘要：

```
set -a; source .env; set +a
python evaluation/evals.py --input_file results/mem0_results_top_15_filter_False_graph_False.json --output_file results/eval.json
```

  如需单独打印已有评测文件：

```
python evaluation/print_eval.py --input_file results/eval.json
```
