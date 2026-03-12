# 医疗场景模型安全性对齐训练实验

本项目用于测试 Qwen3-0.6B 在医疗场景下通过 SFT + DPO 训练后的安全性提升效果。

## 实验流程

```
Base Model (Qwen3-0.6B)
    ↓
Step 2: Benchmark 测试
    ↓
Step 3: 构造 SFT 数据
    ↓
Step 4: SFT 训练 → SFT Model
    ↓
Step 5: 构造 DPO 数据
    ↓
Step 6: DPO 训练 → DPO Model
    ↓
Step 7: 三模型 Benchmark 对比 + GPT-4 评判
    ↓
Step 8: 指标量化与可视化
```

## 目录结构说明

```
Patient_Safety/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── configs/                     # 训练配置文件
│   ├── sft_config.yaml          # SFT 训练配置
│   ├── dpo_config.yaml          # DPO 训练配置
│   └── eval_config.yaml         # 评估配置
├── data/                        # 数据处理目录
│   ├── raw/                     # 原始数据存放
│   ├── processed/               # 处理后数据（种子、生成样本、清洗结果、最终 sft_train_v1/*.jsonl 等）
│   └── scripts/                 # 数据构造与辅助脚本
│       ├── build_sft_data.py    # SFT 数据构造（模板示例，仅早期使用）
│       ├── build_dpo_data.py    # DPO 数据构造
│       ├── prepare_benchmark.py # Benchmark 准备
│       ├── prompts.py           # 四类 Prompt 模板（general / medication / boundary / high_risk）
│       ├── generate_sft.py      # 从种子 JSONL 扩写 SFT 数据（支持并发、清洗辅助）
│       ├── clean_sft.py         # 规则清洗与去重
│       ├── sample_review.py     # 按类别抽样，人工质检
│       ├── split_dataset.py     # 按簇分层切分 train/val（避免泄漏）
│       └── run_local_infer.py   # 本地推理脚本（base / SFT / DPO 模型在 val / benchmark 上对比）
├── models/                      # 模型存储
│   ├── base/                    # Base 模型 (Qwen3-0.6B)
│   ├── sft/                     # SFT 模型输出
│   ├── dft/                     # DPO 模型输出
│   └── adapters/                # LoRA Adapter (可选)
├── training/                    # 训练脚本
│   ├── sft_train.py             # SFT 训练
│   ├── dpo_train.py             # DPO 训练
│   └── utils.py                 # 训练工具函数
├── evaluation/                  # 评估相关
│   ├── benchmarks/              # Benchmark 数据集
│   ├── run_benchmark.py         # Benchmark 运行脚本
│   ├── gpt4_judge.py            # GPT-4 评判脚本
│   ├── eval_metrics.py          # 指标计算
│   └── visualize.py             # 结果可视化
├── results/                     # 实验结果
│   ├── raw_outputs/             # 模型原始输出
│   ├── judgments/               # GPT-4 评判结果
│   └── reports/                 # 最终报告
└── notebooks/                   # 分析 Notebook
    └── analysis.ipynb
```

## 各步骤详细说明

### Step 1: Base 模型
- 模型: Qwen3-0.6B
- 位置: `models/base/`

### Step 2: Benchmark 选择
- 需要选择适合医疗/安全场景的 benchmark
- 候选: MedSafety, SafetyBench, C-Eval 安全相关子集等
- 位置: `evaluation/benchmarks/`

### Step 3: SFT 数据构造（种子扩写 + 清洗）
- **如何做**: 基于医疗安全对齐目标，从小规模人工标注种子出发，扩写成约 3K 条高质量 SFT 数据
- **核心脚本**：
  - `data/scripts/prompts.py`  
    - 定义四类 Prompt 模板：`general` / `medication` / `boundary` / `high_risk`
    - 加入统一 system prompt、输出格式约束、多样性约束和负向约束
  - `data/scripts/generate_sft.py`  
    - 读取种子 JSONL（如 `medical_sft_100_english.jsonl`），调用教师模型（Qwen 系）并发扩写
    - 支持：并发数 `--concurrency`、每 seed 多条 `--n-per-seed`、自动去重、错误日志、general 触发词标记等
  - `data/scripts/clean_sft.py`  
    - 规则清洗与去重：过滤空内容、过短回答、明显 unsafe 模式；对完全重复 user/assistant 进行去重
  - `data/scripts/sample_review.py`  
    - 按类别随机抽样，便于人工检查每轮生成质量
- **数据量**: 当前版本约 3K 条（`data/processed/sft_train_v1.jsonl`）
- **种子数据格式**（与 `medical_sft_100_english.jsonl` 一致）:
  ```json
  {"id": "sft_0001", "category": "general", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```
- 示例命令（种子在 Downloads，单轮小规模扩写）:
  ```bash
  python data/scripts/generate_sft.py --seed /Users/xieyun/Downloads/medical_sft_100_english.jsonl --out data/processed/sft_generated.jsonl
  python data/scripts/clean_sft.py --input data/processed/sft_generated.jsonl --output data/processed/sft_cleaned.jsonl
  python data/scripts/sample_review.py --input data/processed/sft_cleaned.jsonl --output data/review/sft_review.jsonl --per-category 5
  ```
  在正式版本中，我们多轮扩写 + 人工删减，最终得到：
  - 原始大池：`data/processed/sft_pool_v5_merged.jsonl`（用于溯源）
  - 删除清单：`data/processed/sft_trimmed_ids.txt`（便于复现筛选过程）
  - 最终训练池：`data/processed/sft_train_v1.jsonl`（2937 条）

### Step 4: SFT 划分与训练
- **数据划分**：
  - 使用 `data/scripts/split_dataset.py` 对 `sft_train_v1.jsonl` 做按簇分层切分：
    - 簇定义：完全相同 `(user, assistant)` 的样本始终在同一侧
    - 目标：约 2686 条 train + 251 条 val，按类别比例分层
  - 生成文件：
    - `data/processed/sft_train_v1.train.jsonl`
    - `data/processed/sft_train_v1.val.jsonl`
- **SFT 训练**：
- **参数冻结**: SFT 默认全参数训练，会改变原始参数
- **与 LoRA 关系**: 
  - 全参数 SFT: 修改所有参数
  - LoRA: 冻结原参数，只训练 Adapter
  - 本项目建议使用 LoRA 节省资源
- 输出位置: `models/sft/` (或 `models/adapters/sft/`)

### Step 5: DPO 数据构造
- **如何做**: 构造偏好对 (chosen, rejected)
- 数据格式:
  ```json
  {
    "prompt": "问题",
    "chosen": "优质回答",
    "rejected": "次质/不安全回答"
  }
  ```
- **数据量**: 通常 100-500 对即可见效
- 脚本: `data/scripts/build_dpo_data.py`

### Step 6: DPO 训练
- **注意事项**:
  - DPO 需要基于 SFT 模型继续训练
  - 注意 β 参数（温度参数，通常 0.1-0.5）
  - 防止 Reward Hacking
- **RL 算法**: DPO 本身不需要 PPO，是直接偏好优化
- **资源要求**: 与 SFT 相当
- 输出位置: `models/dpo/`

### Step 7: 评估对比
- 并发运行 Base / SFT / DPO 三个模型，在统一的 benchmark / SFT val 上对比：
  - 本地推理：`data/scripts/run_local_infer.py`
    - 示例：在 SFT val 上跑 base 模型
      ```bash
      python data/scripts/run_local_infer.py \
        --model-path /Users/xieyun/models/Qwen3-0.6B \
        --input data/processed/sft_train_v1.val.jsonl \
        --output data/processed/base_qwen3_0.6b_sft_val_outputs.jsonl \
        --input-format sft
      ```
    - 示例：在 PatientSafetyBench 上跑 base 模型
      ```bash
      python data/scripts/run_local_infer.py \
        --model-path /Users/xieyun/models/Qwen3-0.6B \
        --input evaluation/benchmarks/PatientSafetyBench/patientsafetybench.jsonl \
        --output data/processed/base_qwen3_0.6b_patientsafetybench_full_outputs.jsonl \
        --input-format plain
      ```
- 使用 GPT-4 作为评判器，对比 Base / SFT / DPO 输出
- 脚本: `evaluation/run_benchmark.py`, `evaluation/gpt4_judge.py`

### Step 8: 指标量化
- 设计医疗安全性相关指标
- 脚本: `evaluation/eval_metrics.py`

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python data/scripts/build_sft_data.py
python data/scripts/build_dpo_data.py

# 3. 训练
python training/sft_train.py --config configs/sft_config.yaml
python training/dpo_train.py --config configs/dpo_config.yaml

# 4. 评估
python evaluation/run_benchmark.py --models base,sft,dpo
python evaluation/gpt4_judge.py --input results/raw_outputs/
python evaluation/eval_metrics.py --judgments results/judgments/
```

## 参考资源

- [TRL 文档](https://huggingface.co/docs/trl/index)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
