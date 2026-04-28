# IGD (Integrated Gradients-based Defense) Reimplementation

目标：按 `文档/IGD技术路线.txt` 复现 IGD（IG 伪样本 + 混合训练 + GRL + AM-Softmax），并在 AGNEWS / IMDB 上用 TextFooler、TextBugger 做鲁棒性评估。支持 **先 CPU 跑通**，再在 **GPU 环境**（如腾讯云 Cloud Studio、本地 CUDA 等）完整训练。

## 目录

```text
IGD-Reimplement/
  igd/
    __init__.py
    config.py
    data.py
    ig.py
    synonyms.py
    model.py
    losses.py
    train.py
    pseudo.py
    attack_eval.py
    utils.py
  scripts/
    cpu_smoke_test.ps1
  requirements.txt
  run_train.py
  run_attack_eval.py
  config.yaml
```

## 快速开始（CPU 跑通）

1) 安装依赖

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2) 训练 baseline（用于 IG 打分）

```bash
python run_train.py --config config.yaml --stage baseline --device cpu --max_train_samples 800 --max_eval_samples 400 --ig_steps 10
```

3) 训练 IGD（会先按当前训练集 **在内存中** 生成伪样本，不落盘）

```bash
python run_train.py --config config.yaml --stage igd --device cpu --max_train_samples 800 --max_eval_samples 400
```

4) TextAttack 评估（TextFooler / TextBugger）

```bash
python run_attack_eval.py --config config.yaml --attacks textfooler textbugger --device cpu --max_eval_samples 200
```

## Cloud Studio / GPU 环境（概要）

1. 将仓库导入工作空间后进入 `IGD-Reimplement/`（或保证 `config.yaml` 路径正确）。
2. 可选：把模型与日志写到数据盘等处时，设置环境变量覆盖输出目录，例如  
   `export IGD_OUTPUT_DIR=/path/to/your_outputs`
3. `pip install -r requirements.txt` 与 `python -m spacy download en_core_web_sm`
4. 依次运行：`run_train.py --stage baseline` → `run_train.py --stage igd` → `run_attack_eval.py`

仓库内若有在 Kaggle 下保存的 checkpoint，`latest_checkpoint.json` 中的旧绝对路径会自动尝试映射到当前 `paths.output_dir`（或 `IGD_OUTPUT_DIR`）下对应子路径，便于断点续跑。
