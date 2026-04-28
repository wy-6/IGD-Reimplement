$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..

python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1) baseline（小样本 + IG steps=10）
python run_train.py --config config.yaml --stage baseline --device cpu --dataset ag_news --max_train_samples 800 --max_eval_samples 400 --ig_steps 10

# 2) igd（伪样本在内存生成）
python run_train.py --config config.yaml --stage igd --device cpu --dataset ag_news --max_train_samples 800 --max_eval_samples 400

# 3) attack eval（小样本）
python run_attack_eval.py --config config.yaml --device cpu --dataset ag_news --attacks textfooler textbugger --max_eval_samples 50

