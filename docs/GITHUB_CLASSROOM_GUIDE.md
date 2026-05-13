# GitHub Classroom Guide

## Clone Repository

```bash
git clone <repo_url>

conda create -n csc4005-dl python=3.10
conda activate csc4005-dl

pip install -r requirements.txt

python -m src.train `
--config configs/baseline_mfcc_1dcnn.json `
--data_dir "UrbanSound8K"

