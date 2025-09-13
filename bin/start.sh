#!/bin/bash

BASE_PATH=$(cd "$(dirname -- "$0")" && pwd)
APP_PATH=$(dirname "$BASE_PATH")

source $APP_PATH/.venv/bin/activate

# 设置 Python 路径
export PYTHONPATH=$APP_PATH/src


python src/generate.py --mode cds --ckpt_path checkpoints/gemorna_cds.pt --protein_seq MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIF
python src/generate.py --mode 5utr --ckpt_path checkpoints/gemorna_5utr.pt --utr_length short
python src/generate.py --mode 3utr --ckpt_path checkpoints/gemorna_3utr.pt --utr_length long
python src/main_pred5UTR.py --ckpt_path checkpoints/5utr.pt --sequence TACGTTTTGACCTTCGTTCATTTTG
python src/main_pred3UTR.py --ckpt_path checkpoints/3utr.pt --sequence TGTCCCCGGGTCTTCCAACGGACTGGCGTTGCCCCGGTTCACTGGGGACTGCCCTTGGGGTCTCGCTCACCTTCAGCACACATTATCGGGAGCAGTGTCTTCCATAATGT

# 保持进程运行
echo "所有任务执行完成，进程将保持运行..."

# 执行完成后保持 bash 会话
exec bash