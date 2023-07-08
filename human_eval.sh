#!/bin/sh
model="./checkpoints"
temp=0.0
max_len=2048
pred_num=1
num_seqs_per_iter=1

output_path=preds/T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# Default Sampling: temp=0.2, pred_num=200, num_seqs_per_iter=2
# CUDA_VISIBLE_DEVICES=0 python humaneval_gen.py --model ${model} \
#       --temperature ${temp} --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} \
#       --max_len ${max_len} --output_path ${output_path}

# Greedy Decoding: Also set temp=0.0, pred_num=1, and num_seqs_per_iter=1
CUDA_VISIBLE_DEVICES=0 python humaneval_gen.py --model ${model} \
      --temperature ${temp} --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} \
      --max_len ${max_len} --output_path ${output_path} --greedy_decode

output_path=preds/T${temp}_N${pred_num}

echo 'Output path: '$output_path
python process_humaneval.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt

evaluate_functional_correctness ${output_path}.jsonl