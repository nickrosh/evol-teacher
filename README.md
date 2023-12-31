# Evol-Teacher: Recreating WizardCoder

This repo implements the code generation instruction process described in the [WizardCoder Paper](https://arxiv.org/pdf/2306.08568.pdf). Currently, WizardCoder is one the most performant Code Generation models, being beaten only by ChatGPT. This takes the Code Alpaca 20k dataset and evolves each instruction through a randomly chosen evolution prompt to increase instruction complexity. These prompts range from increase time/space complexity, to increasing requirements, to adding erroneus code to improve robustness, etc. This is done three times with pruning and post processing to remove unwanted instructions and responses. The iterative addition of more complexity gives higher quality and more in-depth instructions than what is ususally generated in Alpaca methods. This, like in the case of WizardCoder and WizardLM, can lead to strong performance that gets very close to RLHF model performance.

`generate_evol.py` allows you to generate an Evolution-Instruct dataset from any instruction dataset in the format `Instruction`/`Response`. Alpaca style datasets that contain `input` fields can be converted to Evolution format with `convert_alpaca_to_evol()`. The high level overview of the evolution process is as follows:
1. A seed instruction is taken and evolved with a randomly chosen evolution prompt using GPT3.5.
2. Responses are generated to each of these new evolved prompts also with GPT3.5.
3. Poor quality instructions and responses are pruned and also prevented from further evolution.
4. This evolution process repeats M times. In the paper and the default value in this repo, M=3.

As described in the paper, I performed this process on the full 20k Code Alpaca dataset with three evolutions, resulting in a total of 80k instruction-response pairs. Over 120,000 API calls were made to OpenAI to create this dataset, and due to the rate limit, it took around three days to complete.

## Getting the full 80k Dataset

The full dataset can be found on [HuggingFace Hub](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1).

You can also easily recreate the full dataset by running `merge_evolutions(output_dir="./data/EvolInstruct-Code-80k/")` within `generate_evol.py`. This will merge the seed dataset and the three evolutions. The evolutions are saved here separately for later analysis on generational differences.

## Fine Tuning

We can instruct-tune a model using this dataset very similarly to Alpaca tuning. Simply run `train.py` with your desired parameters. If you set the model max length to 512, it will have a much smaller memory footprint and you will be able to train faster. I instruct-tuned [ReplitLM](https://github.com/replit/ReplitLM) on the full 80k dataset using the following parameters:
```bash
    --model_name_or_path replit/replit-code-v1-3b \
    --data_path ./data/EvolInstruct-Code-80k/EvolInstruct-Code-80k.json \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2000 \
    --bf16 True \
    --tf32 True
```

The instruction tuned version of the model can be found on [HuggingFace Hub](https://huggingface.co/nickrosh/Evol-Replit-v1)

## Evaluation

When this model is run with Greedy Decoding, it achieves a HumanEval Pass@1 Score of 31.1%, up from the initial value of 21% for ReplitLM. Note that this is not the same foundational model as the original WizardCoder, which was fine tuned on top of StarCoder, which has a score of 34% out of the box. To run HumanEval on this model, follow the package download instructions on the HumanEval repo and then run the following script `sh human_eval.sh`.

```bash
#!/bin/sh
model="./checkpoints" # Path to your model
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
```
Please refer to the WizardCoder repo for an example of how to run HumanEval on multiple GPUs.

## Citation


Thanks to the original WizardCoder team
```
@misc{luo2023wizardcoder,
      title={WizardCoder: Empowering Code Large Language Models with Evol-Instruct}, 
      author={Ziyang Luo and Can Xu and Pu Zhao and Qingfeng Sun and Xiubo Geng and Wenxiang Hu and Chongyang Tao and Jing Ma and Qingwei Lin and Daxin Jiang},
      year={2023},
      eprint={2306.08568},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

And thanks to the original Alpaca team.
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```

And thanks to sahil280114 for the CodeAlpaca project and seed dataset.
```
@misc{codealpaca,
  author = {Sahil Chaudhary},
  title = {Code Alpaca: An Instruction-following LLaMA model for code generation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sahil280114/codealpaca}},
}
```


Naturally, you should also cite the original LLaMA paper [1] and the Self-Instruct paper [2].
Also thanks to [Teknium1](https://github.com/teknium1/stanford_alpaca-replit) for the Replit Training Script. I made some changes for the Evolution Instruct format input.
