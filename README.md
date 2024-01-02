# InRank: Incremental Low-Rank Learning
This repository is the official implementation of the "[InRank: Incremental Low-Rank Learning](https://arxiv.org/abs/2306.11250)".

We evaluate InRank-Efficient on GPT-2, and our results indicate that InRank-Efficient achieves comparable prediction performance as the full-rank counterpart while requiring at most 33% of the total ranks throughout training. This renders GPT-2 medium and GPT-2 large models up to 1.7× smaller and attains a 1.6× faster pretraining than full-rank models with even superior prediction performance.

Furthermore, the regularization capabilities of InRank-Efficient contribute to the stability of training GPT-2 medium and GPT-2 large models under single-precision, allowing for the achievement of up to a 3.5× reduction in model size and a 2.2× speedup in pretraining.

We use the template from https://github.com/ashleve/lightning-hydra-template. Please read the instructions there to understand the repo structure.

# Requirements
Make a Docker image from the provided Dockerfile to start the experiments.

# Implementation & Experiments
First, we provide a minimum demo to run GPT-2 on NVIDIA 8 V100 16GB GPUs. You also could change the attribute `trainer.devices` to downsize the number of GPUs to use.  </br>
In the first stage, the intrinsic rank will be determined and the compressed low-rank GPT-2 will be saved in the checkpoint.
```sh
python run.py experiment=wt103/demo-in_rank_efficient_stage_1 trainer.devices=8
```
In the second stage, the low-rank GPT-2 could continue to train.
```sh
python run.py experiment=wt103/demo-in_rank_efficient_stage_2 trainer.devices=8
```

Incremental Low-Rank training on GPT-2 (NVIDIA 8 V100 32GB GPUs):
```sh
# 19.6 ppl, 1.4× smaller, 1.3× faster
python run.py experiment=wt103/gpt2-in_rank_efficient_stage_1
python run.py experiment=wt103/gpt2-in_rank_efficient_stage_2
```

Incremental Low-Rank training on GPT-2 Medium (NVIDIA 8 A100 80GB GPUs): 
```sh
# 20.2 ppl, 1.6× smaller, 1.6× faster
python run.py experiment=wt103/gpt2m-in_rank_efficient_stage_1
python run.py experiment=wt103/gpt2m-in_rank_efficient_stage_2
```

Incremental Low-Rank training on GPT-2 Large (NVIDIA 8 A100 80GB GPUs):
```sh
# 20.4 ppl, 1.7× smaller, 1.6× faster
python run.py experiment=wt103/gpt2l-in_rank_efficient_stage_1
python run.py experiment=wt103/gpt2l-in_rank_efficient_stage_2
```

For comparison, Full-Rank baseline on GPT-2, GPT-2 Medium, GPT-2 Large:
```sh
# 19.3 ppl
python run.py experiment=wt103/gpt2
```
```sh
# 20.6 ppl
python run.py experiment=wt103/gpt2m
```
```sh
# 20.9 ppl
python run.py experiment=wt103/gpt2l
```

For the same Fixed-Rank comparison on GPT-2, GPT-2 Medium, GPT-2 Large:
```sh
# 19.7 ppl
python run.py experiment=wt103/gpt2-in_rank_efficient_stage_1 model.mlp_cfg.linear1_cfg.init_modes=254 model.mlp_cfg.linear1_cfg.buffer_modes=0 model.mlp_cfg.linear1_cfg.warmup_iter=999999999
```
```sh
# 20.5 ppl
python run.py experiment=wt103/gpt2m-in_rank_efficient_stage_1 model.mlp_cfg.linear1_cfg.init_modes=287 model.mlp_cfg.linear1_cfg.buffer_modes=0 model.mlp_cfg.linear1_cfg.warmup_iter=999999999
```
```sh
# 20.8 ppl
python run.py experiment=wt103/gpt2l-in_rank_efficient_stage_1 model.mlp_cfg.linear1_cfg.init_modes=313 model.mlp_cfg.linear1_cfg.buffer_modes=0 model.mlp_cfg.linear1_cfg.warmup_iter=999999999
```

For ablation experiments on Explained Ratio, you just need to change the attribute `model.mlp_cfg.linear1_cfg.explained_ratio_threshold` to control how many ranks you desire:
```sh
# 20.2 ppl, 1.5× smaller, 1.3× faster
python run.py experiment=wt103/gpt2-in_rank_efficient_stage_1 model.mlp_cfg.linear1_cfg.explained_ratio_threshold=0.8
python run.py experiment=wt103/gpt2-in_rank_efficient_stage_2
```

Lastly, we found Full-Rank baseline of GPT-2 Medium and GPT-2 Large is not stable for training under single-precision 16 (will mostly diverge). To be fair comparison, in the above experiments, we use double-precision 32 to compare. However, our InRank algorithm has the capability to regularize the training process to avoid divergence. Therefore, you could further compress the model and speed up the training under single-precision 16 by changing the attribute `trainer.precision`:
```sh
# 20.3 ppl, 3.1× smaller, 2.2× faster
python run.py experiment=wt103/gpt2m-in_rank_efficient_stage_1 trainer.precision=16
python run.py experiment=wt103/gpt2m-in_rank_efficient_stage_2 trainer.precision=16
```
```sh
# 20.5 ppl, 3.5× smaller, 2.2× faster
python run.py experiment=wt103/gpt2l-in_rank_efficient_stage_1 trainer.precision=16
python run.py experiment=wt103/gpt2l-in_rank_efficient_stage_2 trainer.precision=16
```

# Acknowledgement
This repo is directly implemented based on the repo [Fly](https://github.com/HazyResearch/fly) by leveraging on the pretraining framework of foundation models. We are immensely grateful to the authors of that project.

# Citing InRank
If you found the code/scripts here are useful to your work, please cite InRank by
```sh
@inproceedings{zhao2023inrank,
  title={InRank: Incremental Low-Rank Learning},
  author={Zhao, Jiawei and Zhang, Yifei and Chen, Beidi and Schäfer, Florian and Anandkumar, Anima},
  year={2023}
}
```


