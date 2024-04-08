# -*- coding: utf-8 -*-
# @Time    : 11/6/2023 2:49 PM
# @Author  : Gang Qu
# @FileName: Ablation_Experiments.py

import os

seeds = [200]
models = ['MGCN']
if __name__ == '__main__':
    for model in models:
        for seed in seeds:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_FC --seed {s} --x_attributes FC '.format(
                e=int(50), m=model, s=seed)
            os.system(cmd)
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_SC --seed {s} --x_attributes SC'.format(
                e=int(100), m=model, s=seed)
            os.system(cmd)
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_3.py --max_epochs {e} --config {m}_regression_FCSC  --seed {s} --x_attributes FC --x_attributes SC '.format(
                e=int(60), m=model, s=seed)
            os.system(cmd)
