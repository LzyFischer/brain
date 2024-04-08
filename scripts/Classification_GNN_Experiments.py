# -*- coding: utf-8 -*-
# @Time    : 11/6/2023 7:53 AM
# @Author  : Gang Qu
# @FileName: Classification_GNN_Experiments.py
# -*- coding: utf-8 -*-
# @Time    : 11/3/2023 8:25 PM
# @Author  : Gang Qu
# @FileName: Regression_GNN_Experiments.py
import os

seeds = [10]
models = ['GCN']
if __name__ == '__main__':
    for model in models:
        for seed in seeds:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_2.py --max_epochs {e} --config {m}_classification --seed {s}'.format(e=int(100), m=model, s=seed)
            os.system(cmd)
