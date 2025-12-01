# -*- coding: utf-8 -*-
# @Time    : 10/31/2023 12:09 AM
# @Author  : Gang Qu
# @FileName: main_2.py
# -*- coding: utf-8 -*-
# @Time    : 10/14/2023 6:54 PM
# @Author  : Gang Qu
# @FileName: main_1.py
# Age classification GNN experiments
import argparse
import utils
from tqdm import tqdm
import os
from os.path import join as pj
from os.path import abspath
import json
import datasets
import torch
from models.load_model import load_model
import time
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
from torch_geometric.explain import Explainer, CaptumExplainer, GNNExplainer
from train.train import (
    train_epoch,
    evaluate_network,
    train_epoch_anomaly_detection,
    evaluate_network_anomaly_detection,
)
import sys
from models.explain_model import BrainCFExplainer, SurrogateGCN

sys.path.append(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
import pdb
from torch_geometric.data import Data

from scripts.datasets.Brain_dataset import Brain
from torch_geometric.loader import DataLoader
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)

from imblearn.under_sampling import RandomUnderSampler

import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
import torch.nn as nn

from sklearn.model_selection import KFold
from utils.utils import plot_edge_weight



def main(args):

    now = str(time.strftime("%Y_%m_%d_%H_%M"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # set random seed for reproducing results
    if args.seed:
        utils.seed_it(args.seed)

    # import config files and update args
    config_file = pj("./scripts/configs", args.config + ".json")
    with open(config_file) as f:
        config = json.load(f)
    # combine the config and args

    config["dataset"] = args.data_name
    if args.dropout:
        config["net_params"]["dropout"] = args.dropout
    if args.x_attributes is None:
        args.x_attributes = ["FC", "SC"]

    # log the config and args information
    wandb.init(
        project="brain", name=f"posw{args.if_pos_weight}_mod{args.modality}_lr{args.lr}"
    )
    wandb.config.update(args)
    wandb.config.update(config)

    save_name = (
        now
        + config["dataset"]
        + "_"
        + config["model_save_suffix"]
        + "_"
        + config["model"]
        + "_"
        + args.task
        + "_"
        + str(args.sparse)
    )
    if not os.path.exists(abspath("results/brain")):
        os.makedirs(abspath("results/brain"))
    if not os.path.exists(abspath("results/brain/pretrained")):
        os.makedirs(abspath("results/brain/pretrained"))
    if not os.path.exists(abspath("results/brain/loggers")):
        os.makedirs(abspath("results/brain/loggers"))

    # print the config and args information
    # logger = utils.get_logger(name=save_name, path="results/loggers")
    # logger.info(args)
    # logger.info(config)

    # define tensorboard for visualization of the training
    if not os.path.exists(abspath("results/runs")):
        os.makedirs(abspath("results/runs"))
    writer = SummaryWriter(
        log_dir=pj(abspath("results/runs"), save_name), flush_secs=30
    )

    # define dataset
    dataset = Brain(
        task=args.task,
        x_attributes=args.x_attributes,
        args=args,
        rawdata_path=f"./dataset/processed/{args.data_name}.pkl"
    )
    print("Dataset length: ", len(dataset))

    """
    Dataset definition:
    1. 5 cross validation folds without validation set
    Training:
    1. only Train 3 folds (0, 1, 2)
    start with 
    for fold in range(3):
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(kf.split(dataset))

    test_results = {}  # filled after 1st evaluate call in fold 0
    for fold in range(args.num_run):  # Only train on fold 0, 1, 2
        logger.info(f"{'#' * 20} Fold {fold} {'#' * 20}")

        train_indices, test_indices = folds[fold]
        train_subset = dataset[train_indices]
        test_subset = dataset[test_indices]
        
        # under sample the train dataset
        train_subset = utils.undersample_dataset(train_subset)
        test_subset = utils.undersample_dataset(test_subset)
        # test_subset = utils.oversample_dataset(test_subset)
        # train_subset = utils.oversample_dataset(train_subset)

       # Create PyG DataLoaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        # Model setup
        model = load_model(config, args)
        model.to(device)
        logger.info(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        early_stopping = utils.EarlyStopping(tolerance=10, min_delta=0)

        # Training loop
        # will be filled after first evaluation call, based on returned metric keys
        best_by_metric = None
        best_epoch_by_metric = None
        best_ckpt_by_metric = {}  # optional: keep separate checkpoint per metric
        for epoch in range(args.max_epochs):
            epoch_loss_train, optimizer = train_epoch(
                model=model,
                optimizer=optimizer,
                device=device,
                data_loader=train_loader,
                epoch=epoch,
                task_type=args.task,
                logger=logger,
                writer=writer,
                weight_score=args.weight_score,
                args=args,
            )

            epoch_loss_test, metrics = evaluate_network(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                task_type=args.task,
                logger=logger,
                phase="Test",
                writer=writer,
                weight_score=args.weight_score,
                args=args,
            )

            # bootstrap metric trackers on first pass using the keys we actually get back
            if best_by_metric is None:
                metric_names = list(metrics.keys())  # e.g., ["AUC","Accuracy","F1","Precision","Recall"]
                best_by_metric = {m: float("-inf") for m in metric_names}
                best_epoch_by_metric = {m: -1 for m in metric_names}
                # also initialize the cross-fold aggregator once
                if not test_results:
                    test_results = {m: [] for m in metric_names}

            # update per-metric bests (and optional per-metric checkpoints)
            for m, val in metrics.items():
                if val > best_by_metric[m]:
                    best_by_metric[m] = val
                    best_epoch_by_metric[m] = epoch
                    best_ckpt_by_metric[m] = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "metric": m,
                        "value": val,
                    }

            # (keep your existing scalar logging)
            metric = metrics[args.final_metric]  # if you still want a single “driving” metric for LR plots
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalars(
                main_tag=f"Loss/epoch_losses_fold{fold}",
                tag_scalar_dict={"Train": epoch_loss_train, "Test": epoch_loss_test},
                global_step=epoch,
            )

            early_stopping(epoch_loss_train, epoch_loss_test)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch} for fold {fold}")
                break
        
        # log per-metric best for this fold
        for m in best_by_metric:
            logger.info(f"[Fold {fold}] Best {m}: {best_by_metric[m]:.4f} @ epoch {best_epoch_by_metric[m]}")
            test_results[m].append(best_by_metric[m])

        # OPTIONAL: save separate best checkpoint per metric
        # save_dir_base = pj(abspath("scripts/results"), f"{save_name}_fold{fold}")
        # for m, ckpt in best_ckpt_by_metric.items():
        #     torch.save(ckpt, f"{save_dir_base}_best_{m}.pth")

        writer.close()
        # model_save_dir = pj(abspath("scripts/results"), f"{save_name}_fold{fold}.pth")
        # torch.save(checkpoint, model_save_dir)



        #### Plot feature importance
        # learnable_part = model.learnable_weights
        # # plot the weigth
        # import matplotlib.pyplot as plt
        # plt.style.use("default")
        # x = np.arange(0, len(learnable_part))
        # y = learnable_part.detach().cpu().numpy()
        # x_masked = x[model.node_mask.cpu().numpy() == 1]
        # y_masked = y[model.node_mask.cpu().numpy() == 1]
        # x_unmasked = x[model.node_mask.cpu().numpy() == 0]
        # y_unmasked = y[model.node_mask.cpu().numpy() == 0]
        # plt.scatter(x_masked, y_masked, s=15, alpha=0.7, label=f"fold {fold} auc {best_metric:.4f} avg {np.mean(y_masked):.4f} masked")
        # plt.scatter(x_unmasked, y_unmasked, s=15, alpha=0.7, label=f"fold {fold} auc {best_metric:.4f} avg {np.mean(y_unmasked):.4f} unmasked")
        # plt.xlabel("Node index")
        # plt.ylabel("Learnable weight")
        # plt.tight_layout()
        # plt.legend()
        # plt.savefig(f"learnable_weights_fold{fold}.png")
        # plt.close()


        # plot the edge weight in heatmap
        
        # edge_weight = model.learnable_edge_weights
        # edge_weight = (edge_weight + edge_weight.T) / 2
        # edge_weight = torch.sigmoid(model.learnable_edge_weights) * 2 
        # np.save(f"results/learnable_weight/{args.data_name}_fold{fold}.npy", edge_weight.detach().cpu().numpy())
        # plot_edge_weight(edge_weight, model.edge_mask, fold, best_metric, save_name, suffix=args.data_name)
        
        # # plot the weigth
        # import matplotlib.pyplot as plt
        # plt.style.use("default")
        # plt.figure(figsize=(10, 8))
        # plt.imshow((model.edge_mask * edge_weight).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # # lim
        # plt.clim(0.6, 1.2)
        # plt.title(f"fold {fold} auc {best_metric:.4f} avg {(model.edge_mask * edge_weight).sum() / model.edge_mask.sum()}")
        # plt.xlabel("Node index")
        # plt.ylabel("Node index")
        # plt.tight_layout()
        # plt.savefig(f"edge_weights_fold{fold}_masked.png")
        # plt.close()
        # plt.figure(figsize=(10, 8))
        # plt.imshow(((1 - model.edge_mask) *edge_weight).detach().cpu().numpy(), cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.title(f"fold {fold} auc {best_metric:.4f} avg {((1 - model.edge_mask)*edge_weight).sum() / (1 - model.edge_mask).sum()}")
        # plt.xlabel("Node index")
        # plt.ylabel("Node index")
        # plt.tight_layout()
        # plt.savefig(f"edge_mask_fold{fold}_unmasked.png")
        # plt.close()

    
    for m, vals in test_results.items():
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        logger.info(f"{m}: {mean_v:.4f} ± {std_v:.4f}")
        print(f"{m}: {mean_v:.4f} ± {std_v:.4f}")
    

    # # print top 20 weights
    # top_20 = torch.topk(learnable_part, 20)
    # logger.info(top_20)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="HCDP Classification")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--max_epochs", default=15, type=int, help="max number of epochs"
    )
    parser.add_argument("--L2", default=1e-6, type=float, help="L2 regularization")
    parser.add_argument("--dropout", default=0.1, help="dropout rate")
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--config", default="Transformers_classification", help="config file name"
    )
    parser.add_argument(
        "--task",
        default="classification",
        choices=["regression", "classification", "anomaly_detection"],
        help="task type",
    )
    parser.add_argument("--x_attributes", default=None, help="modalities")
    parser.add_argument(
        "--y_attribute",
        default=["nih_crycogcomp_ageadjusted", "nih_fluidcogcomp_ageadjusted"],
        help="target attribute",
    )
    parser.add_argument(
        "--attributes_group", default=["Age in year"], help="attributes to group"
    )
    parser.add_argument(
        "--train_val_test", default=[0.6, 0.2, 0.2], help="train, val, test split"
    )
    # parser.add_argument("--dataset", default="dglHCP", help="dataset name")
    parser.add_argument("--sparse", default=30, type=int, help="sparsity for knn graph")
    parser.add_argument("--gl", default=False, help="graph learning beta")
    parser.add_argument("--weight_score", default=[0.5, 0.5], help="weight score")
    parser.add_argument(
        "--if_pos_weight", default="False", help="if pos weight", type=str
    )
    parser.add_argument(
        "--modality",
        default="FC",
        help="modality",
        type=str,
        choices=["FC", "SC", "Both"],
    )
    parser.add_argument("--task_idx", default=0, help="pretrain", type=int)
    parser.add_argument(
        "--loss_type",
        default="focal",
        help="loss type",
        choices=["bce", "focal", "weighted_bce", "multi_class"],
    )
    parser.add_argument("--focal_gamma", default=2, help="focal gamma")
    parser.add_argument("--label_index", default=None, type=int, help="y attribute")
    parser.add_argument(
        "--final_metric",
        default="AUC",
        help="final metric for evaluation",
        choices=["AUC", "Accuracy", "F1", "Precision", "Recall"],
    )
    parser.add_argument(
        "--num_run", default=1, type=int, help="number of runs for cross validation"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="device to use for training (default: cuda:0)",
    )
    parser.add_argument("--data_name", type=str, default="disease_Anx_nosf", help="Unique ID for the dataset")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for FC edge construction")
    parser.add_argument("--original_edge_weight", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--attention_weights_path", type=float, default=0.5, help="Edge weight for FC edge construction")
    parser.add_argument("--disable_mutual_distill", default=False,
                    help="If set, remove mutual distillation for ablation.")
    parser.add_argument("--use_global", default=True,
                    help="If set, use global mask for classification.")
    parser.add_argument("--use_personal", default=True,
                    help="Ablation: remove APF personalized rank-1 mask (hypernet).")
    args = parser.parse_args()

    if args.if_pos_weight == "True":
        args.if_pos_weight = True
    else:
        args.if_pos_weight = False

    main(args)





