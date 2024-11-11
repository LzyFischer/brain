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
from train.train import train_epoch, evaluate_network
from datasets.Brain_dataset import Brain
from torch_geometric.loader import DataLoader
import sys
import pdb
import wandb

sys.path.append(os.path.dirname((os.path.abspath(__file__))))



def main(args):

    now = str(time.strftime("%Y_%m_%d_%H_%M"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set random seed for reproducing results
    if args.seed:
        utils.seed_it(args.seed)

    # import config files and update args
    config_file = pj("./scripts/configs", args.config + ".json")
    with open(config_file) as f:
        config = json.load(f)
    config["dataset"] = args.dataset
    if args.dropout:
        config["net_params"]["dropout"] = args.dropout
    if args.x_attributes is None:
        args.x_attributes = ["FC", "SC"]

    wandb.init(project="brain", name=f"posw{args.if_pos_weight}_mod{args.modality}_lr{args.lr}")
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
    logger = utils.get_logger(name=save_name, path="results/loggers")
    logger.info(args)
    logger.info(config)

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
    )
    """modify"""
    # non_zero_indices = torch.where(dataset.y.sum(1) != 0)[0]
    # zero_indices = torch.where(dataset.y.sum(1) == 0)[0][: len(non_zero_indices)]
    # non_zero_indices = torch.cat((non_zero_indices, zero_indices), dim=0)
    # dataset = dataset[non_zero_indices]
    """modify end"""
    print("Dataset length: ", len(dataset))


    model_save_dir = pj(abspath("scripts/results"), save_name + ".pth")
    # split the dataset and define the dataloader
    train_size = int(args.train_val_test[0] * len(dataset))
    val_size = int(args.train_val_test[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    """modify"""
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    """modify end"""

    # Prepare the dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    logger.info("#" * 60)

    # Define the model
    model = load_model(config, args)
    model.to(device)
    logger.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    if config["pretrain"]:
        checkpoint = torch.load(abspath(pj("results", config["pretrain_model_name"])))
        model.load_state_dict(checkpoint["model"])
        print("Using pretrained model: ", config["pretrain_model_name"])

    # Training preparation
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10)
    min_test_loss = 1e12
    early_stopping = utils.EarlyStopping(tolerance=10, min_delta=0)

    # Training
    for epoch in range(args.max_epochs):
        epoch_loss_train, optimizer = train_epoch(
            model,
            optimizer,
            device,
            data_loader=train_loader,
            epoch=epoch,
            task_type=args.task,
            logger=logger,
            writer=writer,
            weight_score=args.weight_score,
            args=args,
        )  
        epoch_loss_val = evaluate_network(
            model,
            device,
            val_loader,
            epoch,
            task_type=args.task,
            logger=logger,
            phase="Val",
            writer=writer,
            weight_score=args.weight_score,
            args=args,
        )
        epoch_loss_test = evaluate_network(
            model,
            device,
            test_loader,
            epoch,
            task_type=args.task,
            logger=logger,
            phase="Test",
            writer=writer,
            weight_score=args.weight_score,
            args=args,
        )
        scheduler.step(epoch_loss_val)

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalars(
            main_tag="Loss/epoch_losses",
            tag_scalar_dict={
                "Train": epoch_loss_train,
                "Val": epoch_loss_val,
                "Test": epoch_loss_test,
            },
            global_step=epoch,
        )

        scheduler.step(epoch_loss_train)
        epoch_loss_test = epoch_loss_train
        if epoch_loss_test < min_test_loss:
            min_test_loss = epoch_loss_test
            # model_state = copy.deepcopy(model.state_dict())
            # optimizer_state = copy.deepcopy(optimizer.state_dict())
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        early_stopping(epoch_loss_train, epoch_loss_val)
        if early_stopping.early_stop:
            logger.info("We are at epoch: {}".format(epoch))
            break
    writer.close()
    torch.save(checkpoint, model_save_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="HCDP Classification")
    parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--max_epochs", default=100, type=int, help="max number of epochs"
    )
    parser.add_argument("--L2", default=1e-4, type=float, help="L2 regularization")
    parser.add_argument("--dropout", default=0.1, help="dropout rate")
    parser.add_argument("--seed", default=100, type=int, help="random seed")
    parser.add_argument(
        "--config", default="GIN_classification", help="config file name"
    )
    parser.add_argument(
        "--task",
        default="classification",
        choices=["regression", "classification"],
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
        "--train_val_test", default=[0.7, 0.1, 0.2], help="train, val, test split"
    )
    parser.add_argument("--dataset", default="dglHCP", help="dataset name")
    parser.add_argument("--sparse", default=30, type=int, help="sparsity for knn graph")
    parser.add_argument("--gl", default=False, help="graph learning beta")
    parser.add_argument("--weight_score", default=[0.5, 0.5], help="weight score")
    parser.add_argument("--if_pos_weight", default="False", help="if pos weight", type=str)
    parser.add_argument("--modality", default="FC", help="modality", type=str, choices=["FC", "SC", "Both"])
    parser.add_argument("--task_idx", default=0, help="pretrain", type=int)

    args = parser.parse_args()

    if args.if_pos_weight == "True":
        args.if_pos_weight = True
    else:
        args.if_pos_weight = False

    main(args)
