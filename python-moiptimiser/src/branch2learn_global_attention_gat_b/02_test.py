"""
Model training script.

File adapted from https://github.com/ds4dm/ecole
by Lars Sandberg @Sandbergo
May 2021
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch_geometric

from utilities.general import Logger
from utilities.model import process
from utilities.data import GraphDataset
from models.mlp import MLP1Policy, MLP2Policy, MLP3Policy
from models.gnn import GNN1Policy, GNN2Policy, GIN1Policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Model name.",
        choices=["gnn1", "gnn2", "mlp1", "mlp2", "mlp3", "gin1"],
    )
    parser.add_argument(
        "-p",
        "--problem",
        help="MILP instance type to process.",
        choices=["KP", "AP", "TSP"],
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="CUDA GPU id (-1 for CPU).",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Random generator seed.",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    LEARNING_RATE = 0.001
    NB_EPOCHS = 100
    PATIENCE = 8
    EARLY_STOPPING = 16
    POLICY_DICT = {
        "mlp1": MLP1Policy(),
        "mlp2": MLP2Policy(),
        "mlp3": MLP3Policy(),
        "gnn1": GNN1Policy(),
        "gnn2": GNN2Policy(),
        "gin1": GIN1Policy(),
    }
    PROBLEM = args.problem

    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        DEVICE = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = POLICY_DICT[args.model].to(DEVICE)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    Path("branch2learn_global_attention_gat_b/log/").mkdir(exist_ok=True)
    log = Logger(filename="branch2learn_global_attention_gat_b/log/02_train")

    log(f"Model:   {args.model}")
    log(f"Problem: {PROBLEM}")
    log(f"Device:  {DEVICE}")
    log(f"Lr:      {LEARNING_RATE}")
    log(f"Epochs:  {NB_EPOCHS}")

    # --- TRAIN --- #
    train_files = [
        str(path)
        for path in Path(f"/data/yxwu/python-moiptimiser/src/training_data_oc_sol/{PROBLEM}").glob(
            "sample_*.pkl"
        )
    ]#[:404915]
    valid_files = [
        str(path)
        for path in Path(f"/data/yxwu/python-moiptimiser/src/valid_data_oc_sol/{PROBLEM}").glob(
            "sample_*.pkl"
        )
    ]




    test_files = [
        str(path)
        for path in Path(f"/data/yxwu/python-moiptimiser/src/valid_data_oc_sol/{PROBLEM}_{str(5)}_{str(100)}_{'test'}").glob(
            "sample_*.pkl"
        )
    ]

    log(
        f"Training with {len(train_files)} samples, validating with {len(valid_files)} samples"
    )

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(
        train_data, batch_size=64, shuffle=True, follow_batch=['x_c', 'x_v', 'x_o']
    )
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(
        valid_data, batch_size=64, shuffle=False, follow_batch=['x_c', 'x_v', 'x_o']
    )

    test_data = GraphDataset(test_files)
    test_loader = torch_geometric.data.DataLoader(
        test_data, batch_size=64, shuffle=False, follow_batch=['x_c', 'x_v', 'x_o']
    )


    model_filename = f"/data/yxwu/python-moiptimiser/src/branch2learn_global_attention_gat_b/models/{args.model}/{args.model}_{PROBLEM}_{'1500s'}.pkl"
    policy.load_state_dict(torch.load(model_filename))
    policy.eval()


    log("Beginning testing")

    policy.load_state_dict(torch.load(model_filename))
    valid_loss, valid_acc = process(
        policy=policy, data_loader=test_loader, device=DEVICE, optimizer=None
    )
    log(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    log("End of training.\n\n")
