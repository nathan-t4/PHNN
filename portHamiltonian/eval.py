import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import PortHamiltonianDataset
from model import PHNODE
from train import validate

def eval(mode: str = "open"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = "Mar25_09-51-59_ece-895237.austin.utexas.edu"
    # exp_name = "04-07_16:28:43_closed"
    # exp_name = "Apr07_17:07:32_closed"
    exp_name = "Apr08_11:19:20_closed"
    # exp_name = "Apr08_10:53:59_closed"
    # exp_name = "Apr08_10:53:20_open"
    exp_name = "Apr08_13:29:48_open"
    path = os.path.join(os.curdir, "runs", exp_name, "best_model.pth")
    state_dict = torch.load(path, weights_only=True)

    net_cfg = {"hidden_dim": 16, "in_dim": 2}

    duty = 15 / 35
    r = 30
    # J = torch.tensor([[0, -duty], [duty, 0]], device=device)
    R = torch.tensor([[0, 0], [0, 1/r]], device=device)
    B = torch.tensor([[1], [0]], device=device)
    E = torch.tensor(15.0)

    model = PHNODE(R, B, E, net_cfg, device)
    model.load_state_dict(state_dict)
    model.to(device)

    data_dir = "OpenLoop" if mode == "open" else "ClosedLoop"
    data_dir = os.path.join(os.path.abspath(os.curdir), "data", data_dir)
    test_data = PortHamiltonianDataset(os.path.join(data_dir, "test_data.pkl"), device, training=False, dataset=mode)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    loss_fn = nn.MSELoss()

    eval_loss = validate(0, None, test_dataloader, model, loss_fn, plot_dir=".", log_scale=False)

    print(f"Evaluation Loss: {eval_loss}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="open")
    args = parser.parse_args()

    assert(args.mode in ["open", "closed"]), f"Invalid argument {args.mode}"

    eval(args.mode)