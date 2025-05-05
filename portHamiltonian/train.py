import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint

from datetime import datetime

from portHamiltonian.datasets.boost_converter import PortHamiltonianDataset
from model import PHNODE
from utils import *
from configs.get_config import get_config

# Default to float64 since small scale
torch.set_default_dtype(torch.float64)
# The control
E = torch.tensor(15.0)

def train_one_epoch(iteration, writer: SummaryWriter, dataloader, model: nn.Module, loss_fn, optimizer):
    model.train()

    batch_loss = 0.0
    for batch, (x0, t, y, u) in enumerate(dataloader):
        optimizer.zero_grad()

        loss = 0.0
        batch_size = x0.shape[0]
        # Integrate ODE for each initial condition

        for i in range(batch_size):
            control = lambda tt : input_interp(tt, t[i], u[i])
            y_pred = model.solve_ode(t[i], x0[i].unsqueeze(0), E, control)
            # Compute loss over full trajectories
            loss += loss_fn(y_pred.squeeze(), y[i])

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
    
    print(f"Iter {iteration} batch loss: {batch_loss}")
    if writer is not None:
        writer.add_scalar("train/loss", batch_loss, iteration)

    return batch_loss / len(dataloader)

def validate(iteration, writer, dataloader, model, loss_fn, plot_dir=None, log_scale=True):
    model.eval()

    ys = []
    y_preds = []
    T = []
    x0 = None

    for batch, (x,dt,y,u) in enumerate(dataloader):
        # enumerating over trajectories
        if batch == 0: # TODO: only validate over the first trajectory
            x0 = x
            T = dt.squeeze()
            ys = y.squeeze(dim=0)
            u = u.squeeze()

    # y_preds = odeint(model, x0, T).squeeze(dim=1)
    control = lambda t : input_interp(t, T, u)
    y_preds = model.solve_ode(T, x0, E, control).squeeze(dim=1)

    val_loss = loss_fn(y_preds, ys) / len(T)
    T = T.tolist()

    ys = ys.detach().tolist()
    y_preds = y_preds.detach().tolist()
    
    plot_dir = os.path.join(writer.log_dir, "plots") if plot_dir is None else plot_dir
    plot_path = os.path.join(plot_dir, f"val_{iteration}.png")

    if log_scale:
        ys = np.log(ys)
        y_preds = np.log(y_preds)
    
    plt.plot(T, ys, label=[r"$q$", r"$\phi$"])
    plt.plot(T, y_preds, label=[r"$\hat{q}$", r"$\hat\phi$"])
    plt.legend()
    plt.xlabel(r"Time $[s]$")
    if log_scale:
        plt.ylabel(r"$\log{x}$")
    else:
        plt.ylabel(r"$x$")
    plt.savefig(plot_path)
    plt.clf()
    plt.close()

    if writer is not None:
        writer.add_scalar("val/loss", val_loss, iteration)

    return val_loss

def train(config, mode: str = "open"):
    device = config.training.device
    print(f"Device: {device}")

    data_dir = "OpenLoop" if mode == "open" else "ClosedLoop"
    data_dir = os.path.join(os.path.abspath(os.curdir), "data", data_dir)
    train_data = PortHamiltonianDataset(os.path.join(data_dir, "train_data.pkl"), device, training=True, dataset=mode, min_sequence_length=config.data.sequence_length)
    val_data = PortHamiltonianDataset(os.path.join(data_dir, "val_data.pkl"), device, training=False, dataset=mode, traj_scale=train_data.traj_scale)
    test_data = PortHamiltonianDataset(os.path.join(data_dir, "test_data.pkl"), device, training=False, dataset=mode, traj_scale=train_data.traj_scale)

    log_dir = os.path.join("runs", datetime.now().strftime("%b%d_%H:%M:%S") + "_" + mode)
    writer = SummaryWriter(log_dir)
    os.makedirs(os.path.join(writer.log_dir, "plots"))

    scale = 1 / train_data.traj_scale if config.data.scale else 1.0 # TODO divide by t1 - t0
    model = PHNODE(scale, config.system_matrices, config.net_cfg, device)

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    loss_fn = nn.MSELoss()

    train_dataloader = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    best_val_loss = validate(0, writer, val_dataloader, model, loss_fn, log_scale=True)
    best_val_loss = 1.0

    for epoch in range(config.training.total_epochs):
        train_one_epoch(epoch, writer, train_dataloader, model, loss_fn, optimizer)
        if epoch % config.training.val_interval == 0:
            val_loss = validate(epoch, writer, val_dataloader, model, loss_fn)
            if val_loss < best_val_loss:
                print(f"Iter {epoch} val loss: {val_loss} < best val loss {best_val_loss}")
                best_val_loss = val_loss
                print("Saving model...")
                torch.save(model.state_dict(), os.path.join(writer.log_dir, "best_model.pth"))
            else:
                print(f"Iter {epoch} val loss: {val_loss} > best val loss {best_val_loss}")
        
        if best_val_loss < 1e-10:
            print(f"Loss is sufficiently small. Terminating training at epoch {epoch}...")
            break

    writer.flush()
    writer.close()

    test_loss = validate(epoch, writer, test_dataloader, model, loss_fn)
    print(f"Test loss: {test_loss}")
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="open")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    assert(args.mode in ["open", "closed"]), f"Invalid argument {args.mode}"
    assert(args.dataset in ["boost_converter"]), f"Invalid dataset {args.dataset}"
    
    config = get_config(args.dataset)
    train(config, args.mode)