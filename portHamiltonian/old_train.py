import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial

from utils import *
from portHamiltonian.models.get_model import get_model
from portHamiltonian.datasets.get_dataset import get_dataset
from portHamiltonian.configs.get_config import get_config

from portHamiltonian.models.model_utils import *

# Torch backends
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False

def train_one_epoch(iteration, writer: SummaryWriter, dataloader, model: nn.Module, loss_fn, optimizer):
    global config
    model.train()

    batch_loss = 0.0
    total_norm = 0.0
    for batch, (x0, t, y, u, info) in enumerate(dataloader):
        optimizer.zero_grad()
        model.zero_grad()

        loss = 0.0
        batch_size = x0.shape[0]
        # Integrate ODE for each initial condition
        for i in range(batch_size):
            control_input = lambda tt : input_interp(tt, t[i], u[i]) if len(u[i].shape) > 0 else u[i]
            matrix_input = partial(get_matrix_input(config.system_name, t=t, duty_factor=info, state=x0), idx=i)
            y_pred = model.solve(t[i], x0[i].unsqueeze(0), control_input) if is_baseline_model(model) else model.solve(t[i], x0[i].unsqueeze(0), control_input, matrix_input)
            # Compute loss over full trajectories
            if isinstance(model, NSDE):
                # for NSDE, loss should be average over multiple sampled trajectories.
                for i in range(config.training.np):
                    y_pred = model.solve(t[i], x0[i].unsqueeze(0), control_input) if is_baseline_model(model) else model.solve(t[i], x0[i].unsqueeze(0), control_input, matrix_input)
                    loss += (1/config.training.np) * loss_fn(y_pred.squeeze(), y[i])
            else:
                loss += loss_fn(y_pred.squeeze(), y[i])

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        loss.backward()
        optimizer.step()

        # Compute gradient norm
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        batch_loss += loss.item()
    
    print(f"Iter {iteration} batch loss: {batch_loss}")
    if writer is not None:
        writer.add_scalar("train/loss", batch_loss, iteration)
        writer.add_scalar('train/gradient_norm', total_norm ** 0.5, iteration)

    return batch_loss / len(dataloader)

def validate(iteration, writer, dataloader, model, loss_fn, plot_dir=None, log_scale=True):
    global config
    model.eval()

    ys = []
    y_preds = []
    T = []
    u = []
    info = []
    x0 = None

    for batch, (x,dt,y,uu,i) in enumerate(dataloader):
        # enumerating over trajectories
        if batch == 0: # TODO: only validate over the first trajectory
            x0 = x
            T = dt.squeeze()
            ys = y.squeeze(dim=0)
            u = uu.squeeze()
            info = i.squeeze()
    
    control_input = lambda tt : input_interp(tt, T, u) if len(u.shape) > 0 else u
    matrix_input = get_matrix_input(config.system_name, t=T, duty_factor=info, state=ys, validation=True)
    y_preds = model.solve(T, x0, control_input).squeeze(dim=1) if is_baseline_model(model) else model.solve(T, x0, control_input, matrix_input).squeeze(dim=1)

    val_loss = loss_fn(y_preds, ys) / len(T)
    T = T.tolist()

    num_states = x0.shape[1]
    ys = ys.detach().tolist()
    y_preds = y_preds.detach().tolist()
    
    plot_dir = os.path.join(writer.log_dir, "plots") if plot_dir is None else plot_dir
    plot_path = os.path.join(plot_dir, f"val_{iteration}.png")

    if log_scale:
        ys = np.log(ys)
        y_preds = np.log(y_preds)
    
    plt.plot(T, ys, label=[rf"$x_{i}$" for i in range(num_states)])
    plt.plot(T, y_preds, label=[rf"$\hat x_{i}$" for i in range(num_states)])
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

def train():
    global config
    device = config.training.device
    print(f"Device: {device}")

    Dataset = get_dataset(config.system_name)
    train_data = Dataset(os.path.join(config.data.dir, "train_data.pkl"), device, training=True, dataset=config.data.mode, min_sequence_length=config.data.sequence_length)
    val_data = Dataset(os.path.join(config.data.dir, "val_data.pkl"), device, training=False, dataset=config.data.mode, traj_scale=train_data.traj_scale)
    test_data = Dataset(os.path.join(config.data.dir, "test_data.pkl"), device, training=False, dataset=config.data.mode, traj_scale=train_data.traj_scale)

    log_dir = os.path.join("runs", config.system_name, config.net_cfg.model_name, datetime.now().strftime("%b%d_%H:%M:%S") + "_" + config.experiment_name)
    writer = SummaryWriter(log_dir)
    os.makedirs(os.path.join(writer.log_dir, "plots"))

    scale = (1 / train_data.traj_scale) if config.data.scale else 1.0 # TODO divide by t1 - t0
    if config.net_cfg.model_name in BASELINE_MODELS:
        model = get_model(config.net_cfg.model_name)(scale, config.net_cfg, default_dtype, device)
    else:
        model = get_model(config.net_cfg.model_name)(scale, config.system_matrices, config.net_cfg, default_dtype, device)

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    loss_fn = nn.MSELoss()

    train_dataloader = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    best_val_loss = validate(0, writer, val_dataloader, model, loss_fn, log_scale=True)

    for epoch in range(config.training.total_epochs):
        train_one_epoch(epoch, writer, train_dataloader, model, loss_fn, optimizer)
        if epoch % config.training.val_interval == 0:
            val_loss = validate(epoch, writer, val_dataloader, model, loss_fn)
            if val_loss < best_val_loss:
                print(f"Iter {epoch} val loss: {val_loss} < best val loss {best_val_loss}")
                best_val_loss = val_loss
                print(f"Saving model {model}")
                torch.save(model.state_dict(), os.path.join(writer.log_dir, "best_model.pth"))

                with open(os.path.join(writer.log_dir, "matrices.txt"), 'w') as f:
                    f.write(f"iteration {epoch}")
                    f.write(f"model {model}")
            else:
                print(f"Iter {epoch} val loss: {val_loss} > best val loss {best_val_loss}")
                print(f"model {model}")
        
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
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    config = get_config(args.dataset)
    train()