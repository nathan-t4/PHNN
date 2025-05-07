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
from utils import *
from portHamiltonian.models.get_model import get_model
from portHamiltonian.datasets.get_dataset import get_dataset
from portHamiltonian.configs.get_config import get_config

# Torch backends
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
# The control
control_input = torch.tensor(15.0)

def train_one_epoch(iteration, writer: SummaryWriter, dataloader, model: nn.Module, loss_fn, optimizer):
    model.train()

    batch_loss = 0.0
    total_norm = 0.0
    for batch, (x0, t, y, u) in enumerate(dataloader):
        optimizer.zero_grad()

        loss = 0.0
        batch_size = x0.shape[0]
        # Integrate ODE for each initial condition

        for i in range(batch_size):
            matrix_input = lambda tt : input_interp(tt, t[i], u[i])
            y_pred = model.solve_ode(t[i], x0[i].unsqueeze(0), control_input, matrix_input)
            # Compute loss over full trajectories
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
    matrix_input = lambda t : input_interp(t, T, u)
    y_preds = model.solve_ode(T, x0, control_input, matrix_input).squeeze(dim=1)

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

def train(config):
    device = config.training.device
    print(f"Device: {device}")

    Dataset = get_dataset(config.system_name)
    train_data = Dataset(os.path.join(config.data.dir, "train_data.pkl"), device, training=True, dataset=config.data.mode, min_sequence_length=config.data.sequence_length)
    val_data = Dataset(os.path.join(config.data.dir, "val_data.pkl"), device, training=False, dataset=config.data.mode, traj_scale=train_data.traj_scale)
    test_data = Dataset(os.path.join(config.data.dir, "test_data.pkl"), device, training=False, dataset=config.data.mode, traj_scale=train_data.traj_scale)

    log_dir = os.path.join("runs", datetime.now().strftime("%b%d_%H:%M:%S") + "_" + config.system_name + "_" + config.experiment_name)
    writer = SummaryWriter(log_dir)
    os.makedirs(os.path.join(writer.log_dir, "plots"))

    scale = 1 / train_data.traj_scale if config.data.scale else 1.0 # TODO divide by t1 - t0
    model = get_model(config.net_cfg.model_name)(scale, config.system_matrices, config.net_cfg, device)

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
    train(config)