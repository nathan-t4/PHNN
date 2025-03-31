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

from data import PortHamiltonianDataset
from model import PHNODE

# Default to float64 since small scale
torch.set_default_dtype(torch.float64)

def train_one_epoch(iteration, writer: SummaryWriter, dataloader, model: nn.Module, loss_fn, optimizer):
    model.train()

    batch_loss = 0.0
    for batch, (x0, t, y) in enumerate(dataloader):
        optimizer.zero_grad()

        loss = 0.0
        batch_size = x0.shape[0]
        # Integrate ODE for each initial condition

        for i in range(batch_size):
            x_batch = x0[i].unsqueeze(0)
            y_batch = y[i].unsqueeze(0)
            y_pred = odeint(model, x_batch, t[i], method='rk4')
            # Compute loss over full trajectories
            loss += loss_fn(y_pred.squeeze(), y_batch.squeeze())

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
    
    print(f"Iter {iteration} batch loss: {batch_loss}")
    writer.add_scalar("train/loss", batch_loss, iteration)

    return batch_loss / len(dataloader)

def validate(iteration, writer, dataloader, model, loss_fn):
    model.eval()

    ys = []
    y_preds = []
    T = []
    x0 = None

    for batch, (x,dt,y) in enumerate(dataloader):
        # enumerating over trajectories
        if batch == 0: # TODO: only validate over the first trajectory
            x0 = x
            T = dt.squeeze()
            ys = y.squeeze(dim=0)


    y_preds = odeint(model, x0, T).squeeze(dim=1)

    val_loss = loss_fn(y_preds, ys) / len(T)
    T = T.tolist()

    ys = ys.detach().tolist()
    y_preds = y_preds.detach().tolist()
    
    plt.plot(T, np.log(ys), label=[r"$q$", r"$\phi$"])
    plt.plot(T, np.log(y_preds), label=[r"$\hat{q}$", r"$\hat\phi$"])
    plt.legend()
    plt.xlabel(r"Time $[s]$")
    plt.ylabel(r"$\log{x}$")
    plt.savefig(os.path.join(writer.log_dir, "plots", f"val_{iteration}.png"))
    plt.clf()
    plt.close()

    writer.add_scalar("val/loss", val_loss, iteration)

    return val_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = os.path.join(os.path.abspath(os.curdir), "data")
    train_data = PortHamiltonianDataset(os.path.join(data_dir, "train_data.pkl"), device, training=True)
    val_data = PortHamiltonianDataset(os.path.join(data_dir, "val_data.pkl"), device, training=False)
    test_data = PortHamiltonianDataset(os.path.join(data_dir, "test_data.pkl"), device, training=False)

    writer = SummaryWriter()
    os.makedirs(os.path.join(writer.log_dir, "plots"))

    total_epochs = int(1.5e3)
    batch_size = 128
    val_interval = 10
    learning_rate = 4e-3

    net_cfg = {"hidden_dim": 16, "in_dim": 2}

    duty = 15 / 35
    r = 30
    J = torch.tensor([[0, -duty], [duty, 0]], device=device)
    R = torch.tensor([[0, 0], [0, 1/r]], device=device)
    B = torch.tensor([[1], [0]], device=device)
    E = torch.tensor(15.0)

    model = PHNODE(J, R, B, E, net_cfg, device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    best_val_loss = validate(0, writer, val_dataloader, model, loss_fn)
    best_val_loss = 1.0

    for epoch in range(total_epochs):
        train_one_epoch(epoch, writer, train_dataloader, model, loss_fn, optimizer)
        if epoch % val_interval == 0:
            val_loss = validate(epoch, writer, val_dataloader, model, loss_fn)
            if val_loss < best_val_loss:
                print(f"Iter {epoch} val loss: {val_loss} < best val loss {best_val_loss}")
                best_val_loss = val_loss
                print("Saving model...")
                torch.save(model.state_dict(), os.path.join(writer.log_dir, "best_model.pth"))
            else:
                print(f"Iter {epoch} val loss: {val_loss} > best val loss {best_val_loss}")
        
        if best_val_loss < 1e-9:
            print(f"Loss is sufficiently small. Terminating training at epoch {epoch}...")
            break

    writer.flush()
    writer.close()

    test_loss = validate(epoch, writer, test_dataloader, model, loss_fn)
    print(f"Test loss: {test_loss}")
        

if __name__ == "__main__":
    train()